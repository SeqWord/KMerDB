#!/usr/bin/env python3
"""Benchmark DNA-fragment identification across classifier threshold grids."""
import argparse
import csv
from decimal import Decimal
from pathlib import Path
import sys

from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LIB = Path(__file__).resolve().parent.parent / 'lib'
sys.path.insert(0, str(LIB))
import tools

__version__ = '1.3'


def threshold_grid(start, stop, step):
    a, b, delta = map(lambda v: Decimal(str(v)), (start, stop, step))
    if not (Decimal(0) <= a <= b <= Decimal(1)) or delta <= 0:
        raise ValueError('Threshold ranges must satisfy 0 <= min <= max <= 1 and step > 0')
    values = []
    x = a
    while x <= b:
        values.append(float(x))
        x += delta
    return values


def tag(value):
    return f'{value:.4f}'.rstrip('0').rstrip('.')


def matrix_to_text(matrix):
    lines = ['Record\tTaxon\tIdentity\tPath\tSpecies level']
    for entry in matrix:
        for title, predictions in entry.items():
            lines.append(str(title))
            if predictions and isinstance(predictions[0], (list, tuple)):
                for prediction in sorted(predictions, key=lambda row: float(row[1]), reverse=True):
                    if len(prediction) == 4:
                        lines.append('\t' + '\t'.join(map(str, prediction)))
    return '\n'.join(lines) + '\n'


def taxon_at_level(name: str, level: str) -> str:
    """Return the genus or binomial species name from a taxon string."""
    parts = name.strip().split()
    required = 1 if level == 'genus' else 2
    return ' '.join(parts[:required]) if len(parts) >= required else ''


def truth_from_title(title, trained, level):
    """Extract the query taxon, irrespective of whether it is in training."""
    return taxon_at_level(title, level)


def evaluate(matrix, trained, taxonomic_level='species'):
    trained_taxa = {taxon_at_level(name, taxonomic_level) for name in trained}
    trained_taxa.discard('')
    counts = dict(TP=0, FP=0, TN=0, FN=0)
    details = []
    for entry in matrix:
        for title, predictions in entry.items():
            truth = truth_from_title(str(title), trained, taxonomic_level)
            valid = []
            if predictions and isinstance(predictions[0], (list, tuple)):
                for row in predictions:
                    if (len(row) >= 4
                            and str(row[3]).strip().lower() == 'true'
                            and str(row[0]).strip() not in ('', '0')):
                        valid.append(str(row[0]).strip())
            predicted_taxa = {
                taxon_at_level(name, taxonomic_level) for name in valid
            }
            predicted_taxa.discard('')
            # Each read contributes exactly once to the confusion matrix.
            if truth in trained_taxa:
                result = 'TP' if truth in predicted_taxa else 'FN'
            else:
                result = 'FP' if predicted_taxa else 'TN'
            counts[result] += 1
            details.append((str(title), truth, '; '.join(valid),
                            '; '.join(sorted(predicted_taxa)), result))
    tp, fp, tn, fn = (counts[k] for k in ('TP', 'FP', 'TN', 'FN'))

    def ratio(n, d):
        return n / d if d else float('nan')

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    balanced = (recall + specificity) / 2
    metrics = dict(counts, N=tp + fp + tn + fn, Precision=precision,
                   Recall=recall, Specificity=specificity, F1=f1,
                   Balanced_accuracy=balanced, FPR=ratio(fp, fp + tn))
    return metrics, details


def save_tsv(path, header, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(header)
        writer.writerows(rows)


def draw_metrics(rows, path, accuracy):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    xs = [r['specificity_threshold'] for r in rows]
    for key in ('F1', 'Precision', 'Recall', 'Specificity', 'Balanced_accuracy'):
        ax.plot(xs, [r[key] for r in rows], marker='o', label=key.replace('_', ' '))
    ax.set(xlabel='Classifier specificity threshold', ylabel='Performance metric',
           title=f'Performance vs specificity threshold (accuracy={tag(accuracy)})', ylim=(-0.02, 1.02))
    ax.grid(alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def draw_roc(rows, path, accuracy):
    # Empirical operating points: varying the classifier's specificity
    # threshold; no interpolated per-read score or invented endpoints.
    points = sorted((r['FPR'], r['Recall'], r['specificity_threshold']) for r in rows)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Chance')
    ax.plot([p[0] for p in points], [p[1] for p in points], '-o', label='Observed operating points')
    for fpr, tpr, threshold in points:
        ax.annotate(tag(threshold), (fpr, tpr), xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.set(xlabel='False positive rate (1 − specificity)', ylabel='True positive rate (recall)',
           title=f'ROC operating points (accuracy={tag(accuracy)})', xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--input', default='input')
    parser.add_argument('-o', '--output', default='output')
    parser.add_argument('-f', '--db_folder', default='db')
    parser.add_argument('-d', '--db_file', required=True)
    parser.add_argument('-r', '--reads', required=True)
    parser.add_argument('-p', '--project', default='')
    parser.add_argument('-m', '--min_read_length', type=int, default=5000)
    parser.add_argument('-x', '--max_read_length', type=int, default=0)
    parser.add_argument('--accuracy_min', type=float, default=0.6)
    parser.add_argument('--accuracy_max', type=float, default=0.7)
    parser.add_argument('--accuracy_step', type=float, default=0.05)
    parser.add_argument('--specificity_min', type=float, default=0.7)
    parser.add_argument('--specificity_max', type=float, default=0.9)
    parser.add_argument('--specificity_step', type=float, default=0.02)
    parser.add_argument('--taxonomic_level', choices=('species', 'genus'),
                        default='genus', help='Taxonomic rank used for performance evaluation')
    parser.add_argument('-n', '--use_NN', type=tools.str2bool, default=True)
    parser.add_argument('-e', '--entropy_cutoff', type=float, default=0.5)
    parser.add_argument('-v', '--version', action='version', version=__version__)
    args = parser.parse_args()
    try:
        accuracies = threshold_grid(args.accuracy_min, args.accuracy_max, args.accuracy_step)
        specificities = threshold_grid(args.specificity_min, args.specificity_max, args.specificity_step)
    except ValueError as exc:
        parser.error(str(exc))
    if args.min_read_length <= 0 or args.max_read_length < 0 or (args.max_read_length and args.max_read_length <= args.min_read_length):
        parser.error('Invalid read length bounds')
    if not 0 <= args.entropy_cutoff <= 1:
        parser.error('--entropy_cutoff must be in [0, 1]')
    reads_path = Path(args.input) / args.reads
    db_path = Path(args.db_folder) / args.db_file
    if not reads_path.is_file() or not db_path.is_file():
        parser.error(f'Missing input file: reads={reads_path}, database={db_path}')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    project = f'{args.project or reads_path.stem}_{args.taxonomic_level}'
    _, db, _ = tools.openDBFile(str(db_path))
    trained = set(db.get_leaf_titles())
    print(f'Trained species: {len(trained)}; evaluation level: {args.taxonomic_level}; grid: {len(accuracies)} x {len(specificities)}')
    fields = ['accuracy_threshold', 'specificity_threshold', 'TP', 'FP', 'TN', 'FN', 'N',
              'F1', 'Precision', 'Recall', 'Specificity', 'Balanced_accuracy', 'FPR']
    all_rows = []
    with tqdm(accuracies, desc='Accuracy', position=0, leave=True,
              unit='value', dynamic_ncols=True) as accuracy_bar:
      for accuracy in accuracy_bar:
        rows = []
        with tqdm(specificities, desc=f'Specificity (accuracy={tag(accuracy)})',
                  position=1, leave=False, unit='value', dynamic_ncols=True) as specificity_bar:
          for specificity in specificity_bar:
                prefix = out / f'{project}_accuracy_{tag(accuracy)}_specificity_{tag(specificity)}'
                # Reopen the reads file on each run: identify() consumes the stream.
                with reads_path.open('r', encoding='utf-8') as handle:
                    matrix, log = db.identify(
                        handle, min_length=args.min_read_length, max_length=args.max_read_length,
                        specificity=specificity, accuracy=accuracy,
                        flg_use_NN=args.use_NN, entropy_cutoff=args.entropy_cutoff)
                (Path(str(prefix) + '.txt')).write_text(matrix_to_text(matrix), encoding='utf-8')
                (Path(str(prefix) + '.log.txt')).write_text('\n'.join(map(str, log)) + '\n', encoding='utf-8')
                metrics, details = evaluate(matrix, trained, args.taxonomic_level)
                save_tsv(Path(str(prefix) + '_classification.tsv'),
                         ['Record', 'Expected_taxon', 'Predicted_labels', 'Predicted_taxa', 'Outcome'], details)
                row = dict(accuracy_threshold=accuracy, specificity_threshold=specificity, **metrics)
                rows.append(row)
                all_rows.append(row)
                specificity_bar.set_postfix(F1=f"{row['F1']:.3f}", TP=row['TP'], FP=row['FP'], refresh=False)
          stem = out / f'{project}_accuracy_{tag(accuracy)}'
          save_tsv(Path(str(stem) + '_metrics.tsv'), fields, ([r[k] for k in fields] for r in rows))
          draw_metrics(rows, Path(str(stem) + '_performance.png'), accuracy)
          draw_roc(rows, Path(str(stem) + '_ROC.png'), accuracy)
    save_tsv(out / f'{project}_all_metrics.tsv', fields,
             ([r[k] for k in fields] for r in all_rows))
    print(f'Finished. Results: {out.resolve()}')


if __name__ == '__main__':
    main()
