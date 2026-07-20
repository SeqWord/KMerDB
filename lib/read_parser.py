from __future__ import annotations
import os, io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Union, TextIO, Optional, List

Fileish = Union[str, Path, TextIO, io.BufferedReader]

@dataclass(frozen=True)
class FastaRecord:
    """Minimal SeqIO-like record for FASTA."""
    id: str            # first token of header line (after '>')
    description: str   # full header line without leading '>'
    seq: str           # sequence (concatenated over multiple lines)


def _open_maybe_gzip(handle: Fileish) -> TextIO:
    """
    Accepts a path/Path or an already-open text stream and returns a text stream.
    Supports .gz transparently via lazy import. Caller closes only if a path was given.
    """
    if hasattr(handle, "read"):
        # Already an open text stream; just use it
        return handle  # type: ignore[return-value]

    path = Path(handle)  # type: ignore[arg-type]
    if path.suffix.lower() == ".gz":
        # Lazy import; fail clearly if gzip is unavailable
        try:
            import gzip  # type: ignore
        except Exception as e:
            raise RuntimeError("GZ files cannot be processed: gzip module is unavailable.") from e
        # Open gzip in text mode (universal newlines)
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", newline=None)  # type: ignore[name-defined]
    else:
        return open(path, "rt", encoding="utf-8", newline=None)


def iterate_sequences(handle: Fileish) -> Iterator[Union[FastaRecord, str]]:
    """
    Dispatch to FASTA or FASTQ iterator based on file extension.
    Supports .fa, .fasta, .fq, .fastq and their .gz variants.
    Yields FastaRecord for FASTA; yields str (sequence only) for FASTQ.
    """
    # If it's an open stream, try to use its .name for dispatch; otherwise require a path/str
    name: Optional[str] = None
    if hasattr(handle, "read"):
        name = getattr(handle, "name", None)
        if not name:
            raise TypeError("Cannot infer file type from a stream without a .name; pass a path or set .name.")
    else:
        name = str(handle)  # type: ignore[arg-type]

    p = Path(name)
    suffs = [s.lower() for s in p.suffixes]
    is_gz = suffs and suffs[-1] == ".gz"
    base_ext = (suffs[-2] if is_gz and len(suffs) >= 2 else (suffs[-1] if suffs else "")).lower()

    # If gzipped, ensure gzip is importable now (Windows is fine; this just guards odd builds)
    if is_gz:
        try:
            import gzip  # noqa: F401
        except Exception as e:
            raise RuntimeError("GZ files cannot be processed: gzip module is unavailable.") from e

    fasta_exts = {".fa", ".fasta", ".fst", ".fsa"}
    fastq_exts = {".fq", ".fastq"}

    if base_ext in fasta_exts:
        return iterate_fasta(handle)
    if base_ext in fastq_exts:
        return iterate_fastq(handle)

    raise TypeError(f"File {p} unrecognized as a FASTA/FASTQ (extensions seen: {suffs}).")


def iterate_fasta(handle: Fileish) -> Iterator[FastaRecord]:
    """Streaming FASTA parser (no external deps)."""
    f = _open_maybe_gzip(handle)
    close_after = not hasattr(handle, "read")
    try:
        header: Optional[str] = None
        seq_buf: List[str] = []
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            if line.startswith(">"):
                # flush previous record
                if header is not None:
                    desc = header[1:].strip()
                    rec_id = desc.split(None, 1)[0] if desc else ""
                    yield FastaRecord(id=rec_id, description=desc, seq="".join(seq_buf))
                header = line
                seq_buf = []
            else:
                if header is None:
                    raise ValueError("FASTA format error: sequence line before any header ('>').")
                seq_buf.append(line.strip())
        # flush last record if any
        if header is not None:
            desc = header[1:].strip()
            rec_id = desc.split(None, 1)[0] if desc else ""
            yield FastaRecord(id=rec_id, description=desc, seq="".join(seq_buf))
    finally:
        if close_after:
            f.close()

#### EXAMPLES
'''
# UNKNOWN
for rec in iterate_sequences("reads.*"):
    print(rec.id, len(rec.seq))

# FASTA
for rec in iterate_fasta("reads.fa.gz"):
    print(rec.id, len(rec.seq))

# FASTQ (only sequences)
for seq in iterate_fastq("reads.fastq"):
    print(len(seq))
'''
