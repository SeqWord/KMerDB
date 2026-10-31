import sys, os, shutil
from pathlib import Path, PurePath
from classifier import main as classify
import pandas as pd
from collections import defaultdict
from typing import List, Dict

# In-house modules
import tools, wdb_reader, CSV_merger
from dendro import Root as tree

class kmer_selector:
    def __init__(self, input_path: str, out_path: str, tmp_path: str, project: str, 
            min_k: int = 4, max_k: int = 8, flg_use_NN: bool = True, mmcs: int = 30,    # mmcs - Minimal Median Cluster Size to use Neural Network (NN)
            min_selected: int = 10, top_selected: int = 100,
            diverse_bottom: float = -0.5, diverse_top: float = 0.5, 
            common_bottom: float = -1.0, common_top: float = 1.0, level_increment: float = 0,
            filter_settings: dict = {}):
        self.input_path = input_path
        self.out_path = out_path
        self.tmp_path = tmp_path
        self.project_folder = self.project_name = project
        self.filter_settings = filter_settings
        self.min_k = min_k
        self.max_k = max_k
        self.flg_use_NN = flg_use_NN
        self.median_cluster_size = int(mmcs)
        self.min_selected = min_selected
        self.top_selected = top_selected
        self.oTree = None
        self.failures = []
        self.diverse_bottom = diverse_bottom
        self.diverse_top = diverse_top
        self.common_bottom = common_bottom
        self.common_top = common_top
        self.level_increment = level_increment
        self.file_references = []   # Locations of genomes with accession numbers in different WDB files: [[accession, file_name], ...]
        self.total = 0              # Total number of processed branches
        
    def leaf_wdb_dirs(self):
        """
        Yield all 'leaf' directories (no subdirectories) under `root` that contain ≥1 *.wdb file.
        """
        for dirpath, dirnames, filenames in os.walk(self.input_path):
            # Leaf = no subdirectories
            if dirnames:
                continue
            if any(fn.lower().endswith(".wdb") for fn in filenames):
                yield dirpath

    def execute(self, flg_print_classifier: bool = False, flg_step_by_step: bool = False):
        # === Walk the input tree and process each qualifying leaf directory ===
        leaves = list(self.leaf_wdb_dirs())
        if not leaves:
            print(f"\n❌ No leaf directories with *.wdb files found under: {self.input_path}")
            sys.exit(1)
    
        print(f"🔎 Found {len(leaves)} leaf folder(s) with *.wdb files.\n")
    
        # Create a Tree object
        self.oTree = tree(description=self.project_name)
    
        # Root of the tmp mirror for this project
        tmp_root = self.tmp_path
        # Clean TMP directory if exists
        tools.clean(tmp_root)
        
        os.makedirs(tmp_root, exist_ok=True)
        
        # STEP 1: Process leaves
        self.process_leaves(leaves=leaves, tmp_root=tmp_root)
    
        if flg_step_by_step:
            print("\n\tSTEP 1, WDB->CSV conversion has finished with merging matrices into divers and common \n\t" + 
                f"feature matrices saved in terminal folders of the tree {self.tmp_path}.")
            print("\n\tNext step: calculate intermediate divers and common k-mers.\n")
            input("Press 'ENTER' to continue...")
        
        print("\nSummarise intermediate nodes...\n")
        
        # Collect end-node addresses from oTree
        addresses = [addr for addr in self.oTree.get_addresses() if addr.find(">") > 0]
        
        # STEP 2: Summarize k-mer selections for the intermediate nodes
        self.summarize_nodes(tmp_root, addresses)
        print()
        
        if flg_step_by_step:
            print("\n\tSTEP 2, divers and common k-mers were calculated for \n\t" + 
                f"inytermediate folders of the tree {self.tmp_path}.")
            print("\n\tNext step: identify the most powerful k-mers using random forest clasifier.\n")
            input("Press 'ENTER' to continue...")
        
        # STEP 3: Run classifier (Random Forest) for diverse matrix
        oClassifier = self.classify_nodes(tmp_root, addresses)
        print()
        
        if flg_step_by_step:
            print("\n\tSTEP 3, the most powerful k-mers were identified using random forest clasifier. \n\t" + 
                f"Output files chi2_fdr_results.csv were saved in folders of the tree {self.tmp_path}.")
            print("\n\tNext step: save model file " + f"{self.project_folder}.pkl to folder {self.out_path}.")
            input("\nPress 'ENTER' to continue...")
        
        # Summary
        ok = self.total - len([1 for x in self.failures if os.path.isdir(x[0]) is False])  # count only leaf processing failures
        print(f"\n📊 Summary: {ok}/{self.total} leaf folders processed successfully.")
        if self.failures:
            print("⚠️ Failures:")
            for path_, err in self.failures:
                print(f"   - {path_}: {err}")
                
        if flg_print_classifier:
            # Save classifier model as text to temprary folder
            # Function self.oTree.classifier_to_string() returns a list of lines
            tools.saveTextFile("\n".join(self.oTree.classifier_to_string()), os.path.join(self.out_path, f"{self.project_name}_classifier.txt"))
        # Save classifier as binary objects
        tools.saveDBFile(self.oTree, os.path.join(self.out_path, f"{self.project_name}.pkl"))

    def process_leaves(self, leaves: list, tmp_root: str):
        # === LEAVES: build matrices + leaf-level CSV merge
        for leaf in sorted(leaves):  # leaf is an absolute path to a terminal folder
            # Collect only *.wdb files
            try:
                wdb_files = [fn for fn in os.listdir(leaf) if fn.lower().endswith(".wdb")]
            except FileNotFoundError:
                wdb_files = []
    
            if not wdb_files:
                # Skip if this leaf has no *.wdb after all (defensive)
                continue
    
            rel = os.path.relpath(leaf, start=self.input_path)  # relative path under the project root
            leaf_out = os.path.join(tmp_root, rel)
    
            print(f"Folder {rel} is processed...")
    
            # Create a tree pathway
            pathway = [["root", 1, None]] + [[node, 1, None] for node in rel.split(os.sep)]
            pathway[-1].append(",".join(wdb_files))  # attach file list to the terminal node
            self.oTree.append(pathway)
    
            # Ensure a clean leaf-specific output dir
            tools.clean(leaf_out)
            os.makedirs(leaf_out, exist_ok=True)
    
            print(f"🧩 Processing leaf: {leaf}")
            print(f"   ↳ Output: {leaf_out}")
    
            success, error, file_references = wdb_reader.create_kmer_matrices(
                input_folder=leaf,
                output_folder=leaf_out,
                min_k=self.min_k,
                max_k=self.max_k,
            )
            self.total += 1
            if not success:
                print(f"   ❌ Error: {error}")
                self.failures.append((leaf, error))
            else:
                print(f"   ✅ Done")
            
            # Concatenate file references: [[accession, file_name], ...]
            self.file_references += file_references
    
            # Merge CSVs within this leaf's output folder
            CSV_merger.execute(input_folder=leaf_out, output_dir=leaf_out, filter_settings=self.filter_settings,
                diverse_bottom = self.diverse_bottom,
                diverse_top = self.diverse_top,
                common_bottom = self.common_bottom,
                common_top = self.common_top, 
            )
            
    def address_to_path(self, address: str, root_path: str) -> str:
        # Split on '>', drop empties/spaces
        parts = [p.strip() for p in address.split(">") if p.strip()]
        # If it starts with 'root', drop that marker
        if parts and parts[0].lower() == "root":
            parts = parts[1:]
        # Join whatever remains under root_path
        return str(Path(root_path, *parts))    

    def summarize_nodes(self, root_path: str, addresses: list):
        # Convert addresses into paths
        # address: root>folder_1>folder_2>...
        # root should be replaces with root_path
        # Paths are sorted by intermediate folders
        
        paths = sorted(
            [os.path.normpath(self.address_to_path(addr, root_path)) for addr in addresses if addr],
            key=lambda p: Path(p).parts
        )
        
        # Remove end-folders
        parent_dirs = list(dict.fromkeys(
            os.path.normpath(os.path.dirname(p)) for p in paths
        ))
               
        parent_folder = None
        # All paths in the list lead to end folders. Paths are 
        i = 0
        depth = None
        while i < len(parent_dirs):
            cur_path = parent_dirs[i]
            if depth is None:
                depth = len(cur_path.split(os.sep))
            elif len(cur_path.split(os.sep)) < depth:
                # Update thresholds for upper level (side effect preserved from original)
                self.diverse_bottom -= self.level_increment
                self.diverse_top += self.level_increment
                self.common_bottom -= self.level_increment
                self.common_top += self.level_increment
                depth = len(cur_path.split(os.sep))
            elif len(cur_path.split(os.sep)) > depth:
                # Update thresholds for lower level (side effect preserved from original)
                self.diverse_bottom += self.level_increment
                self.diverse_top -= self.level_increment
                self.common_bottom += self.level_increment
                self.common_top -= self.level_increment
                depth = len(cur_path.split(os.sep))
                
            if cur_path == root_path:
                i += 1
                continue
            intermediate_folder = os.path.dirname(cur_path)
            # If new intermediate (and not the project root), queue it for later
            if intermediate_folder != parent_folder and intermediate_folder != root_path:
                paths.append(intermediate_folder)
                parent_folder = intermediate_folder
    
            # Summarize current node (leaf or appended intermediate)
            try:
                print(f"🧮 Summarizing node: {cur_path}")
                CSV_merger.summarize_node(
                    dirpath=cur_path,
                    _retrieve_value=self._retrieve_value,
                    diverse_bottom = self.diverse_bottom,
                    diverse_top = self.diverse_top,
                    common_bottom = self.common_bottom,
                    common_top = self.common_top,
                    min_selected=self.min_selected, top_selected=self.top_selected,
                    flg_relabel = True
                    )
            except Exception as e:
                print(f"   ⚠️ summarize_node failed for {cur_path}: {e}")
                self.failures.append((cur_path, f"summarize_node: {e}"))
            
            i += 1
                
        # Final run for root_path
        try:
            print(f"🧮 Summarizing node: {root_path}")
            CSV_merger.summarize_node(
                dirpath=root_path,
                _retrieve_value=self._retrieve_value,
                diverse_bottom = self.diverse_bottom,
                diverse_top = self.diverse_top,
                common_bottom = self.common_bottom,
                common_top = self.common_top,
                min_selected=self.min_selected, top_selected=self.top_selected,
                flg_relabel = True
                )
        except Exception as e:
            print(f"   ⚠️ summarize_node failed for {root_path}: {e}")
            self.failures.append((root_path, f"summarize_node: {e}"))          
          
    def _retrieve_value(self, acc_list: List[str], marker_list: List[str]) -> pd.DataFrame:
        """
        Batch-retrieve values for (accession, marker) pairs.
    
        Uses self.file_references = [[accession, file_name], ...] to group accessions
        by source file, calls wdb_reader.get_values(file_name, acc_sublist, marker_list)
        per file, and assembles a single DataFrame with the same row/column order as
        (acc_list, marker_list). Cells default to "0" and are overwritten by retrieved values.
    
        Returns
        -------
        pd.DataFrame
            Shape = (len(acc_list), len(marker_list)), index = acc_list, columns = marker_list.
        """
        # 1) Initialize the full output matrix filled with "0" (as requested)
        result = pd.DataFrame("0", index=list(acc_list), columns=list(marker_list))
    
        # 2) Build a fast lookup: accession -> file_name, from self.file_references
        #    If multiple entries per accession exist, the last one wins (adjust if needed).
        acc_to_file: Dict[str, str] = {}
        for acc, fname in self.file_references:
            acc_to_file[str(acc)] = str(fname)
    
        # 3) Group requested accessions by their file_name (preserving acc_list order per group)
        file_groups: Dict[str, List[str]] = defaultdict(list)
        for acc in acc_list:
            fname = acc_to_file.get(str(acc))
            if fname:
                file_groups[fname].append(str(acc))
            # If an accession has no mapping, we leave its row as "0"s and continue.
    
        # 4) For each file group, fetch the submatrix and fill into result
        #    Assumes an imported module `wdb_reader` with function:
        #    get_values(file_name: str, acc_list: List[str], marker_list: List[str]) -> pd.DataFrame
    
        for fname, acc_sublist in file_groups.items():
            if not acc_sublist:
                continue
    
            try:
                 sub_df = pd.DataFrame(values, index=acc_sublist, columns=marker_list, dtype="string")
            except Exception:
                # If retrieval fails for this file, leave those rows as "0"
                continue
    
            if not isinstance(sub_df, pd.DataFrame):
                # Enforce DataFrame type; skip on unexpected returns
                continue
    
            # Ensure sub_df has exactly the expected index/order and columns/order
            # (rows must match acc_sublist order; columns must match marker_list order)
            sub_df = sub_df.reindex(index=acc_sublist, columns=marker_list)
    
            # Fill back into the corresponding rows/columns of the full matrix.
            # Use .loc assignment; pandas will align 1:1 by index/columns.
            result.loc[acc_sublist, marker_list] = sub_df.values
    
        return result
    

    def classify_nodes(self, root_path: str, addresses: list):
        # Convert addresses into paths
        # address: root>folder_1>folder_2>...
        # root should be replaces with root_path
        # Paths are sorted by intermediate folders
        paths = sorted(
            [os.path.normpath(self.address_to_path(addr, root_path)) for addr in addresses if addr],
            key=lambda p: Path(p).parts
        )
        
        parent_folder = None
        # All paths in the list lead to end folders. Paths are 
        i = 0
        while i < len(paths):
            cur_path = paths[i]
            if cur_path == root_path:
                i += 1
                continue
            intermediate_folder = os.path.dirname(cur_path)
            # If new intermediate (and not the project root), queue it for later
            if intermediate_folder != parent_folder and intermediate_folder != root_path:
                paths.append(intermediate_folder)
                parent_folder = intermediate_folder
    
            # Summarize current node (leaf or appended intermediate)
            decision_model, df_matrix, median_cluster_size = classify(inpath=cur_path, file_name="diverse_features.csv", # data = tuple[np.ndarray, np.ndarray] (labels, data)
                min_selected=self.min_selected, top_selected=self.top_selected)

            try:
                print(f"🧮 Classify node: {cur_path}")
                '''
                min_selected = 0
                path_length = len(cur_path.split(os.sep))
                if path_length < 3:
                    min_selected = 10 * (3 - path_length)
                '''
                decision_model, df_matrix, median_cluster_size = classify(inpath=cur_path, file_name="diverse_features.csv", # data = tuple[np.ndarray, np.ndarray] (labels, data)
                    min_selected=self.min_selected, top_selected=self.top_selected)
                '''
                decision_model : list[dict]
                    Each item:
                    {
                      'title': <column title>,
                      'p': <raw p-value or NaN>,
                      'FDR': <BH corrected p or NaN>,
                      'groups': [[<group_label>, <mean over rows in that group>], ...]
                    }
                    Ordered by ('FDR p-value', 'p-value') ascending.
                '''
                parts = list(PurePath(cur_path).parts)  # robust split, keeps drive/root as a part
                rebased_path = parts[parts.index(self.project_folder) + 1:]
                
                dataset = None
                if self.flg_use_NN and median_cluster_size >= self.median_cluster_size:
                    dataset = df_matrix
                self.oTree.set_classifier(address=">".join(rebased_path), classifier=decision_model, dataset=dataset)
            except Exception as e:
                print(f"   ⚠️ classification failed for {cur_path}: {e}")
                self.failures.append((cur_path, f"summarize_node: {e}"))
            i += 1
                
        # Final run for root_path
        try:
            print(f"🧮 Classify node: {root_path}")
            decision_model, df_matrix, median_cluster_size = classify(inpath=root_path, file_name="diverse_features.csv", 
                min_selected=self.min_selected, top_selected=self.top_selected)
            parts = list(PurePath(root_path).parts)  # robust split, keeps drive/root as a part
            rebased_path = parts[parts.index(self.project_folder) + 1:]
            dataset = None
            if self.flg_use_NN and median_cluster_size >= self.median_cluster_size:
                dataset = df_matrix
            self.oTree.set_classifier(address=">".join(rebased_path), classifier=decision_model, dataset=dataset)
        except Exception as e:
            print(f"   ⚠️ classification failed for {root_path}: {e}")
            self.failures.append((root_path, f"summarize_node: {e}"))
     
    
