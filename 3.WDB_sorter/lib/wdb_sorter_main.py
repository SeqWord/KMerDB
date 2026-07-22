import sys, os, shutil, stat, time, errno
from pathlib import Path, PurePath
from classifier import main as classify
from dendro import Root as tree
import progressbar

# In-house modules
import tools, wdb_reader, CSV_merger
import make_tree
from dendro import Root as tree
from word_db import WordDB

class kmer_selector:
    def __init__(self, input_path: str, out_path: str, tmp_path: str, project: str, 
            min_k: int = 4, max_k: int = 8, chunk_size: int = 8,
            min_selected: int = 10, top_selected: int = 100,
            diverse_bottom: float = -0.5, diverse_top: float = 0.5, 
            common_bottom: float = -1.0, common_top: float = 1.0, level_increment: float = 0,
            output_file: str = "", output_format: str = "pathways", clustering_algorithm: str = "NJ",
            filter_settings: dict = {}):
                
        self.input_path = input_path
        self.out_path = out_path
        self.tmp_path = tmp_path
        self.project_folder = self.project_name = project
        self.filter_settings = filter_settings
        self.min_k = min_k
        self.max_k = max_k
        self.chunk_size = chunk_size
        self.min_selected = min_selected
        self.top_selected = top_selected
        self.oTree = None
        self.failures = []
        self.diverse_bottom = diverse_bottom
        self.diverse_top = diverse_top
        self.common_bottom = common_bottom
        self.common_top = common_top
        self.output_file = output_file
        self.output_format = output_format
        self.clustering_algorithm = clustering_algorithm
        self.level_increment = level_increment
        self.total = 0      # Total number of processed branches
        
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
        # === Infer k-mer based dendrogram from provided WDB files  ===
        
        # Clease previous variant of a tree of folders, if it exists
        self.clean()
        
        output_file = ""
        if self.output_file:
            output_file = os.path.join(self.out_path, self.output_file)
        
        pathways = make_tree.execute(self.input_path, 
            algorithm = "SPECTRAL",
            output_file=output_file, 
            output_format="pathways"
        )
        if self.output_file:
            print(f"Cluster file was saved to {os.path.join(self.out_path, self.output_file)}")
            
        '''
        print()
        print("🧮 Coalesce clusters...")
        print()
        # Create a Tree object
        self.oTree = tree()
        for address in pathways:
            self.oTree.append(address)
            
        clusters = self.oTree.cluster_tree(
             target_fanout = 10,
             sectors = None,
             max_sectors = 64,
             method = "quantile",
             in_place_shortcuts = True,
             shortcut_prefix = "CLUST_",
             oversize_split = "greedy"
             )              
        self.oTree.coalesce(clusters)
        cl = self.oTree.get_node_clusters(True)
        pathways = tools.nested_list_to_newick(cl)
        
        print(pathways)
        '''
        
        components = {}
        folder = []
        bar = progressbar.indicator(len(pathways),"Sorting  WDB  files... ")
        for i in range(len(pathways)):
            parts = pathways[i].split(">")
            # Remove from the path 'Root' and the leafe node number
            folder_path = parts[1:-2]
            if not folder:
                folder = folder_path
            elements = parts[-1].split("_")
            fname = "_".join(elements[:-1])
            accession = elements[-1]
            if fname not in components:
                components[fname] = []
            components[fname].append(accession)
            
            if folder_path != folder:
                subfolder = os.path.join(self.out_path, *folder)
                os.makedirs(subfolder, exist_ok=True)
                self.populate_folder(subfolder, components)
                folder = folder_path
                components = {}
            bar(i)
        
        # Process last subfolder
        subfolder = os.path.join(self.out_path, *folder)
        os.makedirs(subfolder, exist_ok=True)
        self.populate_folder(subfolder, components)
        
        bar.stop()
        
        print()
        print(f"🧮 Folder tree structure: {self.out_path} has successfully been created!")
        print()
        
    def populate_folder(self, folder_path, components):
        for fname in components:
            oDB = self.openDBFile(os.path.join(self.input_path, fname))
            for accession in oDB.get_genomes():
                if accession not in components[fname]:
                    oDB.delete_genome(accession)
            oDB.save_dbfile(os.path.join(folder_path, fname))

    # -------------------------------
    # Open a custom WordDB file
    # -------------------------------
    def openDBFile(self, path):
        oDB = WordDB()
        try:
            oDB.open_dbfile(path)
        except:
            tools.alert(f"Problem with opening {path}!", "Alert!")
            return None
        return oDB

    # -------------------------------
    # Remove tree of folders
    # -------------------------------
    def clean(self, path_to_clean: str = "", retries: int = 5, delay: float = 0.2) -> None:
        """Remove a directory tree even if it contains files, is read-only,
        or the FS is slow to update (ENOTEMPTY)."""
        
        if not path_to_clean:
            path_to_clean = self.out_path
    
        if not os.path.isdir(path_to_clean):
            return
    
        def _onerror(func, path, _exc):
            # Make path writable, then retry the failed operation (remove/rmdir)
            try:
                os.chmod(path, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
            except Exception:
                pass
            try:
                func(path)
            except Exception:
                pass
    
        for i in range(retries):
            try:
                shutil.rmtree(path_to_clean, onerror=_onerror)
                return
            except OSError as e:
                # Retry if the directory appears "not empty" due to FS latency/races
                if e.errno in (errno.ENOTEMPTY, getattr(errno, "EEXIST", 17)):
                    time.sleep(delay * (2 ** i))
                    continue
                raise  # propagate other errors
    
        # Last attempt: ignore any lingering errors (may leave crumbs if something keeps recreating files)
        shutil.rmtree(path_to_clean, ignore_errors=True)
        
    '''
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
    
            success, error = wdb_reader.create_kmer_matrices(
                input_folder=leaf,
                output_folder=leaf_out,
                min_k=self.min_k,
                max_k=self.max_k,
                chunk_size=self.chunk_size
            )
            self.total += 1
            if not success:
                print(f"   ❌ Error: {error}")
                self.failures.append((leaf, error))
            else:
                print(f"   ✅ Done")
    
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
        # Conver addresses into paths
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
        while i < len(parent_dirs):
            cur_path = parent_dirs[i]
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
                CSV_merger.summarize_node(dirpath=cur_path)
            except Exception as e:
                print(f"   ⚠️ summarize_node failed for {cur_path}: {e}")
                self.failures.append((cur_path, f"summarize_node: {e}"))
            i += 1
                
        # Final run for root_path
        try:
            print(f"🧮 Summarizing node: {root_path}")
            CSV_merger.summarize_node(dirpath = root_path)
        except Exception as e:
            print(f"   ⚠️ summarize_node failed for {root_path}: {e}")
            self.failures.append((path, f"summarize_node: {e}"))
          
    def classify_nodes(self, root_path: str, addresses: list):
        # Conver addresses into paths
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
            try:
                print(f"🧮 Classify node: {cur_path}")
                decision_model = classify(inpath=cur_path, file_name="diverse_features.csv", 
                    min_selected=self.min_selected, top_selected=self.top_selected)
                """
                decision_model : list[dict]
                    Each item:
                    {
                      'title': <column title>,
                      'p': <raw p-value or NaN>,
                      'FDR': <BH corrected p or NaN>,
                      'groups': { <group_label>: <mean over rows in that group>, ... }
                    }
                    Ordered by ('FDR p-value', 'p-value') ascending.
                """
                parts = list(PurePath(cur_path).parts)  # robust split, keeps drive/root as a part
                rebased_path = parts[parts.index(self.project_folder) + 1:]
                self.oTree.set_classifier(address=">".join(rebased_path), classifier=decision_model)
            except Exception as e:
                print(f"   ⚠️ classification failed for {cur_path}: {e}")
                self.failures.append((cur_path, f"summarize_node: {e}"))
            i += 1
                
        # Final run for root_path
        try:
            print(f"🧮 Classify node: {root_path}")
            decision_model = classify(inpath=root_path, file_name="diverse_features.csv", min_selected=30, top_selected=100)
            """
            decision_model : list[dict]
                Each item:
                {
                  'title': <column title>,
                  'p': <raw p-value or NaN>,
                  'FDR': <BH corrected p or NaN>,
                  'groups': { <group_label>: <mean over rows in that group>, ... }
                }
                Ordered by ('FDR p-value', 'p-value') ascending.
            """
            parts = list(PurePath(root_path).parts)  # robust split, keeps drive/root as a part
            rebased_path = parts[parts.index(self.project_folder) + 1:]
            self.oTree.set_classifier(address=">".join(rebased_path), classifier=decision_model)
        except Exception as e:
            print(f"   ⚠️ classification failed for {root_path}: {e}")
            self.failures.append((root_path, f"summarize_node: {e}"))
          
     '''
    
