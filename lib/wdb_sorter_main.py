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
            output_file: str = "", output_format: str = "pathways"):
                
        self.input_path = input_path
        self.out_path = out_path
        self.tmp_path = tmp_path
        self.project_folder = self.project_name = project
        self.oTree = None
        self.failures = []
        self.output_file = output_file
        self.output_format = output_format
        
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

    def execute(self, max_cluster_number: int = 5, max_cluster_content: int = 5, 
        max_levels: int = 5, force_k: int = 0):
        # === Infer k-mer based dendrogram from provided WDB files  ===
        
        # Clease previous variant of a tree of folders, if it exists
        self.clean()
        
        output_file = ""
        if self.output_file:
            output_file = os.path.join(self.out_path, self.output_file)
        
        pathways = make_tree.execute(self.input_path, 
            algorithm = "SPECTRAL",
            output_file=output_file, 
            output_format=self.output_format,
            max_cluster_number=max_cluster_number,
            max_cluster_content=max_cluster_content,
            max_levels=max_levels,
            force_k=force_k
        )
        if self.output_file:
            print(f"Cluster file was saved to {os.path.join(self.out_path, self.output_file)}")
            
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
            fname = fname[:fname.rfind('.wdb') + 4]
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
        
    # -------------------------------
    # Populate subfolder
    # -------------------------------
    def populate_folder(self, folder_path, components):
        for fname in components:
            # Expected '*.wdb' file
            oDB = self.openDBFile(os.path.join(self.input_path, fname))
            if not oDB:
                sys.exit()
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
            tools.msg(f"Problem with opening {path}!")
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
        
