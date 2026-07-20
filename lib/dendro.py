import os
from functools import reduce
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import container, tools, progressbar
from NeuralNetwork import RF as NN 

########################################################################
class Node:
    def __init__(self,title):
        self.title = str(title).strip()             # Node title
        self.events = container.Collection()        # Collection of titled muttation object occured at this node
        self.endnode = False                        # endnode marker
        self.branch_length = 0                      # length of the branch leading to this node from the parent node
        self.classifier = {"DTree":[], "NN":None}   # classification models: DTree - decision tree; NN - newral network
        self.qualifiers = {}                        # "branches" like {"1-2":length,...}
                                                    # "seq" - reference sequence as a string
                                                    # "sequence length" - length of the sequence
        
    def __str__(self):
        return self.title
        
    # Info
    def is_endnode(self):
        return self.endnode
    
    def is_terminal(self):
        return self.endnode
    
    # Get methods
    def get_path_length(self,path,path_length=0):
        if self.endnode or len(path)==1:
            return path_length + self.branch_length
        path = path[1:]
        key = path[0]
        try:
            key = int(key)
        except:
            pass
        return self.child_nodes[key].get_path_length(path,path_length + self.branch_length)
        
    def get_titled_node_events(self,path):
        if len(path)==1:
            return str(self.events)
        path = path[1:]
        key = path[0]
        try:
            key = int(key)
        except:
            pass
        return self.child_nodes[key].get_titled_node_events(path)
        
    def get_node_all_events(self,path=[],loci=[]):
        if self.is_endnode():
            return list(map(lambda oEvent: [str(oEvent),[self.title]], self.events.get(loci)))
        if len(path) <= 1:
            return (list(map(lambda oEvent: [str(oEvent),[self.title]], self.events.get(loci))) + 
                reduce(lambda ls1,ls2: ls1+ls2, list(map(lambda oChildNode: oChildNode.get_node_all_events([],loci), self.child_nodes))))
        path = path[1:]
        key = path[0]
        try:
            key = int(key)
        except:
            pass
        return self.child_nodes[key].get_node_all_events(path,loci)
        
    def get_ancestors(self):
        if "seq" not in self.qualifiers:
            return ""
        if self.is_endnode():
            states = tools.dec2bin(self.qualifiers["seq"])
            return [f"{self.title} {states}"]
        states = list(map(lambda oChildNode: oChildNode.get_ancestors(), self.child_nodes))
        seq = tools.dec2bin(self.qualifiers["seq"])
        if len(seq) > 100:
            seq = seq[:97]+"..."
        return [self.title+" "+seq]+reduce(lambda ls1,ls2: ls1+ls2, list(map(lambda ls: list(map(lambda s: "---"+s, ls)),states)))
        
    def get_allelic_states(self):
        return self.get_ancestors()

    def get_node_lineages(self,path):
        if len(path)==1:
            return self.lineages
        path = path[1:]
        key = path[0]
        try:
            key = int(key)
        except:
            pass
        return self.child_nodes[key].get_node_lineages(path)
        
    # Evolution
    def set_ancestor_states(self,length,terminals,bar_operator,refseq="",fasta={}):
        if self.is_endnode():
            bar_operator["bar"](bar_operator["counter"])
            bar_operator["counter"] += 1
            self.qualifiers["seq"] = terminals[self.title].qualifiers["seq"]
            return [self.title,self.title], self.qualifiers["seq"]
        datasets = list(map(lambda oChildNode: oChildNode.set_ancestor_states(length,terminals,bar_operator,refseq,fasta), self.child_nodes)) # datasets = [[endnode title,binary_seq],...]
        if len(datasets) < 2:
            tools.msg("Error with the intermediate node %s! Contains only %d child nodes." % (self.title,len(self.child_nodes)))
            raise IOError("Error with the tree structure!")
        ancestor,mutations = tools.matrix_difference(list(map(lambda item: item[1], datasets)),length)  
        endnode_titles = reduce(lambda ls1,ls2: ls1+ls2, list(map(lambda item: item[0], datasets)))
        self.qualifiers["seq"] = ancestor
        for i in range(len(self.child_nodes)):
            self.child_nodes[i].set_mutations(mutations[i],endnode_titles,refseq,fasta)
        return [[endnode_titles[0],endnode_titles[-1]],ancestor]
        
    def set_mutations(self,mut_ls,endnode_titles=[],refseq="",fasta={}):
        if mut_ls == []:
            return
        self.events.extend(list(map(lambda i: Mutation(str(i)), mut_ls)))

    # Set classifier
    def set_classifier(self, address: str, classifier: list, dataset: Optional[pd.DataFrame] = None) -> None:
        # Create neural network model for an intermediate node
        def create_NN(dataset: pd.DataFrame, classifier: List[Dict]) -> Optional[NN]:
            """
            Build and train an NN from a DataFrame `dataset`, keeping only features
            listed in `classifier` (list of dicts with key 'title').
        
            Returns:
                Trained NN instance, or None if nothing to train.
            """
            if not isinstance(dataset, pd.DataFrame) or dataset.empty:
                return None
        
            # 1) Feature list from classifier
            features = [rec["title"] for rec in classifier if "title" in rec]
        
            # 2) Keep only ['Name','Taxon'] + intersect(features, dataset.columns)
            keep_cols = ["Name", "Taxon"]
            present_features = [f for f in features if f in dataset.columns]
            missing_features = sorted(set(features) - set(present_features))
            if missing_features:
                print(f"[create_NN] Warning: {len(missing_features)} features missing from matrix; they will be ignored.")
        
            keep_cols.extend(present_features)
            filtered_matrix = dataset.loc[:, [c for c in keep_cols if c in dataset.columns]].copy()
        
            # 3) Detect target column
            target_col = "Taxon" if "Taxon" in filtered_matrix.columns else ("taxon" if "taxon" in filtered_matrix.columns else None)
            if target_col is None:
                raise ValueError("DataFrame must contain a 'Taxon' (or 'taxon') column.")
        
            # 4) Build labels (ints) and feature matrix (floats)
            # Non-feature columns to exclude
            non_features = {target_col, "Name", "name", "ID", "id", "Accession", "accession"}
        
            feature_cols = [c for c in filtered_matrix.columns if c not in non_features]
            if not feature_cols:
                print("[create_NN] No usable feature columns after filtering; aborting.")
                return None
        
            # Labels as integers
            try:
                y = pd.to_numeric(filtered_matrix[target_col], errors="raise").astype(int).values
            except Exception as e:
                raise ValueError(f"Could not convert target column '{target_col}' to int: {e}")
        
            # Features as float matrix (coerce any non-numeric cells to 0)
            X = filtered_matrix[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float).values
        
            # 5) Train NN; keep feature titles inside model for later identify()
            oNN = NN().train(X, y, feature_titles=feature_cols)
            return oNN        
            
        address = address.split(">")
        if self.is_terminal() or address == ['']:
            self.classifier["DTree"] = classifier
            
            if isinstance(dataset, pd.DataFrame) and not dataset.empty:
                print("Create Neural Network Clasifier..")
                self.classifier["NN"] = create_NN(dataset, classifier)
            return 
        child_node = address[0]
        address = ">".join(address[1:])
        self.child_nodes[child_node].set_classifier(address, classifier, dataset)
        
    # Return classifier as a string
    def classifier_to_string(self, path: str = "root", counter: int = 1):
        ls_classifier = []
        """        
        classifier : list[dict]
            Each item:
            {
              'title': <column title>,
              'p': <raw p-value or NaN>,
              'FDR': <BH corrected p or NaN>,
              'groups': [[<group_label>, <mean over rows in that group>], ... ]
            }
            Ordered by ('FDR p-value', 'p-value') ascending.
        """
        child_nodes = []
        if not self.is_terminal() and self.child_nodes:
            child_nodes = self.child_nodes.get_titles()
            
        if self.classifier["DTree"]:
            titles = [ls[0] for ls in self.classifier["DTree"][0]['groups']]
            ls_classifier.append("\n" + (30 * "="))
            ls_classifier.append(f"Split: {path}.{counter}: {' | '.join(titles)}")
            for w in self.classifier["DTree"]:
                ls_classifier.append("\t" + f"{w['title']}:" + "\t" + 
                    "|".join([f"{float(w['groups'][i][1]):.4f}" for i in range(len(titles))])
                    )
                    
        for i in range(len(child_nodes)):
            child_node = child_nodes[i]
            ls_classifier += self.child_nodes[child_node].classifier_to_string(path = f"{path}.{counter}", counter = i + 1)
                
        return ls_classifier
        
    def populate_branch_lengths(self,branches,nodename):
        branches[f"{nodename}-{self.title}"] = self.branch_length
        if not self.is_endnode():
            for oChildNode in self.child_nodes:
                oChildNode.populate_branch_lengths(branches,self.title)
        
    def get_lineage_addresses(self,lineage,address=""):
        if self.is_monophyletic(lineage):
            return [(address+"."+self.title).strip(".")]
        if self.is_endnode():
            return []
        values = list(map(lambda oChildNode: oChildNode. get_lineage_addresses(lineage,address+"."+self.title), self.child_nodes))
        if len(values) > 1:
            return list(map(lambda s: s.strip("."), reduce(lambda ls1,ls2: ls1+ls2, values)))
        if len(values)==1:
            values[0][0] = values[0][0].strip(".")
            return values[0]
        return []
    
    def get_lineage_paths(self,lineage):
        if self.is_monophyletic(lineage):
            return [self.path]
        if self.is_endnode():
            return []
        values = list(map(lambda oChildNode: oChildNode. get_lineage_paths(lineage), self.child_nodes))
        if len(values) > 1:
            return reduce(lambda ls1,ls2: ls1+ls2, values)
        if len(values)==1:
            return values[0]
        return []
        
    # Edit nodes
    def rename(self,new_name):
        self.title = new_name
        
    def rename_address(self,current_name,new_name):
        for oChildNode in self.child_nodes:
            oChildNode.rename_address(current_name,new_name)
            
    # Identifier
    def identify_by_kmers(self, seq: str,
            specificity: float = .8, 
            accuracy: float = .65,
            current_value: float = 0,
            identified: list = [],
            path: str = "Root",
            method: str = "average",     # average | avr | max | min
            flg_use_NN: bool = True, 
            entropy_cutoff: float = .5,
            log: list = [],
            ) -> (list, list):
        """
        self.classifier["DTree"] : list[dict]
            Each item:
            {
              'title': <column title>,
              'p': <raw p-value or NaN>,
              'FDR': <BH corrected p or NaN>,
              'groups': [[<group_label>, <mean over rows in that group>], ...]
            }
            Ordered by ('FDR p-value', 'p-value') ascending.
        self.classifier["NN"] : NN_model
        """
        if not self.classifier["DTree"]:
            raise ValueError(f"Node {self.title} has no classifier!")
        
        species = [ls[0] for ls in self.classifier["DTree"][0]['groups']]
        matrix = [{title : 0} for title in species]
        log.append(f"Split: {path}")
        dtree_log = ["\t".join(["Word"] + species + ["Abundance"] + species)]
        
        markers = []    # For NN identification
        for item in self.classifier["DTree"]:
            title = item['title']
            word = title.split("|")[0]
            dtree_log.append("\t".join([word] + [f"{float(ls[1]):.2f}" for ls in item['groups']]))
            
            # Calculate counts of the direct and reverse complement word
            count_dir = tools.count_word(seq, word)
            count_rev = tools.count_word(seq, word, count_reverse_complement=True)
            if method.lower() in ('average', 'avr'):
                count = (count_dir + count_rev) / 2
            elif method.lower() == 'max':
                count = max([count_dir, count_rev])
            elif method.lower() == 'min':
                count = min([count_dir, count_rev])
            else:
                print(f"Unknown argument 'method' = {method}!")
                sys.exit()
                
            abundance = tools.set_value(wlength=len(word), count=count, seqlength=len(seq), 
                coding=[-2, -1, 1, 2])
            dtree_log[-1] += f"\t{abundance}"
            markers.append([title, abundance])
            for i in range(len(species)):
                title = species[i]
                value_match = (4 - int(abs(float(item['groups'][i][1]) - abundance)))
                matrix[i][title] += value_match
                dtree_log[-1] += f"\t{value_match}"
        
        # Possible directions at intermediate splits        
        possible = []
        dtree_log.append(f"{'-' * (2 + 2 * len(species))}")
        dtree_log.append(f"{'\t' * (2 + len(species))}")
        
        # Try neural network classification if a NN model is available
        NN_results = None
        NN_log =[]
        if flg_use_NN and self.classifier["NN"] != None:
            '''
            Example:
            results = {'predictions': [['1', 0.5808236859196959], ['0', 0.4191763140803041]], 'odd-ratios': [0.6639381056194705], 'entropy': 0.9810683250933924}
            odd-ratio = best/(1 - best) / next/(1 - next)
            entropy = -sum([v * log2(v), ...]) / log2(n); 0 - minimal entropy, excelent predition; 1 - maximal entropy
            '''
            NN_results = self.classifier["NN"].identify(markers)
        
        # Use NN predictions
        if NN_results is not None and NN_results['entropy'] <= entropy_cutoff:
            NN_log.append("Neural network classification:")
            NN_log.append("\t".join([""] + species))
            NN_log.append("values:")
            value_recors = ["" for i in range(len(species))]
            odd_ratios = ["" for i in range(len(species))]
            for j in range(len(NN_results['predictions'])):
                title, value = NN_results['predictions'][j]
                i = species.index(title)
                # Add possible directions
                if value >= accuracy:
                    possible.append([species[i], value])
                # Add identified nodes
                if value >= specificity:
                    identified.append([species[i], value, path, self.is_endnode()])
                value_recors[i] = f"{value:.2f}"
                odd_ratios[i] = f"{NN_results['odd-ratios'][j]:.2f}"
                
            NN_log[-1] += "\t" + "\t".join(value_recors)
            NN_log.append("odd-ratios:\t" + "\t".join(odd_ratios))
            NN_log.append(f"entropy: {NN_results['entropy']:.2f}")
            NN_log.append((10 * "-") + "\n")
            log += NN_log
            
        else:   # Use DTree predictions    
            # Normalize counts to range from 0 to 1
            log += dtree_log
            for i in range(len(matrix)):
                value = matrix[i][species[i]]
                value /= (4 * len(self.classifier["DTree"]))
                log[-1] += f"{value:.2f}\t"
                if value >= accuracy:
                    possible.append([species[i], value])
                if value >= specificity:
                    identified.append([species[i], value, path, self.is_endnode()])
                    
        # Continue classification with child nodes            
        if not self.is_endnode():
            if len(possible) == 0:
                return [self.title, current_value, path, False], log
            for node_title, current_value in possible:
                matrix_record, log = self.child_nodes[node_title].identify_by_kmers(seq=seq, 
                    specificity=specificity, 
                    accuracy=accuracy, 
                    current_value=current_value, 
                    identified=identified,
                    path=f"{path}.{node_title}",
                    flg_use_NN=flg_use_NN,
                    entropy_cutoff=entropy_cutoff, 
                    method=method,
                    log=log)
                matrix += matrix_record
        return identified, log
        
    # INFO      
    def is_monophyletic(self,lineage=""):
        if self.is_endnode():
            if self.lineage==lineage:
                return True
            return False
        if len(self.lineages)==1:
            if lineage != "" and self.lineages[0]==lineage:
                return True
            elif lineage == "":
                return True
            else:
                pass
        return False
        
########################################################################        
class EndNode(Node):                            # End or terminal node of the tree
    def __init__(self, title, path, address, branch_length, lineage):
        Node.__init__(self,title)
        self.endnode = True                     # endnode marker
        self.branch_length = branch_length      # length of the branch leading to this node from the parent node
        self.lineage = lineage                  # lineage name
        self.path = path+f".{self.title}"       # path to the end node in indexes
        self.address = address                  # path to the end node in intermediate node titles
        self.index = int(path.split(">")[-1])   # index in the parrent child+node list
        self.resistance_pattern = {}            # antibiotic resistance pattern
        self.predicted_resistance = {}          # predicted resistance

    # Get functions
    def get_lineages(self):
        return [self.lineage]

    def get_lineage(self):
        return self.lineage

    # Edit nodes
    def rename(self,new_name):
        self.rename_address(self.title,new_name)
        self.title = new_name
        
    def rename_address(self,current_name,new_name):
        self.address = self.address.replace(">"+current_name,">"+new_name)
        self.address = self.address.replace(current_name+">",new_name+">")
        self.path = self.path.replace(">"+current_name,">"+new_name)
        self.path = self.path.replace(current_name+">",new_name+">")
        
########################################################################
class Intermediate_Node(Node):
    def __init__(self, title, path):
        Node.__init__(self,title)
        self.child_nodes = container.Collection()   # Collection of named child nodes
        self.confidence = 0                         # branch length confidence
        self.LCA = False                            # lowest common ancestor marker - the last node before at least one end node among the child nodes
        self.lineages = []                          # list of lineages of the child end nodes
        self.path = path                            # Path to the node from the root like 'root.0.0.1.1'
        self.index = int(self.path.split(">")[-1])  # Node index in the parent node: 0 or 1 in a dichotomous tree where 0 is the right branch and 1 is the left branch
    
    def add(self,pathway,path="root",address="root"):                   # add new nodes as child objects
        if len(pathway) == 0:
            return

        # Appending a new end node
        if len(pathway)==1:
            return self.add_endnode(pathway[0], path, address)

        # Add lineage to the list of lineages of the current node
        lineage = pathway[-1][-1]
        if lineage not in self.lineages:
            self.lineages.append(lineage)

        # Setting a new node
        title,branch_length,confidence = pathway[0]
            
        # Check consistency
        if self.title != title.strip():
            raise IOError(f"Error with tree parsing. Node title {self.title} does not match with the submitted title {title}!")
            
        # Removing the current node information from the pathways
        pathway = pathway[1:]
        # Appending a new end node
        if len(pathway)==1:
            return self.add_endnode(pathway[0], path, address)
        # Appending intermediate child nodes
        return self.add_childnode(pathway,path,address)
    
    # Appending intermediate child nodes
    def add_childnode(self,pathway,path,address):
        title,branch_length,confidence = pathway[0]
        title = title.strip()
        try:
            self.branch_length = float(branch_length)
            self.confidence = float(confidence)
        except:
            pass
        address += ">%s" % title
        # Check if the chils node exists already
        if title in self.child_nodes.get_titles():
            path += ">%d" % self.child_nodes.index(title)
            return self.child_nodes[title].add(pathway,path,address)
        # New intermediate child node
        path += ">%d" % len(self.child_nodes)
        oNode = Ancestor(title,path)
        self.child_nodes.append(oNode)
        return self.child_nodes[-1].add(pathway,path,address)
    
    # Appending a new end node
    def add_endnode(self, values, path, address):
        title,branch_length,confidence,lineage = values
        try:
            self.branch_length = float(branch_length)
            self.confidence = float(confidence)
        except:
            pass
        if title.strip() in self.child_nodes.get_titles():
            raise IOError(f"Error with tree parsing. Endnode {title} is not unique!")
        path += ">%d" % len(self.child_nodes)
        address += ">%s" % title
        oEndNode = EndNode(title, path, address, branch_length, lineage)
        self.child_nodes.append(oEndNode)
        self.LCA = True
        return oEndNode
        
    # Get functions
    def get_node(self,path):
        if len(path)==1:
            return self
        path = path[1:]
        key = path[0]
        try:
            key = int(key)
        except:
            pass
        return self.child_nodes[key].get_node(path)
        
    def get_lineages(self):
        return self.lineages

    def get_LCA_numbers(self):
        if self.LCA:
            count = 1
            for oChildNode in self.child_nodes:
                if not oChildNode.endnode:
                    count += oChildNode.get_LCA_numbers()
            return count
        return sum(list(map(lambda oChildNode: oChildNode.get_LCA_numbers(), self.child_nodes)))
    
    def get_admixture_rate(self):
        if self.LCA:
            lineages = tools.dereplicate(reduce(lambda ls1,ls2: ls1+ls2, list(map(lambda oChildNode: oChildNode.get_lineages(), self.child_nodes))))
            count = 0
            if len(lineages) > 1:
                count = 1
            for oChildNode in self.child_nodes:
                if not oChildNode.endnode:
                    count += oChildNode.get_admixture_rate()
            return count
        return sum(list(map(lambda oChildNode: oChildNode.get_admixture_rate(), self.child_nodes)))
        
    def get_lineage_pairs(self):
        if self.is_monophyletic():
            return []
        if len(self.lineages)==2:
            return [[self.title,list(map(lambda lineage: lineage, self.lineages))]]
        pairs = []
        for oChildNode in self.child_nodes:
            if not oChildNode.is_endnode():
                pairs += oChildNode.get_lineage_pairs()
        return pairs
        
    # Edit nodes
    def rename_intermediate_node(self,current_name,new_name,path):
        try:
            i = int(path[0])
            path = path[1:]
        except:
            raise RuntimeError(f"Index {i} must be an integer!")
        if i < 0 or i > len(self.child_nodes):
            raise RuntimeError(f"Address {i} behind the range of child nodes!")
        if self.child_nodes[i].is_endnode():
            self.child_nodes[i].rename(new_name)
        elif len(path)==1:
            self.rename(new_name)
            for oChildNode in self.child_nodes:
                oChildNode.rename_address(current_name,new_name)
                
    # Train neural network
    def train_neural_network(self, csv_fname: str):
        """
        Train a neural network from a CSV file.
        Expects a 'taxon' or 'Taxon' column for class labels.
        """
        if not os.path.exists(csv_fname):
            raise FileNotFoundError(f"CSV not found: {csv_fname}")

        df = pd.read_csv(csv_fname)

        # detect target column
        target_col = None
        for cand in ("taxon", "Taxon"):
            if cand in df.columns:
                target_col = cand
                break
        if target_col is None:
            raise ValueError("CSV must contain a 'taxon' or 'Taxon' column")

        self.target_col = target_col
        y = df[target_col].astype(int).values

        # features = all except target + obvious non-feature columns
        non_features = {target_col, "Name", "name", "ID", "id", "Accession", "accession"}
        feature_cols = [c for c in df.columns if c not in non_features]
        self.feature_cols = feature_cols

        X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float).values

        # create NN and train
        oNN = NN().train(X, y)
        self.classifier["NN"] = oNN
        return oNN    

########################################################################    
class Root(Intermediate_Node):                      # Root of the tree
    def __init__(self, description="", args={}):
        Intermediate_Node.__init__(self, title="root", path="0")
        self.description = description              # Project description
        self.args = args                            # Last call arguments
        self.tree_file_name = ""                    # Original tree file
        try:
            self.tree_file_name = self.args["-t"]
        except:
            pass
        self.terminals = container.Collection()     # Collection of titled end node objects
        self.oTree = None                           # BioPython tree object
        self.flg_was_modified = False               # Indicates made modifications
        self.refseq = ""                            # Reference sequence
        
    def __getitem__(self,nodename):
        if nodename in self.terminals.get_titles():
            return self.terminals[nodename]
        path = self.get_titled_node_path(nodename)
        if path == "":
            return None
        return self.get_node(path)
    
    # Show general info about the tree object
    def info(self):
        message = []
        message.append(f"The source tree is {self.tree_file_name}\n")
        message.append("Total number of leaf nodes:\t\t%d" % self.get_leaf_number())
        message.append("Total number of intermediate nodes:\t%d" % self.get_node_size("intermediate"))
        message.append("Total number of lineages:\t\t%d\nWhich are:" % self.get_number_lineages())
        for lineage in self.get_lineage_titles(True):
            message.append("\t%s%s%d" % (lineage,"\t"*(2-int(len(lineage)//9)),self.get_lineage_number(lineage)))
        message.append("\nAdmixure rate:\t%d/%d" % (self.get_admixture_rate(),self.get_LCA_numbers()))
        return "\n".join(message)

    #### Database_edition
    def append(self, pathway):                       # Append a new path of nodes
        self.terminals.append(self.add(pathway))
        
    def add_endnode_qualifiers(self,input_file_name,qualifier_key):
        if not os.path.exists(input_file_name):
            tools.msg(f"Input file {input_file_name} does not exist!")
            return
        if qualifier_key in self.get_endnode_qualifiers():
            print()
            response = input(f"The key '{qualifier_key}' was already used to save data at the end nodes. Do you want to replace the data Y/N? ").upper()
            if response != "Y":
                return
        data = list(filter(lambda ls: len(ls)==3, tools.open_text_file(input_file_name)))
        if len(data) == 0:
            tools.msg(f"File {input_file_name} is empty, or corrupted, or contains more or less than 3 fields! No data added!")
        strain,state_1,state_2 = data[0]
        k = 1
        available_endnodes = self.get_endnode_titles()
        if strain in available_endnodes:
            response = (input("It looks like input file has not the heading line '','state_1','state_2'."+
                "\nIf headings are there, press Y+Enter, or any other key to set headings").upper())
            if response != "Y":
                headings = []
                while headings == []:
                    resp = input("Enter two comma separated hedings like 'R,S' or 'Resistant,Susceptible': ")
                    resp = resp.split(",")
                    if len(resp)==2 and "" not in resp:
                        heading = resp
                        k = 0
                    else:
                        resp = input("Try again or press Q+Enter to quit: ").upper()
                        if resp=="Q":
                            return
                        continue
        counter = 0
        for i in range(k,len(data),1):
            strain,items_1,items_2 = data[i]
            if strain in available_endnodes:
                self.terminals[strain].qualifiers[qualifier_key] = {state_1:list(map(lambda s: s.strip(),items_1.split(","))),state_2:list(map(lambda s: s.strip(),items_2.split(",")))}
                counter += 1
        tools.msg(f"In total {counter} strains were set with new data.")
        
    # Info methods
    def is_endnode(self,nodename=""):
        if nodename == "":
            return self.endnode
        return nodename in self.terminals.get_titles()
        
    # Get methods
    def get_endnode_number(self):
        return len(self.terminals)
        
    def get_endnode_titles(self):
        return self.terminals.get_titles()
        
    def get_endnode_qualifiers(self):
        return tools.dereplicate(reduce(lambda ls1,ls2: ls1+ls2, list(map(lambda oEndNode: list(oEndNode.qualifiers.keys()), self.terminals))))
        
    def get_lineage_titles(self,flg_sort=False):
        if flg_sort and len(self.lineages) > 1:
            self.lineages.sort()
        return self.lineages
        
    def get_path_length(self,path):
        if type(path)==type(""):
            path = path.split(">")
        # Remove index of the root node
        path = path[1:]
        if len(path)==0:
            return 0
        
        key = path[0]
        try:
            key = int(key)
        except:
            pass
        try:
            return self.child_nodes[key].get_path_length(path)
        except:
            tools.msg("Node path integrity problem!")
            return 0
    
    # End node info
    def get_endnode_depth(self,nodename):
        if not self.is_endnode(nodename):
            tools.msg(f"Name {nodename} is not present among endnode names!")
            return 0
        return self.get_path_length(self.terminals[nodename].path) 
        
    def get_node_depth(self,nodename):
        path = self.get_titled_node_path(nodename)
        return self.get_path_length(path)
        
    def get_endnode_lineage(self,nodename):
        if not self.is_endnode(nodename):
            tools.msg(f"Name {nodename} is not present among endnode names!")
            return ""
        return self.terminals[nodename].lineage
        
    def get_node_lineage(self,nodename):
        path = self.get_titled_node_path(nodename)
        return self.get_node_lineages(path)
        
    def get_endnode_path(self,nodename):
        if not self.is_endnode(nodename):
            tools.msg(f"Name {nodename} is not present among endnode names!")
            return ""
        return self.terminals[nodename].path
        
    def get_node_path(self,nodename):
        return self.get_titled_node_path(nodename)
        
    def get_endnode_address(self,nodename):
        if not self.is_endnode(nodename):
            tools.msg(f"Name {nodename} is not present among endnode names!")
            return ""
        return self.terminals[nodename].address
        
    def get_node_address(self,nodename):
        addresses = self.get_titled_node_addresses(nodename)
        if len(addresses)==0:
            return []
        address = addresses[0]
        return address[:address.index(nodename)+1]
        
    def get_node_events(self,nodename):
        path = self.get_titled_node_path(nodename)
        return self.get_titled_node_events(path)
        
    def get_all_events(self,loci=[],nodename=""):
        path = ""
        if nodename != "":
            path = self.get_titled_node_path(nodename)
        return self.get_node_all_events(path,loci)
        
    def get_modelling_tasks(self):
        return [ModelTask("Identify taxonomic units",self.get_lineage_titles(True))]
        
    def subordinate_node_count(self,nodename):
        if self.is_endnode(nodename):
            return 0
        return len(self.get_subordinate_node_titles(nodename))
        
    def endnode_count(self,nodename):
        if self.is_endnode(nodename):
            return 0
        return len(self.get_titled_node_addresses(nodename))
        
    # General Info
    def has(self,nodename):
        addresses = self.avail_nodes()
        return nodename in addresses
        
    def get_leaf_number(self):
        return len(self.terminals)
        
    def get_number_lineages(self):
        return len(self.lineages)   
    
    def get_lineage_number(self,lineage):
        return len(list(filter(lambda oEndNode: oEndNode.lineage==lineage,self.terminals)))
        
    def get_addresses(self):
        return list(map(lambda oEndNode: oEndNode.address, self.terminals))
        
    def avail_nodes(self):
        return reduce(lambda ls1,ls2: ls1+ls2, list(map(lambda address: address.split(">"), self.get_addresses())))
                
    def get_titled_node_addresses(self,nodename):
        return list(filter(lambda ls: nodename in ls, list(map(lambda address: address.split(">"), self.get_addresses()))))
        
    def get_titled_node_path(self,nodename):
        addresses = self.get_titled_node_addresses(nodename)
        if len(addresses)==0:
            tools.msg(f"Node {nodename} is not present in the three!")
            return ""
        address = addresses[0]
        endnode_title = address[-1]
        try:
            path = self.terminals[endnode_title].path.split(">")[:address.index(nodename)+1]
        except:
            raise IOError(f"End-node name {endnode_title} does not exist!")
        return path
        
    def get_subordinate_node_titles(self,nodename):
        addresses = self.get_titled_node_addresses(nodename)
        return tools.dereplicate(reduce(lambda ls1,ls2: ls1+ls2, list(map(lambda address: address[address.index(nodename):-1], addresses))))
        
    # Evolution
    def set_evolutionary_events(self,bar_operator,refseq="",fasta={}):
        self.refseq = refseq
        self.set_ancestor_states(self.qualifiers["sequence length"],self.terminals,bar_operator,refseq,fasta)
        
    # Operations with the BioPython tree object
    def set_tree_object(self,oTree):
        self.oTree = oTree
        
    def set_branches(self):
        addresses = list(map(lambda address: address.split(">"), self.get_addresses()))
        branches = tools.dereplicate(reduce(lambda ls1,ls2: ls1+ls2, list(map(lambda ls: list(map(lambda i: f"{ls[i]}-{ls[i+1]}", range(len(ls)-1))), addresses))))
        branches = dict(zip(branches,[0]*len(branches)))
        self.populate_branch_lengths(branches)
        self.qualifiers["branches"] = branches
        
    def populate_branch_lengths(self,branches):
        for oChildNode in self.child_nodes:
            oChildNode.populate_branch_lengths(branches,"root")
            
    def rename_node(self,current_name,new_name,checked=False):
        if not self.has(current_name):
            tools.msg(f"Node with the name {current_name} has not been found!")
            return
        if self.has(new_name):
            tools.msg(f"Node {current_name} already exists!")
            return
        if current_name in self.get_endnode_titles():
            self.rename_endnode(current_name,new_name)
            return
        if current_name == self.title:
            self.title = new_name
            self.rename_address(current_name,new_name)
        else:
            path = self.get_node_path(current_name)[1:]
            self.rename_intermediate_node(current_name,new_name,path)
            map(lambda oEndNode: oEndNode.rename_address(current_name,new_name),self.terminals)
        for oEndNode in self.terminals:
            oEndNode.rename_address(current_name,new_name)
        
    def rename_endnode(self,current_name,new_name):
        self.rename_intermediate_node(new_name,self.terminals[current_name].path.split(">")[1:])
        self.terminals[current_name].rename(new_name)
    
    ### Get function
    def get_tree(self):
        return self.oTree
        
    def get_ancestor_states(self):
        return self.get_ancestors()
        
    def get_node_size(self,nodetype="All",nodename="root"): # Node types: all, intermediate, endnode
        if not self.has(nodename):
            tools.msg(f"Node with the name {nodename} does not exist!")
            return 0
        if nodename in self.terminals:
            return 1
        addresses = self.get_addresses()
        if nodename != "root":
            addresses = list(map(lambda s: s[s.find(">"+nodename)+len(nodename)+2:], filter(lambda address: address.find(">"+nodename) > -1, addresses)))
        ls = list(map(lambda address: address.split(">"), addresses))
        if nodetype == "endnode":
            ls = list(map(lambda item: item[-1], ls))
        elif nodetype == "intermediate":
            ls = list(map(lambda item: item[:-1], ls))
        if nodetype != "endnode":
            ls = tools.dereplicate(reduce(lambda ls1,ls2: ls1+ls2, ls))
        size = len(ls)
        if nodename == "root":
            size -= 1       
        return size
        
    def get_longest_branch(self,nodename="root"):
        return self.path_length(nodename,"longest")
        
    def get_shortest_branch(self,nodename="root"):
        return self.path_length(nodename,"shortest")
        
    def get_total_branch_length(self,nodename="root"):
        return self.path_length(nodename,"total")
        
    def get_average_branch_length(self,nodename="root"):
        return self.path_length(nodename,"average")
        
    def path_length(self,nodename,mode):
        if not self.has(nodename):
            tools.msg(f"Node with the name {nodename} does not exist!")
            return ["",0]
        if "branches" not in self.qualifiers:
            self.set_branches()
        if nodename in self.terminals:
            return [nodename,0]
        addresses = self.get_addresses()
        if nodename != "root":
            addresses = list(map(lambda s: s[s.find(">"+nodename)+len(nodename)+2:].strip(">"), filter(lambda address: address.find(">"+nodename) > -1, addresses)))
        addresses = list(map(lambda address: address.split(">"), addresses))
        ls = list(map(lambda address: [address[-1],self.calculate_distance(address,self.qualifiers['branches'])], addresses))
        ls = sorted(ls, key=lambda item: item[1])
        if mode == "longest":
            return ls[-1]
        elif mode == "shortest":
            return ls[0]
        elif mode == "total":
            return [nodename,sum(list(map(lambda item: item[1], ls)))]
        elif mode == "average":
            return [nodename,sum(list(map(lambda item: item[1], ls)))/len(ls)]
            
    def get_distance(self,nodename_1,nodename_2="root"):
        if nodename_1==nodename_2:
            return 0
        if "branches" not in self.qualifiers:
            self.set_branches()
        available_nodes = self.avail_nodes()
        if nodename_1 not in available_nodes or nodename_2 not in available_nodes:
            tools.msg(f"Nodes with the names {nodename_1} or {nodename_2} do not exist!")
            return 0
        addresses = self.get_addresses()
        selection = list(filter(lambda address: address.find(">"+nodename_1) > -1 and (nodename_2=="root" or address.find(">"+nodename_2) > -1), addresses))
        if len(selection) > 0:
            address = selection[0].split(">")
            i = address.index(nodename_1)
            j = address.index(nodename_2)
            return self.calculate_distance(address[min([i,j]):max([i,j])+1],self.qualifiers['branches'])
        address_1 = list(filter(lambda address: address.find(">"+nodename_1) > -1, addresses))[0].split(">")
        address_2 = list(filter(lambda address: address.find(">"+nodename_2) > -1, addresses))[0].split(">")
        distance_1 = self.calculate_distance(address_1[:address_1.index(nodename_1)+1],self.qualifiers['branches'])
        distance_2 = self.calculate_distance(address_2[:address_2.index(nodename_2)+1],self.qualifiers['branches'])
        return distance_1 + distance_2
        
    def get_relative_distance(self,nodename):
        if nodename=="root":
            return 0
        max_branch_length = self.get_longest_branch(nodename)[1]
        if max_branch_length==0:
            return 0
        return 1 - max_branch_length/(max_branch_length + self.get_distance(nodename))
        
    def calculate_distance(self,address,branches):
        distances = list(map(lambda i: branches[f"{address[i]}-{address[i+1]}"], range(len(address)-1)))
        return sum(distances)
        
    #### Summaries
    def mutation_summary(self,output_fname="",loci=[]):
        summary_table = [["Locus","No. permutations","Max time","Min time","Avr time"]]
        ls_mutations = tools.dereplicate_and_concatenate(self.get_all_events(loci))
        ls_mutations = list(map(lambda item: [item[0],list(map(lambda nodename: self.get_relative_distance(nodename), item[1]))], ls_mutations))
        summary_table += list(map(lambda item: [item[0],str(len(item[1]))]+list(map(lambda v: tools.format_number(v),tools.get_min_max_avr(item[1]))), ls_mutations))
        if output_fname != "":
            tools.save_text_file(os.path.join("..",self.args["-o"],output_fname),
                "\n".join(list(map(lambda item: "\t".join(item), summary_table))))
        else:
            for i in range(len(summary_table)):
                print("\t".join(summary_table[i]))
                
    def lineage_overlap_table(self,output_fname="",lineages=[]):
        available_lineages = self.get_lineage_titles()
        if len(lineages) > 0:
            lineages = list(filter(lambda lineage: lineage in available_lineages, lineages))
        else:
            lineages = available_lineages
        table = [[""]+lineages]+list(map(lambda i: [lineages[i]]+(["0"]*len(lineages)), range(len(lineages))))
        lineage_pairs = self.get_lineage_pairs()
        print()
        for i in range(len(lineages)):
            for j in range(i,len(lineages)):
                if i==j:
                    table[i+1][j+1] = str(len(self.get_lineage_addresses(lineages[i])))
                else:
                    selected_pairs = list(filter(lambda ls: lineages[i] in ls[1] and lineages[j] in ls[1], lineage_pairs))
                    table[i+1][j+1] = table[j+1][i+1] = str(len(selected_pairs))
        lineage_pairs = tools.dereplicate_and_concatenate(list(map(lambda item: [sorted(item[1]),[item[0]]], lineage_pairs)))
        if output_fname != "":
            tools.save_text_file(os.path.join(self.args["-o"],output_fname),
                "\n".join(list(map(lambda item: "\t".join(item), table))) + "\n\n" +
                    "\n".join(list(map(lambda pair: f"{pair[0][0]}-{pair[0][1]}:\t"+",".join(pair[1]), lineage_pairs))))
            return
        for pair in lineage_pairs:
            print(f"{pair[0][0]}-{pair[0][1]}: "+",".join(pair[1]))
            
    #### Identification
    def identify(self, handle: str, # handle: str -> path or open(path, 'r')
            min_length: int = 5000, max_length: int = 0,        
            specificity: float = 0.8, accuracy: float = 0.65, 
            flg_use_NN: bool = True, entropy_cutoff: float = .5, method: str = "max") -> (list, list):
        matrix = []
        try:
            import read_parser as SeqIO
        except Exception as e:
            raise RuntimeError("Module 'read_parser' cannot be open! Check dependences.") from e
            
        # Detailed log of identification process
        log = []
        for rec in SeqIO.iterate_sequences(handle):
            seq = rec.seq.upper()
            if len(seq) < min_length or (max_length and len(seq) > max_length):
                continue
            log.append(f"{'=' * 30}\n{rec.description}\nLength = {len(seq)} bp.; accuracy = {accuracy}; specificity = {specificity}.\n")
            species, log_record = self.identify_by_kmers(seq=seq, 
                specificity=specificity, 
                accuracy=accuracy,
                method=method,
                flg_use_NN=flg_use_NN,
                entropy_cutoff=entropy_cutoff,
                identified=[],
                log=[],
            )
            
            log += log_record
            log.append("*" * 10)
            
            if len(species) and any([ls[3] for ls in species]):
                log.append(f"Identified as: {species[0][0]} with identity {species[0][1]}")
                species.sort(key=lambda sp: float(sp[1]), reverse=True)
                matrix.append({rec.description:[sp for sp in species if sp[3]]})   # species = [[species, identity float, path, species_level True/False], ...]
            else:
                log.append("Identification failed")
                if len(species):
                    species.sort(key=lambda sp: [len(sp[2].split('.')), float(sp[1])], reverse=True)
                matrix.append({rec.description: [species[0]] if len(species) else species})   # species = [[species, identity float, path, species_level True/False], ...]
            log.append("\n")
            species = []
        return matrix, log
    
########################################################################
class Ancestor(Intermediate_Node):                  # All intermediate nodes including LCA nodes
    def __init__(self,title,path):
        Intermediate_Node.__init__(self,title,path)

########################################################################
class Mutation:                                     # Mutation or evolutionary event object
    def __init__(self,title,initial_state="0",new_state="1"):
        self.title = title
        self.initial_state = initial_state
        self.new_state = new_state
        
    def __repr__(self):
        return "\t".join([self.title,self.initial_state,self.new_state])
        
    def __str__(self):
        #return f"{self.title}: {self.initial_state}->{self.new_state}"
        if str(self.initial_state) != "0":
            return f"{self.initial_state}{self.title}{self.new_state}"
        return str(self.title)
        
########################################################################
class ModelTask:                                        # Model task description
    def __init__(self,title,units=[]):
        self.title = title
        self.units = {unit:True for unit in units}
        self.qualifiers = {}
        
    def __repr__(self):
        return self.title
    
    def __str__(self):
        return self.__repr__()

########################################################################
if __name__ == "__main__":
    '''
    oRoot = Root()
    oRoot.add("path")
    
    cont = container.Collection()
    ls = [1,2,3,4,5]
    cont.extend(list(map(lambda i: Mutation(i), ls)))
    print(len(cont))
    print(cont)
    '''
    path = os.path.join("..","input","common_polymorphisms.v5.dat")
    #abbreviations = tools.open_text_file(os.path.join(path,"common_polymorphisms.v5_matrix_abbreviations.txt"),True,"\t",True,1)
    oTree = tools.take_from_shelve(path)
    titles = oTree.get_endnode_titles()
    titles = list(map(lambda title: [title,title.replace(".","_")], titles))
    titles = list(filter(lambda ls: ls[0] != ls[1], titles))
    for old_name,new_name in titles:
        oTree.rename_node(old_name,new_name)
    tools.shelve_file(os.path.join("..","output","common_polymorphisms.v5.dat"))
    '''
    print(oTree.get_node_address("ERR067592"))
    print(oTree.get_node_address("2"))
    oTree.rename_node("2","A")
    print(oTree.get_node_address("A"))
    print(oTree.get_node_address("ERR067592"))
    #oTree.lineage_overlap_table()
    #oTree.mutation_summary("mutation_summary.txt",["20","31"])
    #print("\n".join((oTree.get_ancestor_states())))
    
    print("longest",oTree.get_longest_branch())
    print("shortest",oTree.get_shortest_branch())
    print("total",oTree.get_total_branch_length())
    print("average",oTree.get_average_branch_length())
    
    print()
    print("root - ERR067577",oTree.get_distance("ERR067577"))
    print("root - ERR067619",oTree.get_distance("ERR067619"))
    print("ERR067619 - ERR067577",oTree.get_distance("ERR067619","ERR067577"))
    print("ERR067619 - ERR067631",oTree.get_distance("ERR067619","ERR067631"))
    print("relative node 3",oTree.get_relative_distance("3"))
    '''
    
