import os
from functools import reduce
import container, tools, progressbar
from Bio import Phylo

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable, Any
"""
How this fits your architecture

Traversal uses your child_node dict; no reliance on your address strings or get_distance(...).

Branch lengths are read from each node’s branch_length and accumulated to get true depths from the root.

Shortcuts: every committed cluster is exposed at the root as Nick(title, object=node_root) under self.shortcuts[label]. 
That gives you fast navigation to the coalesced zones without altering the original links.

Fan-out control: any cluster whose tip count exceeds target_fanout is split internally (greedy by child subtrees) until 
each subcluster is ≤ target_fanout tips. You can swap oversize_split for a more sophisticated splitter later (e.g., 
depth median or patristic radius).

Adaptive sectors: quantile binning yields narrow sectors where nodes are dense and wide sectors where they’re sparse — 
exactly the behavior you asked for. If you want even more sensitivity, you can replace quantiles with Bayesian Blocks 
later; the rest of the code won’t change much.

Notes / extensions

If you prefer the reduced tree to be materialized (physically collapsed), we can add an in_place_collapse=True mode 
that replaces each subtree by a single Intermediate_Node carrying metadata and rewires child_node accordingly. 
I kept the non-destructive shortcut approach to be safe.

You can seed the number of sectors by passing sectors=... (e.g., equals the number of levels you want), otherwise 
it’s chosen from the number of tips and target_fanout.

"""

########################################################################
@dataclass
class Cluster:
    """Lightweight record of a collapsed in-sector subtree."""
    label: str
    title: str
    root_path: str
    sector: Tuple[float, float]          # (a, b) depth sector
    member_tips: List[str]               # leaf titles under this cluster
    member_internal: List[str]           # internal node titles under this cluster
    min_depth: float
    max_depth: float
    size_tips: int
    size_nodes: int
    
########################################################################
class Nickname:
    __slots__ = ("title", "obj", "child_nodes")

    def __init__(self, title: str, obj: Any):
        self.title: str = title
        self.obj: Any = obj
        # reference the wrapped object's child_nodes if present, else None
        self.child_nodes: Optional[Any] = getattr(obj, "child_nodes", None)

    def is_terminal(self) -> bool:
        # Prefer the wrapped object's API if available
        if hasattr(self.obj, "is_terminal"):
            return bool(self.obj.is_terminal())
        # Fallback heuristic: no children means terminal
        if self.child_nodes is None:
            return True
        try:
            return len(self.child_nodes) == 0
        except TypeError:
            # child_nodes exists but isn't sized; assume non-terminal
            return False
            
########################################################################
class TreeParser:
    def __init__(self,args={},tree_format="",current_dir=""):
        self.current_dir = current_dir
        self.args = args
        self.annotation = []
        if self.args:
            self.set_annoation()
        self.tree_format = tree_format
        self.oTree = None
        
    def __call__(self, args: dict = {}, tree_format: str = "newick"):
        if args != {}:
            self.args = args
            self.set_annoation()
        if tree_format:
            self.tree_format = tree_format
        elif self.args["-t"].split(".")[-1] in ("nwk","newick","tre","tree","phy"):
            self.tree_format = "newick"
        elif self.args["-t"].split(".")[-1] in ("xml","nexml"):
            self.tree_format = "phyloxml"
        else:
            self.tree_format = "newick"
        # Tree file parsing
        tree_file = os.path.join(self.current_dir,self.args["-i"],self.args["-f"])
        try:
            self.oTree = Phylo.read(open(tree_file),tree_format)
        except:
            tools.msg(f"Problem with parsing tree file {tree_file}!")
            return
        return self.parse()
        
    def set_annoation(self,args: dict = {}):
        if args != {}:
            self.args.update(args)
        # Lineage file parsing
        self.annotation = []
        if self.args["-l"]:
            label_file = os.path.join(self.current_dir,self.args['-i'],self.args['-l'])
            if not os.exists(label_file):
                msg(f"Label file {label_file} does not exist!")
                return False
            try:
                self.annotation = tools.open_text_file(label_file, True, "\t", True, 1)
                return True
            except:
                msg("Problem with parsing the lineage file!")
                return False
            if self.annotation == "":
                msg("Problem with parsing the lineage file!")
                return False
    
    def parse(self,oTree=None,args={}):
        if oTree != None:
            self.oTree = oTree
        if self.oTree == None:
            return
        if args:
            self.args.update(args)
            success = self.set_annoation()
            if not success:
                return
        # Processing of clades in the three
        terminals = self.oTree.get_terminals()
        if len(terminals)==0:
            tools.msg("The selected tree is empty!")
            return
        # Creation and filling a tree object
        oRoot = Root(self.args)
        counter = 1
        bar = progressbar.indicator(len(terminals),"Tree parsing")
        for i in range(len(terminals)):
            endnode_title = terminals[i]
            path = self.oTree.get_path(endnode_title)
            pathway = [["Root",0,0]]
            for clade in path:
                if not clade.name:
                    # clade naming
                    clade.name = str(counter)
                    counter += 1
                pathway.append([clade.name,clade.branch_length,clade.confidence])
            endnode_name = pathway[-1][0]
            pathway[-1].append(self._get_lineage(endnode_name))
            oRoot.append(pathway)
            bar(i)
        bar.stop()
        # Add BioPython tree object
        #oRoot.set_tree_object(self.oTree)
        return oRoot

    def _get_lineage(self,title):
        title = title.strip()
        selection = list(filter(lambda ls: title in (ls[0].strip(),ls[1].strip()), self.annotation))
        if len(selection)==0:
            selection = list(filter(lambda ls: title.find(ls[0])==0 or title.find(ls[1])==0, self.annotation))
            if len(selection)==0:
                return "unknown"
        '''
        elif len(selection) > 1:
            raise IOError(f"Entry {title} is not unique in the lineage list!")
        '''
        return selection[0][-1]

########################################################################
class Node:
    def __init__(self,title):
        self.title = str(title).strip()             # Node title
        self.events = container.Collection()        # Collection of titled muttation object occured at this node
        self.endnode = False                        # endnode marker
        self.branch_length = 1                      # length of the branch leading to this node from the parent node
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
        
    def is_root(self):
        if self.title.upper() == "ROOT":
            return True
        return False
    
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
        
    def get_node_clusters(self, flg_use_shortcuts=True):
        if self.is_terminal():
            return self.title
        cl = []
        if flg_use_shortcuts and self.shortcuts:
            for node in self.shortcuts:
                cl.append(node.obj.get_node_clusters(flg_use_shortcuts) if isinstance(node, Nickname) else node.get_node_clusters(flg_use_shortcuts))
        else:
            for node in self.child_nodes:
                cl.append(node.get_node_clusters(flg_use_shortcuts))
        return cl
        
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
        '''
        mutations = []
        if endnode_title and refseq != "" and fasta != {}:
            mutations = list(map(lambda i: [str(i),refseq[i],fasta[endnode_titles[0]][i]], mut_ls))
        #print("db:193",mut_ls,mutations,endnode_title,len(refseq),len(fasta))
        if mutations != []:
            self.events.extend(list(map(lambda ls: Mutation(ls[0],ls[1],ls[2]), mutations)))
        else:
            self.events.extend(list(map(lambda i: Mutation(str(i)), mut_ls)))
        '''
        
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
    def __init__(self,title,path,address,branch_length,lineage):
        Node.__init__(self,title)
        self.endnode = True                     # endnode marker
        self.branch_length = branch_length      # length of the branch leading to this node from the parent node
        self.lineage = lineage                  # lineage name
        self.path = path+f">{self.title}"       # path to the end node in indexes
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
    def __init__(self,title,path):
        Node.__init__(self,title)
        self.child_nodes = container.Collection()   # Collection of named child nodes
        self.shortcuts = container.Collection()     # Shortcuts for clusters
        self.confidence = 0                         # branch length confidence
        self.LCA = False                            # lowest common ancestor marker - the last node before at least one end node among the child nodes
        self.lineages = []                          # list of lineages of the child end nodes
        self.path = path                            # Path to the node from the root like 'root>0>0>1>1'
        self.index = int(self.path.split(">")[-1])  # Node index in the parent node: 0 or 1 in a dichotomous tree where 0 is the right branch and 1 is the left branch
    
    def add(self,pathway, path="Root", address="Root"):                   # add new nodes as child objects
        if len(pathway) == 0:
            return
        # Add lineage to the list of lineages of the current node
        lineage = pathway[-1][-1]
        if lineage not in self.lineages:
            self.lineages.append(lineage)
        # Setting a new node
        title,branch_length,confidence = pathway[0]
        # Check consistency
        if self.title != title.strip():
            print("db:420",self.title,title.strip(),pathway)
            raise IOError(f"Error with tree parsing. Node title {self.title} does not match with the submitted title {title}!")
        # Removing the current node information from the pathways
        pathway = pathway[1:]
        # Appending a new end node
        if len(pathway)==1:
            return self.add_endnode(pathway[0],path,address)
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
    def add_endnode(self,values,path,address):
        title,branch_length,confidence = values
        try:
            self.branch_length = float(branch_length)
            self.confidence = float(confidence)
        except:
            pass
        if title.strip() in self.child_nodes.get_titles():
            raise IOError(f"Error with tree parsing. Endnode {title} is not unique!")
        path += ">%d" % len(self.child_nodes)
        address += ">%s" % title
        oEndNode = EndNode(title,path,address,branch_length,"")
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
            print("db:426",new_name)
            self.rename(new_name)
            for oChildNode in self.child_nodes:
                oChildNode.rename_address(current_name,new_name)
                
    # Set shortcuts
    def set_shortcuts(self, cluster_root_path: str, 
        internal_nodes: list,
        end_nodes: list) -> None:
        # Get child node
        def get_child_node(path: list) -> Node:
            if not path:
                return self
            node = self
            while path:
                ind = path[0]
                try:
                    ind = int(ind)
                except:
                    pass
                path = path[1:]
                node = node.child_nodes[ind]
                if node.is_terminal:
                    return node
            return node
        
        # Remove "Root>" from path
        address = cluster_root_path
        if address.upper().find("ROOT>") == 0:
            address = address[5:]
            
        # Cluster root is reached
        if address == "":
            internal_nodes = [node_path[len(self.path) + 1:] for node_path in internal_nodes]
            end_nodes = [node_path[len(self.path) + 1:] for node_path in end_nodes]
            # Remove internal nodes, if they are represented by end nodes
            internal_nodes = [node_path for node_path in internal_nodes if 
                all([end_node_path.find(node_path) != 0 for end_node_path in end_nodes])]
                
            # Do not allow combining end nodes with intermediate nodes
            # Create an additional intermediate node for end nodes
            if len(end_nodes) > 0 and len(internal_nodes) > 0:
                oINode = Intermediate_Node(title="0", path=cluster_root_path + ">0")
                for node in end_nodes:
                    oINode.child_nodes.append(node)
                self.shortcuts.append(Nickname(title=oINode.title, obj=oINode))
                end_nodes = []
                
            # Link inermediate nodes to shortcuts
            for node_path in internal_nodes:
                node = get_child_node(path = node_path.split(">"))
                self.shortcuts.append(Nickname(title=str(len(self.shortcuts)), obj=node))
            # Link end nodes to shortcuts
            for node_path in end_nodes:
                node = get_child_node(path = node_path.split(">"))
                if node.is_terminal():
                    self.shortcuts.append(node)
                else:
                    self.shortcuts.append(Nickname(title=str(len(self.shortcuts)), obj=node))
            return 
            
        # Walk through child nodes
        path = address.split(">")
        ind = int(path[0])
        path = path[1:]
        self.child_nodes[ind].set_shortcuts(cluster_root_path = ">".join(path),
            internal_nodes = internal_nodes,
            end_nodes = end_nodes)

########################################################################    
class Root(Intermediate_Node):                      # Root of the tree
    def __init__(self,args={}):
        Intermediate_Node.__init__(self,"Root","0")
        self.args = args                            # Last call arguments
        self.tree_file_name = ""                    # Oroginal tree file
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
    def append(self,pathway):                       # Append a new path of nodes
        def parse(item: str) -> tuple:
            title = item
            branch_length = confidence = 1.0
            if item[0] == "(":
                d = item.find(")")
                if d > 1 and d < len(item) - 2:
                    try:
                        branch_length = float(item[1:d])
                        title = item[d + 1:]
                    except:
                        pass 
            return (title, branch_length, confidence)
                
        if isinstance(pathway, str):
            pathway = pathway.split(">")
            
        pathway = [parse(".".join(pathway[:i])) for i in range(1, len(pathway))]
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
        
    # Get parent node for the given node or None for Root
    def get_parent(self, n: Node) -> Node:
        if n.is_root():
            return None
        if n.is_terminal():
            path = self.get_endnode_path(n.title)
        else:
            path = n.path
        # path = 'Root>0>0>1>0>1'
        # Remove address of the current node
        path = path.split(">")[:-1]
        # Check again whether the current not is not the root
        if len(path) == 0:
            return None
        # Top level parent is root
        node = self
        # Remove 'root' from the path
        path = path[1:]
        # Walk through the path to the parent node
        while path:
            ind = int(path[0])
            node = node.child_nodes[ind]
            # Remove the visited node from the path
            path = path[1:]
        return node

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
        #return list(filter(lambda ls: nodename in ls, list(map(lambda address: address.split(">"), self.get_addresses()))))
        return [parts for a in self.get_addresses()
                if (nodename in a) and (nodename in (parts := a.split(">")))]
            
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
            oChildNode.populate_branch_lengths(branches,"Root")
            
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
    
    # Create a Newick tree string
    def get_newick_tree(self, decimals: int = 6, flg_use_shortcuts: bool = False) -> str:
        """
        Return the tree in Newick format as a string.

        - Only leaf (EndNode) names are included.
        - Branch lengths are emitted for every non-root edge.
        - If flg_use_shortcuts is True and an intermediate node has a non-empty
          `shortcuts` container, traversal uses `shortcuts` instead of `child_nodes`
          at that node (locally).
        """

        def fmt_len(x: float) -> str:
            if x is None:
                return ""
            s = f"{round(float(x), decimals):.{decimals}f}"
            s = s.rstrip('0').rstrip('.')
            return s if s else "0"

        def quote_if_needed(name: str) -> str:
            if name is None:
                return ""
            if any(ch in name for ch in " \t\n\r():,;[]'\""):
                return "'" + name.replace("'", "''") + "'"
            return name

        def choose_children_container(node):
            """
            Decide which container to use at this node:
            - If flg_use_shortcuts is True and node.shortcuts is non-empty -> use shortcuts
            - Else -> use child_nodes
            Returns the chosen container or None.
            """
            # Prefer shortcuts only if explicitly requested and available
            if flg_use_shortcuts and hasattr(node, "shortcuts") and node.shortcuts:
                return [node.obj if node.is_terminal() == False else node for node in node.shortcuts]
            # Fallback to child_nodes
            if hasattr(node, "child_nodes") and node.child_nodes:
                return node.child_nodes
            return None

        def iter_children(node):
            """
            Yield child nodes from either the chosen shortcuts or child_nodes.
            Supports list-like or dict-like containers.
            """
            cont = choose_children_container(node)
            if cont is None:
                return
            try:
                # list-like
                for ch in cont:
                    if ch is not None:
                        yield ch
            except TypeError:
                # dict-like
                for ch in cont.values():
                    if ch is not None:
                        yield ch

        def is_leaf(node) -> bool:
            # Your EndNode has is_terminal(); others may not.
            # Treat as leaf if is_terminal() is True OR there are no children to traverse here.
            term = getattr(node, "is_terminal", lambda: False)()
            if term:
                return True
            return choose_children_container(node) is None

        def to_newick(node, is_root=False) -> str:
            if is_leaf(node):
                name = quote_if_needed(getattr(node, "title", ""))
                bl = getattr(node, "branch_length", None)
                return f"{name}{'' if is_root else (':' + fmt_len(bl) if bl is not None else '')}"
            else:
                parts = [to_newick(ch, is_root=False) for ch in iter_children(node)]
                inside = ",".join(parts)
                if is_root:
                    return f"({inside})"
                else:
                    bl = getattr(node, "branch_length", None)
                    return f"({inside}){':' + fmt_len(bl) if bl is not None else ''}"

        return to_newick(self, is_root=True) + ";"
        
    def get_ancestor_states(self):
        return self.get_ancestors()
        
    def get_node_size(self,nodetype="All",nodename="Root"): # Node types: all, intermediate, endnode
        if not self.has(nodename):
            tools.msg(f"Node with the name {nodename} does not exist!")
            return 0
        if nodename in self.terminals:
            return 1
        addresses = self.get_addresses()
        if nodename != "Root":
            addresses = list(map(lambda s: s[s.find(">"+nodename)+len(nodename)+2:], filter(lambda address: address.find(">"+nodename) > -1, addresses)))
        ls = list(map(lambda address: address.split(">"), addresses))
        if nodetype == "endnode":
            ls = list(map(lambda item: item[-1], ls))
        elif nodetype == "intermediate":
            ls = list(map(lambda item: item[:-1], ls))
        if nodetype != "endnode":
            ls = tools.dereplicate(reduce(lambda ls1,ls2: ls1+ls2, ls))
        size = len(ls)
        if nodename == "Root":
            size -= 1       
        return size
        
    def get_longest_branch(self,nodename="Root"):
        return self.path_length(nodename,"longest")
        
    def get_shortest_branch(self,nodename="Root"):
        return self.path_length(nodename,"shortest")
        
    def get_total_branch_length(self,nodename="Root"):
        return self.path_length(nodename,"total")
        
    def get_average_branch_length(self,nodename="Root"):
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
        if nodename != "Root":
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
            
    def get_distance(self,nodename_1,nodename_2="Root"):
        if nodename_1==nodename_2:
            return 0
        if "branches" not in self.qualifiers:
            self.set_branches()
        available_nodes = tools.dereplicate(self.avail_nodes())

        if nodename_1 not in available_nodes or nodename_2 not in available_nodes:
            tools.msg(f"Nodes with the names {nodename_1} or {nodename_2} do not exist!")
            return 0
        addresses = self.get_addresses()
        selection = list(filter(lambda address: address.find(">"+nodename_1) > -1 and (nodename_2=="Root" or address.find(">"+nodename_2) > -1), addresses))
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
        if nodename=="Root":
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
        for line in table:
            print("\t".join(line))
        print()
        for pair in lineage_pairs:
            print(f"{pair[0][0]}-{pair[0][1]}: "+",".join(pair[1]))

    def color_subtree(self,nodename,color="red"):
        titles = self.get_subordinate_node_titles(nodename)
        success = False
        for title in titles:
            try:
                list(self.oTree.find_clades({"name":title}))[0].color = color
                success = True
            except:
                pass
        if success:
            self.flg_was_modified = True
        Phylo.draw(self.oTree)
            
    def color_lineage_branches(self,lineage,color="blue"):
        pass
        
    def coalesce(self, clusters: list) -> True:     # clusters = list of Cluster objects
        # Sort from longest branches to shortest
        clusters.sort(key = lambda cl: len(cl.root_path.split(">")), reverse=True)
        for cl in clusters:
            self.set_shortcuts(cluster_root_path = cl.root_path, 
                internal_nodes = cl.member_internal,
                end_nodes = cl.member_tips)
    
    def cluster_tree(self,
                 target_fanout: int = 5,
                 sectors: Optional[int] = None,
                 max_sectors: int = 64,
                 method: str = "quantile",
                 in_place_shortcuts: bool = True,
                 shortcut_prefix: str = "CLUST_",
                 oversize_split: str = "greedy") -> List[Cluster]:
        """
        Reduce a dichotomic dendrogram by clustering subtrees inside
        depth 'sectors' (variable-width bins along distance-from-root).
    
        Parameters
        ----------
        target_fanout : int
            Desired ~number of descendant tips per cluster (≈5 by default).
        sectors : Optional[int]
            If None, chosen adaptively from tree size. Otherwise, fixed number
            of depth sectors (variable-width by equal-count quantiles).
        max_sectors : int
            Upper bound when auto-selecting sectors.
        method : str
            Currently only 'quantile' (equal-count bins) is implemented.
        in_place_shortcuts : bool
            If True, create Nick-based shortcuts at the root for each cluster.
            The original topology is not destroyed.
        shortcut_prefix : str
            Prefix for new shortcut titles (e.g., 'CLUST_001').
        oversize_split : str
            Strategy to split clusters with too many tips: 'greedy' (by children).
    
        Returns
        -------
        List[Cluster]
            Structured info about each cluster that was created.
        """
        
        def dfs(node, d):
            preorder.append(node)
            depth[node.title] = d
            if not node.is_terminal() and node.child_nodes:
                for child in node.child_nodes:
                    parent[child.title] = node
                    # Accumulate distance using each child's own branch_length
                    child_bl = self.get_distance(child.title)
                    dfs(child, d + child_bl)
    
        def qtile(q):
            idx = int(round(q * (N - 1)))
            return sorted_depths[idx]
        
        def inside_sector(n, a, b):
            # Entire subtree is within sector if all descendants lie in [a,b]
            return (min_d[n.title] >= a) and (max_d[n.title] <= b)
    
        def collect_members(n) -> Tuple[List[Any], List[Any]]:
            tips, inns = [], []
            stack = [n]
            while stack:
                x = stack.pop()
                if x.is_terminal():
                    tips.append(x)
                else:
                    inns.append(x)
                    stack.extend(children[x.title])
            return tips, inns
            
        # Split an oversized cluster by greedily grouping children until tip cap
        def split_oversize(n, cap) -> List[Any]:
            """Return a list of subtree roots each with <= cap tips."""
            # If already small enough, keep as one
            tips, _ = collect_members(n)
            if len(tips) <= cap:
                return [n]
            # Greedy: start from this node's children and group them
            groups = []
            cur_group = []
            cur_count = 0
            for ch in children[n]:
                ch_tips, _ = collect_members(ch)
                tcount = len(ch_tips)
                # If single child too big, recurse on it
                if tcount > cap and children[ch]:
                    groups.extend(split_oversize(ch, cap))
                    continue
                if cur_count + tcount <= cap:
                    cur_group.append(ch)
                    cur_count += tcount
                else:
                    # finalize current group -> make a virtual root by choosing the MRCA (n),
                    # but we'll just return the grouped children as separate cluster roots
                    if cur_group:
                        # return each child as its own cluster root
                        groups.extend(cur_group)
                    # start new group with current child
                    cur_group = [ch]
                    cur_count = tcount
            if cur_group:
                groups.extend(cur_group)
            return groups
    
        # Create a Cluster record and optional Nick shortcut
        def commit_cluster(node_root, a, b):
            nonlocal cluster_ix
            tips, inns = collect_members(node_root)
            # Enforce size cap
            if len(tips) > target_fanout:
                parts = split_oversize(node_root, target_fanout)
                for p in parts:
                    commit_cluster(p, a, b)
                return
            label = f"{shortcut_prefix}{cluster_ix:03d}"
            cl = Cluster(
                label=label,
                title=getattr(node_root, "title", str(node_root)),
                root_path=node_root.path,
                sector=(a, b),
                member_tips=[getattr(t, "path", str(t)) for t in tips],
                member_internal=[getattr(x, "path", str(x)) for x in inns if x is not node_root],
                min_depth=min_d[node_root.title],
                max_depth=max_d[node_root.title],
                size_tips=len(tips),
                size_nodes=len(tips) + len(inns),
            )
            clusters.append(cl)
            if in_place_shortcuts:
                # Expose this cluster via a Nickname at the root
                # Assumes you have Nickname(title=..., object=...)
                self.shortcuts.append(Nickname(title=label, obj=node_root))
            cluster_ix += 1
    
        # --------- 1) Traverse to collect topology, depths, parents ----------
        # We'll do one DFS to compute depth-from-root and parent map
        parent: Dict[Any, Optional[Any]] = {self: None}
        depth: Dict[Any, float] = {self: 0.0}
        preorder: List[Any] = []
        
        dfs(self, 0.0)
    
        all_nodes = preorder
        leaves = [n for n in all_nodes if n.is_terminal()]
        internals = [n for n in all_nodes if not n.is_terminal()]
        
        # --------- 2) Postorder to compute min/max descendant depths ----------
        min_d: Dict[Any, float] = {}
        max_d: Dict[Any, float] = {}
    
        # Build children list helper
        children: Dict[Any, List[Any]] = {}
        for n in all_nodes:
            children[n.title] = n.child_nodes.get() if not n.is_terminal() else []
    
        # Postorder traversal (reverse of preorder with a check)
        for t in [node.title for node in reversed(all_nodes)]:
            if not children[t]:  # leaf
                min_d[t] = max_d[t] = depth[t]
            else:
                md = min([min_d[c.title] for c in children[t]] + [depth[t]])
                xd = max([max_d[c.title] for c in children[t]] + [depth[t]])
                min_d[t], max_d[t] = md, xd
    
        # --------- 3) Build the depth axis and choose sectors ----------
        depth_samples = [depth[n.title] for n in all_nodes]  # nodes density along axis
        N = len(depth_samples)
        
        if sectors is None:
            # Heuristic: aim ~ (total_tips / target_fanout) clusters; cap by max_sectors
            approx_clusters = max(1, round(len(leaves) / max(1, target_fanout)))
            sectors = min(max_sectors, max(1, approx_clusters))
        sectors = max(1, sectors)
        
        # Quantile edges -> variable-width bins w/ roughly equal counts of nodes
        # Add tiny jitter to collapse duplicate edges if depths are identical
        qs = [i / sectors for i in range(sectors + 1)]
        sorted_depths = sorted(depth_samples)
    
        edges = [qtile(q) for q in qs]
        
        # Ensure strictly increasing edges (handle flat regions)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-12  # tiny epsilon to separate
    
        sectors_ab = list(zip(edges[:-1], edges[1:]))
        
        # --------- 4) Find maximal in-sector subtrees (monophyletic) ----------
        clusters: List[Cluster] = []
        cluster_ix = 1
    
        # Helper: collect tips/internal under a node
        # For each sector, find maximal roots inside it
        for (a, b) in sectors_ab:
            # Candidates that are fully inside
            candidates = [n for n in all_nodes if inside_sector(n, a, b)]
            # Keep only those whose parent is not fully inside (maximality)
            for n in candidates:
                p = parent.get(n.title, None)
                #p = get_parent(n)
                if p is None or not inside_sector(p, a, b):
                    commit_cluster(n, a, b)
        
        # Remove single node clusters
        clusters = [cl for cl in clusters if cl.size_nodes > 1]
        return clusters
    
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
    
