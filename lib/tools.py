import sys, os, math, pickle, time, random, shutil, tempfile
from datetime import datetime
from pathlib import Path
from itertools import groupby
# in-house modules
import word_db, nwmapper, bitwiser, progressbar
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def get_current_date():
    return datetime.now().strftime("%d-%m-%Y")

def clean(path):
    """Removes the folder at the specified path, including all its files and subfolders."""
    if os.path.isdir(path):
        try:
            shutil.rmtree(path)
            print(f"🧹 Removed: {path}")
        except Exception as e:
            print(f"Error removing directory {path}: {e}")
            
def copy2(old_name, new_name):
    shutil.copy2(old_name, new_name)

def openDBFile(fname: str, key='$db$', supplkey="$suppl$"):
    with open(fname, 'rb') as file:
        try:
            data = pickle.load(file)
            DB = data[key]
            supplementary = data[supplkey]
            return fname,DB,supplementary
        except:
            raise TypeError(f"File {fname} has wrong formatting or corrupted!")

def saveDBFile(data,
               fname: str,
               supplementary=None,
               key='$db$',
               supplkey="$suppl$",
               flg_appendData=None):
    """
    Atomically write {key: data, supplkey: supplementary} to a .wdb file.
    Creates a temp file in the same directory and os.replace()s it into place.
    Returns the final path as a string.
    """
    path = Path(fname)
    # Ensure the parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {key: data, supplkey: supplementary}

    # Create a temp file in the same directory for atomic replace
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        # Atomic on the same filesystem
        os.replace(tmp_name, str(path))
    except Exception:
        # Best-effort cleanup of the temp file
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise

    return str(path)

def saveTextFile(strText,fname=None):
    if not fname:
        fname = asksaveasfilename([("Text file", "*.txt"),("Text file", "*.out")])
    if not fname:
        return
    with open(fname, "w") as ofp:
        ofp.write(strText)
        ofp.flush()
    return fname

def openTextFile(fname=None):
    if not fname:
        fname = askfilename([("Text file", "*.txt"),("Text file", "*.out")])
    if not fname or not os.path.exists(fname):
        return
    try:
        with open(fname) as f:
            data = f.read().replace("\r","")
        return data
    except:
        return

def openSeqFile(fname, concatenate = True, delimiter = 50 * "N"):
    if not os.path.exists(fname):
        return {}
    if fname[fname.rfind("."):] in (".fa",".fsa",".fas",".fst",".fna",".fasta"):
        records = list(SeqIO.parse(fname, "fasta"))
    elif fname[fname.rfind("."):] in (".gbk", ".gbf", ".gbff", ".gb"):
        records = list(SeqIO.parse(fname, "genbank"))
    
    if concatenate:
        seqname = os.path.basename(fname)
        seqname = seqname[:seqname.rfind(".")] if "." in seqname else seqname
        seqlist = {seqname:delimiter.join([str(seq.seq).upper() for seq in records])}
    else:
        seqlist = dict(zip([seq.description for seq in records], [str(seq.seq).upper() for seq in records]))
        
    return seqlist

def msg(msg,title=""):
    if title:
        title += "\n"
    print(f"\n{title}{msg}\n")
       
def ascertain_integer(v, low_cutoff=0, top_cutoff=None):
    try:
        ivalue = int(v)
        if ivalue > low_cutoff:
            if top_cutoff != None and ivalue > top_cutoff:
                return False
            return ivalue
    except (ValueError, TypeError):
        return False

def ascertain_float(v, low_cutoff=0, top_cutoff=None):
    try:
        ivalue = float(v)
        if ivalue >= low_cutoff:
            if top_cutoff != None and ivalue > top_cutoff:
                return False
            return ivalue
    except (ValueError, TypeError):
        return False

"""
Nicely prints the Newick string representation as a tree.
"""    
def print_newick_tree(newick):
    import ete3  # Install with: pip install ete3
    from ete3 import Tree

    try:
        # Use 'quoted_node_names=True' to handle special characters in node names
        tree = Tree(newick + ";", quoted_node_names=True, format=True)
        print(tree.get_ascii(show_internal=True))
    except Exception as e:
        print(f"Error printing the tree: {e}")
    
# set number of diggets after the dot for a given float num        
def format_number(num,dig,zoom=0):
    return int((10**(dig+zoom))*num)/float(10**dig)
    
def format_file_name(name, echo=True):
    original_name = name
    for symbol in (" ", ":", "/", "?", "#", "=", "$", "*", "[", "]", "@", "|", "<", ">", "\""):
        name = name.replace(symbol, "_")
    if original_name != name and echo:
        tools.msg(f"Original workspace name {original_name} was changed to {name}!","Warning!")
    return name 

def distBinning(report,headers,sequence,seqlength=0):
    convertor = nwmapper.Mapper()
    ranks = {"--":0,"+":.25,"++":.75,"+++":1.0}
    counter = [0]*len(headers)
    maxdist = [0]*len(headers)
    if not seqlength:
        seqlength = float(len(sequence))-sequence.count("N")
    word_number = 0
    lines = []
    for r in range(len(report)):
        if len(report[r])==6:
            num,wlength,x,y,data,table = report[r]
        elif len(report[r])==5:
            num,wlength,x,y,data = report[r]
        else:
            raise Error("Tools:195")
        lines.append("")
        word_number += 1
        word = convertor(wlength,x,y)
        count = count_word(sequence,word)
        wlength,rx,ry = convertor.revcomplement([wlength,x,y])
        rword = convertor(wlength,rx,ry)
        if rword != word:
            count += count_word(sequence,rword)
        if not count:
            count = 1
        frq = 100000.0*count/seqlength
        occurrence = percentile(frq,wlength)
        lines[-1] += word + "\t" + str(format_number(occurrence,4)) + "\t"
        for i in range(len(headers)):
            # percentiles in the range 0.05-0.95
            if type(data)==type([]):
                s = data[i]
                if s in ranks:
                    expectation = ranks[s]
                else:
                    p = s.find("[")
                    if p==-1:
                        expectation = float(s)/10.0
                    else:
                        expectation = float(s[:p])/10.0
            else:
                expectation = (bitwiser.get_rankborder(data,headers[i][1])-1.0)/10.0
            if expectation >= .5:
                maxval = 0
            else:
                maxval = 1.0
            lines[-1] += str(format_number(expectation,4))+"\t"
            counter[i] += (1.0 + (occurrence-expectation)/(2.0-expectation))*(occurrence-expectation)**2
            maxdist[i] += (1.0 + (maxval-expectation)/(2.0-expectation))*(maxval-expectation)**2

    report = {}
    accessions = []
    for i in range(len(headers)):
        if type(headers[i])==type(""):
            key = headers[i]
        else:
            key = headers[i][0]
        # Linear distance
        report[key] = 10.0*math.sqrt(float(counter[i])/maxdist[i])
        # Log distance
        #1
        #report[key] = 10.0*(math.exp(math.sqrt(float(counter[i])/maxdist[i])) - 1)/(math.e - 1)
        #2
        #report[key] = 10.0*(math.exp(float(counter[i])/maxdist[i]) - 1.0)/(math.e - 1)
        #3
        #report[key] = 10.0/math.exp(math.sqrt(1 - float(counter[i])/maxdist[i]))
        #report[key] = 20.0*(math.exp(float(counter[i])/maxdist[i]) - 1)/(math.e - 1)
        accessions.append(headers[i][0])
    #TextEditor("Follow Up",str(int(seqlength))+"\n\t\t"+"\t".join(accessions)+"\n"+"\n".join(lines))
    return report

# Covert all possible values of word frequencies to "--","+","++","+++"
def format_values(value):
    ranks = {"--":2.5,"+":5,"++":7.5,"+++":10}
    if value in ("--","+","++","+++"):
        return value
    try: 
        value = int(value)
        if value < 0 or value > 10:
            print("Error 252 formating value " + str(value))
            return None
    except:
        p = str(value).find("[")
        if p > -1:
            try:
                value = int(value[:p])
            except:
                print("Error 260 formating value " + str(value))
                return None
        else:
            print("Error 263 formating value " + str(value))
            return None
        if value < ranks("--"):
            return "--"
        elif value < ranks("+"):
            return "+"
        elif value < ranks("++"):
            return "++"
        else:
            return "+++"

# Count word instances in the sequence
def count_word(sequence: str, word: str, count_reverse_complement: bool = False):
    if count_reverse_complement:
        word = reverse_complement(word)
    count = 0
    start = 0
    pos = sequence.find(word)
    while pos >= 0:
        count += 1
        start = pos+1
        pos = sequence.find(word,start)
    return count
    
def word_list_to_counts(sorted_words: list) -> list:    # words = ["ACCTG", "ACCTG", "ATG", "GCTA", "GCTA", "GCTA"] -> [['ACCTG', 2], ['ATG', 1], ['GCTA', 3]]
    """
    Convert a sorted list of words into [[word, count], ...].
    
    Args:
        sorted_words (list[str]): Sorted list of words.
        
    Returns:
        list[list]: Each record is [word, count].
    """
    return [[word, sum(1 for _ in group)] for word, group in groupby(sorted_words)]

# Return percentile of word distribution by word frequency
# Percentiles as 0.01, 0.5, 0.95, ...
def percentile(frq,wlength):
    if frq == 0:
        return 0
    percentile = (math.log(frq)+4.5*math.log(wlength)-9.0)/3.0
    if percentile < 0.05:
        return 0
    if percentile > 0.95:
        return 1.0
    return percentile

# Return border word frequency of the given percentile
# frequency of words per 100 Kbp
def border_frequency(percentile,wlength):
    return math.exp(3.0*percentile+9.0)/(wlength**4.5)

def format_genome(line,column_width=25):
    data = line.split("\t")
    level = data[0]
    if level == "0":
        name = data[1]
    elif level == "1":
        name = data[2]
    elif level == "2":
        name = data[2][0]+"."+data[3]+" "+data[4]+" ["+data[5]+"]"
    elif level == "3":
        name = data[2][0]+"."+data[3]
    short_name = name
    if len(name) > column_width:
        short_name = name[:column_width-3]+"..."
    return short_name,name
            
def print_wordList(report):
    words = []
    convertor = nwmapper.Mapper()
    for word_items in report:
        val,wlength,x,y,data,table = word_items
        words.append(convertor(wlength,x,y))
    return "\n".join(words)

def print_words(genomes,wordlist):
    # Headers
    output = "WORDS\nN\tGenomes" + "\t".join(wordlist) + "\n"
    # Table
    genome_counter = 1
    for genome in genomes:
        short_name,name = format_genome(genome)
        output += "\t".join([str(genome_counter),name])
        for i in range(len(wordlist)):
            val = self.trigger("Get word statistics",[wordlist[i],genome])
            if type(val) == type(0.1):
                val = str(format_number(val,2))
            output += "\t"+val
        output += "\n"
        genome_counter += 1
    return output
    
def print_lineages(lineages,title="",fname=""):
    def dict_to_text_tree(data, indent=0):
        """
        Recursively converts a nested dictionary into a text-based tree string.
        Each level is indented with tabs to represent hierarchy.
        """
        lines = []
        for key, value in data.items():
            # Add the current taxon with the appropriate indentation
            lines.append('\t' * indent + key)
            # Recursively process the children if any
            if isinstance(value, dict):
                lines.extend(dict_to_text_tree(value, indent + 1))
        return lines
    output = []
    if title:
        output = [title]
    output += dict_to_text_tree(lineages)
    output = "\n".join(output)
    if fname:
        saveTextFile(output,fname)
    else:
        msg(output)

def transpose_list(matrix):
    """
    Convert [[a1, a2, a3, ...], [b1, b2, b3, ...], [c1, c2, c3, ...], ...]
    into [[a1, b1, c1, ...], [a2, b2, c2, ...], [a3, b3, c3, ...], ...]
    """
    return [list(row) for row in zip(*matrix)]

def print_genomes(report,headers):
    output = "GENOMES\n"+"\t".join(['C','Table','Word'])
    convertor = nwmapper.Mapper()
    # Headers
    for item in headers:
        output += "\t"+item[0]
    output += "\n"
    # Table
    for word_items in report:
        val,wlength,x,y,data,table = word_items
        if type(val)==type(""):
            valText = val
        else:
            valText = str(format_number(val,2))
        output += "\t".join([valText,str(table),convertor(wlength,x,y)])
        for item in headers:
            if type(data)==type(""):
                val = data
            else:
                val = bitwiser.get_textVal(data,item[1])
            output += "\t"+val
        output += "\n"
    return output

def import_genomes(data):
    headers = []
    report = []
    convertor = nwmapper.Mapper()
    headline = data[1].split("\t")
    for i in range(3,len(headline)):
        headers.append([headline[i],i-3])
    for line in data[2:]:
        if not line:
            break
        line = line.split("\t")
        c,table,word = line[:3]
        dataset = [float(c)] + list(convertor(word))
        n = 1
        for i in range(3,len(line)):
            n = bitwiser.insert_textVal(n,i-3,line[i])
        dataset.append(n)
        dataset += [table]
        report.append(dataset)
    return [headers,report]

def print_condencedGroups(report,headers,accessions):
    convertor = nwmapper.Mapper()
    # Headers
    output = "TAXA\n" + "C\tWord"
    for i in range(len(headers)):
        output += "\t" + headers[i].upper()
    output += "\n\t"
    for i in range(len(accessions)):
        if len(accessions[i]) > 1:
            strText = str(len(accessions[i]))+" genomes"
        else:
            strText = str(len(accessions[i]))+" genome"
        output += "\t"+strText
    output += "\n"
    # Table
    for word_items in report:
        val,wlength,x,y,data,table = word_items
        valText = str(format_number(val,2))
        output += "\t".join([valText,convertor(wlength,x,y)])
        for i in range(len(accessions)):
            if i == len(headers):
                break
            if type(data)==type([]):
                output += "\t" + data[i]
            else:
                k = 0
                for j in range(i):
                    k += len(accessions[j])
                avr,disp = bitwiser.get_wordStatistics(data,k,len(accessions[i]))
                std = math.sqrt(disp)
                output += "\t"+str(format_number(avr,2))+" ["+str(format_number(std,2))+"]"
        output += "\n"
    return output

def import_taxa(data):
    report = []
    headers = []
    accessions = []
    convertor = nwmapper.Mapper()
    headers = data[1].split("\t")[2:]
    headline = data[2].split("\t")
    for i in range(2,len(headline)):
        n = int(headline[i][:headline[i].find(" ")])
        accessions.append("*"*n)
    for line in data[3:]:
        if not line:
            break
        line = line.split("\t")
        c,word = line[:2]
        dataset = [float(c)] + list(convertor(word)) + [line[2:],""]
        report.append(dataset)
    return [headers,report,accessions]

def print_extendedGroups(report,outgroups,outgroup_header,counterparts,counterpart_headers):
    convertor = nwmapper.Mapper()
    # Headers
    output = "EXTENDED TAXA\n"+"C\tWord"
    if outgroups:
        output += "\t" + f"{outgroup_header.upper()}" + ('\t' * (len(outgroups)-1))
    else:
        output += "\t"
    for i in range(len(counterpart_headers)):
        output += f"{counterpart_headers[i].upper()}" + ('\t' * (len(counterparts[i])-1))
    output += "\n\t"
    if outgroups:
        for i in range(len(outgroups)):
            output += "\t"+outgroups[i]
    for items in counterparts:
        for i in range(len(items)):
            output += "\t"+items[i]
    output += "\n"
    # Table
    for word_items in report:
        val,wlength,x,y,data,table = word_items
        if type(val)==type(""):
            valText = val
        else:
            valText = str(format_number(val,2))
        output += "\t".join([valText,convertor(wlength,x,y)])
        k = 0
        if outgroups:
            k = len(outgroups)
            for i in range(len(outgroups)):
                val = bitwiser.get_textVal(data,i)
                output += "\t"+val
        for items in counterparts:
            for i in range(len(items)):
                val = bitwiser.get_textVal(data,k+i)
                output += "\t"+val
            k += len(items)
        output += "\n"
    return output

def import_extendedTaxa(data):
    report = []
    headers = []
    accessions = []
    convertor = nwmapper.Mapper()

    headline1 = data[1].split("\t")
    headline2 = data[2].split("\t")

    for i, header in enumerate(headline1[2:], start=2):
        if header:
            headers.append(header)
            accessions.append([headline2[i]])
        else:
            accessions[-1].append(headline2[i])

    for line in data[3:]:
        if not line.strip():
            break
        line = line.split("\t")
        c, word = line[:2]
        dataset = [float(c)] + list(convertor(word)) + [line[2:], ""]
        report.append(dataset)

    return [headers, report, accessions]
    
def import_report(fname=None):
    data = openTextFile(fname)
    if not data:
        return
    data = data.split("\n")
    mode = data[0]
    if mode == "GENOMES":
        result = import_genomes(data)
        if not result:
            return
        return [mode]+list(result)+[""]
    elif mode == "TAXA":
        result = import_taxa(data)
        if not result:
            return
        return [mode]+list(result)
    elif mode == "EXTENDED TAXA":
        result = import_extendedTaxa(data)
        if not result:
            return
        return [mode]+list(result)
    elif mode == "WORDS":
        result = import_words(data)
        if not result:
            return
        return [mode]+list(result)
    else:
        return

# Converting multilevel dictionary to list
def dict_to_list(d, parent_key=''):
    result = []
    for k, v in d.items():
        new_key = f"{parent_key}|{k}" if parent_key else k
        if isinstance(v, dict):
            result.extend(dict_to_list(v, new_key))
        else:
            result.append(f"{new_key}|{v}")
    return result
    
def reverse_complement(seq: str) -> str:
    """
    Return the reverse complement of a DNA sequence (A, T, G, C, N).
    Input is case-insensitive; output is uppercase.
    """
    # Mapping dictionary for complements
    complement = {
        "A": "T", "T": "A",
        "G": "C", "C": "G",
        "N": "N"
    }
    
    # Convert to uppercase, reverse string, and map each base
    return "".join(complement.get(base, "N") for base in reversed(seq.upper()))    

def set_value(wlength, count: int, seqlength: int, binary_code: bool = False, coding: list = []):
    '''
    thresholds = {
        2: [73260, 102188, 115255],
        3: [24183, 31537, 36122],
        4: [4747, 7121, 8979],
        5: [1230, 1759, 2457],
        6: [267, 417, 615],
        7: [61.5, 100.2, 157.0],
        8: [13.6, 23.6, 39.4],
        9: [3.1, 5.78, 10.11],
        10: [0.8628, 1.5429, 2.9282],
        11: [0.43134, 0.62658, 1.024],
    }
    '''
    
    thresholds = {
        2: [48840, 68125, 76837],
        3: [16122, 21025, 24081],
        4: [3165, 4747, 5986],
        5: [820, 1173, 1638],
        6: [178, 278, 410],
        7: [41.0, 66.8, 104.7],
        8: [9.1, 15.7, 26.3],
        9: [2.1, 3.85, 6.74],
        10: [0.5752, 1.0286, 1.9521],
        11: [0.28756, 0.41772, 0.683],
    }
    
    f = 1000000 * count / seqlength
    
    level = 3
    for i in range(3):
        if f <= thresholds[wlength][i]:
            level = i
            break
    if binary_code:
        values = ["000", "001", "011", "111"]
        return values[level]
    if coding and len(coding) == 4:
        return coding[level]
    return level

def str2bool(v: str) -> bool:
    """
    Convert common T/F, Y/N variants into a real boolean.
    """
    if v in ("T", "t", "Y", "y", "True", "true", "1"):
        return True
    elif v in ("F", "f", "N", "n", "False", "false", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(
            "Boolean value expected (T/t/Y/y/F/f/N/n/True/False/0/1)."
        )


