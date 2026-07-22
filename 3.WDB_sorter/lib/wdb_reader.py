import sys, os
from bson import ObjectId
import tools
from word_db import WordDB
import nwmapper

# -------------------------------
# Create matricers of k-mers
# -------------------------------
def create_kmer_matrices(input_folder, output_folder, min_k=4, max_k=12, chunk_size=6):
    # Check if the input folder contains WDB files
    wdb_files = [fn for fn in os.listdir(input_folder) if fn.endswith(".wdb")]
    if not len(wdb_files):
        return (False, f"folder {input_folder} does not contain WDB files")
        
    oKmerGenerator = nwmapper.Mapper()
    value_decoder = {
        "000" : -2,
        "001" : -1,
        "011" : 1,
        "111" : 2
    }
        
    for wdb_file in wdb_files:
        # Open WDB file
        oDB = openDBFile(os.path.join(input_folder, wdb_file))
        if oDB == None:
            continue
        '''
        if len(oDB.genomes["genomes"]) < 5 or len(oDB.genomes["genomes"]) > 20:
            continue
        '''
        print(wdb_file,len(oDB.genomes["genomes"]))

        # Extract data from WDB object
        '''
        data = oDB.export_db()
        data = {
            "info":"info",
            "genomes":[{'lineage': 'Archaea|Fervidicoccales|Fervidicoccus|Fervidicoccus|NC_017461.1', 
                'accession': 'NC_017461.1', 'ID': 1, 'seqlength': 1319206, 
                'source': " 'input/chromosomes/Archaea/Fervidicoccales/Fervidicoccus/Fervidicoccus/NC_017461.1.gbk'"}, ...],
            "title":"title",
            "version":"version",
            "date":"date",
            "values":[['001', '111', '011', ...],...],    # Number of lists = number of words, number of values per list = number of genomes
            "words":[{'word': 'TTTTTTTT', 'length': '8', 'x': '1', 'y': '171', 'index': '0'}, ...]     # Multiple records, should be added to MongoDB by chanks
                                                                                                       # Words can be accessed by length-x-y combinations or by their indices
        }
        '''
        
        data = oDB.export_db(min_k, max_k - 1)
        words = data["words"]

        # Sort genomes by ID
        data['genomes'].sort(key=lambda d: int(d['ID']))

        # Transpose 'values' from [words][genomes] to [genomes][words]
        # "values":[['001', '111', '011', ...],...],    # Number of lists = number of genomes, number of values per list = number of words
        data['values'] = [[ls[int(genome['ID']) - 1] for ls in data['values']] for genome in data['genomes']]

        for k in range(min_k, max_k):
            chunk_index = ""
            step = 0
            if k > chunk_size:
                step = 4**chunk_size
                
            output_file = os.path.join(output_folder, f"{k}_mers.1.csv")    
            
            # Create new matrix file with a title line
            kmers = sorted([kmer for kmer in oKmerGenerator.generate(k) if kmer[1] <= kmer[2]])
            kmer_titles = [f"{oKmerGenerator.restore(int(w[0]),int(w[1]),int(w[2]))}|{w[0]}|{w[1]}|{w[2]}" for w in kmers]
            
            # create empty matrix files with titles
            if not os.path.exists(output_file):
                
                if k <= chunk_size:
                    with open(output_file, "w") as f:
                        f.write(",".join(["Name", "Taxon"] + kmer_titles))
                else:
                    start = 0
                    stop = start + step
                    counter = 1
                    while stop < len(kmer_titles):
                        outfile = f"{k}_mers.{counter}.csv"
                        output_file = os.path.join(output_folder, outfile)
                        with open(output_file, "w") as f:
                            f.write(",".join(["Name", "Taxon"] + kmer_titles[start:stop]))
                        start = stop
                        stop += step
                        counter += 1
            
            # Fill matrix files with data
            used_species = []
            for i in range(len(data["genomes"])):
                genome = data["genomes"][i]
                accession = genome['accession']
                species = genome['lineage'].split("|")[-2].strip()
                
                if species in used_species:
                    continue
                #used_species.append(species)
                
                counter = 1
                values = data["values"][i]
                #record = [0 for kmer in kmers]
                # Matrix is calculate for even k's. If k is an odd number, a half of cells are empty
                if k % 2 == 0:
                    record = ["None" for i in range(4**k)]
                else:
                    record = ["None" for i in range(4**(k + 1))]
                    
                stop = step
                for j in range(len(words)):
                    if stop and j > stop:
                        outfile = f"{k}_mers.{counter}.csv"
                        with open(os.path.join(output_folder, outfile), "a") as f:
                            f.write("\n" + ",".join([accession, species] + [str(v) for v in record]))
                            stop += step
                            counter += 1
                        
                    word = words[j]
                    if int(word['length']) != k or not word['word']:
                        continue
                    if int(word['x']) > int(word['y']):
                        ind = coords_to_linear_index(x=int(word['y']), y=int(word['x']), k=k)
                    else:
                        ind = coords_to_linear_index(x=int(word['x']), y=int(word['y']), k=k)
                    if ind >= 0:  # Check for valid index
                        # record values turning "000" to 0, "001" to 1, "011" to 2, and "111" to 3
                        record[ind] = value_decoder[values[j]]

                # Remove empty cells when k is an odd number
                record = [s for s in record if s != "None"]
                    
                outfile = f"{k}_mers.{counter}.csv"
                with open(os.path.join(output_folder, outfile), "a") as f:
                    f.write("\n" + ",".join([accession, species] + [str(v) for v in record]))
        
    return (True, "Ok")

# -------------------------------
# Convert matrix coordinates to linear index (lower-right triangle)
# -------------------------------
def coords_to_linear_index(x, y, k):
    """
    Convert (x,y) coordinates to linear index for upper triangular matrix with diagonal
    stored in column-major order.
    
    Args:
        x (int): Row coordinate (1-based indexing)
        y (int): Column coordinate (1-based indexing)
        k (int): Matrix size parameter (matrix is 2^k × 2^k)
    
    Returns:
        int: Linear index in the upper triangular storage, -1 if invalid coordinates
        
    Note:
        - Matrix uses 1-based indexing
        - Only upper triangle (including diagonal) is stored
        - For invalid coordinates (x > y), returns -1
        - Storage pattern (column-major):
          Column 1: rows 1-1 (1 element)
          Column 2: rows 1-2 (2 elements)  
          Column 3: rows 1-3 (3 elements)
          Column y: rows 1-y (y elements)
    """
    
    # Matrix is calculate for even k's. If k is an odd number, a half of cells are empty
    if k % 2 == 0:
        n = 2 ** k  # Matrix size
    else:
        n = 2 ** (k + 1)  # Matrix size
    
    # Validate coordinates - for upper triangle, x must be <= y
    if x < 1 or x > n or y < 1 or y > n or x > y:
        return -1
    
    # Calculate using arithmetic series formula (column-major order)
    # Elements in columns 1 to (y-1): sum from i=1 to y-1 of i
    # This equals: (y-1)*y/2
    elements_in_previous_columns = (y - 1) * y // 2
    position_in_current_column = x - 1  # Position within current column (0-based)
    
    return elements_in_previous_columns + position_in_current_column

# -------------------------------
# Open a custom WordDB file
# -------------------------------
def openDBFile(path):
    oDB = WordDB()
    try:
        oDB.open_dbfile(path)
    except:
        tools.alert(f"Problem with opening {path}!", "Alert!")
        return None
    return oDB

"""
Chunk a list of rows into documents of size < max_size when encoded as BSON.
base_doc should be the static part of each document (e.g., {"project_id": ..., "chunk_index": i})
"""
def chunk_documents_by_bson_size(rows, base_doc, max_size=0):
    chunks = []
    current_chunk = []
    current_index = 0

    for row in rows:
        temp_doc = {**base_doc, "values": current_chunk + [row]}
        try:
            size = len(bson_encode(temp_doc))
        except Exception:
            size = max_size + 1  # fallback to force split

        if size > max_size:
            # Commit the current chunk
            if current_chunk:
                chunks.append({**base_doc, "chunk_index": current_index, "values": current_chunk})
                current_chunk = []
                current_index += 1

        current_chunk.append(row)

    # Add the final chunk
    if current_chunk:
        chunks.append({**base_doc, "chunk_index": current_index, "values": current_chunk})

    return chunks
    
# -------------------------------
# Helper to split a list into chunks of specified size
# -------------------------------
def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

# -------------------------------
# Helper to chunk values row-by-row and track chunk indices per genome
# -------------------------------
def chunk_values_by_genome(rows, project_id, chunk_size=10000, start_chunk_index=0):
    all_docs = []
    value_indices = []  # Will contain [start, end] for each genome
    chunk_counter = start_chunk_index

    for row_index, long_row in enumerate(rows):
        start = chunk_counter
        for chunk_index, chunk in enumerate(chunk_list(long_row, chunk_size)):
            doc = {
                "project_id": project_id,
                "row_index": row_index + (start_chunk_index if start_chunk_index > 0 else 0),
                "chunk_index": chunk_index,
                "values": chunk
            }
            all_docs.append(doc)
            chunk_counter += 1
        end = chunk_counter - 1
        value_indices.append([start, end])

    return all_docs, value_indices

# -------------------------------
# Create new daytabase structure with sub-collections for each document
# -------------------------------
def create_database(client, data, db_name, collection_name, word_chunk_size=150000, value_chunk_size=500000):
    db = client[db_name]

    # Subcollections for words and values
    metadata_col = db[collection_name]
    words_col = db[f"{collection_name}_words"]
    values_col = db[f"{collection_name}_values"]

    # Prepare metadata (excluding large arrays)
    metadata_doc = {
        "title": data["title"],
        "version": data["version"],
        "date": data["date"],
        "genomes": data["genomes"]
    }

    # Insert the metadata document
    inserted_id = metadata_col.insert_one(metadata_doc).inserted_id
    project_id = inserted_id

    # Chunk and insert words
    word_chunks = list(chunk_list(data["words"], word_chunk_size))
    word_docs = [{"project_id": project_id, "chunk_index": i, "words": chunk} for i, chunk in enumerate(word_chunks)]
    if word_docs:
        words_col.insert_many(word_docs)

    # Chunk and insert values, track chunk ranges
    value_docs, value_indices = chunk_values_by_genome(
        rows=data["values"],
        project_id=project_id,
        chunk_size=value_chunk_size
    )

    for doc in value_docs:
        values_col.insert_one(doc)

    # Update metadata with chunking info
    metadata_col.update_one(
        {"_id": project_id},
        {"$set": {
            "value_indices": value_indices,
            "max_value_chunk_size": value_chunk_size
        }}
    )

    return inserted_id

# -------------------------------
# Append genomes and values to existing documents in multiple collections
# -------------------------------
def append_data(client, data, db_name, collection_name, document_id, value_chunk_size=500000):
    db = client[db_name]

    # Main and subcollections
    collection = db[collection_name]
    values_col = db[f"{collection_name}_values"]

    # Fetch existing metadata
    doc = collection.find_one({"_id": ObjectId(document_id)})
    if not doc:
        print("Document not found.")
        return None

    project_id = ObjectId(document_id)

    # --- Step 1: Update 'genomes' list and adjust IDs ---
    existing_genomes = doc.get("genomes", [])
    offset = len(existing_genomes)

    new_genomes = data.get("genomes", [])
    for genome in new_genomes:
        genome["ID"] += offset

    collection.update_one(
        {"_id": project_id},
        {"$push": {"genomes": {"$each": new_genomes}}}
    )

    # --- Step 2: Append value_indices for new genomes ---
    existing_indices = doc.get("value_indices", [])
    current_chunk_index = max((end for _, end in existing_indices), default=-1) + 1

    value_docs, new_value_indices = chunk_values_by_genome(
        rows=data.get("values", []),
        project_id=project_id,
        chunk_size=value_chunk_size,
        start_chunk_index=current_chunk_index
    )

    # Append new value_indices to metadata
    collection.update_one(
        {"_id": project_id},
        {"$push": {"value_indices": {"$each": new_value_indices}}}
    )

    # --- Step 3: Insert chunked value documents ---
    for doc in value_docs:
        values_col.insert_one(doc)

    return document_id

if __name__ == "__maine__":

    help_text = ("\n" + f"""
    {'='*80}
    ■ wdb2MongoDB ■
    {'='*80}
    🔬 Purpose:
    Convert WDB files into MongoDB records
    
    ⚙️ Dependencies:
    ┌───────────────────┬───────────────────────┐
    │ Tool              │ Minimum Version       │
    ├───────────────────┼───────────────────────┤
    │ Python            │ 3.9                   │
    │ pymongo           │ 4.11.3                │
    │ word_db           │ local                 │
    │ tools             │ local                 │
    └───────────────────┴───────────────────────┘
    
    💻 Tested Environments:
    - CentOS Linux 7.3.1611
    - Ubuntu 20.04 LTS
    
    🚀 Usage:
        python3 wdb2MongoDB <arguments>
        
    ⚡ Required Arguments:
        --working_directory     First level input folder (default: 'output')
        --db_name               Database name (default: '')
        --collection_name       Name of collection within database (default: '')
        
    🔧 Optional Arguments:
        --version_number        Database version (default: '1.0')
                                Version is added automatically to collection name
        --input_folder          Second level input folder (default: 'chromosomes')
        --project_folder        Third level input folder (default: 'Archaea')
                                These arguments set path to folder with subfolders to process
                                ./working_directory/input_folder/project_folder/
                                project_folder will be reflected in the output LOG file
        --output_folder         Output folder (default: 'log')
                                Log file will be created in this folder
                                If folder does not exist, it will be created
        --mongodb_user          user name if required (default: '')
        --mongodb_password      MongoDB password if required (default: '')
        --drop_collection       No values. If set, the collection /db_name/collection_name/_/version/ will be deleted if exists
        --force                 No values. If set, the database with the name /db_name/ will be deleted if exists
        
    🆘 Help Options:
        -help, --help           Show this help message
        -version, --version     Show version information""" +
    
    """\n📊 Output:
        Create MongoDB /db_name/_/collection_name/ with the structure:
        -------------------------------------------------------------------------------------------------------------------
        field                   Data        Content
        -------------------------------------------------------------------------------------------------------------------
        title                   TXT         Database title, set at database creation
        version                 TXT         Database version, set at database creation
        date                    TXT         Database creation date, set at database creation
        genomes                 ARRAY           [{'lineage': 'Archaea|Fervidicoccales|Fervidicoccus|NC_017461.1', 
                                            'accession': 'NC_017461.1', 'ID': 1, 'seqlength': 1319206, 
                                            'source': " './Archaea/Fervidicoccales/Fervidicoccus/NC_017461.1.gbk'"}, ...]
        words                   ARRAY       [{'word': 'TTTTTTTT', 'length': '8', 'x': '1', 'y': '171', 'index': '0'}, ...]
        values                  CHUNKS      [['001', '111', '011', ...],...]
        max_value_chunk_size    INT         Number of values per chunk
        indices                 ARRAY       [[start_chank_index, end_chunk_index],...] - number of indices = number of genomes
        -------------------------------------------------------------------------------------------------------------------\n\n""" +
                
    f"{'='*80}")
    
    # -------------------------------
    # Print version of the screen
    # -------------------------------
    def show_version(program_version, date_of_creation):
        print("\n" + f"version {program_version} created on {date_of_creation}" + "\n")
    
    # -------------------------------
    # Print help of the screen
    # -------------------------------
    def show_help():
        print(help_text)
    
    '''
    # -------------------------------
    # Command-line argument parser
    # -------------------------------
    def parse_args():
        parser = argparse.ArgumentParser(description="Process genomic data and store in MongoDB.")
        
        # Disable automatic help
        parser = argparse.ArgumentParser(add_help=False)
        
        parser.add_argument("--working_directory", default="output", help="Base working directory")
        parser.add_argument("--input_folder", default="", help="Subfolder in working directory")
        parser.add_argument("--project_folder", default="", help="Project name")
        parser.add_argument("--output_folder", default="log", help="Log file directory")
        parser.add_argument("--version_number", default="1.0", help="MongoDB database version number")
        parser.add_argument("--db_name", default="", help="MongoDB database name")
        parser.add_argument("--collection_name", default="", help="MongoDB collection name")
        parser.add_argument("--mongodb_user", default="", help="MongoDB username if authentication is needed")
        parser.add_argument("--mongodb_password", default="", help="MongoDB password if authentication is needed")
        parser.add_argument("--force", action="store_true", help="Delete the database before processing if it exists")
        parser.add_argument("--drop_collection", action="store_true", help="Delete the database before processing if it exists")
        parser.add_argument("-help","--help","-h", action="store_true", help="Show this help message and exit")
        parser.add_argument("--version", "-v", action="version",
                        version=f"\nversion {__version__} created on {date_of_creation}\n")
    
        return parser.parse_args()
    '''

