import sys, os, math, ast, copy 
import bitwiser, tools
from datetime import datetime
import nwmapper, progressbar, processor
from typing import List

###############################################################################
class Template:
    def __init__(self, title="", date="", version="1.0", matrix_geometry: str = "whole"):
        # {wlength:{'x':{y1:{},y2,...},'y':{x1:{},x2,...}}
        self.db = {}
        # Version of the database
        self.version = version
        # Title of the database
        self.title = title
        # Matrix geometry: whole | lower | upper
        self.matrix_geometry = matrix_geometry
        # Supplementary information
        self.supplementary_data = {}
        # Data of the database creation
        if date:
            self.date = date
        else:
            # Get the current date
            current_date = datetime.now()
            # Format the date as dd/mm/yyyy
            self.date = current_date.strftime('%d/%m/%Y')        
        
    #### ACCESSIONS
    def add(self,wlength,x,y,count=1, echo=False):
        if (self.matrix_geometry.lower() == "lower" and int(x) > int(y)) or (self.matrix_geometry.lower() == "upper" and int(x) < int(y)):
            return wlength, x, y, count
            
        if self.has(wlength,x,y):
            self.db[wlength]['x'][x][y]['counter'] += count
            return wlength,x,y,self.db[wlength]['x'][x][y]['counter']

        new_entry = {'counter':0,'data':[]}
        if wlength not in self.db:
            self.db[wlength] = {'x':{},'y':{}}
        if x not in self.db[wlength]['x']:
            self.db[wlength]['x'][x] = {y:new_entry}
        if y not in self.db[wlength]['y']:
            self.db[wlength]['y'][y] = {x:new_entry}
        if y not in self.db[wlength]['x'][x] or x not in self.db[wlength]['y'][y]:
            self.db[wlength]['x'][x][y] = self.db[wlength]['y'][y][x] = new_entry
        self.db[wlength]['x'][x][y]['counter'] += count
        return wlength,x,y,self.db[wlength]['x'][x][y]['counter']
    
    def add_word(self, word: str, count: int = 1, combine_complements: bool = False) -> None:
        convertor = nwmapper.Mapper()
        wlength,x,y = convertor(word)
            
        if combine_complements:
            y, x = sorted([x, y])
        return self.add(wlength,x,y,count)

    def count_words(self, sequence: str, wlength: int, flg_progress=True, bar_text="Word count: "):
        sequence = sequence.upper()
        words = sorted([
            sequence[i:i + wlength]
            for i in range(len(sequence) - wlength)
            if 'N' not in sequence[i:i + wlength]
        ])
        words = tools.word_list_to_counts(words)
        bar = None
        if flg_progress:
            bar = progressbar.indicator(len(words),bar_text)
        for i in range(len(words)):
            word, count = words[i]
            if sum([word.upper().count(L) for L in ["A", "T", "G", "C"]]) == len(word):
                self.add_word(word,count)
            if bar:
                try:
                    bar(i)
                except:
                    pass
        if bar:
            bar.stop()
        return 
    
    def get(self,wlength,x,y):
        if wlength not in self.db:
            return 0
        if x not in self.db[wlength]['x']:
            return 0
        if y not in self.db[wlength]['x'][x]:
            return 0
        return self.db[wlength]['x'][x][y]['counter']
    
    def getvalue(self,wlength,x,y,flg_complement=False):
        convertor = nwmapper.Mapper()
        count = self.get(wlength,x,y)
        if flg_complement:
            wlength,y,x = convertor.revcomplement([wlength,y,x])
            count += self.get(wlength,y,x)
        return count
    
    def get_word(self,word,flg_complement=False):
        convertor = nwmapper.Mapper()
        wlength,x,y = convertor(word)
        return self.getvalue(wlength,x,y,flg_complement)
    
    def pop(self,wlength,x,y,flg_complement=False):
        if self.has(wlength,x,y):
            count = self.getvalue(wlength,x,y,flg_complement)
            self.delete(wlength,x,y,self.db[wlength]['x'][x][y]['counter'])
        else:
            count = 0
        if flg_complement and self.has(wlength,y,x):
            convertor = nwmapper.Mapper()
            wlength,cx,cy = convertor.revcomplement([wlength,x,y])
            if self.has(wlength,cx,cy):
                self.delete(wlength,cx,cy,self.db[wlength]['x'][cx][cy]['counter'])
        return count
    
    def pop_word(self,word,flg_complement=False):
        convertor = nwmapper.Mapper()
        wlength,x,y = convertor(word)
        return pop(wlength,x,y,flg_complement)
    
    def get_allWords(self,wl_lower=None,wl_upper=None,flg_add_words=False, echo=False):
        output = []
        for wlength in self.db:
            if wl_lower and wl_lower > wlength:
                continue
            if wl_upper and wl_upper < wlength:
                continue

            for x in self.db[wlength]['x']:
                for y in self.db[wlength]['x'][x]:
                    if (self.matrix_geometry.lower() == "lower" and int(x) > int(y)) or (self.matrix_geometry.lower() == "upper" and int(x) < int(y)):
                        continue 
                    output.append([wlength,x,y])
        if flg_add_words:
            oMapper = nwmapper.Mapper()
            output = [[oMapper.restore(wl,x,y),wl,x,y] for wl,x,y in output]
        return output

    def get_word_triplets(self,wl_lower=None,wl_upper=None):
        return self.get_allWords(wl_lower,wl_upper,False)

    def next_word(self,word,longer_words=False):
        try:
            wlength,x,y = word
        except:
            convertor = nwmapper.Mapper()
            wlength,x,y = convertor(word)
        x += 1
        while x <= 2**wlength:
            if x in self.db[wlength]['x'] and y in self.db[wlength]['x'][x]:
                return [wlength,x,y]
            x += 1
        y += 1
        while y <= 2**wlength:
            x = 1
            while x <= 2**wlength:
                if x in self.db[wlength]['x'] and y in self.db[wlength]['x'][x]:
                    return [wlength,x,y]
                x += 1
            y += 1
        if longer_words and wlength < max(list(db.keys())):
            return self.next_word([wlength+1,0,0])
        return [0,0,0]

    def delete(self,wlength,x,y,how_many=1):
        if wlength not in self.db:
            return None
        if x not in self.db[wlength]['x']:
            return None
        if y not in self.db[wlength]['x'][x]:
            return None
        self.db[wlength]['x'][x][y]['counter'] -= how_many
        val_to_return = self.db[wlength]['x'][x][y]['counter']
        if val_to_return <= 0:
            val_to_return = 0
            del self.db[wlength]['x'][x][y]
            try:
                del self.db[wlength]['y'][y][x]
            except:
                pass
        if x in self.db[wlength]['x'] and not self.db[wlength]['x'][x]:
            del self.db[wlength]['x'][x]
        if y in self.db[wlength]['y'] and not self.db[wlength]['y'][y]:
            del self.db[wlength]['y'][y]
        if wlength in self.db and not self.db[wlength]:
            del self.db[wlength]
        return val_to_return

    #### INFO
    def has(self,wlength,x,y,flg_complement=False):
        try:
            count = self.db[wlength]['x'][x][y]
            return True
        except:
            if flg_complement:
                convertor = nwmapper.Mapper()
                wlength,x,y = convertor.revcomplement([wlength,x,y])
                return self.has(wlength,x,y)
            else:
                return False
        
    def has_word(self,word,flg_complement=False):
        convertor = nwmapper.Mapper()
        wlength,x,y = convertor(word)
        return self.has(wlength,x,y,flg_complement)
    
    def search(self,word_template,wl_from,wl_to,flg_complement=False):
        wordlist = []
        wl = []
        convertor = nwmapper.Mapper()
        if not word_template:
            wordlist = self.get_allWords()
            for item in wordlist:
                wlength,x,y = eval(item)
                wl.append(convertor(wlength,x,y))
                if flg_complement:
                    wl.append(convertor(convertor.revcomplement([wlength,y,x])))
            if len(wl) > 1:
                wl.sort()
            return wl
        for i in range(wl_from,wl_to+1):
            word = word_template + "N"*(i-len(word_template))
            result = convertor(word)
            if result and type(result[0])==type([]):
                wordlist.extend(result)
            else:
                wordlist.append(result)
            for item in wordlist:
                wlength,x,y = item
                if self.has(wlength,x,y,flg_complement):
                    wl.append(convertor(wlength,x,y))
            wordlist = []
        if len(wl) > 1:
            wl.sort()
        return wl
        
    def set_version(self,version):
        self.version = version

    def get_version(self):
        return self.version

    def set_date(self,date):
        self.date = date

    def get_date(self):
        return self.date
    
    def set_data(self,wlength,x,y,data):
        if self.has(wlength,x,y):
            self.db[wlength]['x'][x][y]['data'] = data
    
    # append the data value with new set of data
    def merge_data(self,wlength,x,y,data,ID):
        self.set_data(wlength,x,y,bitwiser.merge(self.get_data(wlength,x,y),data,ID))
    
    def get_data(self,wlength,x,y,flg_complement=False):
        if wlength in self.db and x in self.db[wlength]['x'] and y in self.db[wlength]['x'][x]:
            return self.db[wlength]['x'][x][y]['data']
        if flg_complement:
            convertor = nwmapper.Mapper()
            wlength,x,y = convertor.revcomplement([wlength,x,y])
            return self.get_data(wlength,x,y)[ID - 2]
        return None

    def count_entries(self):
        count = 0
        for wlength in self.db:
            for x in self.db[wlength]['x']:
                count += len(self.db[wlength]['x'][x])
        return count

    def count_entries_perX(self,wlength,x):
        return len(self.db[wlength]['x'][x])

    def count_entries_perY(self,wlength,y):
        return len(self.db[wlength]['y'][y])
    
    def size(self):
        count = 0
        for wlength in self.db:
            for x in self.db[wlength]['x']:
                for y in self.db[wlength]['x'][x]:
                    count += 1
        return count

    def get_topX(self,wlength):
        count = 0
        topX = 0
        for x in self.db[wlength]['x']:
            val = self.count_entries_perX(wlength,x)
            if val > count:
                count = val
                topX = x
        return topX,count

    def get_topY(self,wlength):
        count = 0
        topY = 0
        for y in self.db[wlength]['y']:
            val = self.count_entries_perY(wlength,y)
            if val > count:
                count = val
                topY = y
        return topY,count

    def get_bottomX(self,wlength):
        count = "0"
        bottomX = 0
        for x in self.db[wlength]['x']:
            val = self.count_entries_perX(wlength,x)
            if val < count:
                count = val
                bottomX = x
        return bottomX,count

    def get_bottomY(self,wlength):
        count = "0"
        bottomY = 0
        for y in self.db[wlength]['y']:
            val = self.count_entries_perY(wlength,y)
            if val < count:
                count = val
                bottomY = y
        return bottomY,count
    
    def word_distribution(self):
        report = {"scarce":0,"common":0,"frequent":0,"abundant":0}
        bar = progressbar.indicator(len(self.get_allWords()),"Run: ")
        done = 1
        for wlength in self.db:
            for x in self.db[wlength]['x']:
                for y in self.db[wlength]['x'][x]:
                    data = self.get_data(wlength,x,y)
                    result = bitwiser.parse_data(data)
                    try:
                        scarce,common,frequent,abundant = result
                    except:
                        tr = nwmapper.Mapper()
                        print("Error: ", tr(wlength,x,y))
                        for acc in self.genomes:
                            if result == self.genomes[acc]:
                                print(acc)
                        sys.exit(1)
                    report["scarce"] += scarce
                    report["common"] += common
                    report["frequent"] += frequent
                    report["abundant"] += abundant
                    done += 1
                    if done%1000 == 0:
                        bar(done)
        bar.stop()
        return report

    def word_abundance(self):
        report = []
        stat = []
        tr = nwmapper.Mapper()
        bar = progressbar.indicator(len(self.get_allWords()),"Run: ")
        done = 1
        for wlength in self.db:
            w_av = w_count = 0
            for x in self.db[wlength]['x']:
                for y in self.db[wlength]['x'][x]:
                    data = self.get_data(wlength,x,y)
                    result = bitwiser.data_stat(data)
                    try:
                        av,var = result
                        report.append("\t".join([str(wlength),tr(wlength,x,y),str(av),str(var)]))
                        w_av += av
                        w_count += 1
                    except:
                        print("Error in function word.db.word_abundance!")
                        sys.exit(1)
                    done += 1
                    if done%1000 == 0:
                        bar(done)
            stat.append("\t".join([str(wlength),str(w_count),str(float(w_av)/w_count)]))
        bar.stop()
        return stat,report

    #### DATA PERSISTANCE
    def copy(self,version="",date=""):
        new_db = WordDB()
        new_db.fields.extend(self.fields)
        for wlength in self.db:
            for x in self.db[wlength]['x']:
                for y in self.db[wlength]['x'][x]:
                    new_db.add(wlength,x,y,self.db[wlength]['x'][x][y]['counter'])
        for genome in self.genomes:
            new_db.new_genome(genome)
            for word_id in self.genomes[genome]:
                wlength,x,y = eval(word_id)
                for filed in self.genomes[genome][word_id]:
                    new_db.set_field(wlength,x,y,genome,field,self.genomes[genome][word_id][field])
        if version:
            new_db.set_version(version)
        else:
            new_db.set_version(self.get_version())
        if date:
            new_db.set_date(date)
        else:
            new_db.set_date(self.get_date())
        return new_db
    
    def __add__(self,other):
        db = self.copy()
        for wlength,x,y in other.get_allWords():
            db.add(wlength,x,y,other.get(wlength,x,y))
        return db
    
    # This method is used only if it is known for sure that two databases 
    # do not share any comman words
    def update(self,other):
        self.db.update(other.db)

###############################################################################
class Filter(Template):
    def __init__(self, filter_settings={}, sort_para=1):
        Template.__init__(self)
        if not filter_settings:
            self.size_limit = 0
            self.permutations = 0
            self.frameshift = 0
            self.constituents = 0
            self.wl_min = 8
            self.wl_max = 14
            self.threshold = 0
            # collection of values to sort out identical values
            # by default there may be only 10 words that differ by values by 10%
            self.redundancy = 0
            # min_dissimilitude is the percentage of mismatches
            self.min_dissimilitude = 0
        else:
            self.size_limit = filter_settings['top']
            if self.size_limit == "0":
                self.size_limit = 0
            self.permutations = filter_settings['permutations']
            if self.permutations == "0" or self.permutations == "0%":
                self.permutations = 0
            self.frameshift = filter_settings['frameshift']
            if self.frameshift == "0" or self.frameshift == "0%":
                self.frameshift = 0
            self.constituents = filter_settings['constituents']
            if self.constituents == "0":
                self.constituents = 0
            self.wl_min = int(filter_settings['wl_min'])
            self.wl_max = int(filter_settings['wl_max'])
            self.threshold = filter_settings['threshold']
            # collection of values to sort out identical values
            # by default there may be only 10 words that differ by values by 10%
            try:
                self.redundancy = int(filter_settings['redundancy'])
            except:
                self.redundancy = 0
            if self.redundancy < 0:
                self.redundancy = 0
            # min_dissimilitude is the percentage of mismatches
            # by default 10%
            try:
                self.min_dissimilitude = float(filter_settings['similarity'])
            except:
                if not filter_settings['similarity']:
                    self.min_dissimilitude = 0
                elif filter_settings['similarity'][-1] == "%":
                    try:
                        self.min_dissimilitude = float(filter_settings['similarity'][:-1])
                    except:
                        self.min_dissimilitude = 0
            if self.min_dissimilitude < 0 or self.min_dissimilitude > 100:
                self.min_dissimilitude = 0
            
        self.sort_para = sort_para
        self.items = []
        self.values = {}
        
    #### ACCESSIONS
    def add(self,wlength,x,y,value=0,parent=""):
        
        #### TEMP
        # CTATTCC|7|34|192
        echo = False
        if wlength == 7 and x ==34 and y == 192:
            echo = True
            print("CTATTCC|7|34|192")
        ####
            
        if (self.threshold and value <= self.threshold) or wlength < self.wl_min or wlength > self.wl_max:
            return False
        # Check if such word already exists in the database
        if self.has(wlength,x,y):
            # if new value is bigger than old value, the entries must be replaced
            # replacement with a new parent
            if value > self.db[wlength]['x'][x][y]['value'] and not parent:
                pWL,pX,pY = eval(self.db[wlength]['x'][x][y]['parent'])
                if pWL > wlength:
                    return False
                self.replace([pWL,pX,pY],[wlength,x,y],value)
            # replacement with a new neighbour
            elif value > self.db[wlength]['x'][x][y]['value'] and parent:
                self.db[wlength]['x'][x][y]['value'] = value
                self.db[wlength]['x'][x][y]['parent'] = "["+",".join([str(wlength),str(x),str(y)])+"]"
                return True
            else:
                #### TEMP
                if wlength == 7 and x == 34 and y == 192:
                    print("471", value, self.db[wlength]['x'][x][y]['value'], parent)
                ####
                    
                return False
        # create new entry
        flg_neighbours = False
        if not parent:
            parent = "["+",".join([str(wlength),str(x),str(y)])+"]"
            flg_neighbours = True
            self.items.append([value,wlength,x,y])
        new_entry = {'parent':parent,'value':value,'neighbours':[],
            'data':None,'supplement':None,'counter':1}
        if wlength not in self.db:
            self.db[wlength] = {'x':{},'y':{}}
        if x not in self.db[wlength]['x']:
            self.db[wlength]['x'][x] = {y:new_entry}
        if y not in self.db[wlength]['y']:
            self.db[wlength]['y'][y] = {x:new_entry}
        if not self.has(wlength,x,y):
            self.db[wlength]['x'][x][y] = self.db[wlength]['y'][y][x] = new_entry
        # add neighbours
        if flg_neighbours:
            self.db[wlength]['x'][x][y]['neighbours'] = self.get_neighbours(wlength,x,y)
            for wl,nX,nY in self.db[wlength]['x'][x][y]['neighbours']:
                self.add(wl,nX,nY,value,parent)
        #### TEMP        
        if wlength == 7 and x == 34 and y == 192:
            print("492")
        ####
            
        return True
    
    def replace(self,old_entry,new_entry,value):
        wlength,x,y = old_entry
        self.remove(wlength,x,y)
        wlength,x,y = new_entry
        self.add(wlength,x,y,value)
    
    def remove(self,wlength,x,y,flg_removeNeighbours=True):
        if not self.has(wlength,x,y):
            return False
        value = self.db[wlength]['x'][x][y]['value']
        neighbours = []
        neighbours.extend(self.db[wlength]['x'][x][y]['neighbours'])
        self.pop(wlength,x,y)
        try:
            del self.items[self.items.index([value,wlength,x,y])]
        except:
            pass
        if flg_removeNeighbours:
            for wl,nX,nY in neighbours:
                if self.has(wl,nX,nY) and self.db[wl]['x'][nX][nY]['value'] <= value:
                    self.remove(wl,nX,nY)
        return True
        
    def remove_word(self,word):
        try:
            convertor = nwmapper.Mapper()
            wlength,x,y = convertor(word)
            return self.remove(wlength,x,y)
        except:
            return False
        
    def set_data(self,wlength,x,y,data,filter_data=True):
        # check if there are words with similar distribution
        if not self.has(wlength,x,y):
            return
        # save a copy of the word data for the case the word is deleted 
        # when checking for the uniqueness
        word = self.db[wlength]['x'][x][y].copy()
        if not filter_data or self.isDataUnique(wlength,x,y,data):
            if not self.has(wlength,x,y):
                self.add(wlength,x,y)
                self.db[wlength]['x'][x][y] = {}
                self.db[wlength]['x'][x][y].update(word)
            self.db[wlength]['x'][x][y]['data'] = data
    
    def set_supplement(self,wlength,x,y,supplement):
        if self.has(wlength,x,y):
            self.db[wlength]['x'][x][y]['supplement'] = supplement
    
    def get_words(self, flg_includeData = True, flg_tostring = False):
        self.clear_items()
        words = []
        for item in self.items:
            words.append([])
            words[-1].extend(item)
            if flg_includeData:
                val,wlength,x,y = item
                data = self.get_data(wlength,x,y,True)

                if not data:
                    del words[-1]
                    continue
                words[-1].append(data)
                if self.db[wlength]['x'][x][y]['supplement']:
                    words[-1].append(self.db[wlength]['x'][x][y]['supplement'])
                else:
                    words[-1].append("")
        if flg_tostring:
            oMapper = nwmapper.Mapper()
            words = ["|".join([oMapper.restore(wlength, x, y), str(wlength), str(x), str(y)]) for value, wlength, x, y in words]
        return words
    
    def isDataUnique(self,wlength,x,y,data):
        if self.redundancy == 0:
            return True
        echo = False
        val = self.db[wlength]['x'][x][y]['value']
        for entry in self.values:
            count,total = bitwiser.match(data,entry)
            if 100.0*count/total/3.0 <= self.min_dissimilitude:
                if len(self.values[entry]) >= self.redundancy:
                    flg_accepted = False
                    for i in range(len(self.values[entry])):
                        c,wi,xi,yi = self.values[entry][i]
                        if val > c:
                            self.values[entry].insert(i,[val,wlength,x,y])
                            d,wd,xd,yd = self.values[entry].pop()
                            self.remove(wd,xd,yd)
                            flg_accepted = True
                            break
                    if echo:
                        print("word_db:909",flg_accepted)
                    if not flg_accepted:
                        self.remove(wlength,x,y)
                    return flg_accepted
                else:
                    self.values[entry].append(self.items[-1])
                    self.values[entry].sort()
                    if echo:
                        print("word_db:586",flg_accepted)
                    return True
        self.values[data] = [self.items[-1]]
        if echo:
            print("word_db:921",flg_accepted)
            print()
        return True
    
    def clear_items(self):
        if not self.size_limit or len(self.items) <= self.size_limit or len(self.items)==1:
            return
        self.items.sort(key = lambda ls: [ls[0],ls[1]] if self.sort_para else [ls[0],-ls[1]], reverse=True)
        for i in range(len(self.items)-1,0,-1):
            if self.items[i] == self.items[i-1]:
                del self.items[i]
        self.items = self.items[:self.size_limit]
    
    def get_neighbours(self,wlength,x,y):
        parent = [wlength,x,y]
        neighbours = [[wlength,y,x]]
        if self.permutations:
            val = self.getFilterValue(self.permutations,wlength)
            if val:
                neighbours.extend(self.get_permutations(parent,val))
        if self.frameshift:
            val = self.getFilterValue(self.frameshift,wlength)
            if val:
                neighbours.extend(self.get_frameshifts(parent,val))
        if self.constituents:
            val = self.getFilterValue(self.constituents,wlength)
            if val:
                level = wlength - self.wl_min
                if level > val:
                    level = val
                if level:
                    neighbours.extend(self.get_constituents(parent,level))
        if len(neighbours) > 1:
            neighbours.sort(key=lambda ls: list(ls))
        for i in range(len(neighbours)-1,0,-1):
            if neighbours[i] == parent or neighbours[i] == neighbours[i-1]:
                del neighbours[i]
        if neighbours[0] == parent:
            del neighbours[0]
        return neighbours
    
    def get_permutations(self,parent,level):
        convertor = nwmapper.Mapper()
        return convertor.get_permutations(parent,level)
    
    def get_constituents(self,parent,level):
        convertor = nwmapper.Mapper()
        wlength,x,y = parent
        return convertor.constituents(parent,wlength-level)
    
    def get_frameshifts(self,parent,level):
        convertor = nwmapper.Mapper()
        wlength,x,y = parent
        if level >= wlength-1:
            return []
        words = convertor.move_right(parent,level)
        words.extend(convertor.move_left(parent,level))
        return words
    
    def getFilterValue(self,setting,wlength):
        if not setting:
            return 0
        percentage = False
        if str(setting).find("%")==len(str(setting))-1:
            setting = setting[:-1]
            percentage = True
        try:
            val = int(setting)
        except:
            return 0
        if val <= 0:
            return 0
        if percentage:
            if val >= 100:
                val = 99
            val = int(float(wlength*val)/100.0)
        if val >= wlength:
            val = wlength-1
        return val
    
###############################################################################
class WordDB(Template):
    def __init__(self, title="", date="", version="1.0", matrix_geometry: str = "whole"):
        Template.__init__(self, title=title, date=date, version=version, matrix_geometry=matrix_geometry)
        self.lineages = {'acc':{},'genomes':{}}
        self.genomes = {'acc':{},'genomes':{}}
        
    #### METHODS
    # Modes - diverse, abundant and rare
    def get_extremeWords(self,mode,accessions,filter=None,flg_bar=True,wordlist=[]):
        if not self.db:
            return
        ids = []
        convertor = nwmapper.Mapper()
        for acc in accessions:
            genome_id = self.get_genomeID(acc)
            if genome_id != None:
                ids.append([genome_id,acc])
        if not ids:
            return
        if len(ids)>1:
            ids.sort()
        # replace genome IDs with its numbers in the list strarting with 0
        headers = []
        for i in range(len(ids)):
            headers.append([ids[i][1],i])
        if len(ids)==1 and mode=="diverse":
            return self.get_overrepresentedWords(ids[0][0]),headers
        wlength_range = list(self.db.keys())
        wlength_range.sort()
        if mode != "rare":
            wlength_range.reverse()
        sort_para = 1
        if mode=="rare":
            sort_para = 0
        if mode == "compare taxa":
            oFilter = Filter()
        else:
            oFilter = Filter(filter,sort_para)
        bar = None
        if not wordlist:
            wordlist = self.get_allWords()
        if flg_bar:
            bar = progressbar.indicator(len(wordlist),"Run: ")
        done = 0
        for word in wordlist:
            if not word:
                continue
            if type(word) == type("Text"):
                wlength,x,y = convertor(word)
            else:
                wlength,x,y = word
            if mode == "diverse":
                val,data = self.calculate_divergency(self.get_data(wlength,x,y,True),ids)
                added = oFilter.add(wlength,x,y,val)
                if added:
                    oFilter.set_data(wlength,x,y,data)
            elif mode == "abundant" or mode=="rare" or mode=="compare_taxa":
                val,data = self.calculate_commonality(self.get_data(wlength,x,y,True),ids,mode)
                added = oFilter.add(wlength,x,y,val)
                if added:
                    oFilter.set_data(wlength,x,y,data,False)
            done += 1
            if bar and done%1000:
                bar(done)
        if bar:
            bar.stop()
            del bar
        word_list = oFilter.get_words()
        del oFilter
        v = len(headers)
        if len(word_list)>1:
            word_list.sort()
            word_list.reverse()
        for item in word_list:
            val,wlength,x,y,data,table = item
        return word_list,headers
    
    def compare_taxa(self,acc_list,filter,wordlist=[]):
        if len(acc_list)<2:
            tools.alert("Only 1 taxon is present!")
            return None
        taxa = {}
        headers = []
        IDs = []
        taxa_list = []
        taxon_keys = list(acc_list.keys())
        taxon_keys.sort()
        if len(taxon_keys) <= 1000:
            for i in range(len(taxon_keys)):
                taxon = taxon_keys[i]
                print(str(i+1)+"/"+str(len(taxon_keys))+" "+taxon + " is processed")
                result = taxa[taxon] = self.get_extremeWords("compare_taxa",acc_list[taxon],None,True,wordlist)
                if result:
                    taxa[taxon] = result[0]
                    headers.append([])
                    for item in result[1]:
                        headers[-1].append(item[0])
                        IDs.append([self.get_genomeID(item[0]),item[0]])
                    taxa_list.append(taxon)
                else:
                    continue
                if taxon in taxa and len(taxa[taxon])>1:
                    taxa[taxon].sort(key = lambda ls: [ls[0],ls[1]] if self.sort_para else [ls[0],-ls[1]], reverse=True)
        else:
            # implementation using threadings
            acc_num = 0
            for taxon in taxon_keys:
                acc_num += len(acc_list[taxon])
            bar = progressbar.indicator(acc_num,"Taxa processing: ")
            start = 0
            while start < len(taxon_keys):
                acc_num = 0
                stop = start + 11
                if stop > len(taxon_keys):
                    stop = len(taxon_keys)
                thread_list = {}
                for taxon in taxon_keys[start:stop]:
                    thread_list[taxon] = processor.ExtremeWordProcessor(self.get_extremeWords,acc_list[taxon],self.db)
                    thread_list[taxon].start()
                    acc_num += len(acc_list[taxon])
                for taxon in thread_list:
                    thread_list[taxon].join()
                for taxon in thread_list:
                    result = taxa[taxon] = thread_list[taxon].get()
                    if result:
                        taxa[taxon] = result[0]
                        headers.append([])
                        for item in result[1]:
                            headers[-1].append(item[0])
                            IDs.append([self.get_genomeID(item[0]),item[0]])
                        taxa_list.append(taxon)
                    else:
                        continue
                    if taxon in taxa and len(taxa[taxon])>1:
                        taxa[taxon].sort(key = lambda ls: [ls[0],ls[1]] if self.sort_para else [ls[0],-ls[1]], reverse=True)
                bar(acc_num)
            bar.stop()
        oFilter = Filter(filter)
        print("Calculating taxa comparison coefficients")
        bar = progressbar.indicator(self.size(),"Run: ",None)
        done = 0
        max_aov = tools.find_max_aov([bitwiser.get_table(taxa[taxon][0][4],len(acc_list[taxon])) for taxon in taxa_list])
        for i in range(len(taxa[taxa_list[0]])):
            val,wlength,x,y,data,table = taxa[taxa_list[0]][i]
            val = 10.0*tools.aov([bitwiser.get_table(taxa[taxon][i][4],len(acc_list[taxon])) for taxon in taxa_list])/max_aov
            added = oFilter.add(wlength,x,y,val)
            if added:
                oFilter.set_data(wlength,x,y,
                    bitwiser.merge_values([taxa[taxon][i][4] for taxon in taxa_list])
                    )
            if i%10 == 0:
                done = i
                bar(done)
        bar.stop()
        del bar
        # add data
        word_list = oFilter.get_words(False)
        del oFilter
        for item in word_list:
            val,wlength,x,y = item
            item += [self.get_subset(wlength,x,y,IDs),""]
        print("Done...")
        print()
        return word_list,headers,taxon_keys
    
    def confront_taxa(self,outgroups,accessions,mode,filter,wordlist=[]):
        oFilter = Filter(filter)
        convertor = nwmapper.Mapper()
        ids = []
        counterparts = []
        for i in range(len(outgroups)-1,-1,-1):
            acc = outgroups[i]
            genome_id = self.get_genomeID(acc)
            if genome_id != None:
                ids.insert(0,[genome_id,acc])
            else:
                del outgroups[i]
        if not ids:
            tools.alert("Error retrieving the outgroups!")
            return
        for i in range(len(accessions)-1,-1,-1):
            items = accessions[i]
            genome_set = []
            for j in range(len(items)-1,-1,-1):
                acc = items[j]
                genome_id = self.get_genomeID(acc)
                if genome_id != None:
                    genome_set.insert(0,[genome_id,acc])
                else:
                    del items[j]
            if genome_set:
                counterparts.insert(0,[])
                counterparts[0].extend(genome_set)
            else:
                del accessions[i]
        flg_data = False
        for genome_set in counterparts:
            if genome_set:
                flg_data = True
                break        
        if not flg_data:
            tools.alert("Error retrieving the counterpart genomes")
            return
        wlength_range = list(self.db.keys())
        wlength_range.sort()
        print("Confronted comparison")
        if not wordlist:
            wordlist = self.get_allWords()
        bar = progressbar.indicator(len(wordlist),"Run: ")
        done = 0
        for word in wordlist:
            if not word:
                continue
            if type(word) == type("Text"):
                wlength,x,y = convertor(word)
            else:
                wlength,x,y = word
            data = self.get_data(wlength,x,y,True)
            val = self.confrontation_index(data,ids,counterparts,mode)
            oFilter.add(wlength,x,y,val)
            done += 1
            if bar and done%1000:
                bar(done)
        bar.stop()
        del bar
        word_list = oFilter.get_words(False)
        del oFilter
        headers = []
        headers.extend(ids)
        for items in counterparts:
            headers.extend(items)
        for item in word_list:
            val,wlength,x,y = item
            item += [self.get_subset(wlength,x,y,headers),""]
        print("Done...")
        print()
        return word_list,outgroups,accessions
    
    def get_overrepresentedWords(self,ID):
        wlength_range = list(self.db.keys())
        wlength_range.sort()
        word_list = []
        for wlength in range(wlength_range[-1],wlength_range[0]-1,-1):
            for x in list(self.db[wlength]['x'].keys()):
                for y in list(self.db[wlength]['x'][x].keys()):
                    val = bitwiser.get_textVal(self.get_data(wlength,x,y,True),ID)
                    if val == "+++":
                        word_list.append([10.0,wlength,x,y,"",val])
        return word_list
                        
    def calculate_divergency(self,data,ids):
        # The new divergency is simply the varaince of
        # the level of the word in the different organism
        # To get the data into a format so that the new
        # divergency can be calculated, I will just reconstruct
        # the data from the "table" list
        # I will then calculate the variance using my own
        # variance formula, I code myself - this means that
        # I dont have to import numpy.

        n = int(1)
        index = 0
        for genome_id,acc in ids:
            n = bitwiser.insert(n,index,bitwiser.get_value(data,genome_id))
            index += 1

        val = bitwiser.divergency(n,len(ids))
        return val,n
    
    def calculate_commonality(self,data,ids,mode):
        # The new commonality score is just the average
        # "level" -- as defined in measures3.pdf.
        # This average is then transformed to fall in the
        # range 0 to 1
        n = int(1)
        index = 0
        for genome_id,acc in ids:
            n = bitwiser.insert(n,index,bitwiser.get_value(data,genome_id))
            index += 1

        if mode != "compare_taxa":
            # Creation of sample from table:
            val = bitwiser.commonality(n,len(ids))
            if mode=="rare":
                val = 10.0-val
        else:
            val = 0
        return val,n
    
    def confrontation_index(self,data,ids,counterparts,mode):
        outgroup_values = []
        for item in ids:
            genome_id,acc = item
            outgroup_values.append(bitwiser.get_rankborder(data,genome_id))
        n = len(outgroup_values)
        if n==1:
            outgroup_val = outgroup_values[0]
            outgroup_var = 0
        else:
            outgroup_val = float(sum(outgroup_values))/n
            outgroup_var = tools.variance(outgroup_values)

        counterpart_values = []
        for items in counterparts:
            if not items:
                continue
            item_values = []
            for item in items:
                genome_id,acc = item
                item_values.append(bitwiser.get_rankborder(data,genome_id))
            counterpart_values.append(float(sum(item_values))/len(item_values))
        counterpart_val = float(sum(counterpart_values))/len(counterpart_values)
        # Stat error
        mD = math.sqrt(1.0+outgroup_var/n)
        if mode=="+/-":
            score = abs(outgroup_val-counterpart_val)/mD
        elif mode=="+++":
            score = (10.0+outgroup_val-counterpart_val)/2.0/mD
        else:
            score = (10.0+counterpart_val-outgroup_val)/2.0/mD
        return score
    
    def get_wordstat(self,word,genome_list):
        if not genome_list:
            return ""
        convertor = nwmapper.Mapper()
        wlength,x,y = convertor(word)
        if wlength not in self.db and x in self.db[wlength]['x'] and y in self.db[wlength]['x'][x]:
            return ""
        if len(genome_list)==1:
            return bitwiser.get_textVal(self.get_data(wlength,x,y,True),self.genomes[genome_list[0]])
        else:
            values = []
            for acc in genome_list:
                values.append(bitwiser.get_rankborder(self.get_data(wlength,x,y,True),self.genomes[acc]))
            return float(sum(values)/len(values))-1.0
    
    def get_subset(self,wlength,x,y,ids):
        IDs = []
        for item in ids:
            IDs.append(item[0])
        return bitwiser.subset(self.get_data(wlength,x,y,True),IDs)
        
        
    def get_lineages(self):
        return self.lineages['genomes']

    """
    Recursively converts a nested dictionary into a Newick format string.
    """
    def get_lineage_tree(self, data='', title=''):
        if not data:
            data=self.lineages['genomes']
        if not title:
            title = self.title
        if not isinstance(data, dict) or len(data) == 0:
            return title
    
        children = []
        for key, value in data.items():
            children.append(self.get_lineage_tree(value, key))
        
        return f"({','.join(children)}){title}"

    #### ACCESSIONS
    def add_genome_ID(self, lineage, acc, ID=None):
        if ID == None:
            ID = self.next_ID()
        self.set_genomeID(lineage, acc, ID)
        return ID
    
    """
    path: path to input fasta or genbank file
    accession: genome accession
    lineage: list of lineage taxa: [class,genus,species,strain]
    """
    def add_genome(self, path: str, dataset: dict, accession: str, lineage: str, wl1: int = 8, wl2: int = 14, targer_seq_length: int = 0, chunk_number: int = 0) -> str:
        seq = dataset['Sequence'].upper()
        if targer_seq_length > 0 and chunk_number > 0:
            seq = self.deplete_sequence(seq, targer_seq_length, chunk_number)
        seqlength = len(seq)
        
        # Remove genome if exists
        if accession in self.genomes['acc']:
            self.delete_genome(accession)
        
        # Set new genome attributes
        ID = self.add_genome_ID(lineage, accession)
        # Count words
        self.process_sequence(lineage, sequence=dataset['Sequence'], ID=ID, wl1=wl1, wl2=wl2)
        # Add lineage path
        #lineage = path[:path.rfind(".")].split(os.sep)[2:]
        lineage = lineage.split("|")
        d = self.lineages['genomes']
        for taxon in lineage:
            if taxon not in d:
                d[taxon] ={}
            d = d[taxon]
        d[accession] = [ID, seqlength, path]  # add ID, sequence length and path
        self.lineages['acc'][accession] = "\t".join(lineage)
        return ID
    
    def deplete_sequence(self, seq: str, targer_seq_length: int, chunk_number: int) -> str:
        """
        Split a sequence into `chunk_number` chunks.
        - If len(seq) > targer_seq_length: select evenly spaced chunks whose combined length = targer_seq_length.
        - If len(seq) <= targer_seq_length: split the full sequence into `chunk_number` chunks.
    
        For both cases: reverse-complement every even-numbered chunk (2, 4, ...).
        Concatenate all chunks and return.
    
        Args:
            seq: Input nucleotide sequence (string).
            targer_seq_length: Target number of bases to keep across all chunks (if len(seq) is larger).
            chunk_number: Number of chunks to extract.
    
        Returns:
            Concatenated string of chunks (with even-numbered chunks RC'ed).
        """
        n = len(seq)
    
        if chunk_number <= 0:
            raise ValueError("chunk_number must be a positive integer.")
        if targer_seq_length < 0:
            raise ValueError("targer_seq_length must be non-negative.")
    
        # Case A: sequence is short enough → split entire sequence
        if n <= targer_seq_length:
            base_chunk_len = n // chunk_number
            rem = n % chunk_number
            chunk_lengths = [
                base_chunk_len + (1 if i < rem else 0)
                for i in range(chunk_number)
            ]
            pieces: List[str] = []
            pos = 0
            for i, clen in enumerate(chunk_lengths):
                chunk = seq[pos:pos + clen]
                if (i + 1) % 2 == 0:
                    chunk = tools.reverse_complement(chunk)
                pieces.append(chunk)
                pos += clen
            return "".join(pieces)
    
        # Case B: sequence longer than target → depletion logic
        if targer_seq_length < chunk_number:
            raise ValueError("targer_seq_length must be at least chunk_number so each chunk has ≥1 base.")
    
        # Split target length into nearly equal chunks
        base_chunk_len = targer_seq_length // chunk_number
        chunk_len_rem = targer_seq_length % chunk_number
        chunk_lengths: List[int] = [
            base_chunk_len + (1 if i < chunk_len_rem else 0)
            for i in range(chunk_number)
        ]
    
        # Compute spacers between chunks (and margins)
        total_gaps = n - targer_seq_length
        gaps_count = chunk_number + 1
        base_gap = total_gaps // gaps_count
        gap_rem = total_gaps % gaps_count
        gaps: List[int] = [
            base_gap + (1 if i < gap_rem else 0)
            for i in range(gaps_count)
        ]
    
        pieces: List[str] = []
        pos = gaps[0]
        for i, clen in enumerate(chunk_lengths):
            chunk = seq[pos:pos + clen]
            if (i + 1) % 2 == 0:
                chunk = tools.reverse_complement(chunk)
            pieces.append(chunk)
            pos += clen
            if i < chunk_number - 1:
                pos += gaps[i + 1]
    
        return "".join(pieces)

    # Calculate words
    def process_sequence(self, lineage, sequence, ID, wl1=4, wl2=11):
        out_string = f"{lineage}\n"
        convertor = nwmapper.Mapper()
        seqlength = len(sequence)
        counts = {}
        for wlength in range(wl1,wl2+1):
            tmpWordDB = Template(matrix_geometry = self.matrix_geometry)
            tmpWordDB.count_words(sequence=sequence, 
                wlength=wlength, 
                flg_progress=True, bar_text="Count "+str(wlength)+"-mers: ")
            words = tmpWordDB.get_allWords(wlength)
            length = len(words)
            
            bar = progressbar.indicator(length, str(wlength)+"-mers stat: ")
            counter = 1
            for wl, x, y in words:
                if not tmpWordDB.has(wl,x,y):
                    continue
                count = tmpWordDB.get(wl,x,y)
                wl,x,y,n = self.add(wl,x,y,count,True)
                value = tools.set_value(wl, count, seqlength, True)
                    
                self.append_value(wl,x,y,ID,value)
                counter += 1
                if counter%99 == 0:
                    try:
                        bar(counter)
                    except:
                        pass
            bar.stop()
            
    def next_ID(self):
        return len(self.genomes)
    
    # return the list of genome accessions sorted by their IDs in the database
    def get_genomes(self):
        accessions = list(self.genomes['acc'].items())
        # Sort by values
        accessions.sort(key = lambda ls: ls[1])
        return accessions
        
    def get_accessions(self):
        accessions = list(self.genomes['acc'].items())
        # Sort by values
        accessions.sort(key = lambda ls: ls[1])
        return accessions
        
    def delete_genome(self,acc):
        if acc not in self.genomes:
            return
        ID = self.get_genomeID(acc)
        for wlength in self.db:
            for x in self.db[wlength]['x']:
                for y in self.db[wlength]['x'][x]:
                    #self.db[wlength]['x'][x][y]['data'] = bitwiser.delete(self.get_data(wlength,x,y,False),ID)
                    if ID < len(self.db[wlength]['x'][x][y]['data']):
                        self.db[wlength]['x'][x][y]['data'] = self.db[wlength]['x'][x][y]['data'][:ID] + self.db[wlength]['x'][x][y]['data'][ID + 1:]
        del self.genomes[acc]
        for key in self.genomes:
            if self.genomes[key] > ID:
                self.genomes[key] -= 1

    def set_genomeID(self, lineage, acc, ID):
        if acc in self.genomes:
            tools.alert(f"Accession {acc} is already in the database")
        self.genomes['acc'][acc] = ID
        self.genomes['genomes'][acc] = lineage

    def get_genomeID(self,acc):
        if acc in self.genomes['acc']:
            return int(self.genomes['acc'][acc])
        else:
            return None
        
    def get_genomeRecord(self,acc):
        if acc not in self.genomes:
            return None
        ID = self.get_genomeID(acc)
        dataset = []
        for wlength in self.db:
            for x in self.db[wlength]['x']:
                for y in self.db[wlength]['x'][x]:
                    dataset.append([wlength,x,y,bitwiser.get_value(self.get_data(wlength,x,y,False),ID)])
        return dataset
    
    def set_genomeRecord(self,dataset,acc):
        if acc not in self.genomes:
            return None
        ID = self.get_genomeID(acc)
        for wlength,x,y,val in dataset:
            self.insert_value(wlength,x,y,ID,val)
    
    def get_accinteger(self,acc):
        if acc not in self.genomes:
            return None
        ID = self.get_genomeID(acc)
        s = []
        d = ["00","01","10","11"]
        for wlength in self.db:
            for x in self.db[wlength]['x']:
                for y in self.db[wlength]['x'][x]:
                    s.append(d[bitwiser.get_value(self.get_data(wlength,x,y,False),ID)])
        return "".join(s)

    def k_mers(self,k):
        convertor = nwmapper.Mapper()
        words = self.get_word_triplets()
        k_db = WordDB()
        for word in words:
            subwords = convertor.constituents(word,k)
            if not subwords:
                continue
            for i in range(len(subwords)):
                wl,x,y = subwords[i]
                k_db.add(wl,x,y)
                #k_db.append_info(subwords[i],word+[i])
        return k_db

    def has_genome(self,acc):
        return acc in self.genomes
    
    def insert_value(self,wlength,x,y,ID,val):
        # Operate with digits
        #self.db[wlength]['x'][x][y]['data'] = bitwiser.insert(self.get_data(wlength,x,y,False),ID,val)
        
        # Operate with strings
        if ID > len(self.db[wlength]['x'][x][y]['data']):
            self.append_value(wlength,x,y,ID,val)
        else:
            self.db[wlength]['x'][x][y]['data'].insert(ID, val)
    
    def append_value(self,wlength,x,y,ID,val):
        # Operate with digits
        # self.db[wlength]['x'][x][y]['data'] = bitwiser.replace(self.get_data(wlength,x,y,False),ID,val)
        
        # Operate with strings
        if not isinstance(self.db[wlength]['x'][x][y]['data'], list):
            self.db[wlength]['x'][x][y]['data'] = []
        self.db[wlength]['x'][x][y]['data'].append(val)
        
    def set_value(self,wlength,x,y,ID,val):
        # Operate with digits
        # self.db[wlength]['x'][x][y]['data'] = bitwiser.replace(self.get_data(wlength,x,y,False),ID,val)
        
        # Operate with strings
        self.db[wlength]['x'][x][y]['data'] = self.db[wlength]['y'][y][x]['data'] = [val]
        
    def update_value(self,wlength,x,y,ID,val):
        # Operate with digits
        # self.db[wlength]['x'][x][y]['data'] = bitwiser.replace(self.get_data(wlength,x,y,False),ID,val)
        
        # Check if 'data' list length equals the number of genomes
        if len(self.db[wlength]['x'][x][y]['data']) != self.next_ID():
            raise ValueError(f"Length of 'data' list {len(self.db[wlength]['x'][x][y]['data'])} does not correspond to the genome number {self.next_ID()}!")
            
        # Operate with strings
        self.db[wlength]['x'][x][y]['data'][-1] = val
        
    def get_value(self,wlength,x,y,acc):
        if acc not in self.genomes['acc']:
            return None
        ID = self.get_genomeID(acc)
        #return bitwiser.get_value(self.get_data(wlength,x,y,False),ID)
        return self.get_data(wlength,x,y,False)
        
    def get_values(self, word_list: list, acc_list: list) -> list:   # words = [[wlength, x, y], ...]
        matrix = [[self.get_value(*word_list[j], acc_list[j]) for j in range(acc_list)] for i in range(acc_list)]
        return matrix
        
    def get_info(self):
        output = []
        output.append(f"Database '{self.title}' version {self.version}")
        output.append("Genomes:\t" + str(len(self.genomes['acc'])))
        output.append("Words:\t\t" + str(len(self.get_allWords())))
        word_lengths = [int(v) for v in list(self.db.keys())]
        if len(word_lengths):
            for w in range(min(word_lengths),max(word_lengths)+1,1):
                output.append("\t"+str(w)+"mers:\t" + str(len(self.get_allWords(w))))
        return "\n".join(output)

    #### THE FOLLOWING FUNCTIONS ARE USED ONLY WHEN WORD_DB IS RUN AS THE MAIN MODULE
    # Save database
    def save_dbfile(self, fname=""):
        tools.saveDBFile(fname=fname, data=self.db, 
            supplementary={'genomes':self.genomes,
            'lineages':self.lineages,
            'version':self.version,
            'date':self.date,
            'title':self.title,
            'matrix_geometry':self.matrix_geometry,
            'supplementary':self.supplementary_data})
    
    # Load data from binary database file
    def open_dbfile(self, fname=""):
        fname,DB,supplementary = tools.openDBFile(fname)
        self.db = DB
        self.genomes = supplementary['genomes']
        self.matrix_geometry = "whole"
        if 'matrix_geometry' in supplementary:
            self.matrix_geometry = supplementary['matrix_geometry']
        self.lineages = {}
        if 'lineages' in supplementary:
            self.lineages = supplementary['lineages']
        self.supplementary_data = {}
        if 'supplementary' in supplementary:
            self.supplementary_data = supplementary['supplementary']
        title = date = version = ""
        if 'date' in supplementary:
            self.date = supplementary['date']
        if 'title' in supplementary:
            self.title = supplementary['title']
        if 'version' in supplementary:
            self.version = supplementary['version']
        self.path = fname
            
    # Load data from text file
    def import_db(self, infile, outfile = ""):
        self.lineages = {'acc':{},'genomes':{}}
        self.genomes = {'acc':{},'genomes':{}}
        # Add genome lineage line
        def add_lineage(line):
            obj = self.lineages['genomes']
            lineage = line.split("|")
            values = [int(v) for v in lineage[-1][1:-1].replace(" ","").split(",")]
            lineage = lineage[:-1]
            for taxon in lineage:
                taxon = taxon.strip()
                if taxon not in obj:
                    obj[taxon] = {}
                obj = obj[taxon]
            obj[taxon] = values
            self.lineages['acc'][lineage[-1]] = "\t".join(lineage[:-1])
            
        # Add accession
        def add_accession(line):
            acc, genome_id = [s.strip() for s in line.split("\t")]
            self.genomes['acc'][acc] = genome_id
            
        # Add k-mer record
        def add_word(line):
            word,wlength,x,y,counter,record = line.split("|")
            wlength,x,y,counter,record = [int(v) for v in [wlength,x,y,counter,record]]
            if wlength not in self.db:
                self.db[wlength] = {'x':{},'y':{}}
            if x not in self.db[wlength]['x']:
                self.db[wlength]['x'][x] = {}
            if y not in self.db[wlength]['y']:
                self.db[wlength]['y'][y] = {}
            if y not in self.db[wlength]['x'][x]:
                self.db[wlength]['x'][x][y] = {}
            self.db[wlength]['x'][x][y]['counter'] = counter
            #self.db[wlength]['x'][x][y]['data'] = record
            self.db[wlength]['x'][x][y]['data'] = record.split(",")
            self.db[wlength]['y'][y][x] = self.db[wlength]['x'][x][y]
            
        # Add supplementary data
        def add_supplementary(line):
            self.supplementary_data = ast.literal_eval(line.strip())
            
        # Create empty dictionary self.lineages and self.db
        self.lineages = {'acc':{},'genomes':{}}
        self.db = {}
        # Parse database file in text format
        mode = ""
        with open(infile,'r') as f:
            for line in f.readlines():
                if line.startswith("#"):
                    if line.startswith("#TITLE"):
                        mode = "title"
                    elif line.startswith("#LINEAGES"):
                        mode = "lineages"
                    elif line.startswith("#ACCESSIONS"):
                        mode = "accessions"
                    elif line.startswith("#WORDS"):
                        mode = "words"
                    elif line.startswith("#VERSION"):
                        mode = "version"
                    elif line.startswith("#SUPPLEMENTARY"):
                        mode = "supplementary"
                    continue
                if mode == "title":
                    self.title = line.strip()
                elif mode == "version":
                    self.version = line.strip()
                elif mode == "lineages":
                    add_lineage(line.strip())
                elif mode == "accessions":
                    add_accession(line.strip())
                elif mode == "words":
                    add_word(line.strip())
                elif mode == "supplementary":
                    add_supplementary(line.strip())
        # Get the current date
        current_date = datetime.now()
        # Format the date as dd/mm/yyyy
        formatted_date = current_date.strftime('%d/%m/%Y')        
        # Save database in inary format to an output file, if provided
        self.date = formatted_date
        # Save outfile
        if outfile:
            self.save_dbfile(fname=outfile)
                
    # Export database to dictionary
    def export_db(self, min_k=4, max_k=0, outfile=''):
        def genome(s):
            ls = s.split("|")
            parts = ls[-1][1:-1].split(",")
            return {"lineage":"|".join(ls[:-2]),
                    "accession":ls[-2],
                    "ID":int(parts[0]) - 1,
                    "seqlength":int(parts[1]),
                    "source":str(parts[2]),
                }
            
        # Add title
        text_db_title = ["#TITLE"]
        try:
            title = self.title
        except:
            title = ""
        text_db_title.append(title)
        # Add version
        text_db_version = ["#VERSION"]
        try:
            version = str(self.version)
        except:
            version = "1.0"
        text_db_version.append(version)
        # Add supplementary data
        text_db_supplementary = ["#SUPPLEMENTARY"]
        try:
            supplementary = str(self.supplementary_data)
        except:
            supplementary = "{}"
        text_db_supplementary.append(supplementary)

        data = {
            "info":self.get_info(),
            "genomes":[genome(g) for g in tools.dict_to_list(self.lineages['genomes'])],
            "title":self.title,
            "version":self.version,
            "date":self.date,
            "matrix_geometry":self.matrix_geometry,
            "values":[],
            "words":[]
        }
        
        text_lineages = ["#LINEAGES","#Lineage|[ID,Count]"] + tools.dict_to_list(self.lineages['genomes'])
        text_accessions = ["#ACCESSIONS","#Accession|Genome_ID"] + ["%s\t%s" % (str(item[0]),str(item[1])) for item in self.genomes.items()]
        text_words = ["#WORDS","#Word|wlength|X|Y|Count|Record"]
        
        words = self.get_allWords(wl_lower=min_k, wl_upper=max_k, flg_add_words=True)
        data['words'] = [{
                        "word" : words[i][0],
                        "length" : str(words[i][1]),
                        "x" : str(words[i][2]),
                        "y" : str(words[i][3]),
                        "index" : str(i),
                        } 
            for i in range(len(words))]
        
        data['values'] = [self.db[w[1]]['x'][w[2]][w[3]]['data'] for w in words]
        
        if outfile:
            # Save DB in text format
            tools.saveTextFile("\n".join(text_db_title + text_db_version + text_db_supplementary + text_lineages + text_accessions + text_words), outfile)

        return data
        
    def get_matrix(self):
        data = self.export_db()
        matrix = [['Genomes'] + [item['word'] for item in data['words']]]
        for i in range(len(data['genomes'])):
            matrix.append([data['genomes'][i]])
            for value in data['values']:
                matrix[-1].append(value)
        return matrix        
        
###################################################
if __name__ == "__main__":
    
    def msg(msg,title=""):
        if title:
            title += "\n"
        print(f"\n{title}{msg}\n")
        
    def do(oDB):
        response = True
        while response:
            show_menue()
            response = input("?").upper()
            print()
            if response.upper() == "Q":
                return
            elif response.upper() == "E":
                export_db(oDB)
            elif response.upper() == "S":
                save_db(oDB)
            elif response.upper() == "I":
                show_info(oDB)
            elif response.upper() == "A":
                show_acc_info(oDB)
            elif response.upper() == "C":
                compare_genomes(oDB)
            else:
                continue
        
    def show_menue():
        print("\nSelect a command from the list or press Q to exit:")
        print("\tE - export database to a text file")
        print("\tS - save database to a binary file")
        print("\tI - show database summary")
        print("\tA - show genome accessions")
        print("\tC - compare genome")
        print()
    
    # Exportdatabase to text file
    def export_db(oDB):
        path = None
        while path == None:
            path = input("Enter path to export database or enter Q to exit: ")
            if path.upper() == "Q":
                return
            if not path.endswith(".txt"):
                path += ".txt"
            try:
                oDB.export_db(path)
            except:
                msg(f"Exporting to {path} was unsuccessful! Check the path and try again.","Alert!")
                path = None
                continue
        msg(f"Database {oDB.title} was successfuly exported to {path}!")

    # Save database to binary file
    def save_db(oDB):
        path = None
        while path == None:
            path = input("Enter path to save database or enter Q to exit: ")
            if path.upper() == "Q":
                return
            if not path.endswith(".wdb"):
                path += ".wdb"
            try:
                oDB.save_dbfile(path)
            except:
                msg(f"Saving to {path} was unsuccessful! Check the path and try again.","Alert!")
                path = None
                continue
        msg(f"Database {oDB.title} was successfuly saved to {path}!")
    
    # Get Info
    def show_info(oDB):
        output = []
        output.append("Genomes:\t" + str(len(oDB.genomes['acc'])))
        output.append("Words:\t\t" + str(len(oDB.get_allWords())))
        for w in range(8,15,1):
            output.append("\t"+str(w)+"mers:\t" + str(len(oDB.get_allWords(w))))
        msg("\n".join(output))
        
    # Show list of genome accessions
    def show_acc_info(oDB):
        msg(", ".join(list(oDB.genomes['acc'].keys())))

    def compare_genomes(oDB):
        success = False
        while success == False:
            accessions = input("Input comma-separated list of accessions or Q to abort: ").strip()
            if not accessions:
                continue
            if accessions.upper() == "Q":
                return
            accessions = [acc.strip() for acc in accessions.split(",")]
            not_found = []
            for acc in accessions:
                if acc not in oDB.genomes['acc']:
                    not_found.append(acc)
            if not_found:
                msg(f"The following accessions were not found in the database: {','.join(not_found)}!","Alert!")
                continue
            print("\nSelect mode of comparison:")
            print("\tD - select diverse words")
            print("\tA - select commonly abundant words")
            print("\tR - select commonly rare words")
            print("\tQ - exit genome comparison")
            response = input("? ")
            if response.upper() == "Q":
                return
            if response.upper() == "A":
                select_words(oDB,accessions,"abundant")
            elif response.upper() == "R":
                select_words(oDB,accessions,"rare")
            elif response.upper() == "D":
                select_words(oDB,accessions,"diverse")
            else:
                continue
            
    def select_words(oDB,accessions,mode):
        filter_settings = set_word_filter()
        output_file = input("\nEnter name of the output file or keep empty and press ENTER: ")
        result = oDB.get_extremeWords(mode,accessions,filter_settings,flg_bar=True)
        if result:
            word_list,headers = result
        else:
            msg("Error during genome comparison!","Alert")
            return
        #print(result)
        report = tools.print_genomes(word_list,headers)
        if output_file:
            try:
                with open(output_file,"w") as f:
                    f.write(report)
            except:
                msg(f"Error when saving genome comparison report to file {output_file}","Alert!")
                return
        else:
            print(report)
        
    def set_word_filter():
        return None
        
    #### MAIN PROCESS
    path = None
    while path == None:
        #path = input("Enter path to database file in 'wdb' or 'txt' formats or enter Q to exit: ").strip().replace('\\x00','')
        path = "../db/Bacillus.wdb"
        if path.upper() == "Q":
            exit()
        if not os.path.exists(path):
            msg(f"Path {path} does not exist!","Alert!")
            path = None
            continue
        # Try to open DB file
        oDB = WordDB()
        success = False
        # Pickle binary file
        if path.endswith(".wdb"):
            try:
                oDB.open_dbfile(path)
                success = True
            except:
                pass
        # Import from text file
        elif path.endswith(".txt"):
            try:
                oDB.import_db(infile=path)
                success = True
            except:
                pass
        if not success:
            msg(f"File {path} is in a wrong format or corrupted! Try again.","Alert!")
            path = None
            continue
            
        print(oDB.genomes.keys())
        print(oDB.genomes['acc'])
        print(oDB.genomes['genomes'])
        sys.exit()

        do(oDB)
        

    '''
    # CREATION OF NEW DATABASE AND LOADING FROM TEXT FILE
    # Database version
    version = "1.0"
    # Database title
    title = "Bacteria"
    # Create an empty database
    oDB = WordDB(title=title, version=version)
    
    # Load data from text file and save to binary outfile
    oDB.import_db(infile=os.path.join("..","db","bacteria_db.txt"),
        outfile=os.path.join("..","db","bacteria.wdb"))
    
    # LOADING DATA FROM BINARY FILE TO EMPTY DATABASE
    # Create an empty database
    oDB = WordDB()
    # Load data from DB file
    oDB.open_dbfile(os.path.join("..","db","Bacillus.wdb"))
    
    # Exporting database to text format file
    oDB.export_db(os.path.join("..","db","Bacillus_py3_export.txt"))
    
    print (list(oDB.genomes['genomes'].keys()))
    print(oDB.get_genomeID("UCMB5140"))
    '''
