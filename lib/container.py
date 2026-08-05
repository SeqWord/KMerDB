import tools
########################################################################
class Collection:
    def __init__(self, title: str = "", version: str = "0", date: str = ""):
        self.title = title
        self.version = version
        self.date = date
        # Format the date as dd/mm/yyyy
        if not self.date:
            self.date = tools.get_current_date()
            
        self.container = []
        self.para = {}
        
    def _get_key(self, key: str | int):
        if isinstance(key, int):
            if len(self) <= key:
                return None
            return key
        elif isinstance(key, str):
            return self.index(key)
        else:
            sys.exit(f"Unsupported index type {type(key)}!")
    
    def _get_title(self, obj):
        if hasattr(obj, 'title'):
            return obj.title
        return ""

    def __len__(self):
        return len(self.container)
        
    def __contains__(self, key: str | int):
        if self._get_key(key) is not None:
            return True
        return False
    
    def __iter__(self):
        if not self.container:
            return iter([])
        records = []
        for record in self.container:
            records.append(record)
        return iter(records)
    
    def __getitem__(self, key: str | int):
        key = self._get_key(key)
        if key is not None:
            return self.container[key]
        return None
    
    def __setitem__(self, key: str | int, value):
        key = self._get_key(key)
        if key is not None:
            self.container[key] = value
    
    def __delitem__(self, key: str | int):
        key = self._get_key(key)
        if key is not None:
            del self.container[key]
            
    def __repr__(self):
        #return "\n".join(list(map(lambda i: "%d\t%s" % (i+1,str(self.container[i])), range(len(self.container)))))
        return "\n".join([f"{i + 1}\t{self.container[i]}" for i in range(len(self.container))])
            
    def __str__(self):
        #return ";".join(list(map(lambda Obj: str(Obj), self.container)))
        return ";".join([str(Obj) for Obj in self.container])
            
    def has(self, title: str):
        return title in self.get_titles()
        
    def get_next_ID(self):
        return len(self.container)
    
    def index(self, title: int | str):
        if isinstance(title, int):
            index = title
            if abs(index) >= len(self):
                return None
            return index
        elif isinstance(title, str):
            titles = self.get_titles()
            if title in titles:
                return titles.index(title)
            return None
        else:
            sys.exit(f"Unsupported index type {type(title)}!")
    
    def append(self, obj):
        self.container.append(obj)
        
    def extend(self, ls: list):
        for obj in ls:
            if hasattr(obj, "ID"):
                obj.ID = self.get_next_ID()
            self.append(obj)
        
    def get_titles(self):
        #return list(map(lambda obj: obj.title, self.container))
        return [obj.title for obj in self.container]
        
    def keys(self):
        return self.get_titles()
    
    def get(self, titles: list = []):
        if titles:
            #return list(filter(lambda Obj: Obj.title in titles, self.container))
            return [Obj for Obj in self.container if Obj.title in titles]
        else:
            return self.container
            
    def get_length(self):
        return len(self.container)
            
    def copy(self):
        oCollection = Collection(self.title)
        oCollection.para.update(self.para)
        for record in self.container:
            oCollection.append(record.copy())
        return oCollection
            
