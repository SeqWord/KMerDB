
########################################################################
class Collection:
	def __init__(self):
		self.container = []
		
	def _get_key(self,key):
		if type(key)==type(0):
			if len(self) <= key:
				return
			return key
		elif type(key)==type(""):
			return self.index(key)
	
	def _get_title(self,obj):
		if hasattr(obj, 'title'):
			return obj.title
		return ""

	def __len__(self):
		return len(self.container)
		
	def __contains__(self,key):
		if self._get_key(key) != None:
			return True
		return False
	
	def __iter__(self):
		if not self.container:
			return iter([])
		records = []
		for record in self.container:
			records.append(record)
		return iter(records)
	
	def __getitem__(self,key):
		key = self._get_key(key)
		if key != None:
			return self.container[key]
	
	def __setitem__(self,key,value):
		key = self._get_key(key)
		if key != None:
			self.container[key] = value
	
	def __delitem__(self,key):
		key = self._get_key(key)
		if key != None:
			del self.container[key]
			
	def __repr__(self):
		return "\n".join(list(map(lambda i: "%d\t%s" % (i+1,str(self.container[i])), range(len(self.container)))))
			
	def __str__(self):
		return ";".join(list(map(lambda Obj: str(Obj), self.container)))
			
	def has(self,title):
		return title in self.get_titles()
	
	def index(self,title):
		if type(title) == type(0):
			index = title
			if abs(index) >= len(self):
				return
			return index
		titles = self.get_titles()
		if title in titles:
			return titles.index(title)
		return
	
	def append(self,obj):
		self.container.append(obj)
		
	def extend(self,ls):
		self.container.extend(ls)
		
	def get_titles(self):
		return list(map(lambda obj: obj.title, self.container))
	
	def get(self,titles=[]):
		if titles:
			return list(filter(lambda Obj: Obj.title in titles, self.container))
		else:
			return self.container
			
