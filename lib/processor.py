from threading import Thread

###############################################################################
class ExtremeWordProcessor(Thread):
    def __init__(self,function,accessions=[],db=None,filter=None):
        Thread.__init__(self)
        self.main = function
        self.accessions = accessions
        self.db = db
        self.filter = filter
        self.output = None
        
    def run(self):
        self.output = self.main("compare_taxa",self.accessions,self.filter,None)
        
    def get(self):
        return self.output

###############################################################################
class WordCountProcessor(Thread):
    def __init__(self,function,sequence,wlength,db,flg_progress=False):
        Thread.__init__(self)
        self.main = function
        self.sequence = sequence
        self.wlength = wlength
        self.db = db
        self.flg_progress = flg_progress
        
    def run(self):
        self.output = self.main(self.sequence,self.wlength,self.db,self.flg_progress)
        
    def get(self):
        return self.db

