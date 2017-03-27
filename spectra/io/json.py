from cfjson.xrdataset import *

def to_json(self,filename,attributes={}):
    strout=self.dset.cfjson.json_dumps(indent=2,attributes=attributes)
    with open(filename,'w') as f:
        f.write(strout)

def read_json(self,filename):
    raise NotImplementedError('Cannot read CFJSON format')