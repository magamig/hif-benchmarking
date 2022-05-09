import lmdb
import numpy as np
from get_name import get_name
path = '/home/zhu_19/data/lmdb/data/'
env = lmdb.open(path)
# txn = env.begin(write= True)
print(1)
# txn.put('00231'.encode(),'asdadwararf'.encode())
# txn.commit()
txn = env.begin()
i = 0
namelist = []
for key,value in txn.cursor():
    i += 1
    # if i > 2:
    print(key.decode(),i)
    print(np.frombuffer(value).shape)
name_ = get_name()
print(len(name_))