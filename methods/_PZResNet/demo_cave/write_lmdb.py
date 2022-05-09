import lmdb
import os
import pickle
import numpy as np
path = '/home/zhu_19/data/'
path1 = '/home/zhu_19/data/lmdb/'
name = 'data.lmdb'
new_load =  lambda *a,**k: np.load(*a, allow_pickle=True, **k)
for _,_,name_ in os.walk(path):
    if len(name_)>1:
        break
with open('filename.txt','wb') as f:
    pickle.dump(name_,f)

env = lmdb.open(path = path1+name, map_size=107374182400)
txn = env.begin(write=True)
for index , temp in enumerate(name_):
    n,_ = temp.split('.')
    data = new_load(path + temp)[()]
    hsi = data['hsi']
    rgb = data['rgb']
    nhsi = temp + '_hsi'
    nrgb = temp +'_rgb'
    txn.put(nhsi.encode(),hsi)
    txn.put(nrgb.encode(),rgb)
    print('{} save {}'.format(index,n))
txn.commit()
