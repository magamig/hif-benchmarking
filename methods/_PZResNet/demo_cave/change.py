import h5py
from get_name import get_name
import numpy as np
import time
import threading
import os
import lmdb
# name = get_name(itera = True)
path = '/home/zhu_19/Hyperspectral_image/data/ICVL/'
path_1 ='/home/zhu_19/data/'
path1 = '/home/zhu_19/data/lmdb/'
i = 0
thread_num = 100
# class image_name_safe():
#     def __init__(self):
#         super(image_name_safe,self).__init__()
#         self.lock = threading.Lock()
#         self.iter = get_name()
#     def __iter__(self):
#         return self
#     def next(self):
#         with self.lock:
#             return next(self.iter)

for _,_,temp_name in os.walk(path_1):
    if len(temp_name) == 199:
        break

# lock = threading.Lock()
img_gen = get_name()
env = lmdb.open(path = path1+'data', map_size=107374182400)
txn = env.begin(write=True)
namelist = []

for key,value in txn.cursor():
    i += 1
    namelist.append(key.decode())

def loop(txn):
    i = 0
    while True:
        i+=1
        # for n in name
        n = next(img_gen)
        print(n)
        if n == None:
            break
        # try:
        a,_ = n.split('.')
        name = a
        nhsi = name +'_hsi'
        nrgb = name +'_rgb'
        if nhsi in namelist:
            print('pass{}'.format(i))
            continue

        with h5py.File(path+n,'r') as f:
            hsi_g = f['rad'].value
            rgb = f['rgb'].value
            hsi_g = np.flip(np.transpose(hsi_g,axes=(0,2,1)),2)
            rgb = np.ascontiguousarray(rgb)
            hsi_g = np.ascontiguousarray(hsi_g)
            if hsi_g.shape[1]>1024 and hsi_g.shape[2]>1024 and rgb.shape[1] >1024 and rgb.shape[2] >1024:
                hsi_g = hsi_g[:,:1024,:1024]
                hsi_g = np.clip(hsi_g,0,4000)
                rgb = rgb[:,:1024,:1024]
                txn.put(nhsi.encode(),hsi_g.tobytes())
                txn.put(nrgb.encode(),rgb.tobytes())
                # i += 1
            else:
                print('size of {} is not enough'.format(name))
        if i % 1 == 0:
            txn.commit()
            txn = env.begin(write=True)
        print('{} done {}'.format(i,name))

loop(txn)
# loop()
# for _,_,temp_name in os.walk(path_1):
#     if len(temp_name) != 0:
#         break
# # print(len(temp_name))
# while True:
#     a = next(img_gen)
#     if a == None:
#         break
#     a,_ = a.split('.')
#     name = a+'.npy'
#     if name in temp_name:
#         # print('exit_{}'.format(name))
#         pass
#     else:
#         print('produce_{}'.format(name))
# for i in range(0, thread_num):
#     time.sleep(0.1)
#     t = threading.Thread(target=loop, name='thread{}'.format(i),)
#     t.start()
# n = 'prk_0328-1034.mat'
# a = 'prk_0328-1034'
# with h5py.File(path+n,'r') as f:
#     hsi_g = f['rad'].value
#     rgb = f['rgb'].value
#     hsi_g = np.flip(np.transpose(hsi_g,axes=(0,2,1)),2)
#     np.save( path_1+a+'.npy',{'hsi':hsi_g,'rgb':rgb})
#     print('done_{}'.format(a+'.npy'))