# coding=UTF-8
import os


path = './data/CAVE/train/HSI'
imgs = os.listdir(os.path.join(path))
imgs.sort()
num = len(imgs)
for im in range(0,num):
    im_path = os.path.join(os.path.join(path),imgs[im])
    print(im_path)
    fp = open(os.path.join('./data/pathlist/datalist_NSSR_P.txt'), 'a')
    fp.write(im_path)
    fp.write('\n')
    fp.close()