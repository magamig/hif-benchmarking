# Hyperspectral Image Super-Resolution with Optimized RGB Guidance

This repository is a Caffe implementation of the paper "Hyperspectral Image Super-Resolution with Optimized RGB Guidance" from CVPR 2019

If you find our work useful in your research or publication, please cite our work:

[1] Ying Fu, Tao Zhang, Yinqiang Zheng, Debing Zhang, and Hua Huang, "Hyperspectral Image Super-Resolution with Optimal RGB Guidance", CVPR 2019.

```
@InProceedings{Fu_2019_CVPR,
  author = {Fu, Ying and Zhang, Tao and Zheng, Yinqiang and Zhang, Debing and Huang, Hua},
  title = {Hyperspectral Image Super-Resolution with Optimized RGB Guidance},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2019}
}
```

## Note
Please add 
```
optional float decay_mult2 = 5 [default = 0.0];
```
to `message ParamSpec` in `caffe/src/proto/caffe.proto`.
