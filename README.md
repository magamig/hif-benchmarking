# Multispectral and Hyperspectral Image Fusion Benchmarking

Comparison the MS and HS Image Fusion techniques used for the spatial resolution enhancement of hyperspectral images.

## Instructions

Download and process dataset(s) (e.g.: CAVE):

```
bash aux/CAVE/download.sh
python aux/CAVE/process.py
```

Post process dataset(s) to create MS image and downsampled HS image by a factor of 4 (or any other power of 2):

```
python aux/postprocess.py CAVE 4
```

Run all algorithms over the dataset:

```
python run.py
```

You can edit ``run.py`` to customize the combinatory that you wish to process in terms of datasets, methods and scaling factors.