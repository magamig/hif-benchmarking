deep_cnn.py: model
eval_deep_cnn.py: test
HSI_getdata.m： produce the training data

train:
python deep_cnn.py

test:
python eval_deep_cnn.py


Requirements:

python:2.7
keras:2.0.8(tf back_end)
numpy:1.14.1
h5py:2.7.1

matlab: