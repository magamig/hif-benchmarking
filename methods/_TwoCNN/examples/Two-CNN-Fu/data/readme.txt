put the data for training or testing in the fold. 

The data can be leveldb or hdf5

The data in HSI branch is of size spectral_bands x 1 x number_of_samples 
The data in MSI branch is of size spatial_window x spatial_window x number_of_samples
The output data is spectrum in HR HSI with size spectral_bands x 1 x number_of_samples