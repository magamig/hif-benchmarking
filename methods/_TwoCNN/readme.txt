1. Put the prototxt in examples folder of caffe, training the caffemodel
2. Put the caffemodel in matlab\fusion folder, and run "saveFIlters_fusion.m", caffemodel could be read in matlab
3. Run "main.m" in matlab, you could reconstruct HR HSI in matlab. 

Notes: The recostruction of HR HSI is more clear in matlab, but the running time would be long since there is no GPU here.