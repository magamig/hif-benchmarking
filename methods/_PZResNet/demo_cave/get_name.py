import os
import pickle
# def get_name(path):
#     for _,_ ,temp_name in os.walk(path):
#         if len(temp_name) != 0:
#             break
#     return temp_name

# def get_name():
#     with open('filename.txt','rb') as f:
#         name = pickle.load(f)
#     return name
def get_name(itera=True):
    name = []
    f = open("./full_name.txt")
    line = f.readline()
    while line:
        line = line.split()
        try:
            if line[0].find('_')>0 or line[0].find('-')>0 or line[0].find('00')>0:
                name.append(line)
        except:
            pass
        line = f.readline()
    return_name = []
    for name_ in name:
        # if itera == True:
        #     yield(name_[0]+'.mat')
        # else:
        return_name.append(name_[0]+'.mat')
    return return_name