import os
import time 
start = time.time()
#DIR = "/home/g08410099/Documents/aic19/mtmc-vt-master/src/aic19-track1-mtmc/adjust_c_cropped_imgs2020/backup_2/"
DIR = "/home/g08410099/Documents/aic19/mtmc-vt-master/src/aic19-track1-mtmc/adjust_c_cropped_imgs2020/"
for dir in os.listdir(DIR):
    #if dir != "c014":
    #    continue
    
    if not dir.startswith("c0") or dir != "c0cgta":
        continue
    print(dir)
    
    if os.path.isdir(DIR + dir):
        #if os.path.isfile("/home/g08410099/Documents/aic19/mtmc-vt-master/src/aic19-track1-mtmc/test2020/S01/" + dir + "/deep_features.txt" ):
            
        #    continue
        print(dir)
        
        
        
        
        os.system("python tools/inference.py  --config_file=" + "configs//track2_softmax_triple.yml " + "TEST.WEIGHT " + "export_dir/aic_track2/softmax_triplet/resnet50_checkpoint_36859.pth " + \
             "DATASETS.DATA_PATH " + DIR + dir)
        os.system("mv feature.txt deep_features.txt")
        os.system("mv deep_features.txt " + "/home/g08410099/Documents/aic19/mtmc-vt-master/src/aic19-track1-mtmc/test2020/S01/" + dir)
end = time.time()
print(end - start)