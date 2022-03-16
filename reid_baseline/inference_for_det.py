import os
import time 
start = time.time()
DIR = "/home/g08410099/Documents/Yet-Another-EfficientDet-Pytorch-master/DETRAC-test-data/Insight-MVT_Annotation_Test/"
for dir in os.listdir(DIR):
    #if dir != "c014":
    #    continue
  
    if not dir.startswith("OUT") and  dir != "OUT_MVI_39311":
        continue
    print(dir)
    
    if os.path.isdir(DIR + dir):
        #if os.path.isfile("/home/g08410099/Documents/aic19/mtmc-vt-master/src/aic19-track1-mtmc/test2020/S01/" + dir + "/deep_features.txt" ):
            
        #continue
        print(dir)
        
        os.system("python tools/inference.py  --config_file=" + "configs//track2_softmax_triple.yml " + "TEST.WEIGHT " + "export_dir/aic_track2/softmax_triplet/resnet50_checkpoint_36859.pth " + \
             "DATASETS.DATA_PATH " + DIR + dir)
         
        os.system("mv feature.txt deep_features.txt")
         
        os.system("mv deep_features.txt " + DIR  + dir[4:] + "/det")
         
end = time.time()
print(end - start)