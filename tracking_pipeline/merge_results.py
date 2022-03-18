import numpy as np
import os
import cv2
import math
from PIL import Image
import PIL.Image
import testr
import time
def read_cam_dict():
    d = {}
    with open("list_video_id.txt") as f:
        for line in f:
            
            line = line.strip().split(" ")
            
            d[line[1]] = line[0]
        
    return d    
input_dir = "../aic19-track1-mtmc/test2020_2"
def main():
    scene_dirs = []
    scene_fds = os.listdir(input_dir)
    video_number = 1
    for scene_fd in scene_fds:
        scene_dirs.append(os.path.join(input_dir, scene_fd))

    for scene_dir in scene_dirs:
        camera_dirs = []
        fds = os.listdir(scene_dir)
        cam_dict = read_cam_dict()
        for fd in fds:
            #if fd.startswith('c0'):
                camera_dirs.append(os.path.join(scene_dir, fd))
        counter = 0
        file = open("final_output.txt","w")
        for camera_dir in camera_dirs:
            if not camera_dir.split("/")[-1].startswith("c0") or not os.path.isfile(camera_dir + "/final_ouput.txt"):
                counter+=1
                continue
            print(camera_dir)
            f_output = open(camera_dir+"/final_ouput.txt")
            for line in f_output:
                file.write(cam_dict[camera_dir.split("/")[-1]] + " " + line)
            f_output.close()
        file.close()
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(end - start)