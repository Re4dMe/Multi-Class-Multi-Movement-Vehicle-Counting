#load track
#
#for each track
#   test if track match any predifine road
#       if match:
#           car of this road += 1
#       elif not match:
#           discard this track
#   
import numpy as np
import os
import cv2
import math
from PIL import Image
import PIL.Image
import testr
import time
MATCHED = True
NO_MATCHED = False
SIMILAR_TH = 700
SPEED_TH = 100
TRACK_TH = 2
MOVE_TH = 20000
eps = 0.000001

input_dir = "../aic19-track1-mtmc/test2020_2"

PAD_SIZE = 10


class Box(object):
    def __init__(self, frame_index, id, box, score, gps_coor, feature):
        self.frame_index = frame_index
        self.id = id
        self.box = box
        self.score = score
        self.gps_coor = gps_coor
        self.feature = feature
        self.center = (self.box[0] + self.box[2] / 2, self.box[1] + self.box[3] / 2)
        self.match_state = NO_MATCHED
        self.floor_center = (self.box[0] + self.box[2] / 2, self.box[1] + self.box[3] * 0.8)

    def get_area(self):
        return self.box[2]*self.box[3]


def sub_tuple(sp, ep):
    return ep[0] - sp[0], ep[1] - sp[1]

class Track(object):

    def __init__(self, id, sequence):
        self.id = id
        self.sequence = sequence
        self.match_state = MATCHED
        # self.leak_time = int(0)   

    # determine whether the movement is a straight line or a curve 
    def is_straight(self):
        if len(self.sequence)<4:
            return True

        sp = self.sequence[0].center  # start point
        ep = self.sequence[-1].center  # end point
        # move_vec = sub_tuple(sp, ep)
        # Ax+By+C=0
        A = ep[1]-sp[1]
        B = sp[0]-ep[0]
        C = ep[0]*sp[1] - ep[1]*sp[0]
        D = math.sqrt(A*A+B*B)
        dis_sum = 0
        for item in self.sequence:
            p = item.center
            dis_sum = dis_sum + abs(A*p[0]+B*p[1]+C)/(D+eps)
        dis_mean = dis_sum/(len(self.sequence)-2+eps)

        if dis_mean<150:
            return True
        else:
            return False

    def area_stable(self):
        s = self.sequence[0].get_area()
        e = self.sequence[-1].get_area()
        k = s/float(e+eps)
        # return k
        if k>0.333 and k<3:
            return True
        else:
            return False

    def calu_vec_orientation(self, vec):
        x = vec[0]
        y = vec[1]
        AREA_STABLE = self.area_stable()
        STRAIGHT = self.is_straight()

        if x > 0 and abs(x / (y + eps)) > 4 and AREA_STABLE and STRAIGHT:
            return 'c'
        if x < 0 and abs(x / (y + eps)) > 4 and AREA_STABLE and STRAIGHT:
            return 'c'
        if y > 0 and abs(y / (x + eps)) > 1.7 and STRAIGHT:
            return 'f'
        if y < 0 and abs(y / (x + eps)) > 1.7 and STRAIGHT:
            return 'b'
        if y > 0:
            return 'fc'
        else:
            return 'bc'

    def append(self, box):
        self.sequence.append(box)

    def get_orientation(self):
        l = self.get_length()

        # return if trajectory is not long enough
        if l<2:
            return 'fbc'

        move_dis = calu_moving_distance(self.sequence[0], self.sequence[-1])

        if move_dis > MOVE_TH:
            move_vec = calu_moving_vec(self.sequence[0], self.sequence[-1])
            ori = self.calu_vec_orientation(move_vec)

            return ori
        else:
            return 'fbc'

         


    def remove_edge_box(self, roi):
        boxes = self.sequence
        l = len(boxes)
        start = 0
        for i in range(0, l):
            bx = boxes[i]
            if edge_box(bx, roi):
                start = i
            else:
                break
        end = l-1
        for i in range(l-1, -1, -1):
            bx = boxes[i]
            if edge_box(bx, roi):
                end = i
            else:
                break
        end += 1

        if start >= end:
            self.sequence = boxes[0:1]
        else:
            self.sequence = boxes[start:end]

    # connect two tracks
    def link(self, tk):
        for bx in tk.sequence:
            self.sequence.append(bx)

    # if the box(detection) appears halfway
    def halfway_appear(self, roi):
        side_th = 100
        h, w, _ = roi.shape
        apr_box = self.sequence[0]
        bx = apr_box.box
        # print 'x,y,w,h'
        # print bx
        c_x, c_y = apr_box.center
        c_x = int(c_x)
        c_y = int(c_y)
        if bx[0]>side_th and bx[1]>side_th and bx[0]+bx[2]<w-side_th and bx[1]+bx[3]<h-side_th and roi[c_y][c_x][0]>128:
            return True
        else:
            return False

    def halfway_lost(self, roi):
        side_th = 100
        h, w, _ = roi.shape
        apr_box = self.sequence[-1]
        bx = apr_box.box
        # print 'x,y,w,h'
        # print bx
        c_x, c_y = apr_box.center
        c_x = int(c_x)
        c_y = int(c_y)
        if bx[0] > side_th and bx[1] > side_th and bx[0] + bx[2] < w - side_th and bx[1] + bx[3] < h - side_th and roi[c_y][c_x][0] > 128:
            return True
        else:
            return False

    def get_last(self):
        return self.sequence[-1]

    def get_first(self):
        return self.sequence[0]

    def get_last_feature(self):
        return self.sequence[-1].feature

    def get_first_feature(self):
        return self.sequence[0].feature

    def get_length(self):
        return len(self.sequence)

    # get where the trajectory ends 
    def get_last_gps(self):
        return self.sequence[-1].gps_coor

    def get_first_gps(self):
        return self.sequence[0].gps_coor

    # get moving distance to filter out stop vehicles 
    def get_moving_distance(self):
        start_p = self.sequence[0].center
        end_p = self.sequence[-1].center
        gps_dis_vec = ((end_p[0] - start_p[0]), (end_p[1] - start_p[1]))
        gps_dis = gps_dis_vec[0]**2 + gps_dis_vec[1]**2
        return gps_dis

  
    def get_moving_vector(self):
        start_p = self.sequence[0].gps_coor
        end_p = self.sequence[-1].gps_coor
        gps_dis_vec = ((end_p[0] - start_p[0]), (end_p[1] - start_p[1]))
        return gps_dis_vec

    def show(self):
        print("For track-" + str(self.id) + ' : ', "length-" + str(len(self.sequence)), ", matchState-", str(self.match_state))
        print(self.get_moving_distance())

def analysis_to_track_dict(file_path):
    track_dict = {}
    lines = open(file_path, 'r').readlines()
    for line in lines:
        words = line.strip('\n').split(',')
        index = int(words[0])
        id = int(words[1])
        box = [int(float(words[2])), int(float(words[3])), int(float(words[4])), int(float(words[5]))]
        score = float(words[6])
        gps = words[7].split('-')
        # print gps
        gps_x = float(gps[0])
        #gps_y = float(gps[2])
        gps_y = float(gps[1])
        ft = np.zeros(len(words) - 10)
        for i in range(10, len(words)):
            ft[i - 10] = float(words[i])
        cur_box = Box(index, id, box, score, (gps_x, gps_y), ft)
        if id not in track_dict:
            track_dict[id] = Track(id, [])
        track_dict[id].append(cur_box)
    return track_dict

class road(object):

    def __init__(self,id,start,end):
        self.id = id
        self.start = start
        self.end = end

def frame_num(p):
    cap = cv2.VideoCapture(p)
    return int(cap.get(7))
def subt(x,y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])
def main():
    
      
    scene_dirs = []
    scene_fds = os.listdir(input_dir)
    video_number = 1
    for scene_fd in scene_fds:
        scene_dirs.append(os.path.join(input_dir, scene_fd))

    for scene_dir in scene_dirs:
        camera_dirs = []
        fds = os.listdir(scene_dir)
     
        for fd in fds:
            #if fd.startswith('c0'):
                camera_dirs.append(os.path.join(scene_dir, fd))
        counter = 0
        for camera_dir in camera_dirs:
            print(camera_dir)
            if not camera_dir.split("/")[-1].startswith("c0") or not os.path.isfile(camera_dir + "/road.txt"):
                counter+=1
                continue
            if not os.path.isfile(camera_dir + "/optimized_track.txt"):
                continue
            print(camera_dir)
            track_path = os.path.join(camera_dir, 'optimized_track.txt')
            roi_path = os.path.join(camera_dir, 'roi.jpg')
            out_path = os.path.join(camera_dir, 'final_ouput.txt')
            track_dict = analysis_to_track_dict(track_path)
            road_path  = os.path.join(camera_dir,'road.txt')
            video_path = camera_dir + '/vdo.avi'
            outfile   =  open(out_path,"w")
            FRAME = []
            
            
            cap = cv2.VideoCapture(video_path)
            """
            for i in range(0,frame_num( video_path)):
                ret, I = cap.read()
                FRAME.append(I)
            """
            # delete track has only 1 box：
            delete_list = []
            #print(len(track_dict))
            f = open(out_path,'w')
            road_list = []
            road_num = 0
            road_id  = 0
            with open(road_path,'r') as f2:
                first_line = True
                for line in f2:
                    if first_line:
                       road_num = int(line)
                       first_line = False
                       continue
                    
                    line = line.strip().split(" ")
                    
                    if len(line) < 5:
                        road_list.append(road(road_id,(int(line[0]),int(line[1])),(int(line[2]),int(line[3]))))
                    else:
                        road_id = int(line[0])
                        road_list.append(road(int(line[0]),(int(line[1]),int(line[2])),(int(line[3]),int(line[4]))))
           
            score = {}
            traffic = []
            
            
            for i in range(2*road_num):
                traffic.append(0)
            for id in track_dict:

                track = track_dict[id]
                #print(len(FRAME[track.get_first().frame_index]))
                #print(track.get_first().box[1],track.get_first().box[1]+track.get_first().box[3])
                #print(track.get_first().box[0],track.get_first().box[0]+track.get_first().box[2])
                #print(track.get_first().box[0]+track.get_first().box[2],track.get_first().box[1]+track.get_first().box[3])
                #FRAME[track.get_first().frame_index][track.get_first().box[1]:track.get_first().box[1]+track.get_first().box[3],
                #            track.get_first().box[0]:track.get_first().box[0]+track.get_first().box[2]]
            
                #boxes,preds = testr.object_detection_write_api(Image.fromarray(cv2.cvtColor(FRAME[track.get_first().frame_index],cv2.COLOR_BGR2RGB)), threshold=0.4)
                
                predict = ""
                #print("object:",track.get_first().box[0],track.get_first().box[1],track.get_first().box[2],track.get_first().box[3])
                md = 999999
                """
                for i in range(len(boxes)):
                    b = boxes[i]
                    #print("detect:",b[0][0],b[0][1],b[1][0] - b[0][0],b[1][1] - b[0][1])
                    dis = abs(b[0][0] - track.get_first().box[0]) + abs(b[0][1] - track.get_first().box[1]) + \
                        abs((b[1][0] - b[0][0]) - track.get_first().box[2]) + abs((b[1][1] - b[0][1]) - track.get_first().box[3])
                    if md > dis:
                        predict = preds[i]
                        md = dis
                """
                if predict != "car" and predict != "truck":
                    predict = "car"
                
                if track.id not in score:
                    #score[id] = []
                    dis = 99999999999
                assign = -1
                dis = 99999999999
                for r_n in range(len(road_list)):
                    r = road_list[r_n]
                    #print(track.get_first().center,r.start,r.end)
                   
                    
                    new_dis = subt(r.start, track.get_first().center) #\
                            #+ subt(r.end, track.get_last().center)
                    if dis > new_dis:
                        dis = new_dis
                        assign = road_list[r_n].id
                if dis <= 400:  
                    if predict == "car":
                        outfile.write(str(track.get_last().frame_index) + " " + str(assign) + " "+ str(1) + '\n') 
                    else:
                        outfile.write(str(track.get_last().frame_index) + " " + str(assign) + " "+ str(2) + '\n') 
            
            counter += 1       
            outfile.close()
            video_number += 1
         
            f.close()
if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end - start)