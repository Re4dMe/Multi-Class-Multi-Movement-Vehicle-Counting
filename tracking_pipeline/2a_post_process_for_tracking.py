# -*- coding: utf-8 -*-
 
import numpy as np
import os
import cv2
import math
import time
MATCHED = True
NO_MATCHED = False
SIMILAR_TH = 700
SPEED_TH = 100
TRACK_TH = 2
MOVE_TH = 20000
eps = 0.000001

input_dir = "../aic19-track1-mtmc/test2020_3"

PAD_SIZE = 10


class Box(object):
    """
    match_state:一个box是否匹配到一个track中，若没有，应该生成新的track
    """
    def __init__(self, frame_index, id, box, score, v_type, gps_coor, feature):
        self.frame_index = frame_index
        self.id = id
        self.box = box
        self.score = score
        self.pred = v_type
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

        # 当轨迹长度不够时，不判定车脸朝向，
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
 
    def link(self, tk):
        for bx in tk.sequence:
            self.sequence.append(bx)
 
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

   
    def get_last_gps(self):
        return self.sequence[-1].gps_coor

    def get_first_gps(self):
        return self.sequence[0].gps_coor
 
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


def edge_box(bbox, roi):
    side_th = 30
    h, w, _ = roi.shape
    bx = bbox.box
    if bx[0] > side_th and bx[1] > side_th and bx[0] + bx[2] < w - side_th and bx[1] + bx[3] < h - side_th:
        return False
    else:
        return True


def calu_moving_distance(box1, box2):
    start_p = box1.center
    end_p = box2.center
    dis_vec = ((end_p[0] - start_p[0]), (end_p[1] - start_p[1]))
    dis = dis_vec[0]**2 + dis_vec[1]**2
    return dis


def calu_moving_vec(box1, box2):
    start_p = box1.center
    end_p = box2.center
    dis_vec = ((end_p[0] - start_p[0]), (end_p[1] - start_p[1]))
    return dis_vec


def preprocess_roi(roi):
    h, w, _ = roi.shape
    width_erode = int(w * 0.1)
    height_erode = int(h * 0.1)
    left = roi[:, 0:width_erode, :]
    right = roi[:, w-width_erode:w, :]
    top = roi[0:height_erode, :, :]
    bottom = roi[h-height_erode:h, :, :]

    left = left*0
    right = right*0
    top = top*0
    bottom = bottom*0

    return roi


def calu_feature_distance(ft0, ft1):
    feature_dis_vec = ft1 - ft0
    feature_dis = np.dot(feature_dis_vec.T, feature_dis_vec)
    return feature_dis


def calu_track_distance(pre_tk, back_tk):
    #return abs(pre_tk.sequence[-1].center[0] - back_tk.sequence[-1].center[0])+
    #    abs(pre_tk.sequence[-1].center[1] - back_tk.sequence[-1].center[1])
    lp = min(5, len(pre_tk.sequence))*-1
    lb = min(5, len(back_tk.sequence))

    pre_seq = pre_tk.sequence[lp:]
    back_seq = back_tk.sequence[:lb]

    mindis = 999999
    for bx0 in pre_seq:
        for bx1 in back_seq:
            ft0 = bx0.feature
            ft1 = bx1.feature
            curdis = calu_feature_distance(ft0, ft1)
            mindis = min(curdis, mindis)
    gdistance = calu_moving_distance(pre_tk.sequence[-1],back_tk.sequence[0])
    Max = max(pre_tk.sequence[-1].get_area(),back_tk.sequence[0].get_area())
    Min = min(pre_tk.sequence[-1].get_area(),back_tk.sequence[0].get_area())
    adis = float(Min/Max)
    fdis = abs(pre_tk.sequence[-1].frame_index - back_tk.sequence[0].frame_index)
    return mindis, gdistance,adis,fdis



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
        pred  = str(words[10])
        words = words[:10] + words[11:]
        ft = np.zeros(len(words) - 10)
        for i in range(10, len(words)):
            ft[i - 10] = float(words[i])
        cur_box = Box(index, id, box, score, pred, (gps_x, gps_y), ft)
        if id not in track_dict:
            track_dict[id] = Track(id, [])
        track_dict[id].append(cur_box)
    return track_dict


def analysis_time_stamp(time_stamp_file):
    time_stamp_dict = {}
    lines = open(time_stamp_file).readlines()
    for line in lines:
        words = line.strip('\n').split()
        camera_id = words[0]
        time_stamp = float(words[1])
        time_stamp_dict[camera_id] = time_stamp
    return time_stamp_dict


def tk_time_fit(tk0, tk1):
    # TTTHHH = 10
    # near_th = 400
    # far_th = 650
    near_th = 70
    far_th = 100
    # near_th = 0
    # far_th = 0
    # near_th = 20
    # mid_th = 75
    # far_th = 150
    time = tk1.get_first().frame_index - tk0.get_last().frame_index
    if time < 5:
        return far_th
    else:
        return near_th


def analysis_transfrom_mat(cali_path):
    first_line = open(cali_path).readlines()[0].strip('\r\n')
    cols = first_line.lstrip('Homography matrix: ').split(';')
    transfrom_mat = np.ones((3, 3))
    for i in range(3):
        values_string = cols[i].split()
        for j in range(3):
            value = float(values_string[j])
            transfrom_mat[i][j] = value
    inv_transfrom_mat = np.linalg.inv(transfrom_mat)
    return inv_transfrom_mat


def load_fps_dict(fps_file):
    fps_dict = {}
    lines = open(fps_file).readlines()
    for line in lines:
        words = line.split()
        cam = words[0]
       
        fps = float(words[1])
        fps_dict[cam] = fps
    return fps_dict


def main():
    scene_dirs = []
    scene_fds = os.listdir(input_dir)
    for scene_fd in scene_fds:
        scene_dirs.append(os.path.join(input_dir, scene_fd))

    #time_stamp_file = '../aic19-track1-mtmc/cam_timestamp/eval.txt'
    #time_stamp_file = '../aic19-track1-mtmc/cam_timestamp/S02.txt'
    time_stamp_file = "test_fps.txt"
    time_stamp_dict = analysis_time_stamp(time_stamp_file)
    # fps_file = "./aic19-track1-mtmc/cam_timestamp/eval_fps.txt"
    fps_file = "test_fps.txt"
    fps_dict = load_fps_dict(fps_file)
    
    
    
    for scene_dir in scene_dirs:
        # if scene_dir != './aic19-track1-mtmc/test/S02':
        #     continue

        camera_dirs = []
        fds = os.listdir(scene_dir)
        for fd in fds:
            if fd.startswith('c0'):
                camera_dirs.append(os.path.join(scene_dir, fd))

        for camera_dir in camera_dirs:   
            # print camera_dir
            # if camera_dir != './aic19-track1-mtmc/test/S02/c007':
            #     continue
            if not os.path.isfile(camera_dir+"/det_reid_track.txt"): #or os.path.isfile(camera_dir+"/optimized_track.txt"):
                continue
            
            if not camera_dir.split('/')[-1] == "c002":
                continue

            print(camera_dir)

            cali_path = camera_dir + '/calibration.txt'
            trans_mat = analysis_transfrom_mat(cali_path)
            cam = camera_dir.split('/')[-1]
            vdo_path = os.path.join(camera_dir, 'vdo.avi')
            cap = cv2.VideoCapture(vdo_path)
            # fps
            time_stamp = time_stamp_dict[cam]
            fps = fps_dict[cam]

            track_path = os.path.join(camera_dir, 'det_reid_track.txt')
            roi_path = os.path.join(camera_dir, 'roi.jpg')
            out_path = os.path.join(camera_dir, 'optimized_track.txt')

            track_dict = analysis_to_track_dict(track_path)
     
            
            delete_list = []
            for id in track_dict:
                track = track_dict[id]
                # if track.get_length() < TRACK_TH or track.get_first().frame_index > track.get_last().frame_index:
                if track.get_length() < TRACK_TH:
                    delete_list.append(id)
            for id in delete_list:
                track_dict.pop(id)
            
            print("dict1: ", len(track_dict))
         
            roi_src = cv2.imread(roi_path)
            roi = preprocess_roi(roi_src)
        
            halfway_list = []
            delete_list = []   
            for id in track_dict:
                track = track_dict[id]
                halfway_list.append(track)
            """    
            for id in track_dict:
                track = track_dict[id]
                if track.halfway_appear(roi) or track.halfway_lost(roi):
                    # print track.id
                    halfway_list.append(track)
                else:
                    continue
            """
            halfway_list = sorted(halfway_list, key=lambda tk: tk.sequence[0].frame_index)

            print("length of lost: ", len(halfway_list))
            for lost_tk in halfway_list:
                if lost_tk.id in delete_list:
                    continue
                min_match = 999999
                match_list = 1
                while match_list != -1:
                    min_match = 999999
                    match_list = -1
                    for apr_tk in halfway_list:
                        if apr_tk.id in delete_list:
                            continue
                        if lost_tk.get_last().frame_index < apr_tk.get_first().frame_index:# and apr_tk.get_first().frame_index - lost_tk.get_last().frame_index < 25:
                            # apr_ft = apr_tk.get_first_feature()
                            # lost_ft = lost_tk.get_last_feature()
                            # dis = calu_feature_distance(apr_ft, lost_ft)
                            
                            dis, gdis, adis, fdis = calu_track_distance(lost_tk, apr_tk)
                            #print(dis)
                            th = tk_time_fit(lost_tk, apr_tk)
                            """
                            if dis < 400 :
                                min_match = dis
                                match_list = apr_tk
                            """    
                            # print lost_tk.id, apr_tk.id, dis, th
                            #ori_compare = True
                            #if lost_tk.get_orientation() != 'fbc' and apr_tk.get_orientation() != 'fbc':
                            #    if lost_tk.get_orientation() != apr_tk.get_orientation():
                            #        ori_compare = False
                            #if dis < th or (gdis < 80 and adis >= 1.0/3.0 and ori_compare): #and fdis <= 40):
                            if  dis < th or (gdis < 400):
                                
                                # print dis, lost_tk.id, apr_tk.id
                                lost_tk.link(apr_tk)
                                if apr_tk.id not in delete_list:
                                    delete_list.append(apr_tk.id)
                    """        
                    if match_list != -1:
                        lost_tk.link(match_list)
                        if match_list.id not in delete_list:
                            delete_list.append(match_list.id)
                    """        
            for id in delete_list:
                track_dict.pop(id)

            print("length of success linked: ", len(delete_list))
            print("length of all: ", len(track_dict))
            
           
            """
            delete_list = []
            for id in track_dict:
                track = track_dict[id]
                moving_dis = track.get_moving_distance()
                moving_steps = track.get_length()
                speed = moving_dis/moving_steps
                # print 'speed: ', speed
                # track.show()
                if speed < SPEED_TH:
                    # print id, speed
                    delete_list.append(id)
            for i in delete_list:
                track = track_dict[i]
                track_dict.pop(i)
            """
            """
            
            for id in track_dict:
                track = track_dict[id]
                track.remove_edge_box(roi)
            """
           
            delete_list = []
            print(len(track_dict))
            for id in track_dict:
                track = track_dict[id]
                if track.get_length() < 2:
                    delete_list.append(id)
            for id in delete_list:
                track_dict.pop(id)

            f = open(out_path, 'w')
            for k in track_dict:
               
                tk = track_dict[k]
                # print tk.id, tk.get_orientation(), '  dis: ', tk.is_straight(), '  area: ', tk.area_stable()

                for bx in tk.sequence:

               
                    coor = bx.floor_center
                    image_coor = [coor[0], coor[1], 1]
                    new_GPS_coor = np.dot(trans_mat, image_coor)
                    new_GPS_coor = new_GPS_coor / new_GPS_coor[2]

                    # ww = str(bx.frame_index) + ',' + str(tk.id) + ',' + str(bx.box[0]) + ',' + str(bx.box[1]) + \
                    #      ',' + str(bx.box[2]) + ',' + str(bx.box[3]) + ',-1,-1\n'

                    ww = str(bx.frame_index) + ',' + str(tk.id) + ',' + str(bx.box[0]) + ',' + str(bx.box[1]) + \
                         ',' + str(bx.box[2]) + ',' + str(bx.box[3]) + ',' + str(bx.score) + ',' + \
                         str(new_GPS_coor[0]) + '-' + str(new_GPS_coor[1])+ ',' + bx.pred + ',' + tk.get_orientation() + ',' + str(time_stamp+bx.frame_index/fps)
                    for i in bx.feature:
                        ww += ',' + str(i)
                    ww += '\n'

                    f.write(ww)
            f.close()


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end - start)