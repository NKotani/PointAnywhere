import argparse
import Test
import PointingVector
import Equi2Pers as E2P
import subprocess
import RegionOfInterst as ROI
import os
import glob
import GreatCircle as GC
import datetime
import Pers2Equi as P2E
import numpy as np
from utils.general import increment_path
from pathlib import Path
import cv2
import Correct
from pytorch_openpose.src import util
from pytorch_openpose.src.body import Body
# from ml import Data

body_estimation = Body('pytorch_openpose/model/body_pose_model.pth') # OpenPose (pytorch)

parser = argparse.ArgumentParser(description='Estimate a pointing object')
parser.add_argument('-input', default='inputOmni', type=str, help='path of equirectangular images')
parser.add_argument('-saveimg', default=1, type=int, help='save images when this is 1')
parser.add_argument('-skelton', default='', type=str, help='Path of the folder containing the results of the pre-estimated user, i.e. skelton/')

args = parser.parse_args()

input =  args.input
saveimg = args.saveimg
skelton_path = args.skelton

dt_now = datetime.datetime.now()
stdout = dt_now.strftime('%Y%m%dg%H%M%S')

save_dir = increment_path(Path('./Experiment') / 'result', exist_ok=False)  # increment run
print(save_dir)
outputOmni_f = save_dir / 'outputOmni'
jointPers_f = save_dir / 'jointPers'
jointPersFinger_f = save_dir / 'jointPers/finger'
greatOmni_f = save_dir / 'greatOmni'
outDirect_f = save_dir / 'outDirect'
obj_f = save_dir / 'obj'
testResult_f = save_dir / 'testResult'
testResult_distance_f = save_dir / 'testResult/distance'
testResult_vec_f = save_dir / 'testResult/vec'
(outputOmni_f).mkdir(parents=True, exist_ok=True)  # make dir
(outDirect_f).mkdir(parents=True, exist_ok=True)  # make dir
(obj_f).mkdir(parents=True, exist_ok=True)  # make dir
(testResult_f).mkdir(parents=True, exist_ok=True)  # make dir

if saveimg:
    (jointPers_f).mkdir(parents=True, exist_ok=True)
    (jointPersFinger_f).mkdir(parents=True, exist_ok=True)
    (greatOmni_f).mkdir(parents=True, exist_ok=True)
    (testResult_distance_f).mkdir(parents=True, exist_ok=True)
    (testResult_vec_f).mkdir(parents=True, exist_ok=True)

if os.path.isdir(input):
    files = glob.glob(input + "/*")
else:
    files = [input]

for file in files:
    sphere = os.path.splitext(os.path.basename(file))[0] # name of equirectangular image (R0010068)
    Test.write_stdout(sphere, stdout, testResult_f)

    command = []
    if not skelton_path:
        if not skelton_path and saveimg == 1:
            command = ['python','yolov5/detect.py','--weights','yolov5/yolov5s.pt','--source',file,'--save-txt','--exist-ok','--save-conf','--project',obj_f,'--name',sphere,'--classes','0','--imgsz','640','1280']
        elif not skelton_path and saveimg == 0:
            command = ['python','yolov5/detect.py','--weights','yolov5/yolov5s.pt','--source',file,'--save-txt','--exist-ok','--save-conf','--project',obj_f,'--name',sphere,'--classes','0','--imgsz','640','1280','--nosave']
        proc = subprocess.run(command, capture_output=True)
        # print('return code: {}'.format(proc.returncode))
        # print('captured stdout: {}'.format(proc.stdout.decode()))
        # print('captured stderr: {}'.format(proc.stderr.decode()))
    people = {}
    name_txt = ''
    if not skelton_path:
        name_txt = os.path.join(obj_f, sphere+ '/labels/' + sphere + '.txt') # 'obj/'+sphere+'/labels/'
    else:
        name_txt = os.path.join(skelton_path, sphere+ '/labels/' + sphere + '.txt')
    try:
        with open(name_txt, 'r') as f:
            persons = f.read().splitlines() # ['person 0.571615 0.548177 0.0279018 0.18936 0.789395']
            for person in persons:
                person = person.split() # ['person', '0.571615', '0.548177', '0.0279018', '0.18936', '0.789395']
                people[float(person[5])] = [float(person[1]),float(person[2]),float(person[3]),float(person[4])]
            people = sorted(people.items()) # in order of decreasing confidence
    except FileNotFoundError as e:
        Test.write_stdout(f'{-1} {sphere}', 'score', testResult_f) # fail to detect human from the equirectangular image
        continue
    print(people[-1]) # the most hight confidence human:(0.885246, [0.0642671, 0.530506, 0.0191592, 0.0959821])
    high_person = people[-1]
    
    pers_w = 432 # width of perspective image
    human_x = high_person[1][0] - 0.5 # x-coordinate of human center(-0.5~0.5)
    theta = int(human_x * 360)
    human_y = 0.5 - high_person[1][1] # y-coordinate of human center(-0.5~0.5)
    phi = int(human_y * 180)
    
    human_w = high_person[1][2]*2
    fov = int(human_w*360) # Variable below 120
    human_h = high_person[1][3]*1.2 # 1.2 times the height of a person
    hfov = int(human_h*180) # fov of vertical (-150)
    if fov > 90:
        fov = 120 # when the person is large, the YOLO is not accurate, so align it wider.
        if high_person[0] < 0.65: # when confidence is low
            phi = 50 # hands often pass over the camera
            hfov = 150 # If joint detection fails, cut out the part of the body that is 180 degrees opposite to the human
    elif hfov > 150:
            hfov = 150
    pers_h = int(pers_w * hfov / fov) # hight of perspective image

    candidate, subset = np.empty((1,1)), np.empty(1)
    Test.write_stdout(f'horizontal fov={fov}, vartical fov={hfov}', stdout, testResult_f)
    if not skelton_path:
        E2P.get_perspective(file, fov, theta, phi, pers_h, pers_w, outputOmni_f)
        pers_path = os.path.join(outputOmni_f, sphere + '_' + str(theta) + '_' + str(phi) + '.jpg') # './outputOmni/'
        persImg = cv2.imread(pers_path)
        candidate, subset = body_estimation(persImg)
    else:
        human_filename = os.path.join(skelton_path, sphere + '/human')
        npz_kw = np.load(human_filename+'.npz')
        candidate = npz_kw['candidate']
        subset = npz_kw['subset']
    if saveimg:
        E2P.get_perspective(file, fov, theta, phi, pers_h, pers_w, outputOmni_f)
        pers_path = os.path.join(outputOmni_f, sphere + '_' + str(theta) + '_' + str(phi) + '.jpg') # './outputOmni/'
        persImg = cv2.imread(pers_path)

    shoulder, wrist, vec = [], [], []
    joints_num, head, joints, msg = PointingVector.rightLeftArm_head_pytorch(candidate, subset)
    if saveimg:
        persImg = util.draw_bodypose(persImg, candidate, subset)
        output = os.path.join(jointPers_f,  sphere + '_' + str(theta) + '_' + str(phi) + '.png')
        cv2.imwrite(output, persImg) # result of OpenPose
        Test.draw_finger(joints, joints_num, output, jointPersFinger_f) # result of rightLeftArm_head_pytorch
    print(joints)
    print(msg)
    Test.write_stdout(msg, stdout, testResult_f)
    if joints_num == [-1]:
        shoulder = [theta + 170, 50] 
        wrist = [theta + 170, 20]
        vec = [0, -30] # go downwards
    else:
        vec = PointingVector.get_vec_mid(joints, joints_num, head, theta, phi, pers_w, pers_h, fov)
        wrist = np.rad2deg(P2E.get_angle([joints[joints_num[-1]][0], joints[joints_num[-1]][1]], theta, phi, pers_w, pers_h, fov)) # the far end of the joint the human is pointing at
        shoulder = [wrist[0]-vec[0], wrist[1]-vec[1]]
    Test.write_stdout(f'The arm of pointing: {joints_num}, direction: {vec}, start(shoulder): ({wrist})', stdout, testResult_f)
    print(f'shoulder:{shoulder}, wrist:{wrist}')
    z, y = [], []

    if  saveimg:
        z, y = GC.get(shoulder, wrist, file, greatOmni_f, True) # Number of elements 1080 → 30 degrees for 90 elements
    else:
        z, y = GC.get(shoulder, wrist, file, save=False) # z will always return

    rot_z = []
    rot_y = []
    start_num = 0 # what element of z/y is the angle of the wrist
    last = 0 # number of the last element
    first, final = 0, 0
    opposite = 0 # 180 degrees opposite of the starting point
    if vec[0] == 0: # Period 60 degrees
        start_num = int((wrist[0]+175) * 1080 / 360) % 1080
        rot_z.append(int(z[start_num]))
        rot_y.append(int(y[start_num]))
        E2P.get_perspective(file, 60, rot_z[0], rot_y[0], 640, 640, os.path.join(outDirect_f,sphere+'/'))
        for i in [1,2,3,4]:
            index = (start_num + i * 30) % 1080 # advance by 10 degrees
            rot_z.append(int(z[index]))
            rot_y.append(int(y[index]))
            E2P.get_perspective(file, 60, rot_z[i], rot_y[i], 640, 640, os.path.join(outDirect_f,sphere+'/'))
        first = (start_num - 30) % 1080 #  10 degrees for 30 elements
        final = (start_num + 30*5) % 1080
    else:  
        if vec[0] > 0: # right
            const = 30 # fingertip should be on the right edge of the image.
            adjust = 27 # shift it by 27 degrees at first
        else:
            const = -30 # fingertip is on the left side of the image
            adjust = -27
        for i in range(9): # keep it wide because it may not advance 30 degree
            if i == 0:
                if wrist[0] < 180:
                    start_num = int((wrist[0]+180+adjust) * 1080 / 360) - 1
                else:
                    start_num = int((wrist[0]-180+adjust) * 1080 / 360) - 1
                start_num = start_num % 1080
                if abs(y[start_num]-y[(start_num-const*3)%1080]) > 60: # proceed in a large, longitudinal direction
                    adjust = const * 12 / 30 # shift by 12 degrees
                    start_num = int(start_num - const*1.5) % 1080 # 12=27-15
                rot_z.append(int(z[start_num]))
                rot_y.append(int(y[start_num]))
                last = start_num
                Test.write_stdout(f'z-direction:{rot_z[i]}, y-direction:{rot_y[i]}', stdout, testResult_f)
                E2P.get_perspective(file, 60, rot_z[i], rot_y[i], 640, 640, os.path.join(outDirect_f,sphere+'/'))
            else:
                last = (last + const * 3) % 1080
                opposite += 30
                if abs(y[last]) > 60 and abs(rot_y[i-1]) > 60: # ceiling or floor twice in a row
                    last = (last + const*2) % 1080 # 50 degrees forward
                    opposite += 20
                while abs(y[last]-rot_y[i-1]) >= 60: # loop until no significant vertical progress is made.
                    last = int(last - const*0.5) % 1080 # shift by 5 degrees
                    opposite -= 5
                rot_z.append(int(z[last]))
                rot_y.append(int(y[last]))
                Test.write_stdout(f'z-direction:{rot_z[i]}, y-direction:{rot_y[i]}', stdout, testResult_f)
                E2P.get_perspective(file, 60, rot_z[i], rot_y[i], 640, 640, os.path.join(outDirect_f,sphere+'/'))
                if  opposite >= 180: # from the starting point you are pointing to the opposite side 180° to the left and right.
                    print('Left or right 180° too far.')
                    Test.write_stdout('Left or right 180° too far.', stdout, testResult_f)
                    break
     
        first = (start_num-const*3) % 1080
        final = (last+const*3) %1080

    detected = [] # [Number of cropped images len(rot_z)][Number of objects detected]
    for i in range(len(rot_z)):
        name = os.path.join(outDirect_f, sphere + '/' + sphere + '_' + str(int(rot_z[i])) + '_' + str(int(rot_y[i])) + '.jpg') # 'outDirect/'
        command = []
        if  saveimg:
            command = ['python','yolov5/detect.py','--weights','yolov5/yolov5s.pt','--source',name,'--save-txt','--exist-ok','--save-conf','--project',obj_f,'--name', sphere, '--conf-thres', '0.28'] # Only 0.29 or higher
        else:
            command = ['python','yolov5/detect.py','--weights','yolov5/yolov5s.pt','--source',name,'--save-txt','--exist-ok','--save-conf','--project',obj_f,'--name', sphere, '--conf-thres', '0.28', '--nosave']
        proc = subprocess.run(command, capture_output=True)
    
        name_txt = os.path.join(obj_f,sphere+'/labels/' + sphere + '_' + str(int(rot_z[i])) + '_' + str(int(rot_y[i])) + '.txt') # 'obj/'+sphere+'/labels/'
        try:
            with open(name_txt, 'r') as f:
                data = f.read().splitlines() # label, xywh, conf(except \n)
        except FileNotFoundError as e:
            data = [] # nothing detected
        detected.append(data)
    
    # [Number of cropped images len(rot_z)][Number of objects detected][label,[x,y,w,h],conf]
    obj = ROI.match_obj(detected, rot_z, rot_y, vec)
    result = ROI.distance_circle(obj, rot_z, rot_y, [z[first],y[first]], [z[final],y[final]])
    # print(result) # [(8.069252459121573,['chair', [2862, 1597, 545, 610], 0.57567]), ...]
    ground_truth = os.path.join(os.path.split(os.path.dirname(file))[0], 'ROI', sphere+'.txt')
    test_file = 'dummy'
    if  saveimg:
        # Test.crop_obj(obj, file)
        test_file = os.path.join(testResult_vec_f, sphere + '.jpg') # 'test_result/vec/' the image created by distance_circle
        Test.distance_circle(rot_z, rot_y, [z[first],y[first]], [z[final],y[final]], file, testResult_vec_f)
        Test.roi2img(obj, test_file, testResult_distance_f) # the pointing vector itself is represented by a distance_circle
    Correct.run(result, ground_truth, stdout, testResult_f, sphere, saveimg, test_file, theta)
    # make dataset
    # score, _ = Correct.in_order_of_distance(result, ground_truth)
    # result = Data.make(result, theta, sphere, score, stdout)

dt_now = datetime.datetime.now() # run end time
finish = dt_now.strftime('%Y%m%d-%H%M%S')
Test.write_stdout(finish, stdout, testResult_f)
Correct.cal_top1_5_10ALL(testResult_f, len(files))
