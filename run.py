import Test
import PointingVector
import Equi2Pers as E2P
import subprocess
import RegionOfInterst as ROI
import os
import glob
import GreatCircle as GC
import datetime
import Per2Eq as P2E
import numpy as np
from utils.general import increment_path
from pathlib import Path
import sys
import cv2
import Correct
from pytorch_openpose.src import util
from pytorch_openpose.src.body import Body
from ml import Data

body_estimation = Body('pytorch_openpose/model/body_pose_model.pth')

dt_now = datetime.datetime.now()
stdout = dt_now.strftime('%Y%m%dg%H%M%S')

input = sys.argv[1] # './inputOmni/R0010095.JPG', '../../dataset/image' image,2,3で全データセット
testMODE = int(sys.argv[2]) # 1のとき余計な画像保存しない

save_dir = increment_path(Path('./Experiment') / 'result', exist_ok=False)  # increment run
print(save_dir)
outputOmni_f = save_dir / 'outputOmni' # 必須
jointPers_f = save_dir / 'jointPers'
jointPersFinger_f = save_dir / 'jointPers/finger'
greatOmni_f = save_dir / 'greatOmni'
outDirect_f = save_dir / 'outDirect' # 必須
obj_f = save_dir / 'obj' # 必須, labelを作るのに必要だけどimgはいらない
testResult_f = save_dir / 'testResult' # 必須
testResult_distance_f = save_dir / 'testResult/distance'
testResult_vec_f = save_dir / 'testResult/vec'
(outputOmni_f).mkdir(parents=True, exist_ok=True)  # make dir
(outDirect_f).mkdir(parents=True, exist_ok=True)  # make dir
(obj_f).mkdir(parents=True, exist_ok=True)  # make dir
(testResult_f).mkdir(parents=True, exist_ok=True)  # make dir

if testMODE == 0: # 定性評価のとき
    (jointPers_f).mkdir(parents=True, exist_ok=True)  # make dir
    (jointPersFinger_f).mkdir(parents=True, exist_ok=True)
    (greatOmni_f).mkdir(parents=True, exist_ok=True)  # make dir
    (testResult_distance_f).mkdir(parents=True, exist_ok=True)  # make dir
    (testResult_vec_f).mkdir(parents=True, exist_ok=True)  # make dir

# OpenPose pytorch版
if os.path.isdir(input):
    files = glob.glob(input + "/*")
else:
    files = [input]

for file in files:
    sphere = os.path.splitext(os.path.basename(file))[0] # 全天球画像の名前(R0010068)
    Test.write_stdout(sphere, stdout, testResult_f)

    command = []
    if  testMODE == 0: # 定性評価のとき
        command = ['python','yolov5/detect.py','--weights','yolov5/yolov5s.pt','--source',file,'--save-txt','--exist-ok','--save-conf','--project',obj_f,'--name',sphere,'--classes','0','--imgsz','640','1280']
    else: # 定量評価
        command = ['python','yolov5/detect.py','--weights','yolov5/yolov5s.pt','--source',file,'--save-txt','--exist-ok','--save-conf','--project',obj_f,'--name',sphere,'--classes','0','--imgsz','640','1280','--nosave']
    proc = subprocess.run(command, capture_output=True)
    # print('return code: {}'.format(proc.returncode))
    # print('captured stdout: {}'.format(proc.stdout.decode()))
    # print('captured stderr: {}'.format(proc.stderr.decode()))
    people = {}
    name_txt = os.path.join(obj_f, sphere+ '/labels/' + sphere + '.txt') # 'obj/'+sphere+'/labels/'
    try:
        with open(name_txt, 'r') as f:
            persons = f.read().splitlines() # ['person 0.571615 0.548177 0.0279018 0.18936 0.789395']
            for person in persons:
                person = person.split() # ['person', '0.571615', '0.548177', '0.0279018', '0.18936', '0.789395']
                people[float(person[5])] = [float(person[1]),float(person[2]),float(person[3]),float(person[4])]
            people = sorted(people.items()) # confidenceが小さい順に並ぶ
    except FileNotFoundError as e:
        Test.write_stdout(f'{-1} {sphere}', 'score', testResult_f) # 全天球画像から人検出できなかった
        continue
    print(people[-1]) # 一番confidenceが高い人間:(0.885246, [0.0642671, 0.530506, 0.0191592, 0.0959821])
    high_person = people[-1]
    
    omni_w = 5376 # 全天球画像の横幅
    pers_w = 432 # perspective画像の横幅
    human_x = high_person[1][0] - 0.5 # 人間の中心のx座標(-0.5~0.5)
    theta = int(human_x * 360)
    human_y = 0.5 - high_person[1][1] # 人間の中心のy座標(-0.5~0.5)
    phi = int(human_y * 180)
    
    human_w = high_person[1][2]*2
    fov = int(human_w*360) # -120の間で可変
    human_h = high_person[1][3]*1.2 # 人の縦幅の1.2倍
    hfov = int(human_h*180) # 縦のfov(-150)
    if fov > 90:
        fov = 120 # 人物が大きい時YOLOが正確ではないので広めに揃える
        if high_person[0] < 0.65: # confidenceが低い時
            phi = 50 # 手はカメラの上を通過することが多い
            hfov = 150 # 関節検出失敗したら人間と180度反対の部分を切り取る
    elif hfov > 150:
            hfov = 150
    pers_h = int(pers_w * hfov / fov) # perspective画像の縦幅

    Test.write_stdout(f'横fov={fov}, 縦fov={hfov}', stdout, testResult_f)
    E2P.get_perspective(file, fov, theta, phi, pers_h, pers_w, outputOmni_f) # imgがreturnされるけど使わない
    
    pers_path = os.path.join(outputOmni_f, sphere + '_' + str(theta) + '_' + str(phi) + '.jpg') # './outputOmni/'
    persImg = cv2.imread(pers_path)
    candidate, subset = body_estimation(persImg)

    sholder, wrist, vec = [], [], []
    joints_num, head, joints, msg = PointingVector.rightLeftArm_head_pytorch(candidate, subset)
    if  testMODE ==  0: # 定性評価のとき
        persImg = util.draw_bodypose(persImg, candidate, subset)
        output = os.path.join(jointPers_f,  sphere + '_' + str(theta) + '_' + str(phi) + '.png')
        cv2.imwrite(output, persImg) # OpenPoseの結果
        Test.draw_finger(joints, joints_num, output, jointPersFinger_f) # rightLeftArm_head_pytorchの結果
    print(joints)
    print(msg)
    Test.write_stdout(msg, stdout, testResult_f)
    if joints_num == [-1]:
        sholder = [theta + 170, 50] 
        wrist = [theta + 170, 20]
        vec = [0, -30] # 下向きに進む
    else:
        vec = PointingVector.get_vec_mid(joints, joints_num, head, theta, phi, pers_w, pers_h, fov)
        wrist = np.rad2deg(P2E.get_angle([joints[joints_num[-1]][0], joints[joints_num[-1]][1]], theta, phi, pers_w, pers_h, fov)) # 指差している関節の一番端
        sholder = [wrist[0]-vec[0], wrist[1]-vec[1]]
    Test.write_stdout(f'指差している腕{joints_num}, 向き{vec}, 起点=肩({wrist})', stdout, testResult_f)
    print(f'肩{sholder}手首{wrist}')
    z, y = [], []

    if  testMODE == 0: # 定性評価のとき
        z, y = GC.get(sholder, wrist, file, greatOmni_f, True) # 要素数1080個→30度は90個分
    else: # 定量評価
        z, y = GC.get(sholder, wrist, file, save=False) # zは必ず返ってくる
    if (np.isnan(y)).any():
        Test.write_stdout(f'z={z},y={y}:GC計算失敗', stdout, testResult_f)
        Test.write_stdout(f'{-3} {sphere}', 'score', testResult_f)
        continue

    rot_z = []
    rot_y = []
    start_num = 0 # 手首の角度がz,yの何番目の要素か
    last = 0 # 最後の要素の番号
    first, final = 0, 0
    opposite = 0 # スタート地点の180度反対
    if vec[0] == 0: # 周期60度
        start_num = int((wrist[0]+175) * 1080 / 360) % 1080
        rot_z.append(int(z[start_num]))
        rot_y.append(int(y[start_num]))
        E2P.get_perspective(file, 60, rot_z[0], rot_y[0], 640, 640, os.path.join(outDirect_f,sphere+'/'))
        for i in [1,2,3,4]:
            index = (start_num + i * 30) % 1080 # 10度ずつ進む
            rot_z.append(int(z[index]))
            rot_y.append(int(y[index]))
            E2P.get_perspective(file, 60, rot_z[i], rot_y[i], 640, 640, os.path.join(outDirect_f,sphere+'/'))
        first = (start_num - 30) % 1080 # 10度は30個分
        final = (start_num + 30*5) % 1080
    else:  
        if vec[0] > 0: # 右向き
            const = 30 # 指先が画像の右端にする
            adjust = 27 # 最初は27度分ずらす
        else:
            const = -30 # 指先が画像の左端
            adjust = -27
        for i in range(9): # 上端と下端に到達すると30°進まないので幅を持たせておく
            if i == 0:
                if wrist[0] < 180:
                    start_num = int((wrist[0]+180+adjust) * 1080 / 360) - 1
                else:
                    start_num = int((wrist[0]-180+adjust) * 1080 / 360) - 1
                start_num = start_num % 1080
                if abs(y[start_num]-y[(start_num-const*3)%1080]) > 60: # 縦に大きく進む
                    adjust = const * 12 / 30 # 12度分ずらす
                    start_num = int(start_num - const*1.5) % 1080 # 12=27-15
                rot_z.append(int(z[start_num]))
                rot_y.append(int(y[start_num]))
                last = start_num
                Test.write_stdout(f'z方向:{rot_z[i]}, y方向:{rot_y[i]}', stdout, testResult_f)
                E2P.get_perspective(file, 60, rot_z[i], rot_y[i], 640, 640, os.path.join(outDirect_f,sphere+'/'))
            else:
                last = (last + const * 3) % 1080 # i*90を+/-する
                opposite += 30
                if abs(y[last]) > 60 and abs(rot_y[i-1]) > 60: # 天井か床が2回連続
                    last = (last + const*2) % 1080 # 50度進む
                    opposite += 20
                while abs(y[last]-rot_y[i-1]) >= 60: # 縦に大きく進まなくなるまでループ
                    last = int(last - const*0.5) % 1080 # 5度ぶん減らす
                    opposite -= 5
                rot_z.append(int(z[last]))
                rot_y.append(int(y[last]))
                Test.write_stdout(f'z方向:{rot_z[i]}, y方向:{rot_y[i]}', stdout, testResult_f)
                E2P.get_perspective(file, 60, rot_z[i], rot_y[i], 640, 640, os.path.join(outDirect_f,sphere+'/'))
                if  opposite >= 180: # 指差している起点から左右180°反対まで
                    print('左右180°過ぎました')
                    Test.write_stdout('左右180°過ぎました', stdout, testResult_f)
                    break
     
        first = (start_num-const*3) % 1080
        final = (last+const*3) %1080

    detected = [] # [切り取った画像数len(rot_z)][検出されたオブジェクト数]
    for i in range(len(rot_z)):
        name = os.path.join(outDirect_f, sphere + '/' + sphere + '_' + str(int(rot_z[i])) + '_' + str(int(rot_y[i])) + '.jpg') # 'outDirect/'
        command = []
        if  testMODE == 0: # 定性評価のとき
            command = ['python','yolov5/detect.py','--weights','yolov5/yolov5s.pt','--source',name,'--save-txt','--exist-ok','--save-conf','--project',obj_f,'--name', sphere, '--conf-thres', '0.28'] # 0.29以上のみ
        else: # 定量評価
            command = ['python','yolov5/detect.py','--weights','yolov5/yolov5s.pt','--source',name,'--save-txt','--exist-ok','--save-conf','--project',obj_f,'--name', sphere, '--conf-thres', '0.28', '--nosave']
        proc = subprocess.run(command, capture_output=True)
    
        name_txt = os.path.join(obj_f,sphere+'/labels/' + sphere + '_' + str(int(rot_z[i])) + '_' + str(int(rot_y[i])) + '.txt') # 'obj/'+sphere+'/labels/'
        try:
            with open(name_txt, 'r') as f:
                data = f.read().splitlines() # ラベル, xywh, conf(改行を除く)
        except FileNotFoundError as e:
            data = [] # 何も検出されなかった
        detected.append(data)
        # logging.info(data)
    
    # [切り取った画像数len(rot_z)][検出されたオブジェクト数][ラベル,[x,y,w,h],conf]
    obj = ROI.match_obj(detected, rot_z, rot_y, vec)
    result = ROI.distance_circle(obj, rot_z, rot_y, [z[first],y[first]], [z[final],y[final]])
    # print(result) # [(8.069252459121573,['chair', [2862, 1597, 545, 610], 0.57567]), ...]
    ground_truth = 'ROI/' + sphere + '.txt'
    test_file = 'dummy'
    if  testMODE == 0: # 定性評価のとき
        # Test.crop_obj(obj, file)
        test_file = os.path.join(testResult_vec_f, sphere + '.jpg') # 'test_result/vec/' distance_circleでかいた画像
        Test.distance_circle(rot_z, rot_y, [z[first],y[first]], [z[final],y[final]], file, testResult_vec_f)
        Test.roi2img(obj, test_file, testResult_distance_f) # 指差しベクトル自体はdistance_circleでかく
    Correct.run(result, ground_truth, stdout, testResult_f, sphere, testMODE, test_file, theta)
    # データセットの作成
    # score, _ = Correct.in_order_of_distance(result, ground_truth)
    # result = Data.make(result, theta, sphere, score, stdout)

dt_now = datetime.datetime.now() # 実行終了時間
finish = dt_now.strftime('%Y%m%d-%H%M%S')
Test.write_stdout(finish, stdout, testResult_f)
Correct.cal_top1_5_10ALL(testResult_f, len(files))