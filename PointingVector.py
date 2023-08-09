import Pers2Equi as P2E
import numpy as np

# 関節からベクトルを求める(角度) 頭ー指先または肩ー指先のベクトルを求める
def get_vec(joints:dict, joints_num:list, theta:float, phi:float, pers_w:int, pers_h:int, fov:int):
    # joints_num: 指差している右か左の肩・肘・手首の番号(2-7)が格納されている(+頭)
    vector = [0,0]
    origin = P2E.get_angle([joints[joints_num[0]][0],joints[joints_num[0]][1]], theta, phi, pers_w, pers_h, fov) # 肩側
    terminus = P2E.get_angle([joints[joints_num[-1]][0],joints[joints_num[-1]][1]], theta, phi, pers_w, pers_h, fov) # 手首側
    if origin == terminus: # 関節1つのとき始点と終点同じ
        terminus[0] += np.pi / 2
        terminus[1] = 0 # 取れてる関節をinclination(傾斜)になる点としたGCを描く。右向きに進む
    vector = [np.rad2deg(t - o) for (t, o) in zip(terminus, origin)]
    if abs(vector[0]) == 0: # 経線に沿って進む
        k = 30 / abs(vector[1]) # y方向のベクトルを大きさ30°に正規化
        vector = [int(n*k) for n in vector]
    return vector

# 関節からベクトルを求める(角度) 頭ー指先と肩ー指先の平均を取る
def get_vec_mid(joints:dict, joints_num:list, head_num:list, theta:float, phi:float, pers_w:int, pers_h:int, fov:int):
    # joints_num: 指差している右か左の肩・肘・手首の番号(2-7)が格納されている
    # head: 頭・右目・左目
    vector = [0,0]
    if len(joints_num) < 1:  # 頭ー指先 or どこか関節1つ
        vector = get_vec(joints, head_num+joints_num, theta, phi, pers_w, pers_h, fov)
    elif len(head_num) == 0: # 頭取れなかった
        vector = get_vec(joints, joints_num, theta, phi, pers_w, pers_h, fov) # 肩ー指先
    else:
        sholder = get_vec(joints, joints_num, theta, phi, pers_w, pers_h, fov) # 肩ー指先
        head = get_vec(joints, head_num+joints_num, theta, phi, pers_w, pers_h, fov) # 頭ー指先
        vector = [(s+h)/2 for (s, h) in zip(sholder, head)]
    return vector

# 関節からベクトルを求める(角度) 頭ー指先または肩ー指先のベクトルを求める:equirectangular版
def get_vec_eq(joints:dict, joints_num:list, eq_w:int, eq_h:int):
    # joints_num: 指差している右か左の肩・肘・手首の番号(2-7)が格納されている(+頭)
    vector = [0,0]
    origin = px2angle(joints[joints_num[0]][0], joints[joints_num[0]][1], eq_w, eq_h) # 肩側
    terminus = px2angle(joints[joints_num[-1]][0],joints[joints_num[-1]][1], eq_w, eq_h) # 手首側
    if origin == terminus: # 関節1つのとき始点と終点同じ
        terminus[0] += 90
        terminus[1] = 0 # 取れてる関節をinclination(傾斜)になる点としたGCを描く。右向きに進む
    vector = [t - o for (t, o) in zip(terminus, origin)]
    if abs(vector[0]) == 0: # 経線に沿って進む
        k = 30 / abs(vector[1]) # y方向のベクトルを大きさ30°に正規化
        vector = [int(n*k) for n in vector]
    return vector # degree

# 関節からベクトルを求める(角度) 頭ー指先と肩ー指先の平均を取る:equirectangular版
def get_vec_mid_eq(joints:dict, joints_num:list, head_num:list, eq_w:int, eq_h:int):
    # joints_num: 指差している右か左の肩・肘・手首の番号(2-7)が格納されている
    # head: 頭・右目・左目
    vector = [0,0]
    if len(joints_num) < 1:  # 頭ー指先 or どこか関節1つ
        vector = get_vec_eq(joints, head_num+joints_num, eq_w, eq_h)
    elif len(head_num) == 0: # 頭取れなかった
        vector = get_vec_eq(joints, joints_num, eq_w, eq_h) # 肩ー指先
    else:
        sholder = get_vec_eq(joints, joints_num, eq_w, eq_h) # 肩ー指先
        head = get_vec_eq(joints, head_num+joints_num, eq_w, eq_h) # 頭ー指先
        vector = [(s+h)/2 for (s, h) in zip(sholder, head)]
    return vector

# equirectangularの画像のpxを緯度経度に変換(x：横, y：縦)
def px2angle(x:int, y:int, eq_w:int, eq_h:int):
    # 右上原点から、中心原点にする
    angle = [0,0]
    angle[0] = (x / eq_w)*360 + 180 # 横
    angle[1] = 90 - (y / eq_h)*180 # 縦
    return angle # degree

# 指差している腕の判定と頭
def rightLeftArm_head_pytorch(candidate, subset):
    # print(candidate.shape, subset.shape) # (18, 4) (2, 20)
    joints = {} # indexをキーにして座標をvalueに入れて返す
    point_arm =[]
    # 腕の格納
    right = []
    left = []
    head = []
    msg = 'rightLeftArm_head_pytorch'
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i]) # indexは格納されている位置で関節番号自体は1
            if index == -1:
                continue
            x= int(candidate[index][0])
            y= int(candidate[index][1])
            if i in [2,3,4]:
                if len(right) > 0 and abs(joints[right[-1]][0]-x) <= 1 and abs(joints[right[-1]][1]- y) <= 1:
                    continue # 関節が重なってる
                right.append(i)
                joints[i] = (x,y)
            elif i in [5,6,7]:
                if len(left) > 0 and abs(joints[left[-1]][0]-x) <= 1 and abs(joints[left[-1]][1]- y) <= 1:
                    continue # 関節が重なってる
                left.append(i)
                joints[i] = (x,y)
            elif i in [0,14,15]: # 鼻、右目、左目
                head.append(i)
                joints[i] = (x,y)
    lR = len(right)
    lL = len(left)
    # Judgment of the arm the person is pointing at
    if lL <= 1 and lR <= 1: # 関節数0-1点
        point_arm = [-1]
        msg = 'Less than 1 point detected on both arms.'
    # 片方は関節2つ以上
    elif lR <= 1:
        point_arm = left # 右は肩しか取れてない
        msg = 'Only 1 point for the right arm.'
    elif lL <= 1:
        point_arm = right # 左は肩しか取れてない
        msg = 'Only 1 point for the left arm.'
    # 以下右も左も関節2つ以上
    elif lR <= 2 and lL == 3:
        point_arm = left
        msg = '2 points for the right arm and 3 points for the left arm.'
    elif lL <= 2 and lR == 3:
        point_arm = right
        msg = '2 points for the left arm and 3 points for the right arm.'
    # 両腕とも2点か両腕とも3点
    elif lR == 3 and (joints[3][0] - joints[2][0])*(joints[4][0] - joints[3][0]) < 0:
        point_arm = left
        msg = 'Right arm is bent.'
        if lL == 3 and (joints[6][0] - joints[5][0])*(joints[7][0] - joints[6][0]) < 0: # 左腕も曲がっている
            if joints[right[-1]][1] < joints[left[-1]][1]:
                point_arm = right
                msg = 'Both arms are bent and right arm is up.'
            else:
                point_arm = left
                msg = 'Both arms are bent and left arm is up.'
    elif lL == 3 and (joints[6][0] - joints[5][0])*(joints[7][0] - joints[6][0]) < 0:
        point_arm = right
        msg = 'Left arm is bent'
    elif joints[right[-1]][1] < joints[left[-1]][1]:
        point_arm = right
        msg = 'Right arm is up.'
        if (lL > lR) and (abs(joints[3][0]-joints[6][0])<15 and abs(joints[3][1]-joints[6][1])<15): # 右腕が上でも左の関節の方が取れてる
            point_arm = left # 右肘と左肘の位置に差がないので多く関節が取れている方を使う
            msg = 'The left arm is better detected.'
    else:
        point_arm = left # 左腕が上
        msg = 'Left arm is up'
        if (lL < lR) and (abs(joints[3][0]-joints[6][0])<15 and abs(joints[3][1]-joints[6][1])<15): # 左腕が上でも右の関節の方が取れてる
            point_arm = right # 左肘と右肘を比べている
            msg = 'The right arm is better detected.'
    # 腕の長さの補正
    limb = 703.3 # 肩から指先
    arm = 303.95 # 肩から肘
    forearm = 238.85 # 肘から手首
    extend = 0
    if (2 in point_arm and 4 in point_arm) or (5 in point_arm and 7 in point_arm): # 肩と手首
        extend = limb / (arm + forearm)
    elif (2 in point_arm and 3 in point_arm) or (5 in point_arm and 6 in point_arm): # 肩と肘
        extend = limb / arm
    elif (3 in point_arm and 4 in point_arm) or (6 in point_arm and 7 in point_arm): # 肘と手首
        extend = limb / forearm
    if extend > 0: # 2点以上検出されると補正
        vec = [joints[point_arm[-1]][0] - joints[point_arm[0]][0], joints[point_arm[-1]][1] - joints[point_arm[0]][1]]
        joints[point_arm[-1]] =  [int(v*extend+o) for (v, o) in zip(vec, joints[point_arm[0]])]
    return point_arm, head, joints, msg