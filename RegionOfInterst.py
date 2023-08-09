import numpy
import Pers2Equi as P2E

# perspective画像のYOLO形式ROIをEq形式に変換する
def pers2eq(x:float, y:float, w:float, h:float, rot_z:int, rot_y:int, pers_w=1, pers_h=1, fov=60, eq_w=5376):
    # w_eq = w * eq_w / 6 # persのfovが60/360
    # h_eq = h * eq_h / 3 # persのfovが60/180
    x_small = x - w/2 # 右上の座標
    y_small = y - h/2
    x_big = x + w/2
    y_big = y + h/2
    right_up = P2E.per2eq([x_small, y_small], rot_z, rot_y, pers_w, pers_h, fov) # pers_w=640でも1のままでも同じ
    # right_down = P2E.per2eq([x_small, y_big], rot_z, rot_y, pers_w, pers_h, fov) # 右下
    # left_up = P2E.per2eq([x_big, y_small],rot_z, rot_y, pers_w, pers_h, fov) # 左下の座標
    left_down = P2E.per2eq([x_big, y_big],rot_z, rot_y, pers_w, pers_h, fov) # 左下の座標
    # x_eq = min(right_up[0], left_up[0]) # あまり変わらない
    # y_eq = min(right_up[1], left_up[1])
    # w_eq = max(right_down[0], left_down[0]) - x_eq
    # h_eq = max(right_down[1], left_down[1]) - y_eq
    x_eq = min(right_up[0], left_down[0])
    y_eq = right_up[1]
    w_eq = max(right_up[0], left_down[0]) - x_eq # /形のときがあるから右上が小さいとは限らない
    h_eq = left_down[1] - y_eq
    x_eq = x_eq % eq_w # P2E.per2eqでは右側に飛び出した座標
    return [x_eq, y_eq, w_eq, h_eq]

# Eq形式のROI(xywh)をもらってIoUを計算
def IoU(roi1:list, roi2:list):
    area1 = roi1[2] * roi1[3] # WxH
    area2 = roi2[2] * roi2[3]
    x_min = max(roi1[0], roi2[0]) # 重なっている領域の左上の点
    y_min = max(roi1[1], roi2[1])
    x_max = min(roi1[0]+roi1[2], roi2[0]+roi2[2]) # 重なっている領域の右下の点
    y_max = min(roi1[1]+roi1[3], roi2[1]+roi2[3])
    w = max(0, x_max - x_min) # 重ならない時x_max - x_minは負の値
    h = max(0, y_max - y_min)
    intersect = w * h # 重なっている領域の面積
    iou = intersect / (area1 + area2 - intersect)
    # print(f'{roi1}, {roi2}, w={w}, h={h}, a1={area1}, a2={area2}, 共通{intersect}, iou={iou}')
    return iou

# 1つ前のpers画像と一致しているものを削除し、データ形式を整える
def match_obj(data:list, rot_z:list, rot_y:list, vec:list):
    result = []

    for i in range(len(data)):
        line = []
        for label in data[i]:
            tmp = label.split() # ['bottle', '0.40625', '0.866406', '0.071875', '0.148438', '0.35455']
            if tmp[0] == 'person':
                continue # 人なら飛ばす
            if len(tmp) == 7:
                # dining tableみたいな2単語
                tmp = [tmp[0] + '_' + tmp[1]] + tmp[2:]
            roi = pers2eq(float(tmp[1]), float(tmp[2]), float(tmp[3]), float(tmp[4]), rot_z[i], rot_y[i])
            if i != 0:
                # 前の画像で被っているものを探す
                for k in range(i):
                    rotZ = rot_z[k]
                    rotI = rot_z[i]
                    if vec[0] >= 0 and rotZ > 0 and rotI < 0: # 今いる画像は左側に突入
                        rotI += 360
                    elif vec[0] < 0 and rotZ < 0 and rotI > 0: # 今いる画像は右側に突入
                        rotZ += 360
                    if abs(rotZ - rotI) > 50 or abs(rot_y[k] - rot_y[i]) > 50:
                        continue
                    # print(f'{rot_z[i]}, {rot_y[i]}にいて、{rot_z[k]}, {rot_y[k]}を探します{i, k}')
                    for j in range(len(result[k])):
                        if tmp[0] == result[k][j][0] and IoU(roi, result[k][j][1])>0.04: # 0.03未満のものを統合するな
                            RoI = [0,0,0,0] # xywh
                            # 座標を統合
                            # print(f'{roi}と{result[k][j]}を統合')
                            if roi[0] < 1000 and result[k][j][1][0] > 4000: # またがっているときの統合
                                RoI[0] = result[k][j][1][0]
                                RoI[2] = roi[0]+roi[2] - result[k][j][1][0] # roiが画像右側
                            elif roi[0] > 4000 and result[k][j][1][0] < 1000: # またがっているときの統合
                                RoI[0] = roi[0]
                                RoI[2] = result[k][j][1][0]+result[k][j][1][2] - roi[0] # roiが画像左側に切れてる
                            else:
                                RoI[0] = min(roi[0], result[k][j][1][0])
                                RoI[2] = max(roi[0]+roi[2], result[k][j][1][0]+result[k][j][1][2]) - RoI[0] # max(x+w) - x_min
                            RoI[1] = min(roi[1], result[k][j][1][1]) # y
                            RoI[3] = max(roi[1]+roi[3], result[k][j][1][1]+result[k][j][1][3]) - RoI[1] # h
                            tmp[5] = max(float(tmp[5]), result[k][j][2]) # confidence高い方に揃える
                            roi = RoI
                            result[k].pop(j) # 前の被りを消す
                            break
            line.append([tmp[0], roi, float(tmp[5])]) # ['bottle', [0.40625, 0.866406, 0.071875, 0.148438], 0.35455]
        result.append(line)
    return result

# equirectangular画像で検出されたデータをリサイズしなおしてデータ形式を整える
def resize_obj(data:list, eq_w=5376, eq_h=2688):
    result = []
    for label in data:
        tmp = label.split() # ['bottle', '0.40625', '0.866406', '0.071875', '0.148438', '0.35455']
        if tmp[0] == 'person':
            continue # 人なら飛ばす
        if len(tmp) == 7:
            # dining tableみたいな2単語
            tmp = [tmp[0] + '_' + tmp[1]] + tmp[2:]
        x = float(tmp[1]) -  float(tmp[3]) / 2 # 右上
        y = float(tmp[2]) -  float(tmp[4]) / 2 # 右上
        w = float(tmp[3])
        h = float(tmp[4])
        roi = [int(x*eq_w), int(y*eq_h), int(w*eq_w), int(h*eq_h)]
        line = [tmp[0], roi, float(tmp[5])]
        result.append(line) # ['bottle', [0.40625, 0.866406, 0.071875, 0.148438], 0.35455]
    return result

# 指差しベクトルとROIの中心との距離を求める(直線)
def distance(data:list, rot_z:list, rot_y:list, vec:list, eq_w=5376, eq_h=2688):
    result = {}
    i = 0
    reconst = 0
    if vec[0] > 0: # 右向き
        reconst = -27 # 指先が画像の右端にしたのを戻す
    else:
        reconst = 27 # 指先が画像の左端
    for obj in data:
        for roi in obj: # ['cup', [612, 1661, 36, 45], 0.252065]
            if i == len(rot_z) - 1:
                # a(x1, y1) 指差しベクトルの起点
                x1 = ((rot_z[i]+180+reconst) % 360) / 360 * eq_w
                y1 = (90 - rot_y[i]) / 180 * eq_h
                # b(x2, y2) 指差しベクトルの先
                x2 = x1 + (vec[0] * eq_w / 360)
                y2 = y1 - (vec[1] * eq_h / 180) # y座標は符号が逆
            else:
                # a(x1, y1) 指差しベクトルの起点
                x1 = ((rot_z[i]+180+reconst) % 360) / 360 * eq_w
                y1 = (90 - rot_y[i]) / 180 * eq_h
                # b(x2, y2) 指差しベクトルの先
                x2 = ((rot_z[i+1]+180+reconst) % 360) / 360 * eq_w
                y2 = (90 - rot_y[i+1]) / 180 * eq_h
            # c(x3, y3) roiの中心
            x3 = roi[1][0] + roi[1][2]/2
            y3 = roi[1][1] + roi[1][3]/2
            # 切れ目にいるとき修正
            if x1 > 4000 and x2 < 1000: # 右向き
                x2 += eq_w # 10度→370度
                if x3 < 2500:
                    x3 += eq_w
            elif  x1 < 1000 and x2 > 4000: # 左向き
                x1 += eq_w
                if x3 < 2500:
                    x3 += eq_w
            u = numpy.array([x2 - x1, y2 - y1])
            v = numpy.array([x3 - x1, y3 - y1])
            distance = abs(numpy.cross(u, v) / numpy.linalg.norm(u)) # 1267以下
            # print(f'点({int(x1)},{int(y1)})({int(x2)},{int(y2)})とroi({int(x3)},{int(y3)})とのベクトルは{u},{v})')
            result[distance] =  roi
        i += 1
    result_list = sorted(result.items())
    return result_list

# 指差しベクトルとROIの中心との距離を求める(直線, equirectangular版)
def distance_eq(data:list, rot_z:list, rot_y:list, vec:list, eq_w=5376, eq_h=2688):
    result = {}
    reconst = 0
    if vec[0] > 0: # 右向き
        reconst = -27 # 指先が画像の右端にしたのを戻す
    else:
        reconst = 27 # 指先が画像の左端
    for roi in data: # ['cup', [612, 1661, 36, 45], 0.252065]
        first = 0 # ベクトル1つ目
        last = 0 # ベクトルの一番最後
        # a(x1, y1) 指差しベクトルの起点
        x1 = ((rot_z[0]+180+reconst) % 360) / 360 * eq_w
        y1 = (90 - rot_y[0]) / 180 * eq_h
        # b(x2, y2) 指差しベクトルの先
        x2 = ((rot_z[1]+180+reconst) % 360) / 360 * eq_w
        y2 = (90 - rot_y[1]) / 180 * eq_h
        # c(x3, y3) roiの中心
        x3 = roi[1][0] + roi[1][2]/2
        y3 = roi[1][1] + roi[1][3]/2
        # 切れ目にいるとき修正
        if x1 > 4000 and x2 < 1000: # 右向き
            x2 += eq_w # 10度→370度
            if x3 < 2500:
                x3 += eq_w
        elif  x1 < 1000 and x2 > 4000: # 左向き
            x1 += eq_w
            if x3 < 2500:
                x3 += eq_w
        u = numpy.array([x2 - x1, y2 - y1])
        v = numpy.array([x3 - x1, y3 - y1])
        first = abs(numpy.cross(u, v) / numpy.linalg.norm(u))

        # a(x1, y1) 指差しベクトルの起点
        x1 = ((rot_z[-2]+180+reconst) % 360) / 360 * eq_w
        y1 = (90 - rot_y[-2]) / 180 * eq_h
        # b(x2, y2) 指差しベクトルの先
        x2 = ((rot_z[-1]+180+reconst) % 360) / 360 * eq_w
        y2 = (90 - rot_y[-1]) / 180 * eq_h
        # 切れ目にいるとき修正
        if x1 > 4000 and x2 < 1000: # 右向き
            x2 += eq_w # 10度→370度
        elif  x1 < 1000 and x2 > 4000: # 左向き
            x1 += eq_w
        u = numpy.array([x2 - x1, y2 - y1])
        v = numpy.array([x3 - x1, y3 - y1])
        last = abs(numpy.cross(u, v) / numpy.linalg.norm(u))
        distance = min(first, last)
        result[distance] = roi
    result_list = sorted(result.items())
    return result_list

# Test.pyのdistance_circleからdistanceに必要なベクトル計算部分のみにした
def cal_circle(rot_z:list, rot_y:list, first:list, last:list, eq_w=5376, eq_h=2688):
    X = []
    Y = []
    for i in range(len(rot_z)):
        if i == 0:
            # 指差しベクトルの起点
            x = ((first[0]+180) % 360) / 360 * eq_w
            y = (90 - first[1]) / 180 * eq_h
            X.append(x)
            Y.append(y)
            x = ((rot_z[i]+180) % 360) / 360 * eq_w
            y = (90 - rot_y[i]) / 180 * eq_h
            X.append(x)
            Y.append(y)
        elif i == len(rot_z) -1:
            # a(x1, y1) 最後のpersの中心座標
            x = ((rot_z[i]+180) % 360) / 360 * eq_w
            y = (90 - rot_y[i]) / 180 * eq_h
            X.append(x)
            Y.append(y)
            # 指差しベクトルの先
            x = ((last[0]+180) % 360) / 360 * eq_w
            y = (90 - last[1]) / 180 * eq_h
            X.append(x)
            Y.append(y)
        else:
            x = ((rot_z[i]+180) % 360) / 360 * eq_w
            y = (90 - rot_y[i]) / 180 * eq_h
            X.append(x)
            Y.append(y)
        # 切れ目にいるとき修正は距離の計算(distance_circle)でやる
    return X,Y

# 指差しベクトルとROIの中心との距離を求める(great ciecle ver)
def distance_circle(data:list, rot_z:list, rot_y:list, first:list, last:list, eq_w=5376):
    result = {}
    i = 0
    X, Y = cal_circle(rot_z, rot_y, first, last)
    vec = 0
    if X[0] < X[1]:
        vec = 1 # 右向き
    else:
        vec = -1
    for obj in data:
        for roi in obj: # ['cup', [612, 1661, 36, 45], 0.252065]
            x1, y1, x2, y2 = 0, 0, 0, 0
            # c(x3, y3) roiの中心
            x3 = roi[1][0] + roi[1][2]/2
            y3 = roi[1][1] + roi[1][3]/2
            if (vec==1 and X[i+1]>x3) or (vec==-1 and X[i+1]<x3): # 画像の前半部分
                x1 = X[i]
                y1 = Y[i]
                x2 = X[i+1]
                y2 = Y[i+1]
            else:
                x1 = X[i+1]
                y1 = Y[i+1]
                x2 = X[i+2]
                y2 = Y[i+2]
            # 切れ目にいるとき修正
            if x1 > 4000 and x2 < 1000: # 右向き
                x2 += eq_w # 10度→370度
                if x3 < 2500:
                    x3 += eq_w
            elif  x1 < 1000 and x2 > 4000: # 左向き
                x1 += eq_w
                if x3 < 2500:
                    x3 += eq_w
            u = numpy.array([x2 - x1, y2 - y1])
            v = numpy.array([x3 - x1, y3 - y1])
            distance = abs(numpy.cross(u, v) / numpy.linalg.norm(u)) # 633.5以下
            # print(f'点({int(x1)},{int(y1)})({int(x2)},{int(y2)})とroi({int(x3)},{int(y3)})とのベクトルは{u},{v})')
            result[distance] =  roi
        i += 1
    result_list = sorted(result.items())
    return result_list

# 指差しベクトルとROIの中心との距離を求める(great ciecle and equirectangular ver)
def distance_circle_eq(data:list, rot_z:list, rot_y:list, first:list, last:list, eq_w=5376):
    result = {}
    X, Y = cal_circle(rot_z, rot_y, first, last)
    for roi in data: # ['cup', [612, 1661, 36, 45], 0.252065]
        distance_list = [] # great circleのすべての線と距離をもとめて最小値を採用
        for i in range(len(X)-1):
            # c(x3, y3) roiの中心
            x3 = roi[1][0] + roi[1][2]/2
            y3 = roi[1][1] + roi[1][3]/2

            x1 = X[i]
            y1 = Y[i]
            x2 = X[i+1]
            y2 = Y[i+1]
            # 切れ目にいるとき修正
            if x1 > 4000 and x2 < 1000: # 右向き
                x2 += eq_w # 10度→370度
                if x3 < 2500:
                    x3 += eq_w
            elif  x1 < 1000 and x2 > 4000: # 左向き
                x1 += eq_w
                if x3 < 2500:
                    x3 += eq_w
            u = numpy.array([x2 - x1, y2 - y1])
            v = numpy.array([x3 - x1, y3 - y1])
            distance_list.append(abs(numpy.cross(u, v) / numpy.linalg.norm(u)))

        distance = min(distance_list)
        result[distance] =  roi
    result_list = sorted(result.items())
    return result_list 

if __name__ == '__main__':
    image = './inputOmni/R0010182.JPG'
    sphere = 'R0010182'
    rot_z = [375]
    rot_y = [-13]
    fov = 60
    # pers_wh = 640
    eq_w = 5376
    for i in range(len(rot_z)):
        name_txt = 'obj/'+sphere+'/labels/' + sphere + '_' + str(rot_z[i]) + '_' + str(rot_y[i]) + '.txt'
        try:
            with open(name_txt, 'r') as f:
                data = f.read().splitlines() # ラベル, xywh, conf(改行を除く)
                for d in data:
                    tmp = d.split() # ['bottle', '0.40625', '0.866406', '0.071875', '0.148438', '0.35455']
                    if len(tmp) == 7:
                        # dining tableみたいな2単語
                        tmp = [tmp[0] + '_' + tmp[1]] + tmp[2:]
                    roi = pers2eq(float(tmp[1]), float(tmp[2]), float(tmp[3]), float(tmp[4]), rot_z[i], rot_y[i])
                    print(f'tmp={tmp},roi={roi}')
        except FileNotFoundError as e:
            print(e)
            continue