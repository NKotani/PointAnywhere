import cv2
import os
from utils.general import (colors, name2cls, draw_text)

LINE_THICKNESS = 30 # 指示ベクトルの太さ
OBJ_THICKNESS = 13 # 長方形の物体領域の太さ

# [切り取った画像数len(rot_z)][検出されたオブジェクト数][ラベル,[x,y,w,h],conf]
def crop_obj(objs:list, image:str):
    image = cv2.imread(image)
    for obj in objs:
        for roi in obj:
            print(roi)
            img_crop = image[int(roi[1][1]):int(roi[1][1]+roi[1][3]),int(roi[1][0]):int(roi[1][0]+roi[1][2])]
            cv2.imshow("crop", img_crop)
            cv2.waitKey(0)
    return

def write_obj(joint:list, vec:list, objs:list, sphere:str, path='test_result/'):
    file = os.path.join(path, sphere + '.txt')
    with open(file, 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(f'指差している腕{joint}, 向き{vec}')
        f.write('\n')
        for obj in objs:
            for roi in obj:
                f.writelines(str(roi))
                f.write('\n')
    return

# 指差しベクトル(直線)を描画するだけ
def distance_line(rot_z:list, rot_y:list, vec:list, image:str, path='test_result/vec/', eq_w=5376, eq_h=2688):
    reconst = 0
    if vec[0] > 0: # 右向き
        reconst = -27 # 指先が画像の右端にしたのを戻す
    else:
        reconst = 27 # 指先が画像の左端
    img = cv2.imread(image)
    for i in range(len(rot_z)):
        if i == len(rot_z) - 1:
            # a(x1, y1) 指差しベクトルの起点
            x1 = ((rot_z[i]+180+reconst) % 360) / 360 * eq_w
            y1 = (90 - rot_y[i]) / 180 * eq_h
            # b(x2, y2) 指差しベクトルの先
            x2 = x1 + (vec[0] * eq_w / 360)
            y2 = y1 - (vec[1] * eq_h / 180)
        else:
            # a(x1, y1) 指差しベクトルの起点
            x1 = ((rot_z[i]+180+reconst) % 360) / 360 * eq_w
            y1 = (90 - rot_y[i]) / 180 * eq_h
            # b(x2, y2) 指差しベクトルの先
            x2 = ((rot_z[i+1]+180+reconst) % 360) / 360 * eq_w
            y2 = (90 - rot_y[i+1]) / 180 * eq_h
        # 切れ目にいるとき修正
        if x1 > 4000 and x2 < 1000: # 右向き
            cv2.arrowedLine(img, (int(x1-eq_w),int(y1)), (int(x2),int(y2)), (0,241,255), thickness=LINE_THICKNESS)
            x2 += eq_w # 10度→370度
        elif  x1 < 1000 and x2 > 4000: # 左向き
            cv2.arrowedLine(img, (int(x1),int(y1)), (int(x2-eq_w),int(y2)), (0,241,255), thickness=LINE_THICKNESS)
            x1 += eq_w
        cv2.arrowedLine(img, (int(x1),int(y1)), (int(x2),int(y2)), (0,241,255), thickness=LINE_THICKNESS) #bgr
        # cv2.imshow("step", img)
        # cv2.waitKey(0)
    cv2.imwrite(os.path.join(path, os.path.splitext(os.path.basename(image))[0] + ".jpg"), img)
    return

# 指差しベクトルを描画するだけ(great cilcle ver)
def distance_circle(rot_z:list, rot_y:list, first:list, last:list, image:str, path='test_result/vec_great/', eq_w=5376, eq_h=2688):
    img = cv2.imread(image)
    for i in range(len(rot_z)):
        if i == 0:
            # a(x1, y1) 指差しベクトルの起点
            x1 = ((first[0]+180) % 360) / 360 * eq_w
            y1 = (90 - first[1]) / 180 * eq_h
            # b(x2, y2) 指差しベクトルの先
            x2 = ((rot_z[i]+180) % 360) / 360 * eq_w
            y2 = (90 - rot_y[i]) / 180 * eq_h
        else:
            # a(x1, y1) 指差しベクトルの起点
            x1 = ((rot_z[i-1]+180) % 360) / 360 * eq_w
            y1 = (90 - rot_y[i-1]) / 180 * eq_h
            # b(x2, y2) 指差しベクトルの先
            x2 = ((rot_z[i]+180) % 360) / 360 * eq_w
            y2 = (90 - rot_y[i]) / 180 * eq_h
        # 切れ目にいるとき修正
        if x1 > 4000 and x2 < 1000: # 右向き
            cv2.arrowedLine(img, (int(x1-eq_w),int(y1)), (int(x2),int(y2)), (0,241,255), thickness=LINE_THICKNESS)
            x2 += eq_w # 10度→370度
        elif  x1 < 1000 and x2 > 4000: # 左向き
            cv2.arrowedLine(img, (int(x1),int(y1)), (int(x2-eq_w),int(y2)), (0,241,255), thickness=LINE_THICKNESS)
            x1 += eq_w
        cv2.arrowedLine(img, (int(x1),int(y1)), (int(x2),int(y2)), (0,241,255), thickness=LINE_THICKNESS) #bgr
        # cv2.imshow("step", img)
        # cv2.waitKey(0)
        if i == len(rot_z) -1:
            # a(x1, y1) 最後のpersの中心座標
            x1 = ((rot_z[i]+180) % 360) / 360 * eq_w
            y1 = (90 - rot_y[i]) / 180 * eq_h
            # b(x2, y2) 指差しベクトルの先
            x2 = ((last[0]+180) % 360) / 360 * eq_w
            y2 = (90 - last[1]) / 180 * eq_h
            # 切れ目にいるとき修正
            if x1 > 4000 and x2 < 1000: # 右向き
                cv2.arrowedLine(img, (int(x1-eq_w),int(y1)), (int(x2),int(y2)), (0,241,255), thickness=LINE_THICKNESS)
                x2 += eq_w # 10度→370度
            elif  x1 < 1000 and x2 > 4000: # 左向き
                cv2.arrowedLine(img, (int(x1),int(y1)), (int(x2-eq_w),int(y2)), (0,241,255), thickness=LINE_THICKNESS)
                x1 += eq_w
            cv2.arrowedLine(img, (int(x1),int(y1)), (int(x2),int(y2)), (0,241,255), thickness=LINE_THICKNESS) #bgr
            # cv2.imshow("step", img)
            # cv2.waitKey(0)
    cv2.imwrite(os.path.join(path, os.path.splitext(os.path.basename(image))[0] + ".jpg"), img)
    return 

# 全部のROIを描くだけ(指差しベクトルはdistance_lineかdistance_circleで描く)
def roi2img(data:list, image:str, output='test_result/distance/', eq_w=5376):
    img = cv2.imread(image)
    for obj in data:
        for roi in obj: # ['cup', [612, 1661, 36, 45], 0.252065]
            cls = name2cls(roi[0])
            color = colors(cls, True)
            # c(x3, y3) roiの中心
            x3 = roi[1][0] + roi[1][2]/2
            y3 = roi[1][1] + roi[1][3]/2
            cv2.drawMarker(img, (int(x3), int(y3)), color, markerSize=40, thickness=8) # roiの中心
            cv2.rectangle(img, (int(roi[1][0]), int(roi[1][1])), (int(roi[1][0]+roi[1][2]), int(roi[1][1]+roi[1][3])), color, thickness=OBJ_THICKNESS) # roi
            # 切れ目にいるとき追加
            if roi[1][0]+roi[1][2] > eq_w:
                cv2.drawMarker(img, (int(x3)-eq_w, int(y3)), color, markerSize=40, thickness=8) # マーカーは右端の可能性もあるけど処理を簡単にするためやっとく
                cv2.rectangle(img, (int(roi[1][0])-eq_w, int(roi[1][1])), (int(roi[1][0]+roi[1][2])-eq_w, int(roi[1][1]+roi[1][3])), color, thickness=OBJ_THICKNESS) # roi
        # cv2.imshow("step", img)
        # cv2.waitKey(0)
    cv2.imwrite(os.path.join(output, os.path.splitext(os.path.basename(image))[0] + ".jpg"), img)
    return

# 上位5個のROIを描くだけ(指差しベクトルはdistance_lineかdistance_circleで描く)
def roi2imgTop5(data:list, image:str, output='test_result/distance/', eq_w=5376):
    img = cv2.imread(image)
    for i, obj in reversed(list(enumerate(data))): # (54.2771666751422, ['bowl', [2593, 1686, 77, 67], 0.859447])
        roi = obj[1]
        cls = name2cls(roi[0])
        color = colors(cls, True)
        # txt = '{} {:.2f}'.format(roi[0], obj[0]) # 物体名 (補正した)距離
        txt = str(i+1) + ' ' + roi[0]
        cv2.rectangle(img, (int(roi[1][0]), int(roi[1][1])), (int(roi[1][0]+roi[1][2]), int(roi[1][1]+roi[1][3])), color, thickness=OBJ_THICKNESS) # roi
        draw_text(img, txt, pos=(int(roi[1][0]), int(roi[1][1])), text_color_bg=color)
        # 切れ目にいるとき追加
        if roi[1][0]+roi[1][2] > eq_w:
            # cv2.drawMarker(img, (int(x3)-eq_w, int(y3)), color, markerSize=40, thickness=7) # マーカーは右端の可能性もあるけど処理を簡単にするためやっとく
            cv2.rectangle(img, (int(roi[1][0])-eq_w, int(roi[1][1])), (int(roi[1][0]+roi[1][2])-eq_w, int(roi[1][1]+roi[1][3])), color, thickness=OBJ_THICKNESS) # roi
            draw_text(img, txt, pos=(int(roi[1][0])-eq_w, int(roi[1][1])), text_color_bg=color)
    cv2.imwrite(os.path.join(output, os.path.splitext(os.path.basename(image))[0] + '_top5.jpg'), img)
    return

# 上位1個のROIを描くだけ(指差しベクトルはdistance_lineかdistance_circleで描く)
def roi2imgTop1(data:list, image:str, output='test_result/distance/', eq_w=5376):
    img = cv2.imread(image)
    obj = data[0]
    roi = obj[1]
    cls = name2cls(roi[0])
    color = colors(cls, True)
    txt = '{} {:.2f}'.format(roi[0], obj[0])
    cv2.rectangle(img, (int(roi[1][0]), int(roi[1][1])), (int(roi[1][0]+roi[1][2]), int(roi[1][1]+roi[1][3])), color, thickness=OBJ_THICKNESS) # roi
    draw_text(img, txt, pos=(int(roi[1][0]), int(roi[1][1])), text_color_bg=color)
    # 切れ目にいるとき追加
    if roi[1][0]+roi[1][2] > eq_w:
        # cv2.drawMarker(img, (int(x3)-eq_w, int(y3)), color, markerSize=40, thickness=7) # マーカーは右端の可能性もあるけど処理を簡単にするためやっとく
        cv2.rectangle(img, (int(roi[1][0])-eq_w, int(roi[1][1])), (int(roi[1][0]+roi[1][2])-eq_w, int(roi[1][1]+roi[1][3])), color, thickness=OBJ_THICKNESS) # roi
        draw_text(img, txt, pos=(int(roi[1][0]), int(roi[1][1])), text_color_bg=color)
    cv2.imwrite(os.path.join(output, os.path.splitext(os.path.basename(image))[0] + '_top1.jpg'), img)
    return

def write_stdout(text:str, name:str, path='test_result/'):
    if name != 'NO':
        file = os.path.join(path, name + '.txt')
        with open(file, 'a', encoding='utf-8', newline='\n') as f:
            f.writelines(text)
            f.write('\n')
    return

# 延長した指先の確認
def draw_finger(joints:dict, pointing_num:list, image:str, output= 'jointPers/fingers'):
    img = cv2.imread(image)
    if len(pointing_num) == 2:
        cv2.line(img, joints[pointing_num[0]], joints[pointing_num[1]], (255,90,0), thickness=7)
    elif len(pointing_num) == 3:
        cv2.line(img, joints[pointing_num[0]], joints[pointing_num[1]], (255,90,0), thickness=7)
        cv2.line(img, joints[pointing_num[1]], joints[pointing_num[2]], (255,90,0), thickness=7)
    
    for joint in joints.values():
        cv2.circle(img, joint, 5, (0, 75, 255), thickness=5)
    cv2.imwrite(os.path.join(output, os.path.splitext(os.path.basename(image))[0] + ".jpg"), img)
    return