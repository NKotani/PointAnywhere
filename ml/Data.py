import numpy as np
import os

# 出現数をカウントしたリストを返す
def count_obj(data:list):
    labels = {}
    for obj in data: # (54.2771666751422, ['bowl', [2593, 1686, 77, 67], 0.859447])
        # ラベルの登場回数をカウント
        if obj[1][0] in labels:
            labels[obj[1][0]] += 1
        else:
            labels[obj[1][0]] = 1 # 初期化
    return labels

# 指示者とROIの中心との横方向の差分の距離を求める(斜めの厳密な距離ではない)
def distance_human(obj:list, theta_human:int, eq_w=5376):
    theta_human += 180 # 0度~360度、画像の右端原点
    xywh_obj = obj[1][1] # (54.2771666751422, ['bowl', [2593, 1686, 77, 67], 0.859447])
    theta_obj = (xywh_obj[0] + xywh_obj[2]) * 360 / eq_w # ROI横方向の中心座標
    distance = abs(np.sin(np.deg2rad(theta_human)) - np.sin(np.deg2rad(theta_obj))) # sinにすることで0度と360度のとき差が0になる
    return distance # 0-2


# データセットの作成 pers-大円-pers前提, scoreはCorret.in_order_of_distance(距離の短い順)の番号
def make(data:list, theta_human:int, sphere:str, score:int, file:str):
    file = os.path.join('ml', file + '.txt')
    i = 0
    labels = count_obj(data)
    result = [] # [sphere, この画像に正解は存在するか？, この物体は正解か？, 指示ベクトル距離, 希少度, 信頼度, 人距離, 面積]
    exist_flag = 0 # この画像に正解は存在するか？(全候補の中に正解物体が検出できているか)
    if score > 0:
        exist_flag = 1 # 正解は存在
    for obj in data: # (54.2771666751422, ['bowl', [2593, 1686, 77, 67], 0.859447])
        i += 1
        correct_flag = 0 # この物体が正解か？
        if exist_flag and i == score:
            correct_flag = 1 # 正解の物体の番
        rarity = labels[obj[1][0]] / len(data) # 希少度=同じ物体数/全物体数
        Hdistance = distance_human(obj, theta_human) # ROIと人の距離
        area = abs(obj[1][1][2] * obj[1][1][3]) # 面積
        tmp = [sphere, exist_flag, correct_flag, obj[0], rarity, obj[1][2], Hdistance, area]
        result.append(tmp)
        with open(file, 'a', encoding='utf-8', newline='\n') as f:
            f.writelines(f'{sphere} {exist_flag} {correct_flag} {obj[0]} {rarity} {obj[1][2]} {Hdistance} {area}\n')
    return result

# 評価用のパラメータ計算 pers-大円-pers前提
def cal(data:list, theta_human:int):
    labels = count_obj(data)
    result = [] # [指示ベクトル距離, 出現頻度, 信頼度, 人距離, 面積]
    obj_name = [] # ['bowl', [2593, 1686, 77, 67], 0.859447]
    for obj in data: # (54.2771666751422, ['bowl', [2593, 1686, 77, 67], 0.859447])
        rarity = labels[obj[1][0]] / len(data) # 重要度=同じ物体数/全物体数
        Hdistance = distance_human(obj, theta_human) # ROIと人の距離
        area = abs(obj[1][1][2] * obj[1][1][3]) # 面積
        result.append([obj[0], rarity, obj[1][2], Hdistance, area])
        obj_name.append(obj[1])
    result = np.array(result)
    np.nan_to_num(result, copy=False) # NaNを0に変換
    return result, obj_name