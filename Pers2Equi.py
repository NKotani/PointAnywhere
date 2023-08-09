import numpy as np
import cv2

"""
# perspective画像中心からの角度を求める(coordinates:右上原点の座標)
def get_angle(coordinates:list, pers_w:int, pers_h:int, fov:int):
    z = pers_w * 0.5 / np.tan(np.deg2rad(0.5*fov)) # 焦点距離
    x =  coordinates[0] - pers_w / 2 # 原点を中心にする
    y =  pers_h / 2 - coordinates[1]
    lon = np.arctan2(x, z) # x/z
    lat = np.arcsin(y / np.sqrt(x**2+y**2+z**2))
    # print(f'{coordinates}, x={x}, y={y}, z={z}, lon={lon}, lat={lat}')
    return lon, lat # 横(右が＋)、縦(上が＋) (radian)
"""

# 緯度(radian)をもらってeqのy座標に変換する
def lat2eq(lat:float, eq_w=5376, eq_h=2688):
    radius = eq_w * 0.5 / np.pi # 855.616
    eq = radius * lat # 中央原点
    eq = (eq_h/2 - eq) % eq_h
    return int(eq) # 右上原点、下が正

# 経度(radian)をもらってeqのx座標に変換する
def lon2eq(lon:float, eq_w=5376):
    radius = eq_w * 0.5 / np.pi # 855.616
    eq = radius * lon # 中央原点
    # print(f'中央原点{eq}')
    eq = eq_w/2 + eq
    # print(f'右上原点{eq}')
    return int(eq) # 右上原点、右が正

# perspective画像の点をeq形式の座標にするを求める(coordinates:右上原点の座標, rot_z,rot_y,fovはdegree)
def per2eq(coordinates:list, rot_z:float, rot_y:float, pers_w:int, pers_h:int, fov:int):
    lon, lat = get_angle(coordinates, rot_z, rot_y, pers_w, pers_h, fov) # radain
    x = lon2eq(lon)
    y = lat2eq(lat)
    return x, y # 右上原点のeq座標(z座標は5376以上もあり得る)

# equirectangularの画像中心からの角度を求める(coordinates:右上原点の座標)
def get_angle(coordinates:list, rot_z:float, rot_y:float, pers_w:int, pers_h:int, fov:int):
    # 半径1の単位円に正規化
    k = np.tan(np.deg2rad(fov/2))*2 / pers_w
    x = (coordinates[0] - pers_w/2) * k
    y = (pers_h/2 - coordinates[1]) * k

    rou = np.sqrt(x ** 2 + y ** 2)
    c = np.arctan(rou)
    sin_c = np.sin(c)
    cos_c = np.cos(c)

    rot_y = np.deg2rad(rot_y)
    rot_z = np.deg2rad(rot_z)
    lat = np.arcsin(cos_c * np.sin(rot_y) + (y * sin_c * np.cos(rot_y)) / rou)
    lon = rot_z + np.arctan2(x * sin_c, rou * np.cos(rot_y) * cos_c - y * np.sin(rot_y) * sin_c)

    return [lon, lat] # 横(右が＋)、縦(上が＋) (radian)

if __name__ == '__main__':
    image = './inputOmni/R0010182.JPG'
    sphere = 'R0010182'
    rot_z = [372, 485]
    rot_y = [-10, 2]
    fov = 60
    pers_wh = 640
    eq_w = 5376
    img = cv2.imread(image)
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
                    x = float(tmp[1])
                    y = float(tmp[2])
                    lon, lat = get_angle([x,y], rot_z[i], rot_y[i], 1, 1, fov)
                    x = lon2eq(lon)
                    y = lat2eq(lat)
                    print(f'lon={lon},lat={lat}x={x},y={y}')
                    cv2.drawMarker(img, (int(x), int(y)), (0, 75, 255), markerSize=40, thickness=7) # roiの中心
                    cv2.imshow("step", img)
                    cv2.waitKey(0)
        except FileNotFoundError as e:
            print(e)
            continue