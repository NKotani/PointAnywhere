import RegionOfInterst as ROI
import os
import Test
from pathlib import Path
from ml import Data as mlData
from ml import Svm as mlsvm

IOU_THRESHOLD = 0.5

# 距離が近い順に判定
def in_order_of_distance(data:list, sphere:str, threshold=IOU_THRESHOLD):
    result = 0
    with open(sphere, 'r', encoding='utf-8') as f:
        roi = f.readline().rstrip('\n') # (x, y, w, h)(改行を除く)
        roi = tuple(map(int, roi.strip('()').split(',')))
        print(f'{sphere}{roi}')
    i, iou = 0, 0
    for obj in data: # (54.2771666751422, ['bowl', [2593, 1686, 77, 67], 0.859447])
        i += 1
        iou = ROI.IoU(roi, obj[1][1])
        if iou >= threshold: # 正解
            result = i
            break
    return result, iou # 1-5:正解 0:正解なし

# 検出された物体の個数に基づいて補正
def regularized_based_number_of_objects(data:list, sphere:str, threshold=IOU_THRESHOLD):
    result = 0
    labels = {}
    with open(sphere, 'r', encoding='utf-8') as f:
        roi = f.readline().rstrip('\n') # (x, y, w, h)(改行を除く)
        roi = tuple(map(int, roi.strip('()').split(',')))
        # print(f'{sphere}{roi}')
    
    # ラベルの登場回数をカウント
    for obj in data: # (54.2771666751422, ['bowl', [2593, 1686, 77, 67], 0.859447])
        if obj[1][0] in labels:
            labels[obj[1][0]] += 1
        else:
            labels[obj[1][0]] = 1 # 初期化

    # 登場回数に基づいて距離を補正
    new_data = {}
    for obj in data: # (54.2771666751422, ['bowl', [2593, 1686, 77, 67], 0.859447])
        distance = obj[0] * labels[obj[1][0]] / len(data) # 距離*そのラベルの登場回数/総物体数
        new_data[distance] = obj[1]
    new_data = sorted(new_data.items())

    i, iou = 0, 0
    for obj in new_data: # (54.2771666751422, ['bowl', [2593, 1686, 77, 67], 0.859447])
        i += 1
        iou = ROI.IoU(roi, obj[1][1])
        if iou >= threshold: # 正解
            result = i
            break
    return result, iou, new_data # 1-5:正解 0:正解なし

# 検出された物体の個数に基づいて補正(confifdence低い1個の物体は3個あることにして重要度下げる) 正解平均confidenceが0.68
def regularized_based_number_of_objects_confi(data:list, sphere:str, threshold=IOU_THRESHOLD, confidence=0.6):
    result = 0
    labels = {} # key: label, value:個数
    with open(sphere, 'r', encoding='utf-8') as f:
        roi = f.readline().rstrip('\n') # (x, y, w, h)(改行を除く)
        roi = tuple(map(int, roi.strip('()').split(',')))
    
    # ラベルの登場回数をカウント
    for obj in data: # (54.2771666751422, ['bowl', [2593, 1686, 77, 67], 0.859447])
        if obj[1][0] in labels:
            labels[obj[1][0]] += 1
        else:
            labels[obj[1][0]] = 1 # 初期化
    
    # 登場回数に基づいて距離を補正
    new_data = {}
    for obj in data: # (54.2771666751422, ['bowl', [2593, 1686, 77, 67], 0.859447])
        distance = obj[0] * labels[obj[1][0]] / len(data) # 距離*そのラベルの登場回数/総物体数
        if labels[obj[1][0]] == 1 and obj[1][2] < confidence:
            distance *= 3 # confidenceが低い1個の物体は3個あることにする
        new_data[distance] = obj[1]
    new_data = sorted(new_data.items())

    i, iou = 0, 0
    for obj in new_data: # (54.2771666751422, ['bowl', [2593, 1686, 77, 67], 0.859447])
        i += 1
        iou = ROI.IoU(roi, obj[1][1])
        if iou >= threshold: # 正解
            result = i
            break # 上位5個以内に正解するとskip_dataの数が5個以下になる
    return result, iou, new_data # 1-5:正解 0:正解なし

# 同じ物体を1つしか選ばない
def only_one_of_same_object_select(data:list, sphere:str, threshold=IOU_THRESHOLD):
    result = 0
    labels = [] # 既にでたラベルをカウント
    with open(sphere, 'r', encoding='utf-8') as f:
        roi = f.readline().rstrip('\n') # (x, y, w, h)(改行を除く)
        roi = tuple(map(int, roi.strip('()').split(',')))
    
    # 同じ物体が既に出て来てたら距離+10000
    new_data = {}
    for obj in data: # (54.2771666751422, ['bowl', [2593, 1686, 77, 67], 0.859447])
        distance = obj[0]
        if obj[1][0] in labels:
            distance += 10000
        else:
            labels.append(obj[1][0])
        new_data[distance] = obj[1]
    new_data = sorted(new_data.items())

    i, iou = 0, 0
    for obj in new_data: # (54.2771666751422, ['bowl', [2593, 1686, 77, 67], 0.859447])
        i += 1
        iou = ROI.IoU(roi, obj[1][1])
        if iou >= threshold: # 正解
            result = i
            break
    return result, iou, new_data # 1-5:正解 0:正解なし

# 面積補正割り算版
# 距離が近い順に判定
def in_order_of_distanceAREA(data:list, sphere:str, threshold=IOU_THRESHOLD):
    result = 0
    with open(sphere, 'r', encoding='utf-8') as f:
        roi = f.readline().rstrip('\n') # (x, y, w, h)(改行を除く)
        roi = tuple(map(int, roi.strip('()').split(',')))

    # 面積に基づいて距離を補正
    area_data = regularized_area(data)

    i, iou = 0, 0
    for obj in area_data: # (54.2771666751422, ['bowl', [2593, 1686, 77, 67], 0.859447])
        i += 1
        iou = ROI.IoU(roi, obj[1][1])
        if iou >= threshold: # 正解
            result = i
            break
    return result, iou, area_data # 1-5:正解 0:正解なし

# 検出された物体の個数に基づいて補正
def regularized_based_number_of_objectsAREA(data:list, sphere:str, threshold=IOU_THRESHOLD):
    result = 0
    labels = {}
    with open(sphere, 'r', encoding='utf-8') as f:
        roi = f.readline().rstrip('\n') # (x, y, w, h)(改行を除く)
        roi = tuple(map(int, roi.strip('()').split(',')))
    
    # ラベルの登場回数をカウント
    for obj in data: # (54.2771666751422, ['bowl', [2593, 1686, 77, 67], 0.859447])
        if obj[1][0] in labels:
            labels[obj[1][0]] += 1
        else:
            labels[obj[1][0]] = 1 # 初期化

    # 登場回数に基づいて距離を補正
    new_data = {}
    for obj in data: # (54.2771666751422, ['bowl', [2593, 1686, 77, 67], 0.859447])
        distance = obj[0] * labels[obj[1][0]] / len(data) # 距離*そのラベルの登場回数/総物体数
        new_data[distance] = obj[1]
    new_data = sorted(new_data.items())

    # 面積に基づいて距離を補正
    new_data = regularized_area(new_data)

    i, iou = 0, 0
    for obj in new_data: # (54.2771666751422, ['bowl', [2593, 1686, 77, 67], 0.859447])
        i += 1
        iou = ROI.IoU(roi, obj[1][1])
        if iou >= threshold: # 正解
            result = i
            break
    return result, iou, new_data # 1-5:正解 0:正解なし

# 検出された物体の個数に基づいて補正
def regularized_based_number_of_objects_confiAREA(data:list, sphere:str, threshold=IOU_THRESHOLD, confidence=0.5):
    result = 0
    labels = {} # key: label, value:個数
    with open(sphere, 'r', encoding='utf-8') as f:
        roi = f.readline().rstrip('\n') # (x, y, w, h)(改行を除く)
        roi = tuple(map(int, roi.strip('()').split(',')))
    
    # ラベルの登場回数をカウント
    for obj in data: # (54.2771666751422, ['bowl', [2593, 1686, 77, 67], 0.859447])
        if obj[1][0] in labels:
            labels[obj[1][0]] += 1
        else:
            labels[obj[1][0]] = 1 # 初期化
    
    # 登場回数に基づいて距離を補正
    new_data = {}
    for obj in data: # (54.2771666751422, ['bowl', [2593, 1686, 77, 67], 0.859447])
        distance = obj[0] * labels[obj[1][0]] / len(data) # 距離*そのラベルの登場回数/総物体数
        if labels[obj[1][0]] == 1 and obj[1][2] < confidence:
            distance *= 3 # confidenceが低い1個の物体は3個あることにする
        new_data[distance] = obj[1]
    new_data = sorted(new_data.items())

    # 面積に基づいて距離を補正
    new_data = regularized_area(new_data)

    i, iou = 0, 0
    for obj in new_data: # (54.2771666751422, ['bowl', [2593, 1686, 77, 67], 0.859447])
        i += 1
        iou = ROI.IoU(roi, obj[1][1])
        if iou >= threshold: # 正解
            result = i
            break
    return result, iou, new_data # 1-5:正解 0:正解なし

# 同じ物体を1つしか選ばない
def only_one_of_same_object_selectAREA(data:list, sphere:str, threshold=IOU_THRESHOLD):    
    result = 0
    labels = [] # 既にでたラベルをカウント
    with open(sphere, 'r', encoding='utf-8') as f:
        roi = f.readline().rstrip('\n') # (x, y, w, h)(改行を除く)
        roi = tuple(map(int, roi.strip('()').split(',')))
    
    # 同じ物体が既に出て来てたら距離+10000
    new_data = {}
    for obj in data: # (54.2771666751422, ['bowl', [2593, 1686, 77, 67], 0.859447])
        distance = obj[0]
        if obj[1][0] in labels:
            distance += 10000
        else:
            labels.append(obj[1][0])
        new_data[distance] = obj[1]
    new_data = sorted(new_data.items())

     # 面積に基づいて距離を補正
    area_data = regularized_area(new_data)

    i, iou = 0, 0
    for obj in area_data: # (54.2771666751422, ['bowl', [2593, 1686, 77, 67], 0.859447])
        i += 1
        iou = ROI.IoU(roi, obj[1][1])
        if iou >= threshold: # 正解
            result = i
            break
    return result, iou, area_data # 1-5:正解 0:正解なし

# 面積を計算 roi:x,y,w,h
def cal_area(roi:list):
    if roi[2] <= 0:
        roi[2] = 1
    if roi[3] <= 0:
        roi[2] = 1
    return roi[2]*roi[3]

# 面積に基づいて距離を補正したリスト返す(割り算版)
def regularized_area(data:list):
    new_data = {}
    for obj in data: # (54.2771666751422, ['bowl', [2593, 1686, 77, 67], 0.859447])
        distance = obj[0] / cal_area(obj[1][1]) # 距離*そのラベルの登場回数/総物体数
        new_data[distance] = obj[1]
    return sorted(new_data.items())

def svm(data:list, sphere:str, threshold=IOU_THRESHOLD):
    result = 0
    with open(sphere, 'r', encoding='utf-8') as f:
        roi = f.readline().rstrip('\n') # (x, y, w, h)(改行を除く)
        roi = tuple(map(int, roi.strip('()').split(',')))
    i, iou = 0, 0
    for obj in data: # (-67.7724263084868, ['mouse', [2262, 1586, 42, 30], 0.841083])
        i += 1
        iou = ROI.IoU(roi, obj[1][1])
        if iou >= threshold: # 正解
            result = i
            break
    return result, iou, data # 1-5:正解 0:正解なし

# Top_5 Accuracyを計算
def cal_top1_5_10(score='Result/great/testResult/score.txt', path='testResult', dataset=125):
    order = {} # key:順位, value:個数
    top1 = 0
    top5 = 0 # top5 accuracy
    top10 = 0
    direction_right = 0 # 方向性は合ってたものの個数
    # min_confidence = 100 # 最低confidence

    scores = []
    with open(score, 'r', encoding='utf-8') as f:
        scores = f.read().splitlines() # ['2 0.6906746031746032 R0010074', ..]

    scores = [s.split() for s in scores] # [['2', '0.6906746031746032', 'R0010074'], 
    for s in scores:
        if int(s[0]) in order:
            order[int(s[0])] += 1
        else:
            order[int(s[0])] = 1 # 初期化

        if int(s[0]) == 1 :
            top1 += 1

        if int(s[0]) >= 1 and int(s[0]) <= 5:
            top5 += 1
        elif int(s[0]) > 5:
            direction_right += 1

        if int(s[0]) >= 1 and int(s[0]) <= 10:
            top10 += 1
        
        # if int(s[0]) > 0:
        #     # 正解か方向性が合っている時
        #     min_confidence = min(float(s[3]), min_confidence)

    # result = result / len(scores)

    file = os.path.join(path, 'result.txt')
    with open(file, 'a', encoding='utf-8', newline='\n') as f:
        f.writelines(score)
        f.write('\n')
        f.writelines(f'top-1 accuracy = {round(top1/dataset, 2)} 個数{top1}\n')
        f.writelines(f'top-5 accuracy = {round(top5/dataset, 2)} 個数{top5}\n')
        f.writelines(f'top-10 accuracy = {round(top10/dataset, 2)} 個数{top10}\n')
        f.writelines(f'方向性は合ってる(top5) = {direction_right}\n')
        f.writelines(str(order))
        f.write('\n')
        # f.writelines(f'最低confidence = {min_confidence}\n')
    return

# Top-5accuracyをすべての選定手法で適用 dataset:データセット数
def cal_top1_5_10ALL(testResult_f='Result/great/testResult', dataset=125):
    score_file = os.path.join(testResult_f, 'score.txt')
    cal_top1_5_10(score_file, testResult_f, dataset)
    score_file = os.path.join(testResult_f, 'score_number.txt')
    cal_top1_5_10(score_file, testResult_f, dataset)
    score_file = os.path.join(testResult_f, 'score_confi.txt')
    cal_top1_5_10(score_file, testResult_f, dataset)
    # svm
    # 正規化
    score_file = os.path.join(testResult_f, 'score_svmnorm.txt') # 
    cal_top1_5_10(score_file, testResult_f, dataset)
    score_file = os.path.join(testResult_f, 'score_svmnormrbf.txt') # (rbfカーネル)
    cal_top1_5_10(score_file, testResult_f, dataset)
    # 標準化
    score_file = os.path.join(testResult_f, 'score_svmstd.txt') # 
    cal_top1_5_10(score_file, testResult_f, dataset)
    score_file = os.path.join(testResult_f, 'score_svmstdrbf.txt') # (rbfカーネル)
    cal_top1_5_10(score_file, testResult_f, dataset)
    return

# すべての選定手法を実行
def run(result:list, ground_truth='../../dataset/ROI/R0010066.txt', stdout='22021231p1234', testResult_f='Result/great/testResult', sphere='ROO10066', testMODE=1, test_file='test_result/vec/R00100.jpg', theta=0):
    score, iou = in_order_of_distance(result, ground_truth)
    Test.write_stdout(f'上位{score}個目が正解で、iou={iou}', stdout, testResult_f)
    try:
        Test.write_stdout(f'{score} {iou} {sphere} {result[score-1][1][2]}', 'score', testResult_f) # 4個目はconfidence
    except IndexError:
        Test.write_stdout(f'{score} {iou} {sphere} {100}', 'score', testResult_f)
    score, iou, num_result = regularized_based_number_of_objects(result, ground_truth) # 物体の個数で補正
    # Test.write_stdout(f'上位{score}個目が正解で、iou={iou}', stdout, testResult_f)
    try:
        Test.write_stdout(f'{score} {iou} {sphere} {num_result[score-1][1][2]}', 'score_number', testResult_f)
    except IndexError:
        Test.write_stdout(f'{score} {iou} {sphere} {100}', 'score_number', testResult_f)
    score, iou, confi_result = regularized_based_number_of_objects_confi(result, ground_truth) # 物体の個数で補正(confidence低いのはスキップする版)
    # Test.write_stdout(f'上位{score}個目が正解で、iou={iou}', stdout, testResult_f)
    try:
        Test.write_stdout(f'{score} {iou} {sphere} {confi_result[score-1][1][2]}', 'score_confi', testResult_f)
    except IndexError:
         Test.write_stdout(f'{score} {iou} {sphere} {100}', 'score_confi', testResult_f)
    # svm
    dataset = 'dataset290'
    svm_data, obj_name = mlData.cal(result, theta) # [[3.9602220990045156, 0.08333333333333333, 0.343917, 0.08352847660943236, 54, ['cat', [2983, 1225, 1, 54], 0.343917]], ...]
    # 正規化
    svm_result0norm = mlsvm.estimate(svm_data, obj_name, 'ml/'+dataset+'norm.pickle', 1) # [[3.119450434853303 'tv' list([4788, 1626, 74, 123]) 0.847969]...]
    score, iou, svm_result0norm = svm(svm_result0norm, ground_truth) # [(3.119450434853303, ['tv', [4788, 1626, 74, 123], 0.847969])...]
    Test.write_stdout(f'{score} {iou} {sphere}', 'score_svmnorm', testResult_f)
    svm_result0normrbf = mlsvm.estimate(svm_data, obj_name, 'ml/'+dataset+'normrbf.pickle', 1) # rbfカーネル全学習版
    score, iou, svm_result0normrbf = svm(svm_result0normrbf, ground_truth)
    Test.write_stdout(f'{score} {iou} {sphere}', 'score_svmnormrbf', testResult_f)
    # 標準化
    svm_result0std = mlsvm.estimate(svm_data, obj_name, 'ml/'+dataset+'std.pickle', 2) # [[3.119450434853303 'tv' list([4788, 1626, 74, 123]) 0.847969]...]
    score, iou, svm_result0std = svm(svm_result0std, ground_truth) # [(3.119450434853303, ['tv', [4788, 1626, 74, 123], 0.847969])...]
    Test.write_stdout(f'{score} {iou} {sphere}', 'score_svmstd', testResult_f)
    svm_result0stdrbf = mlsvm.estimate(svm_data, obj_name, 'ml/'+dataset+'stdrbf.pickle', 2) # rbfカーネル全学習版
    score, iou, svm_result0stdrbf = svm(svm_result0stdrbf, ground_truth)
    Test.write_stdout(f'{score} {iou} {sphere}', 'score_svmstdrbf', testResult_f)

    if  testMODE == 0: # 定性評価のとき
        testResult_shortest_f = Path(os.path.join(testResult_f, 'shortest')) # 距離の短い順を格納
        testResult_number_f = Path(os.path.join(testResult_f, 'number')) # 物体の個数に基づく補正
        testResult_numberConfi_f = Path(os.path.join(testResult_f, 'numberConfi')) # 物体の個数に基づく補正(confidence低いのは3倍)
        # svm
        # 正規化
        testResult_svmnorm_f = Path(os.path.join(testResult_f, 'svmN')) # 
        testResult_svmnormrbf_f = Path(os.path.join(testResult_f, 'svmNrbf')) # (rbfカーネル)
        # 標準化
        testResult_svmstd_f = Path(os.path.join(testResult_f, 'svmS')) # 
        testResult_svmstdrbf_f = Path(os.path.join(testResult_f, 'svmSrbf')) # (rbfカーネル)

        (testResult_shortest_f).mkdir(parents=True, exist_ok=True)  # make dir
        (testResult_number_f).mkdir(parents=True, exist_ok=True)
        (testResult_numberConfi_f).mkdir(parents=True, exist_ok=True)
        (testResult_svmnorm_f).mkdir(parents=True, exist_ok=True)
        (testResult_svmnormrbf_f).mkdir(parents=True, exist_ok=True)
        (testResult_svmstd_f).mkdir(parents=True, exist_ok=True)
        (testResult_svmstdrbf_f).mkdir(parents=True, exist_ok=True)
        
        Test.roi2imgTop5(result[:5], test_file, testResult_shortest_f) # 指差しベクトル自体はdistance_circleでかく
        Test.roi2imgTop5(num_result[:5], test_file, testResult_number_f) # 物体の個数で補正
        Test.roi2imgTop5(confi_result[:5], test_file, testResult_numberConfi_f) # 物体の個数で補正(confidence低いのは3倍)
        # svm正規化
        Test.roi2imgTop5(svm_result0norm[:5], test_file, testResult_svmnorm_f) # (線形)
        Test.roi2imgTop5(svm_result0normrbf[:5], test_file, testResult_svmnormrbf_f) # (rbf)
        # svm標準化
        Test.roi2imgTop5(svm_result0std[:5], test_file, testResult_svmstd_f) # (線形)
        Test.roi2imgTop5(svm_result0stdrbf[:5], test_file, testResult_svmstdrbf_f) # (rbf)

        # Top1
        Test.roi2imgTop1(result[:5], test_file, testResult_shortest_f) # 指差しベクトル自体はdistance_circleでかく
    return
