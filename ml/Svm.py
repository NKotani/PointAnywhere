import numpy as np
import sklearn.svm as svm
from sklearn.preprocessing import MinMaxScaler # 正規化
from sklearn.preprocessing import StandardScaler # 標準化
import pickle # モデルの保存

def make_dataset(dataset_file='inputOmni.txt', liner=1, train=['R0010066','R0010068','R0010074','R0010304'], fig_path='ground_truth2', sc='none'):
    # データの読み込み
    # [sphere, この画像に正解は存在するか？, この物体は正解か？, 指示ベクトル距離, 希少度, 信頼度, 人距離, 面積]
    with open(dataset_file, 'r') as f:
        data = f.read().splitlines() # ['R0010074 1 0 21.185607273702335 0.391304347826087 0.510216 0.4612904231819535 31992', 'R0010074 1 1 37.77999999999972 0.043478260869565216 0.866853 0.26064539711372137 3481', ...]
    
    data = [d.split() for d in data] # [['R0010074', '1', '0', '21.185607273702335', '0.391304347826087', '0.510216', '0.4612904231819535', '31992'],]

    # データをテスト用と学習用に分ける
    train_data = []
    train_image_num = [] # スキップしないで使った枚数(YOLO人検出に失敗している画像を除く)
    test_data = []
    for d in data:
        if d[0] in train:
            train_data.append(d)
            if (d[0] in train_image_num) == False:
                train_image_num.append(d[0]) # 初めてその画像名が出てきたときのみ追加
        else:
            test_data.append(d)
    
    print(f'学習に使った画像数は{len(train_image_num)}枚。学習事例数は{len(train_data)}、テスト事例数は{len(test_data)}')

    # 説明変数X: 指示ベクトル距離, 出現頻度, 信頼度, 人距離, 面積
    train_data = np.array(train_data)
    Xtrain = train_data[:, 3:] # [['21.185607273702335' '0.391304347826087' '0.510216' '0.4612904231819535' '31992'] ...]
    Xtrain = Xtrain.astype(float) # [[2.11856073e+01 3.91304348e-01 5.10216000e-01 4.61290423e-01 3.19920000e+04] ...]
    Ytrain = train_data[:, 2].astype(int) # [0 1 0 0 0 0 0 0 ....]

    test_data = np.array(test_data)
    Xtest = test_data[:, 3:] # [['21.185607273702335' '0.391304347826087' '0.510216' '0.4612904231819535' '31992'] ...]
    Xtest = Xtest.astype(float) # [[2.11856073e+01 3.91304348e-01 5.10216000e-01 4.61290423e-01 3.19920000e+04] ...]
    Ytest = test_data[:, 2].astype(int) # [0 1 0 0 0 0 0 0 ....]

    if sc == 'norm':
        scaler = MinMaxScaler([0,1]) # 0~1の正規化
        Xtrain = scaler.fit_transform(Xtrain)
        Xtest = scaler.fit_transform(Xtest)
    elif sc == 'std':
        scaler = StandardScaler() # 平均0、標準偏差1
        Xtrain = scaler.fit_transform(Xtrain)
        Xtest = scaler.fit_transform(Xtest)
    
    np.nan_to_num(Xtrain, copy=False) # NaNを0に変換
    np.nan_to_num(Xtest, copy=False) # NaNを0に変換
    return Xtrain, Ytrain, Xtest, Ytest
    
def train(Xtrain, Ytrain, liner=1, model_file='model.pickle', fig_path='support'):
    # 線形識別
    model = svm.SVC(kernel='linear')
    if liner != 1:
        model = svm.SVC(kernel='rbf')
    model.fit(Xtrain, Ytrain) # サポートベクトルマシーンに学習させる
    #学習モデルの保存
    with open(model_file, mode='wb') as f:
        pickle.dump(model, f, protocol=2)
    
    if liner == 1:
        # 学習によって求められたwとbの確認
        print(f'model.coef_[0] {model.coef_[0]}') # [-0.0711648  -0.74056212  2.48978766  0.31786319 -0.02336892]
        print(f'model.intercept_[0] {model.intercept_[0]}') # 3.8420832052798746
        fig, ax = UG.init_graph(Xtrain[:, 0:2], Ytrain, figsize=(7, 6))
        UG.plot_data(ax, Xtrain[:, 0:2], Ytrain)
        UG.plot_support_vectors(ax, Xtrain[:, 0:2], Ytrain, model.support_, s=100)
        UG.draw_line(ax, model.coef_[0][0], model.coef_[0][1], model.intercept_[0], '-', 'wx+b') # 2次元データじゃないので正しくないかも
        # plt.show()
        fig.savefig('ml/figTrain/'+fig_path+'.png')
    return

# SVMの性能の評価(機械学習的な評価)
def eval(Xtest, Ytest, model_file='model.pickle'):
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    # モデルのオープン
    with open(model_file, mode='rb') as f:
        model = pickle.load(f)
    Ytest_pred = model.predict(Xtest)
    cmatrix = confusion_matrix(Ytest, Ytest_pred) # Ytestは真の値,  Ytest_predは予測値
    print('混同行列')
    print(cmatrix) # 混同行列 [[tn fp][fn tp]]
    accuracy = model.score(Xtest, Ytest) # 正解率
    ac = accuracy_score(Ytest, Ytest_pred) # 正解率
    precision = precision_score(Ytest, Ytest_pred, zero_division=0) # 適合率[0.95695364 0.]
    recall = recall_score(Ytest, Ytest_pred) # 再現率
    f1 = f1_score(Ytest, Ytest_pred) # F1スコア
    print(f'正解率{round(accuracy,5)}={round(ac,3)}, 適合率{round(precision,3)}, 再現率{round(recall,3)}, F1スコア{round(f1,3)}')
    # score = model.decision_function(Xtest).tolist() # rbfのとき[-1.0003946922658549, -1.0002101327914756
    return

# 確信度を計算して、確信度の高い順に物体をソート(指示物体推定)
# data=[指示ベクトル距離, 希少度, 信頼度, 人距離, 面積]のnp.array, obj_name=['label', [x,y,w,h], confidence]
def estimate(data, obj_name:list, model_file='./ml/model.pickle', sc=0):
    with open(model_file, mode='rb') as f:
        model = pickle.load(f)
    # data = np.array(data) # [[2.11856073e+01 3.91304348e-01 5.10216000e-01 4.61290423e-01 3.19920000e+04] ...]
    score = []
    if data.size != 0: # 空配列で1次元になってないとき
        if sc == 1:
            scaler = MinMaxScaler([0,1]) # 0~1の正規化
            data = scaler.fit_transform(data) # [[0.00000000e+00 1.00000000e+00 3.53377647e-01 2.67628556e-01 1.06046531e-01]...]
        elif sc == 2:
            scaler = StandardScaler() # 平均0、標準偏差1
            data = scaler.fit_transform(data) # [[-1.52941525  1.2282132  -0.40772556 -0.64662884 -0.46873461]...]
        np.nan_to_num(data, copy=False) # NaNを0に変換
        score = model.decision_function(data).tolist() # [ 3.11945043, -70.84610062, .... -520.35225726]
    result = dict(zip(score, obj_name)) # {-1543.3319085998276: ['chair', [2523, 1439, 172, 186], 0.510216]...}
    result = sorted(result.items(), reverse=True) # 降順にソート [(-67.7724263084868, ['mouse', [2262, 1586, 42, 30], 0.841083]),...]
    return result