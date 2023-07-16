# 全天球画像からの指示物体推定
## 環境構築
* 骨格検出: [OpenPose](https://github.com/Hzzone/pytorch-openpose)
* 物体検出: [YOLOv5](https://github.com/ultralytics/yolov5)
* 線形SVC: scikit-learn

これらの全てが動く環境が必要です。下記のコマンドで環境構築できます。
```
conda create -n hogehoge python=3.7
conda activate hogehoge
cd pytorch-openpose
pip install -r requirements.txt # openpose
conda install pytorch
pip install torchvision
cd ..
pip install -r requirements.txt # yolo
conda install -c anaconda scikit-learn
```

骨格検出や物体検出のモデルもダウンロードする必要があります。全体のコードはこのレポジトリに含まれているのでクローンする必要はありません。

## 実行方法
定性評価(画像が生成されます)
```
cd pytorch-openpose
python run.py inputOmni 0
```
testResult内の画像について
* distance: 全指示物体候補
* shortest: 物体領域中心と指示ベクトルとの距離(補正なし)
* number: 物体の出現頻度に基づく補正
* numberConfi: 物体の出現頻度に基づく補正 + 物体検出の信頼度に基づく補正
* only: 同じカテゴリの物体が1度しか選ばればいようにする補正
* shortestAREA: 物体領域の面積に基づく補正
* svm: 線形SVC
* svmrbf: rbfカーネルのSVC
* svmN: 線形SVCでデータを正規化
* svmBrbf: rbfカーネルのSVCでデータを正規化
* svm: 線形SVCでデータを標準化
* svmrbf: rbfカーネルのSVCでデータを標準化

定量評価(画像は生成されません)
```
cd pytorch-openpose
python run.py inputOmni 1
```

結果はExperiment/resultN (Nは数字)に保存されます。
Experiment/resultN/testResult/result.txt にtop-k accuracyが表示されます。

## データセット
[こちらからダウンロードできます](https://drive.google.com/drive/folders/17BXn-vFv390EeBbiVqhUBWeIOnqt3th0)

ここでは、全天球画像名をsphere.jpgとして説明します。
* image.zip: 全天球画像290枚
    * 個人情報保護のため一部モザイクや黒塗り処理をしています。
    * image/sphere.jpgが含まれています
* ROI.zip: 正解の指示物体の物体領域
    * ROI/sphere.txtが含まれています
    * 物体領域の表記形式は[YOLOv5](https://github.com/ultralytics/yolov5)と同じ(x,y,w,h)です。
* skeleton.zip: モザイクや黒塗り処理する前の元の全天球画像で、人検出・骨格推定をした結果
    * skeleton/sphere/labels/sphere.txt: 全天球画像のままYOLOv5を適用して、人検出した結果の物体領域
        * 表記形式は[YOLOv5](https://github.com/ultralytics/yolov5)を参照してください。
        * YOLOv5で人を検出できなかった場合はこのtxtファイルは存在しません。
    * skeleton/sphere/human.txt: 人周辺領域のみをperspective画像にしたときの変数
        * 表記形式は(fov_person, theta_person, phi_person, h_perspective_person, w_perspective_person)です。
    * skeleton/sphere/human.npz    
        * candidate, subset: OpenPoseによって推定された関節
        * 表記形式は[OpenPose](https://github.com/Hzzone/pytorch-openpose)を参照してください。

ROI.zipに含まれる値は正解なのに対し、skeleton.zipに含まれる値は推定結果であることに注意してください。
