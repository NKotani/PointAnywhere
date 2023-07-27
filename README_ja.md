PointAnywhere
====
全天球画像からの指示物体推定を行うコードです。英語の説明は[こちら](README.md)。
![The equirectangular image of a user pointing to a white car outdoors.](Experiment/result/testResult/svmS/R0010498_top5.jpg "A Successful example")

Nanami Kotani and Asako Kanezaki. 2023. Point Anywhere: Directed Object Estimation from Omnidirectional Images. In ACM SIGGRAPH 2023 Posters (SIGGRAPH '23). Association for Computing Machinery, New York, NY, USA, Article 33, 1–2. https://doi.org/10.1145/3588028.3603650
* [ACM showcase on Kudos](https://link.growkudos.com/1cvv7ucfim8)

## Installation and Requirement
* 骨格検出: [OpenPose](https://github.com/Hzzone/pytorch-openpose)
* 物体検出: [YOLOv5](https://github.com/ultralytics/yolov5)
* SVC: scikit-learn

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

OpenPoseやYOLOv5のモデルもダウンロードする必要があります。全体のコードはこのレポジトリに含まれているのでクローンする必要はありません。

## Usage
```
python run.py
```
testResult内の画像について
* distance: 全指示物体候補
* shortest: 物体領域中心と指示ベクトルとの距離(補正なし)
* number: 物体の出現頻度に基づく補正
* numberConfi: 物体の出現頻度に基づく補正 + 物体検出の信頼度に基づく補正
* svmN: 線形SVCでデータを正規化
* svmNrbf: rbfカーネルのSVCでデータを正規化
* svmS: 線形SVCでデータを標準化
* svmSrbf: rbfカーネルのSVCでデータを標準化

結果はExperiment/resultN (Nは数字)に保存されます。
Experiment/resultN/testResult/result.txt にtop-k accuracyが表示されます。

## Dataset
[こちらからダウンロードできます。](https://drive.google.com/drive/folders/17BXn-vFv390EeBbiVqhUBWeIOnqt3th0)

ここでは、全天球画像名をsphere.jpgとして説明します。
* image.zip: 全天球画像290枚
    * 個人情報保護のため一部モザイクや黒塗り処理をしています。
    * image/sphere.jpgが含まれています
* ROI.zip: 正解の指示物体の物体領域
    * ROI/sphere.txtが含まれています
    * 物体領域の表記形式は[YOLOv5](https://github.com/ultralytics/yolov5)と同じ(x,y,w,h)です。
* skeleton.zip: モザイクや黒塗り処理する前の元の全天球画像で、人検出・骨格推定をした結果
    * skeleton/sphere/labels/sphere.txt: 全天球画像のままYOLOv5を適用して、人検出した結果の物体領域
        * 表記形式は[YOLOv5](https://github.com/ultralytics/yolov5)と同じです。
        * YOLOv5で人を検出できなかった場合はこのtxtファイルは存在しません。
    * skeleton/sphere/human.txt: 人周辺領域のみをperspective画像にしたときの変数
        * 表記形式は(fov_person, theta_person, phi_person, h_perspective_person, w_perspective_person)です。
        * labels/sphere.txtを元に算出される値です。
    * skeleton/sphere/human.npz
        * candidate, subset: OpenPoseによって推定された関節
        * 表記形式は[OpenPose](https://github.com/Hzzone/pytorch-openpose)を参照してください。

ROI.zipに含まれる値は正解なのに対し、skeleton.zipに含まれる値は推定結果であることに注意してください。

### How to use the dataset
```
python run.py -input ../dataset/image  -skelton ../dataset/skeleton
```
