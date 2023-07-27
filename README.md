PointAnywhere
====
This code performs directed object estimation from omnidirectional images. [日本語版](README_ja.md)
![The equirectangular image of a user pointing to a white car outdoors.](Experiment/result/testResult/svnS/R0010498_top5.jpg "A Successful example")

Nanami Kotani and Asako Kanezaki. 2023. Point Anywhere: Directed Object Estimation from Omnidirectional Images. In ACM SIGGRAPH 2023 Posters (SIGGRAPH '23). Association for Computing Machinery, New York, NY, USA, Article 33, 1–2. https://doi.org/10.1145/3588028.3603650
* [ACM showcase on Kudos](https://link.growkudos.com/1cvv7ucfim8)

## Installation and Requirement
* Skeltal detection: [OpenPose](https://github.com/Hzzone/pytorch-openpose)
* Object detection: [YOLOv5](https://github.com/ultralytics/yolov5)
* SVC: scikit-learn

An environment in which all of these things work is required. The environment can be built with the following commands.
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

You will also need to download the OpenPose and YOLOv5 models. The entire code is included in this repository and does not need to be cloned.

## Usage
```
python run.py
```
About images in "testResult"
* distance: Candidates for all pointing objects
* shortest: Distance between object area center and pointing vector (uncorrected)
* number: Correction based on frequency of occurrence of objects
* numberConfi: Correction based on object appearance frequency and object detection confidence
* svmN: Normalize data with linear SVC
* svmNrbf: Normalize data with SVC in rbf kernel
* svmS: Standardize data with linear SVC
* svmSrbf: Standardize data with SVC in rbf kernel

Results are stored in "Experiment/resultN" (N is a number).
The top-k accuracy is displayed in "Experiment/resultN/testResult/result.txt".

## Dataset
[You can download dataset here.](https://drive.google.com/drive/folders/17BXn-vFv390EeBbiVqhUBWeIOnqt3th0)

Here, the name of the equirectangular image is "sphere.jpg".
* image.zip: 290 equirectangular images
    * Some parts have been mosaicked or blacked out to protect personal information.
    * Contains image/sphere.jpg
* ROI.zip: Object area of the ground truth pointing object
    * Contains ROI/sphere.txt
    * The notation format for object area is (x,y,w,h), the same as in [YOLOv5](https://github.com/ultralytics/yolov5).
* skeleton.zip: Results of human detection and skeletal estimation on the original equirectangular image before mosaicing and blackening process.
    * skeleton/sphere/labels/sphere.txt: Object area resulting from person detection by applying YOLOv5 with the equirectangular image
        * The notation format for object area is the same as in [YOLOv5](https://github.com/ultralytics/yolov5).
        * If YOLOv5 fails to detect a person, this txt file does not exist.
    * skeleton/sphere/human.txt: Variable when only the area around the person is used as a perspective image.
        * The notation format is (fov_person, theta_person, phi_person, h_perspective_person, w_perspective_person).
        * The value is calculated based on "labels/sphere.txt".
    * skeleton/sphere/human.npz
        * candidate, subset: Joints estimated by OpenPose
        * See [OpenPose](https://github.com/Hzzone/pytorch-openpose) for notation format.

Note that the values in ROI.zip are the correct answers, whereas the values in skeleton.zip are the estimated results.

### How to use the dataset
```
python run.py -input ../dataset/image  -skelton ../dataset/skeleton
```
