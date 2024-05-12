# RED_FEPSY

### Dataset
* RAF-DB : /nas_homes/jihyun/RAF_DB/
* AffectNet : /nas_homes/jihyun/datasets/AffectNet/
* FERPlus : /nas_homes/jihyun/datasets/FERPlus/

### Pretrained backbone model

Download the pretrained [ResNet-50](https://drive.google.com/file/d/1yQRdhSnlocOsZA4uT_8VO0-ZeLXF4gKd/view) model and then put it under the model directory.

### Train
* RAF_DB train : sh train_RAF.sh
* AffectNet train : sh train_AFFECTNET.sh

* hyperparameter Tuning 필요
#####Best
  * Focal Gamma : 2, 5
  * Rank Margin : 0.1, 0.2
  * Rank Alpha : 0.6
