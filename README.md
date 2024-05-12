# RED_FEPSY

### Is Single Face Enough for FER?: A Rescue to Miscalibrated FER by Ranking-based Confidence Calibration
### Dataset
* RAF-DB : /nas_homes/jihyun/RAF_DB/
* AffectNet : /nas_homes/jihyun/datasets/AffectNet/
* FERPlus : /nas_homes/jihyun/datasets/FERPlus/

### Pretrained backbone model

/home/jihyun/code/eccv/model/resnet50_ft_weight.pkl

### Train
* RAF_DB train : sh train_RAF.sh
* AffectNet train : sh train_AFFECTNET.sh

#### Hyperparameter Tuning 필요
##### *Best
  * Focal Gamma : 2, 5
  * Rank Margin : 0.1, 0.2
  * Rank Alpha : 0.6
