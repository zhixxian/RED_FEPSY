import torch
import pickle
from resnet import *
from torch.autograd import Variable
from abc_modules import ABC_Model
import torch.nn.functional as F
#from __init__ import *

import torch
import pickle

class FixedBatchNorm(nn.BatchNorm2d):
    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps)

def group_norm(features):
    return nn.GroupNorm(4, features)

class Model(nn.Module, ABC_Model):
    
    def __init__(self, args, pretrained=True, num_classes=7, mode='fix'):
        super(Model, self).__init__()
        
        self.mode = mode
        self.num_classes = args.num_classes
        
        if self.mode == 'fix':
            self.norm_fn = FixedBatchNorm
        else:
            self.norm_fn = nn.BatchNorm2d
        
        self.resnet50 = ResNet(Bottleneck, [3, 4, 6, 3], strides=(2, 2, 2, 1), batch_norm_fn=self.norm_fn)
        resnet50_path = '/home/jihyun/code/eccv/model/resnet50_ft_weight.pkl'
        if pretrained:
            with open(resnet50_path, 'rb') as f:
                obj = f.read()
            weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
            weights.pop('fc.weight')
            weights.pop('fc.bias')
        
            self.resnet50.load_state_dict(weights)

        
        self.classifier = nn.Conv2d(2048, self.num_classes, 1, bias=False) # 2048 -> 7
        # self.num_classes = args.num_classes

        self.initialize([self.classifier])
        
        self.stage1 = nn.Sequential(self.resnet50.conv1, 
                                    self.resnet50.bn1, 
                                    self.resnet50.relu, 
                                    self.resnet50.maxpool)
        self.stage2 = nn.Sequential(self.resnet50.layer1)
        self.stage3 = nn.Sequential(self.resnet50.layer2)
        self.stage4 = nn.Sequential(self.resnet50.layer3)
        self.stage5 = nn.Sequential(self.resnet50.layer4)
                    
        
    def forward(self, x, with_cam=False):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        # print(x.shape) # [batch, 2048, 7, 7]

        if with_cam:
            features = self.classifier(x) # 8 7 14 14

            logits = self.global_average_pooling_2d(features) # 8 7 
            # logits.shape # [batch, num_classes]
            # logits = torch.nn.functional.gumbel_softmax(logits, tau=10, hard=False, eps=1e-10, dim=- 1)
            return logits, features
        else:
            features = self.classifier(x) # [batch, num_classes, 7, 7]
            logits = self.global_average_pooling_2d(features)
            
            return logits, features
