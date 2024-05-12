from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import math
import wandb
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
from collections import defaultdict, Counter
from torch.nn.functional import softmax
from tqdm import tqdm
from torch.utils.data import Subset
from torch.utils.data.sampler import WeightedRandomSampler , RandomSampler
from model import Model
from dataset import RafDataset, RafCDataset, FERPlusDataset, LFW, AffectNet

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, plot_overlap, get_mean_and_std, init_params, mapping_func

from focal import FocalLoss
from ranking import RankingLoss

from calibration import CalibrateEvaluator

"""Get arguments from cmd."""
parser = argparse.ArgumentParser(description='PyTorch CutMix_Pseudo Training')

# cutmix options
parser.add_argument('--wh', default='width', type=str, help='cutmix width or height')


parser.add_argument('--eval', default=1, type=int, metavar='N',
                    help='evaluate model on validation set (default: 0 (=False))')

# Optimization options
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', default=0.0005, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--workers', type=int, default=8,
                        help='num of workers to use')
parser.add_argument('--ema_decay', type=float, default=0.999,
                        help="Exponential moving average of model weights")
parser.add_argument('--iterations', type=int, default=2*200,
                        help='Number of iteration per epoch')
parser.add_argument('--out', default='result',
                        help='Directory to output the result')
#Device options
parser.add_argument('--gpu', default='1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

#Method options
parser.add_argument('--train_iteration', type=int, default=800,
                        help='Number of iteration per epoch')
parser.add_argument('--num_classes', type=int, default=7,
                        help='raf : 7, affectnet : 8, ferplus : 8')

# arguments related to pseudo labeling algorithm
parser.add_argument('--lu_weight', type=float, default=1.0,
                        help="Weight for unsupervised loss")
parser.add_argument('--mapping', default='convex', choices=['convex', 'concave', 'linear'],
                        help="Beta Mapping function")
parser.add_argument('--beta', default=1, type=float, help='hyperparam for beta distribution')
parser.add_argument('--cutmix_prob', default=0.5, type=float, help='mixup probability')

# misc options
parser.add_argument('--amp', action='store_true', default=False,
                        help="Use mixed precision training")
parser.add_argument('--manualSeed', type=int, default=5, help='manual seed')

# Path options
parser.add_argument('--datasets', default='raf', type=str, help='dataset',
                    choices=['raf', 'affectnet', 'ferplus'])
parser.add_argument('--load_path', type=str, default="/home/jihyun/code/eccv/src/2_0.1_0.3.pth", #####################################
                        help='Checkpoint to load path from.')
# parser.add_argument('--eval', action='store_true', default=False,
#                         help='Evaluate the model.')
parser.add_argument('--log', action='store_true', default=False,
                        help='Log progress.')
parser.add_argument('--log_interval', type=int, default=10,
                        help='Log interval.')
parser.add_argument('--save_interval', type=int, default=1,
                        help='Save interval.')
parser.add_argument('--seed', type=int, default=5,
                        help='Random seed.')
parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use.')
parser.add_argument('--mode', type=str, default='train',
                        help='Mode to run.')
parser.add_argument('--mean', type=list, default=[0.485, 0.456, 0.406],
                        help='Mean of dataset.')
parser.add_argument('--std', type=list, default=[0.229, 0.224, 0.225],
                        help='Std of dataset.')
parser.add_argument('--focal_gamma', type=float, default=5.0,
                        help='Focal Loss gamma')
parser.add_argument('--focal_alpha', type=float, default=1.0,
                        help='Focal Loss alpha')
parser.add_argument('--rank_alpha', type=float, default=0.4,
                        help='Ranking Loss gamma')
parser.add_argument('--rank_margin', type=float, default=0.1,
                        help='Ranking Loss alpha')
parser.add_argument('--num_bins', type=int, default=15,
                        help='Number of bins for adafocal')

# parser.add_argument('--alpha', type=float, default=0.25,
#                         help='Focal Loss alpha')
parser.add_argument('--threshold', default=0.95, type=float,
                    help='pseudo label threshold')
# data
parser.add_argument('--raf_path', type=str, default='/nas_homes/jihyun/RAF_DB/', 
                    help='raf_dataset_path')
parser.add_argument('--rafc_path', type=str, default=None, # '/nas_homes/jihyun/RAF-DB-C/'
                    help='raf_dataset_path')
parser.add_argument('--label_path', type=str, default='list_patition_label.txt', 
                    help='label_path')
parser.add_argument('--lfw_path', type=str, default='/nas_homes/jihyun/LFW/', 
                    help='lfw_dataset_path')
parser.add_argument('--aff_path', type=str, default='/nas_homes/jihyun/AffectNet/datasets/', 
                    help='raf_dataset_path')
parser.add_argument('--ferplus_path', type=str, default='/nas_homes/jihyun/FERPlus', 
                    help='raf_dataset_path')
parser.add_argument('--resnet50_path', type=str, default='/home/jihyun/code/eccv/model/resnet50_ft_weight.pkl', 
                    help='pretrained_backbone_path')

# for sanity check 
parser.add_argument('--log_freq', type=int, default=50, 
                    help='log print frequency')
parser.add_argument('--wandb', type=str, default='cutmix_fer', 
                    help='wandb project name')
parser.add_argument('--save_path', type=str, 
                    default='/home/jihyun/code/eccv/src/save_path', 
                    help='weight save path')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_LAUNCH_BLOCKING'] = args.gpu
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0

def main():
    # setup_seed(0)
    global best_acc
    
    if not os.path.isdir(args.out):
        mkdir_p(args.out)
    # Data
    print('==> Preparing data..')
    
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std),
    ])

    transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std), 
    ])
    
    transforms_compound = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    
    if args.datasets == 'raf':
        fer_dataset = RafDataset(args, phase='train', transform=transform_train)

            
    fr_dataset = LFW(args, transform=transform_train, strong_transform=True)
    
    if args.datasets == 'raf':
        test_dataset = RafDataset(args, phase='test', transform=transform_val)
    
    if args.rafc_path is not None:
        compound_dataset = RafCDataset(args, phase='train', transform=transforms_compound)
    
    fer_train_loader = torch.utils.data.DataLoader(fer_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.workers,
                                                pin_memory=True,
                                                drop_last=True)
    fr_train_loader = torch.utils.data.DataLoader(fr_dataset,
                                                batch_size=args.batch_size,
                                                #    sampler=sampler,
                                                shuffle=True,
                                                num_workers=args.workers,
                                                pin_memory=True,
                                                drop_last=True)
    
    # Difference with fr_train_loader: shuffle=False, drop_last=False
    fr_test_loader = torch.utils.data.DataLoader(fr_dataset,  
                                                batch_size=args.batch_size,
                                                #    sampler=sampler,
                                                shuffle=False,
                                                num_workers=args.workers,
                                                pin_memory=True,
                                                drop_last=False)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=args.workers,
                                                pin_memory=True)
    
    if args.rafc_path is not None:
        compound_loader = torch.utils.data.DataLoader(compound_dataset, 
                                                    batch_size=1,
                                                    shuffle=False,
                                                    num_workers=args.workers,
                                                    pin_memory=True)
    else:
        compound_loader = None

    N=fr_train_loader.dataset.__len__()
    learning_status = [-1] * N

    fr_samples_num=len(fer_dataset)    
    # Model
    print("==> creating ResNet-50")

    def create_model(ema=False):
        model = Model(args, pretrained=True, num_classes=args.num_classes, mode=args.mode)
        model = torch.nn.DataParallel(model).cuda()

        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)
    
    device = torch.device('cuda')
    # device = torch.device('cuda:{}'.format(args.gpu))

    mapping = mapping_func(args.mapping)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # criterion
    criterion= nn.CrossEntropyLoss(reduction='none')
    criterion_focal = FocalLoss(gamma=args.focal_gamma, ignore_index=100, size_average=False)
    criterion_ranking = RankingLoss(num_classes=args.num_classes, margin=args.rank_margin)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ema_optimizer= WeightEMA(model, ema_model, alpha=args.ema_decay)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    os.makedirs(os.path.join(args.save_path, "results", args.wandb), exist_ok=True)
    
    # Load checkpoint
    ckpt = torch.load(args.load_path, map_location='cpu')
    model.load_state_dict(ckpt, strict=True)
    
    wandb.init(project="ECCV")
    wandb.run.name = args.wandb
    wandb.config.update(args)
    wandb.watch(model)
    
    # N = len(fr_train_loader.dataset.indices)     

    logger = Logger(os.path.join(args.out, 'log.txt'), title='RAF')
    logger.set_names(['train_loss', 'train_loss_focal', 'train_loss_ranking', 'test_loss', 'test_acc', 'top5_acc','nll', 'ece', 'per_class_ece', 'aece', 'oe', 'ue'])
    
    threshold = args.threshold
    amp_flag = args.amp

    test_accs = []
    # threshold = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    start_epoch = 1
    best_acc = 0
    best_epoch = 0
    
    compound_confidence_list=[]
    
    # Train and val
    for epoch in range(start_epoch, args.epochs + 1):
        
        train_loss, cls_thresholds = train(fer_train_loader=fer_train_loader, fr_train_loader=fr_train_loader, pseudo_loader=None,
                                            model=model, optimizer=optimizer, 
                                            ema_optimizer=ema_optimizer, criterion_ce=criterion, criterion_focal=criterion_focal, 
                                            criterion_ranking=criterion_ranking, threshold=threshold, 
                                            epoch=epoch, use_cuda=use_cuda, device=device, 
                                            mapping=mapping, learning_status=learning_status, 
                                            amp_flag=amp_flag)
        # class_threshold 
        pseudo_target, mask_idx = pseudo_labeling(fr_test_loader=fr_test_loader, model=model, thresholds=cls_thresholds, 
                                                use_cuda=use_cuda, device=device)
            
        fr_dataset.update_label(pseudo_target)
        pseudo_dataset = Subset(fr_dataset, mask_idx)
        
        weights = make_weights_for_balanced_classes(pseudo_target, mask_idx, n_classes=args.num_classes)
        weights = torch.DoubleTensor(weights)
        
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        loader = torch.utils.data.DataLoader(pseudo_dataset,
                                            batch_size=args.batch_size,
                                            sampler=sampler,
                                            shuffle=False,
                                            drop_last=True,
                                            num_workers=args.workers)


        train_loss, train_loss_focal, train_loss_ranking = train(fer_train_loader=fer_train_loader, fr_train_loader=fr_train_loader, pseudo_loader=loader, 
                                                                                model=model, optimizer=optimizer, 
                                                                                ema_optimizer=ema_optimizer, criterion_ce=criterion, criterion_focal=criterion_focal, 
                                                                                criterion_ranking=criterion_ranking, threshold=threshold, 
                                                                                epoch=epoch, use_cuda=use_cuda, device=device, 
                                                                                mapping=mapping, learning_status=learning_status, 
                                                                                amp_flag=amp_flag)
        
        fr_dataset.restore_label()

        test_loss, test_acc, top5_acc, nll, ece, mce, per_class_ece, aece, oe, ue = validate(test_loader, model, criterion, device, use_cuda, mode=args.mode)

        if test_acc > best_acc:
            torch.save(model.state_dict(), os.path.join(args.save_path, "results", args.wandb, 'best_model.pth'))
            best_acc = test_acc
            best_epoch = epoch
        else:
            print(f"Still best accuracy is {best_acc} at epoch {best_epoch}\n")
            
        print(f"Epoch : [{epoch}/{args.epochs}] \n Train loss : {train_loss:.4f} \n top1 accuracy : {test_acc:.4f} \n top5 accuracy : {top5_acc}\n Test loss : {test_loss:.4f}\n NLL : {nll}\n ECE : {ece}\n MCE:{mce} \nper_class_ece : {per_class_ece}\n AECE : {aece}\n OE : {oe}\n UE : {ue}\n")                     

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
            'learning_status' : learning_status,
            'iteration' : epoch
            }, is_best)
        test_accs.append(test_acc)
            
        class_correct, class_total, avg_accuracy = eval_per_class(model, device=device)
        output_dict = {
            'train_loss':train_loss,
            'train_loss_focal':train_loss_focal,
            'train_loss_ranking':train_loss_ranking,
            'test_loss':test_loss,
            'test_acc':test_acc,
            'top5_acc':top5_acc,
            'nll':nll,
            'mce':mce,
            'ece':ece,
            # 'per_class_ece':per_class_ece,
            # 'cece':per_class_ece,
            'aece':aece,
            'oe':oe,
            'ue':ue
        }
        per_class_ece=per_class_ece.tolist()
        
        for k in range(args.num_classes):
            output_dict[f"Test accuracy of class {k}"] = 100 * class_correct[k] / class_total[k]
        output_dict["Average accuracy"] = avg_accuracy
        
        for k in range(len(per_class_ece)):
            if k<len(per_class_ece)-2:
                output_dict['class_'+str(k)+'_ece'] = per_class_ece[k]
            elif k==len(per_class_ece)-2:
                output_dict['cece'] = per_class_ece[k]
            else:
                output_dict['var_ece'] = per_class_ece[k]
        
        ###############################################
        if True: # args.rafc_path is not None:
            compound_list = validate_compound(test_loader, model, device)
            for i in range(len(compound_list)):
                print(compound_list[i].avg.tolist())
            compound_output_dict = {
                'epoch': i,
                # 'test_acc': test_acc,
                # 'ece': output_dict['ece'],
                'SU': compound_list[0].avg.tolist(),
                'FE': compound_list[1].avg.tolist(),
                'DI': compound_list[2].avg.tolist(),
                'HA': compound_list[3].avg.tolist(),
                'SA': compound_list[4].avg.tolist(),
                'AN': compound_list[5].avg.tolist(),
                'NE': compound_list[6].avg.tolist()
            }
            #save_dict
            for i in range(len(compound_list)):
                print(compound_list[i].avg.tolist())
            compound_confidence_list.append(compound_output_dict)
        ###############################################
        # if args.wandb:
        #     wandb.log(output_dict, step=epoch)
        
    with open('compound_confidence_list.pkl', 'wb') as f:
        pickle.dump(compound_confidence_list, f)
        
    logger.close()

    print('Best acc:')
    print(best_acc)


def train(fer_train_loader, fr_train_loader, pseudo_loader, model, optimizer, ema_optimizer, criterion_ce, criterion_focal, criterion_ranking, threshold, epoch, use_cuda, device, mapping, learning_status, amp_flag):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_ce = AverageMeter()
    losses_focal = AverageMeter()
    losses_ranking = AverageMeter()
    losses_cutmix = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=len(fer_train_loader))
    
    cls_thresholds = torch.zeros(args.num_classes, device=device)

    model.train()
  
    total_pseudo_target = []
    total_indicator_idx = []
        
    if pseudo_loader==None:
            
        for batch_idx in range(len(fer_train_loader)):
            try:
                # inputs_x, targets_x = labeled_train_iter.next()
                inputs_x, targets_x = next(iter(fer_train_loader))
            except:
                labeled_train_iter = iter(fer_train_loader)
                inputs_x, targets_x = labeled_train_iter.next()

            try:
                # (inputs_u, inputs_strong), _ = unlabeled_train_iter.next()
                (inputs_u, inputs_strong), u_i = next(iter(fr_train_loader))
            except:
                unlabeled_train_iter = iter(fr_train_loader)
                (inputs_u, inputs_strong), u_i = unlabeled_train_iter.next()

            # measure data loading time
            data_time.update(time.time() - end)

            # batch_size = inputs_x.size(0)

            if use_cuda:
                inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda()
                inputs_u = inputs_u.cuda()
                inputs_strong = inputs_strong.cuda()
            
            all_inputs = torch.cat([inputs_x, inputs_u, inputs_strong], dim=0)
            outputs, features = model(all_inputs.to(device))
            x_pred, u_pred, s_pred = torch.split(outputs, [inputs_x.size(0), inputs_u.size(0), inputs_strong.size(0)])
            
            counter = Counter(learning_status)
            # x, u, s = torch.split(outputs, [inputs_x.size(0), inputs_u.size(0), inputs_strong.size(0)])
            
            # normalize the status
            num_unused = counter[-1] # unused data
            if num_unused != len(fr_train_loader): # if there is unused data
                max_counter = max([counter[c] for c in range(args.num_classes)])
                if max_counter < num_unused:
                    # normalize with flexmatch eq.11 
                    sum_counter = sum([counter[c] for c in range(args.num_classes)])
                    denominator = max(max_counter, len(fr_train_loader) - sum_counter)
                else:
                    denominator = max_counter
                # threshold per class
                for c in range(args.num_classes):
                    beta = counter[c] / denominator
                    cls_thresholds[c] = mapping(beta) * threshold
                    # print("Class:", c, "Beta:", beta, "Threshold:", cls_thresholds[c]) 
                    
            # update the pseudo label
            with torch.no_grad():
                # print(u_pred)
                uw_prob = softmax(u_pred, dim=1)
                max_prob, hard_label = torch.max(uw_prob, dim=1)
                over_threshold = max_prob > threshold 
                if over_threshold.any():
                    u_i = u_i.cuda()
                    # inputs_u = inputs_u[over_threshold]
                    sample_index = u_i[over_threshold].tolist()
                    pseudo_label = hard_label[over_threshold].tolist()

                    for i, l in zip(sample_index, pseudo_label):
                        learning_status[i] = l # pseudo label
                      
            batch_threshold = torch.index_select(cls_thresholds, 0, hard_label) 
            indicator = max_prob > batch_threshold 
            
            # pseudo label: supervised loss + unsupervised_loss
            outputs_fer, _ = model(inputs_x)
            loss_fer=criterion_ce(outputs_fer, targets_x).mean() #supervised loss
            total_loss=loss_fer
            #unsupervised loss us_pred & hard_label) & indicator
            loss_fer_un = (criterion_ce(u_pred, hard_label)*indicator).mean()
            total_loss+=loss_fer_un * args.lu_weight
            
            losses.update(total_loss.item(), inputs_x.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            total_loss.backward()
            # losses.backward()
            optimizer.step()
            ema_optimizer.step()
            
            bar.suffix  = '(epoch : {epoch}/{total_epoch} | {batch}/{size}) Total: {total:} | Loss_ce: {losses:.4f} | cls_threshold: {thresholds} '.format(
                epoch=epoch,
                total_epoch=args.epochs,
                batch=batch_idx + 1,
                size=len(fer_train_loader),
                total=bar.eta_td,
                losses=losses.avg,
                thresholds=list(np.round(cls_thresholds.tolist(), 4)))
            bar.next()
            
            break
        
    else:
        ################################################################
        ####################     cutmix     ############################
        ################################################################

        loader = zip(fer_train_loader, pseudo_loader) 
        
        for batch_idx, ((inputs_x, targets_x), ((new_inputs, _), new_targets)) in enumerate(loader):
            
            if use_cuda:
                inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda()
                new_inputs, new_targets = new_inputs.cuda(), new_targets.cuda().long()
            
            data_time.update(time.time() - end)
            
            cutmix_input = inputs_x.clone()
            assert cutmix_input.shape == new_inputs.shape, f"{cutmix_input.shape}, {new_inputs.shape}"
            height = cutmix_input.shape[-2]
            
            lam = 0.5
            
            selected = np.random.choice([0,1], args.batch_size) # 0: FR lower, 1: FR upper
            idx_upper = np.where(selected != 0)
            idx_lower = np.where(selected == 0)
            
            cutmix_input[idx_upper, :, :int(height*lam), :] = new_inputs[idx_upper, :, :int(height*lam), :]
            cutmix_input[idx_lower, :, int(height*lam):, :] = new_inputs[idx_lower, :, int(height*lam):, :]
                                                
            target_re_fer=torch.zeros(args.batch_size, args.num_classes, requires_grad=False).cuda()
            target_re_fer.scatter_(1, targets_x.unsqueeze(1), 1)

            target_re_fr=torch.zeros(args.batch_size, args.num_classes, requires_grad=False).cuda()
            target_re_fr.scatter_(1, new_targets.unsqueeze(1), 1)
            # print(target_re_fr.shape) # 4, 7
            target_re = lam * target_re_fer + (1 - lam) * target_re_fr
            
            #compute fer, fr, cutmix output            
            outputs_fer, _ = model(inputs_x)
            outputs_fr, _ = model(new_inputs)
            outputs_cutmix, _ = model(cutmix_input)
            new_targets = new_targets.long()

            focal_fer = criterion_focal(outputs_fer, targets_x)
            focal_fr = criterion_focal(outputs_fr, new_targets)
            
            # ranking loss
            rank_fer_cutmix = criterion_ranking(outputs_fer, targets_x, outputs_cutmix, target_re,lam)
            rank_fr_cutmix = criterion_ranking(outputs_fr, new_targets, outputs_cutmix, target_re,lam)
            
            # loss
            loss_focal = focal_fr+focal_fer
            loss_ranking=rank_fer_cutmix+rank_fr_cutmix
            total_loss = loss_focal + loss_ranking

            # record loss
            losses_focal.update(loss_focal.item(), inputs_x.size(0))
            losses_ranking.update(loss_ranking.item(), inputs_x.size(0))
            losses.update(total_loss.item(), inputs_x.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            ema_optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
            # eta_td = bar.eta_td
        

            # plot progress
            bar.suffix  = '(epoch : {epoch}/{total_epoch} | {batch}/{size}) Total: {total:} | Loss: {loss:.4f} | Loss Focal: {loss_focal:.4f} | Loss Ranking: {loss_ranking:.4f} '.format(
                    epoch=epoch,
                    total_epoch=args.epochs,
                    batch=batch_idx + 1,
                    size=len(fer_train_loader),
                    total=bar.eta_td,
                    loss=losses.avg,
                    # loss_ce=losses_ce.avg,
                    loss_focal=losses_focal.avg,
                    loss_ranking=losses_ranking.avg,
                    )
            bar.next()
    
    
    bar.finish()
    if pseudo_loader == None:
        return(losses.avg, cls_thresholds)
    else:
        return (losses.avg, losses_focal.avg, losses_ranking.avg)
    
def pseudo_labeling(fr_test_loader, model, thresholds, use_cuda, device):
    model.eval()
    pseudo_target = []
    mask_idx = []
    for batch_idx, ((inputs_u, _), indexes) in tqdm(enumerate(fr_test_loader)):
        if use_cuda:
            inputs_u = inputs_u.cuda()
        # compute output
        outputs, _ = model(inputs_u)
        # _, predicts = torch.max(outputs, 1)
        
        softmax_output = F.softmax(outputs, dim=1)
        tmp_indicator = softmax_output > thresholds
        softmax_output[~tmp_indicator] = 0
        
        max_prob, hard_label = torch.max(softmax_output, dim=1)
        batch_threshold = torch.index_select(thresholds, 0, hard_label)
        indicator = max_prob > batch_threshold

        pseudo_target.extend(hard_label.tolist())
        mask_idx.extend(indexes[indicator.cpu()].tolist())
    
    return pseudo_target, mask_idx

def validate(valloader, model, criterion, device, use_cuda, mode):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    nll = AverageMeter()
    ece = AverageMeter()
    mce = AverageMeter()
    aece = AverageMeter()
    per_class_ece = AverageMeter()
    oe = AverageMeter()
    ue = AverageMeter()
    
    calibrate_evaluator = CalibrateEvaluator(args.num_classes, num_bins=15, device=device)
    calibrate_evaluator.reset()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets).mean()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            # nll, ece, aece, cece, oe, ue = CalibrateEvaluator.update(outputs, targets)  # update calibration evaluator
            
            # measure calibration confidence

            calibrate_evaluator.update(outputs, targets)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Total: {total:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f} | NLL: {nll: .4f} | ECE: {ece: .4f} | MCE: {mce: .4f} | per_class_ece: {per_class_ece: .4f} | AECE: {aece: .4f} | OE: {oe: .4f} | UE: {ue: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        total=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        nll=nll.avg,
                        ece=ece.avg,
                        mce=mce.avg,
                        per_class_ece=per_class_ece.avg,
                        aece=aece.avg,
                        oe=oe.avg,
                        ue=ue.avg
                        )
            bar.next()
        bar.finish()
        
        cal_dict, _ = calibrate_evaluator.mean_score(print_classes=True, all_metric=True)
        nll_item, ece_item, mce_item, aece_item, per_class_ece_item, oe_item, ue_item = cal_dict['nll'], cal_dict['ece'], cal_dict['mce'], cal_dict['aece'], cal_dict['per_class_ece'], cal_dict['oe'], cal_dict['ue']
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        nll.update(nll_item)
        ece.update(ece_item)
        mce.update(mce_item)
        aece.update(aece_item)
        per_class_ece.update(per_class_ece_item)
        oe.update(oe_item)
        ue.update(ue_item)

    return (losses.avg, top1.avg, top5.avg, nll.avg, mce.avg, ece.avg, per_class_ece.avg, aece.avg, oe.avg, ue.avg)

def validate_compound(valloader, model, device):
    SU = AverageMeter()
    FE = AverageMeter()
    DI = AverageMeter()
    HA = AverageMeter()
    SA = AverageMeter()
    AN = AverageMeter()
    NE = AverageMeter()
    compound_list = [SU, FE, DI, HA, SA, AN, NE]
    
    for i in range(len(compound_list)):
        compound_list[i].reset()
    
    calibrate_evaluator = CalibrateEvaluator(args.num_classes, num_bins=15, device=device)
    calibrate_evaluator.reset()
    with torch.no_grad():
        model.eval()

        iter_cnt = 0

        for batch_i, (imgs, labels) in enumerate(tqdm(valloader)):
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs, _ = model(imgs)
            softmaxes = F.softmax(outputs, dim=1) 
            # confidences, _ = torch.max(softmaxes, 1)
            # loss = nn.CrossEntropyLoss()(outputs, labels)/

            iter_cnt += 1
            compound_list[labels].update(softmaxes)
                        
    return compound_list

def eval_per_class(model, device):
    # setup_seed(0)
    
    eval_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    
    test_dataset = RafDataset(args, phase='test', transform=eval_transforms)        
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True)
    
    # model = model(imgs1)
    # model = Model(args)
    # model.load_state_dict(torch.load(os.path.join(args.save_path, "results", args.wandb, "best_model.pth")))

    model.eval()
    
    #####
    total = 0 
    correct = 0
    class_correct = list(0 for _ in range(args.num_classes)) 
    class_total = list(0 for _ in range(args.num_classes)) 
    correct_sum = 0
    data_num = 0
    #####
    
    # device = torch.device('cuda:{}'.format(args.gpu))
    model.to(device)
    
    with torch.no_grad():
        model.eval()

        for batch_i, (imgs1, labels) in enumerate(tqdm(test_loader)):
            imgs1 = imgs1.to(device)
            labels = labels.to(device)

            outputs, _ = model(imgs1)
            _, predicts = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicts == labels).sum().item()
            
            correct_num = torch.eq(predicts, labels).sum()
            correct_sum += correct_num
            data_num += outputs.size(0)
            
            c = (predicts == labels).squeeze()
            for j in range(imgs1.size(0)):
                label = labels[j]
                class_correct[label] += c[j].item()
                class_total[label] += 1
                
    print(class_correct, sum(class_correct))
    print(class_total, sum(class_total))
    
    avg_accuracy = 100 * correct_sum.float() / float(data_num)
    
    for k in range(0, args.num_classes):
        print(f"Accuracy of class {k}: {(100 * class_correct[k] / class_total[k]):.4f}")
    print(f"Average Test Accuracy: {avg_accuracy:.4f}")
    
    return class_correct, class_total, avg_accuracy

def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)

def make_weights(labels, nclasses):
    labels = np.array(labels)   # where, unique 
    weight_list = []
 
    for cls in range(nclasses):
        idx = np.where(labels == cls)[0]
        count = len(idx) 
        weight = 1/count    
        weights = [weight] * count
        weight_list += weights
 
    return weight_list

def make_weights_a(labels, nclasses):
    labels = np.array(labels) 
    weight_arr = np.zeros_like(labels) 
    
    _, counts = np.unique(labels, return_counts=True) 
    for cls in range(nclasses):
        weight_arr = np.where(labels == cls, 1/counts[cls], weight_arr) 
 
    return weight_arr

def make_weights_for_balanced_classes(pseudo_target, mask_idx, n_classes): #images, nclasses):
    global weights
    masked_pseudo_target = np.array(pseudo_target)[mask_idx]

    n_images = len(masked_pseudo_target) #len(images)
    count_per_class = [0] * n_classes

    cnt_pseudo_target = Counter(masked_pseudo_target)
    for key, value in cnt_pseudo_target.items():
        count_per_class[key] = value
    print(count_per_class)
    
    weight_per_class = np.array([0.] * n_classes) 
    
    for i in range(n_classes):
        if count_per_class[i] > 0:
            weight_per_class[i] = float(count_per_class[i]) / float(n_images)
        else:
            weight_per_class[i] = 0
    
    weights = np.array([0.] * n_images)
    
    for target in cnt_pseudo_target.keys():
        weights[masked_pseudo_target == target] = weight_per_class[target]
    
    return weights

if __name__ == '__main__':
    main()
