import os
import sys
import argparse
import shutil
import time
import random
import gc
import json
from distutils.version import LooseVersion
import scipy.misc
import logging


import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F


from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from utils.transforms import ResizeImage, ResizeAnnotation

from dataset.referit_loader import *
from model.grounding_model import *
from utils.parsing_metrics import *
from utils.utils import *
from tqdm import tqdm


def yolo_loss(input, target, gi, gj, best_n_list, w_coord=5., w_neg=1./5, size_average=True):
    mseloss = torch.nn.MSELoss(size_average=True)
    celoss = torch.nn.CrossEntropyLoss(size_average=True)
    batch = input[0].size(0)

    pred_bbox = Variable(torch.zeros(batch,4).cuda())
    gt_bbox = Variable(torch.zeros(batch,4).cuda())
    for ii in range(batch):
        pred_bbox[ii, 0:2] = F.sigmoid(input[best_n_list[ii]//3][ii,best_n_list[ii]%3,0:2,gj[ii],gi[ii]])
        pred_bbox[ii, 2:4] = input[best_n_list[ii]//3][ii,best_n_list[ii]%3,2:4,gj[ii],gi[ii]]
        gt_bbox[ii, :] = target[best_n_list[ii]//3][ii,best_n_list[ii]%3,:4,gj[ii],gi[ii]]
    loss_x = mseloss(pred_bbox[:,0], gt_bbox[:,0])
    loss_y = mseloss(pred_bbox[:,1], gt_bbox[:,1])
    loss_w = mseloss(pred_bbox[:,2], gt_bbox[:,2])
    loss_h = mseloss(pred_bbox[:,3], gt_bbox[:,3])

    pred_conf_list, gt_conf_list = [], []
    for scale_ii in range(len(input)):
        pred_conf_list.append(input[scale_ii][:,:,4,:,:].contiguous().view(batch,-1))
        gt_conf_list.append(target[scale_ii][:,:,4,:,:].contiguous().view(batch,-1))
    pred_conf = torch.cat(pred_conf_list, dim=1)
    gt_conf = torch.cat(gt_conf_list, dim=1)
    loss_conf = celoss(pred_conf, gt_conf.max(1)[1])
    return (loss_x+loss_y+loss_w+loss_h)*w_coord + loss_conf

def refine_loss(input, target):
    celoss = torch.nn.CrossEntropyLoss(size_average=True)
    input = torch.cat(input)
    target = torch.cat(target).long()

    return celoss(input, target)


def save_segmentation_map(bbox, target_bbox, input, mode, batch_start_index, \
    merge_pred=None, pred_conf_visu=None, save_path='./visulizations/'):
    n = input.shape[0]
    save_path=save_path+mode

    input=input.data.cpu().numpy()
    input=input.transpose(0,2,3,1)
    for ii in range(n):
        os.system('mkdir -p %s/sample_%d'%(save_path,batch_start_index+ii))
        imgs = input[ii,:,:,:].copy()
        imgs = (imgs*np.array([0.299, 0.224, 0.225])+np.array([0.485, 0.456, 0.406]))*255.
        # imgs = imgs.transpose(2,0,1)
        imgs = np.array(imgs, dtype=np.float32)
        imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
        cv2.rectangle(imgs, (bbox[ii,0], bbox[ii,1]), (bbox[ii,2], bbox[ii,3]), (255,0,0), 2)
        cv2.rectangle(imgs, (target_bbox[ii,0], target_bbox[ii,1]), (target_bbox[ii,2], target_bbox[ii,3]), (0,255,0), 2)
        cv2.imwrite('%s/sample_%d/pred_yolo.png'%(save_path,batch_start_index+ii),imgs)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter):
    # print(optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])
    if args.power!=0.:
        lr = lr_poly(args.lr, i_iter, args.nb_epoch, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
      optimizer.param_groups[1]['lr'] = lr / 10
        
def save_checkpoint(state, is_best, filename='default'):
    if filename=='default':
        filename = 'model_%s_batch%d'%(args.dataset,args.batch_size)

    checkpoint_name = './saved_models/%s_checkpoint.pth.tar'%(filename)
    best_name = './saved_models/%s_model_best.pth.tar'%(filename)
    torch.save(state, checkpoint_name)
    if is_best:
        shutil.copyfile(checkpoint_name, best_name)

def build_target(raw_coord, pred):
    coord_list, bbox_list = [],[]
    for scale_ii in range(len(pred)): # change gt box coord to coor in feat map
        coord = Variable(torch.zeros(raw_coord.size(0), raw_coord.size(1)).cuda())
        batch, grid = raw_coord.size(0), args.size//(32//(2**scale_ii)) # 1/32, 1/16, 1/8
        coord[:,0] = (raw_coord[:,0] + raw_coord[:,2])/(2*args.size)
        coord[:,1] = (raw_coord[:,1] + raw_coord[:,3])/(2*args.size)
        coord[:,2] = (raw_coord[:,2] - raw_coord[:,0])/(args.size)
        coord[:,3] = (raw_coord[:,3] - raw_coord[:,1])/(args.size)
        coord = coord * grid
        coord_list.append(coord)
        bbox_list.append(torch.zeros(coord.size(0),3,5,grid, grid))

    best_n_list, best_gi, best_gj = [],[],[]
    neg_gi, neg_gj=[],[]


    iou_all=np.zeros((len(pred),batch,3))
    for ii in range(batch):
        anch_ious = []
        iou_scale=[]
        for scale_ii in range(len(pred)):
            batch, grid = raw_coord.size(0), args.size//(32//(2**scale_ii))
            gi = coord_list[scale_ii][ii,0].long()
            gj = coord_list[scale_ii][ii,1].long()
            tx = coord_list[scale_ii][ii,0] - gi.float()
            ty = coord_list[scale_ii][ii,1] - gj.float()

            gw = coord_list[scale_ii][ii,2]
            gh = coord_list[scale_ii][ii,3]

            anchor_idxs = [x + 3*scale_ii for x in [0,1,2]]
            anchors = [anchors_full[i] for i in anchor_idxs]
            scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
                x[1] / (args.anchor_imsize/grid)) for x in anchors]

            ## Get shape of gt box
            gt_box = torch.from_numpy(np.array([0, 0, gw, gh]).astype(np.float32)).unsqueeze(0)
            ## Get shape of anchor box
            anchor_shapes = torch.from_numpy(np.concatenate((np.zeros((len(scaled_anchors), 2)), np.array(scaled_anchors)), 1).astype(np.float32))
            ## Calculate iou between gt and anchor shapes
            iou_list=list(bbox_iou(gt_box, anchor_shapes))
            anch_ious += iou_list
            iou_all[scale_ii,ii,:]=np.array(iou_list)
        ## Find the best matching anchor box
        best_n = np.argmax(np.array(anch_ious)) # select best match anchor box
        best_scale = best_n//3 # select best match scaled feature map

        batch, grid = raw_coord.size(0), args.size//(32/(2**best_scale))
        anchor_idxs = [x + 3*best_scale for x in [0,1,2]]
        anchors = [anchors_full[i] for i in anchor_idxs]
        scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
            x[1] / (args.anchor_imsize/grid)) for x in anchors]

        gi = coord_list[best_scale][ii,0].long()
        gj = coord_list[best_scale][ii,1].long()
        tx = coord_list[best_scale][ii,0] - gi.float()
        ty = coord_list[best_scale][ii,1] - gj.float()
        gw = coord_list[best_scale][ii,2]
        gh = coord_list[best_scale][ii,3]
        tw = torch.log(gw / scaled_anchors[best_n%3][0] + 1e-16)
        th = torch.log(gh / scaled_anchors[best_n%3][1] + 1e-16)

        bbox_list[best_scale][ii, best_n%3, :, gj, gi] = torch.stack([tx, ty, tw, th, torch.ones(1).cuda().squeeze()])
        best_n_list.append(int(best_n))
        best_gi.append(gi)
        best_gj.append(gj)
        # iou_all.append(iou_scale)


    for ii in range(len(bbox_list)):
        bbox_list[ii] = Variable(bbox_list[ii].cuda())
    return bbox_list, best_gi, best_gj, best_n_list,iou_all,coord_list

def main():
    parser = argparse.ArgumentParser(
        description='Dataloader test')
    parser.add_argument('--gpu', default='0', help='gpu id')
    parser.add_argument('--workers', default=16, type=int, help='num workers for data loading')
    parser.add_argument('--nb_epoch', default=100, type=int, help='training epoch')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--power', default=0.9, type=float, help='lr poly power')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--size_average', dest='size_average', 
                        default=False, action='store_true', help='size_average')
    parser.add_argument('--size', default=256, type=int, help='image size')
    parser.add_argument('--anchor_imsize', default=416, type=int,
                        help='scale used to calculate anchors defined in model cfg file')
    parser.add_argument('--data_root', type=str, default='./ln_data/',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='data',
                        help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='referit', type=str,
                        help='referit/flickr/unc/unc+/gref')
    parser.add_argument('--time', default=20, type=int,
                        help='maximum time steps (lang length) per batch')
    parser.add_argument('--emb_size', default=512, type=int,
                        help='fusion module embedding dimensions')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrain', default='/shared/ReferCam/saved_models/bert_referit_model.pth.tar', type=str, metavar='PATH',
                        help='pretrain support load state_dict that are not identical, while have no loss saved as resume')
    parser.add_argument('--optimizer', default='RMSprop', help='optimizer: sgd, adam, RMSprop')
    parser.add_argument('--print_freq', '-p', default=100, type=int,
                        metavar='N', help='print frequency (default: 1e3)')
    parser.add_argument('--savename', default='ReferCam', type=str, help='Name head for saved model')
    parser.add_argument('--save_plot', dest='save_plot', default=False, action='store_true', help='save visulization plots')
    parser.add_argument('--seed', default=13, type=int, help='random seed')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')
    parser.add_argument('--test', dest='test', default=False, action='store_true', help='test')
    parser.add_argument('--light', dest='light', default=False, action='store_true', help='if use smaller model')
    parser.add_argument('--lstm', dest='lstm', default=False, action='store_true', help='if use lstm as language module instead of bert')
    parser.add_argument('--seg', dest='seg', default=False, action='store_true', help='if use lstm as language module instead of bert')

    global args, anchors_full
    args = parser.parse_args()
    print('----------------------------------------------------------------------')
    print(sys.argv[0])
    print(args)
    print('----------------------------------------------------------------------')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    
    ## fix seed
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed+1)
    torch.manual_seed(args.seed+2)
    torch.cuda.manual_seed_all(args.seed+3)

    eps=1e-10
    anchors_full=get_archors_full(args)

    ## save logs
    if args.savename=='default':
        args.savename = 'model_%s_batch%d'%(args.dataset,args.batch_size)
    if not os.path.exists('./logs'):
        os.mkdir('logs')
    logging.basicConfig(level=logging.DEBUG, filename="./logs/%s"%args.savename, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")

    input_transform = Compose([
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ReferDataset(data_root=args.data_root,
                         split_root=args.split_root,
                         dataset=args.dataset,
                         split='train',
                         imsize = args.size,
                         transform=input_transform,
                         max_query_len=args.time,
                         lstm=args.lstm,
                         augment=True)
    val_dataset = ReferDataset(data_root=args.data_root,
                         split_root=args.split_root,
                         dataset=args.dataset,
                         split='val',
                         imsize = args.size,
                         transform=input_transform,
                         max_query_len=args.time,
                         lstm=args.lstm)
    ## note certain dataset does not have 'test' set:
    ## 'unc': {'train', 'val', 'trainval', 'testA', 'testB'}
    test_dataset = ReferDataset(data_root=args.data_root,
                         split_root=args.split_root,
                         dataset=args.dataset,
                         testmode=True,
                         split='test',
                         imsize = args.size,
                         transform=input_transform,
                         max_query_len=args.time,
                         lstm=args.lstm)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, drop_last=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                              pin_memory=True, drop_last=True, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                              pin_memory=True, drop_last=True, num_workers=0)

    ## Model
    ## input ifcorpus=None to use bert as text encoder
    ifcorpus = None
    if args.lstm:
        ifcorpus = train_dataset.corpus
    model = grounding_model(corpus=ifcorpus, light=args.light, emb_size=args.emb_size, coordmap=True,\
        bert_model=args.bert_model, dataset=args.dataset)
    # model=model.cuda()

    model = torch.nn.DataParallel(model).cuda()

    if args.pretrain:
        if os.path.isfile(args.pretrain):
            pretrained_dict = torch.load(args.pretrain)['state_dict']
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            assert (len([k for k, v in pretrained_dict.items()])!=0)
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print("=> loaded pretrain model at {}"
                  .format(args.pretrain))
            logging.info("=> loaded pretrain model at {}"
                  .format(args.pretrain))
        else:
            print(("=> no pretrained file found at '{}'".format(args.pretrain)))
            logging.info("=> no pretrained file found at '{}'".format(args.pretrain))
    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint (epoch {}) Loss{}"
                  .format(checkpoint['epoch'], best_loss)))
            logging.info("=> loaded checkpoint (epoch {}) Loss{}"
                  .format(checkpoint['epoch'], best_loss))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))
            logging.info(("=> no checkpoint found at '{}'".format(args.resume)))




    print('Num of parameters:', sum([param.nelement() for param in model.parameters()]))
    logging.info('Num of parameters:%d'%int(sum([param.nelement() for param in model.parameters()])))

    visu_param = model.module.visumodel.parameters()
    rest_param = [param for param in model.parameters() if param not in visu_param]
    visu_param = list(model.module.visumodel.parameters())
    sum_visu = sum([param.nelement() for param in visu_param])
    sum_text = sum([param.nelement() for param in model.module.textmodel.parameters()])
    sum_fusion = sum([param.nelement() for param in rest_param]) - sum_text
    print('visu, text, fusion module parameters:', sum_visu, sum_text, sum_fusion)

    if args.seg:
        param=model.module.segmentation.parameters()
    else:
        param=model.parameters()



    ## optimizer; rmsprop default
    if args.optimizer=='adam':
        optimizer = torch.optim.Adam(param, lr=args.lr, weight_decay=0.0005)
    elif args.optimizer=='sgd':
        optimizer = torch.optim.SGD(param, lr=args.lr, momentum=0.99)
    else:
        optimizer = torch.optim.RMSprop([{'params': rest_param},
                {'params': visu_param, 'lr': args.lr/10.}], lr=args.lr, weight_decay=0.0005)

    # print([name for name, param in model.named_parameters() if param not in model.module.visumodel.parameters()])


    ## training and testing
    best_accu = -float('Inf')
    # accu_new = validate_epoch(val_loader, model, args.size_average)
    if args.test:
        _ = test_epoch(test_loader, model, args.size_average)
        exit(0)
    for epoch in range(args.nb_epoch):
        adjust_learning_rate(optimizer, epoch)
        train_epoch(train_loader, model, optimizer, epoch, args.size_average)
        accu_new = validate_epoch(val_loader, model, args.size_average)
        ## remember best accu and save checkpoint
        is_best = accu_new > best_accu
        best_accu = max(accu_new, best_accu)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': accu_new,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename=args.savename)
    print('\nBest Accu: %f\n'%best_accu)
    logging.info('\nBest Accu: %f\n'%best_accu)

def get_args():
    return args

def train_epoch(train_loader, model, optimizer, epoch, size_average):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    seg_losses=AverageMeter()
    acc = AverageMeter()
    acc_center = AverageMeter()
    miou = AverageMeter()
    acc_refine = AverageMeter()

    model.train()
    end = time.time()
    tbar=tqdm(train_loader)

    # for batch_idx, (imgs, word_id, word_mask, bbox) in enumerate(train_loader):
    for batch_idx, (imgs, word_id, word_mask, bbox) in enumerate(tbar):
        imgs = imgs.cuda()
        word_id = word_id.cuda()
        word_mask = word_mask.cuda()
        bbox = bbox.cuda()
        image = Variable(imgs)
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        bbox = Variable(bbox)
        bbox = torch.clamp(bbox,min=0,max=args.size-1)

        ## Note LSTM does not use word_mask
        pred_anchor,intmd_fea = model(image, word_id, word_mask)

        ## convert gt box to center+offset format
        gt_param, gi, gj, best_n_list,iou_all,coord_list = build_target(bbox, pred_anchor)
        ## flatten anchor dim at each scale
        # pred_conf_list=[]
        for ii in range(len (pred_anchor)):
            pred_anchor[ii] = pred_anchor[ii].view(pred_anchor[ii].size(0),3,5,pred_anchor[ii].size(2),pred_anchor[ii].size(3))
            # pred_conf_list.append(pred_anchor[ii][:, :, 4, :, :].contiguous().view(args.batch_size, -1))
        pred_conf_list = []
        for ii in range(len(pred_anchor)):
            pred_conf_list.append(pred_anchor[ii][:,:,4,:,:].contiguous().view(args.batch_size,-1))

        cam, bi_score, gt_score = model.module.segmentation((intmd_fea, bbox, pred_anchor, args))



        ## training offset eval: if correct with gt center loc
        ## convert offset pred to boxes
        pred_coord = torch.zeros(args.batch_size,4)
        for ii in range(args.batch_size):
            best_scale_ii = best_n_list[ii]//3
            grid, grid_size = args.size//(32//(2**best_scale_ii)), 32//(2**best_scale_ii)
            anchor_idxs = [x + 3*best_scale_ii for x in [0,1,2]]
            anchors = [anchors_full[i] for i in anchor_idxs]
            scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
                x[1] / (args.anchor_imsize/grid)) for x in anchors]

            pred_coord[ii,0] = F.sigmoid(pred_anchor[best_scale_ii][ii, best_n_list[ii]%3, 0, gj[ii], gi[ii]]) + gi[ii].float()
            pred_coord[ii,1] = F.sigmoid(pred_anchor[best_scale_ii][ii, best_n_list[ii]%3, 1, gj[ii], gi[ii]]) + gj[ii].float()
            pred_coord[ii,2] = torch.exp(pred_anchor[best_scale_ii][ii, best_n_list[ii]%3, 2, gj[ii], gi[ii]]) * scaled_anchors[best_n_list[ii]%3][0]
            pred_coord[ii,3] = torch.exp(pred_anchor[best_scale_ii][ii, best_n_list[ii]%3, 3, gj[ii], gi[ii]]) * scaled_anchors[best_n_list[ii]%3][1]
            pred_coord[ii,:] = pred_coord[ii,:] * grid_size
        pred_coord = xywh2xyxy(pred_coord)

        ## loss
        ref_loss=refine_loss(bi_score, gt_score)
        loss = yolo_loss(pred_anchor, gt_param, gi, gj, best_n_list) + ref_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), imgs.size(0))

        seg_losses.update(ref_loss.item(), imgs.size(0))
        rois_per_image = 8
        for ii in range(len(bi_score)):
            accr=np.sum(np.array(bi_score[ii].max(1)[1].data.cpu().numpy()== gt_score[ii].data.cpu().numpy(),dtype=float))/args.batch_size/rois_per_image/3
            acc_refine.update(accr, imgs.size(0)*rois_per_image*3)

        ## box iou
        target_bbox = bbox
        iou = bbox_iou(pred_coord, target_bbox.data.cpu(), x1y1x2y2=True)
        accu = np.sum(np.array((iou.data.cpu().numpy()>0.5),dtype=float))/args.batch_size
        ## evaluate if center location is correct
        # pred_conf_list, gt_conf_list = [], []
        # for ii in range(len(pred_anchor)):
        #     pred_conf_list.append(pred_anchor[ii][:,:,4,:,:].contiguous().view(args.batch_size,-1))
        #     gt_conf_list.append(gt_param[ii][:,:,4,:,:].contiguous().view(args.batch_size,-1))
        # pred_conf = torch.cat(pred_conf_list, dim=1)
        # gt_conf = torch.cat(gt_conf_list, dim=1)
        # accu_center = np.sum((pred_conf.max(1)[1] == gt_conf.max(1)[1]).cpu().numpy().astype(np.float32))/args.batch_size
        ## metrics
        accu_center=0.
        miou.update(iou.data[0], imgs.size(0))
        acc.update(accu, imgs.size(0))
        acc_center.update(accu_center, imgs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        tbar.set_description(
            'Seg Loss {seg_loss.avg:.4f} '\
                'Accu_R {acc_c.avg:.4f} ' \
            .format(seg_loss=seg_losses, acc_c=acc_refine))

        if args.save_plot:
            # if batch_idx%100==0 and epoch==args.nb_epoch-1:
            if True:
                save_segmentation_map(pred_coord,target_bbox,imgs,'train',batch_idx*imgs.size(0),\
                    save_path='./visulizations/%s/'%args.dataset)

        if (batch_idx+1) % args.print_freq == 0 or (batch_idx+1)==len(train_loader):
            print_str = '\rEpoch: [{0}][{1}/{2}]\t' \
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                'Seg Loss {seg_loss.val:.4f} ({seg_loss.avg:.4f})\t' \
                'Accu {acc.val:.4f} ({acc.avg:.4f})\t' \
                'Mean_iu {miou.val:.4f} ({miou.avg:.4f})\t' \
                'Accu_R {acc_c.val:.4f} ({acc_c.avg:.4f})\t' \
                .format( \
                    epoch, (batch_idx+1), len(train_loader), batch_time=batch_time, \
                    data_time=data_time, loss=losses, seg_loss=seg_losses, miou=miou, acc=acc, acc_c=acc_refine)
            print(print_str, end="\n")
            # print('\n')
            logging.info(print_str)


def validate_epoch(val_loader, model, size_average, mode='val'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    acc_refine= AverageMeter()
    acc_center = AverageMeter()
    miou = AverageMeter()

    model.eval()
    end = time.time()
    tbar = tqdm(val_loader)

    for batch_idx, (imgs, word_id, word_mask, bbox) in enumerate(tbar):
        imgs = imgs.cuda()
        word_id = word_id.cuda()
        word_mask = word_mask.cuda()
        bbox = bbox.cuda()
        image = Variable(imgs)
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        bbox = Variable(bbox)
        bbox = torch.clamp(bbox,min=0,max=args.size-1)

        with torch.no_grad():
            ## Note LSTM does not use word_mask
            pred_anchor,intmd_fea = model(image, word_id, word_mask)
        for ii in range(len(pred_anchor)):
            pred_anchor[ii] = pred_anchor[ii].view(   \
                    pred_anchor[ii].size(0),3,5,pred_anchor[ii].size(2),pred_anchor[ii].size(3))
        gt_param, target_gi, target_gj, best_n_list, _, _ = build_target(bbox, pred_anchor)


        ## eval: convert center+offset to box prediction
        ## calculate at rescaled image during validation for speed-up
        pred_conf_list, gt_conf_list = [], []
        for ii in range(len(pred_anchor)):
            pred_conf_list.append(pred_anchor[ii][:,:,4,:,:].contiguous().view(args.batch_size,-1))
            gt_conf_list.append(gt_param[ii][:,:,4,:,:].contiguous().view(args.batch_size,-1))

        pred_conf = torch.cat(pred_conf_list, dim=1)
        gt_conf = torch.cat(gt_conf_list, dim=1)
        max_conf, max_loc = torch.max(pred_conf, dim=1)

        pred_bbox = torch.zeros(args.batch_size,4)
        seg_bbox= torch.zeros(len(intmd_fea),args.batch_size,4).cuda()
        pred_gi, pred_gj, pred_best_n = [],[],[]
        for ii in range(args.batch_size):
            if max_loc[ii] < 3*(args.size//32)**2:
                best_scale = 0
            elif max_loc[ii] < 3*(args.size//32)**2 + 3*(args.size//16)**2:
                best_scale = 1
            else:
                best_scale = 2

            grid, grid_size = args.size//(32//(2**best_scale)), 32//(2**best_scale)
            anchor_idxs = [x + 3*best_scale for x in [0,1,2]]
            anchors = [anchors_full[i] for i in anchor_idxs]
            scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
                x[1] / (args.anchor_imsize/grid)) for x in anchors]

            pred_conf = pred_conf_list[best_scale].view(args.batch_size,3,grid,grid).data.cpu().numpy()
            max_conf_ii = max_conf.data.cpu().numpy()

            # print(max_conf[ii],max_loc[ii],pred_conf_list[best_scale][ii,max_loc[ii]-64])
            (best_n, gj, gi) = np.where(pred_conf[ii,:,:,:] == max_conf_ii[ii])
            best_n, gi, gj = int(best_n[0]), int(gi[0]), int(gj[0])
            pred_gi.append(gi)
            pred_gj.append(gj)
            pred_best_n.append(best_n+best_scale*3)

            pred_bbox[ii,0] = F.sigmoid(pred_anchor[best_scale][ii, best_n, 0, gj, gi]) + gi
            pred_bbox[ii,1] = F.sigmoid(pred_anchor[best_scale][ii, best_n, 1, gj, gi]) + gj
            pred_bbox[ii,2] = torch.exp(pred_anchor[best_scale][ii, best_n, 2, gj, gi]) * scaled_anchors[best_n][0]
            pred_bbox[ii,3] = torch.exp(pred_anchor[best_scale][ii, best_n, 3, gj, gi]) * scaled_anchors[best_n][1]

            for scale_ii in range(len(intmd_fea)):
                grid_ratio = (2 ** (scale_ii - best_scale))
                seg_bbox[scale_ii,ii,:]=pred_bbox[ii,:] * grid_ratio
            pred_bbox[ii,:] = pred_bbox[ii,:] * grid_size
        pred_bbox = xywh2xyxy(pred_bbox)
        target_bbox = bbox

        cam, bi_score = model.module.segmentation((intmd_fea, pred_bbox.cuda(), args))


        ## metrics
        iou = bbox_iou(pred_bbox, target_bbox.data.cpu(), x1y1x2y2=True)
        accu_center = np.sum(np.array((target_gi == np.array(pred_gi)) * (target_gj == np.array(pred_gj)), dtype=float))/args.batch_size
        accu = np.sum(np.array((iou.data.cpu().numpy()>0.5),dtype=float))/args.batch_size
        gt_onehot=np.array((iou.data.cpu().numpy()>0.5),dtype=float)


        for ii in range(len(bi_score)):
            accr=np.sum(np.array(bi_score[ii].max(1)[1].data.cpu().numpy()== gt_onehot,dtype=float))/args.batch_size
            acc_refine.update(accr, imgs.size(0))

        acc.update(accu, imgs.size(0))
        acc_center.update(accu_center, imgs.size(0))
        miou.update(iou.data[0], imgs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        tbar.set_description(
            'Acc refine {acc_refine.val:.4f} ({acc_refine.avg:.4f})'
            .format(acc_refine=acc_refine))
        if args.save_plot:
            if batch_idx%1==0:
                save_segmentation_map(pred_bbox,target_bbox,imgs,'val',batch_idx*imgs.size(0),\
                    save_path='./visulizations/%s/'%args.dataset)
        
        if (batch_idx+1) % args.print_freq == 0 or (batch_idx+1)==len(val_loader):
            print_str = '\r[{0}/{1}]\t' \
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                'Accu {acc.val:.4f} ({acc.avg:.4f})\t' \
                'Mean_iu {miou.val:.4f} ({miou.avg:.4f})\t' \
                'Accu_Refine {acc_c.val:.4f} ({acc_c.avg:.4f})\t' \
                .format( \
                    batch_idx+1, len(val_loader), batch_time=batch_time, \
                    acc=acc, acc_c=acc_refine, miou=miou)
            print(print_str, end="\n")
            logging.info(print_str)


    # print(best_n_list, pred_best_n)
    # print(np.array(target_gi), np.array(pred_gi))
    # print(np.array(target_gj), np.array(pred_gj),'-')
    # print(acc.avg, miou.avg,acc_center.avg)
    # print_str = '[{0}/{1}]\t' \
    #             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
    #             'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
    #             'Accu {acc.val:.4f} ({acc.avg:.4f})\t' \
    #             'Mean_iu {miou.val:.4f} ({miou.avg:.4f})\t' \
    #             'Accu_c {acc_c.val:.4f} ({acc_c.avg:.4f})\t' \
    #     .format( \
    #     batch_idx, len(val_loader), batch_time=batch_time, \
    #     data_time=data_time, \
    #     acc=acc, acc_c=acc_center, miou=miou)
    # print(print_str)
    #
    # logging.info("%f,%f,%f"%(acc.avg, float(miou.avg),acc_center.avg))
    return acc.avg

def test_epoch(val_loader, model, size_average, mode='test'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    acc_center = AverageMeter()
    miou = AverageMeter()

    model.eval()
    end = time.time()

    for batch_idx, (imgs, word_id, word_mask, bbox, ratio, dw, dh, im_id) in enumerate(val_loader):
        imgs = imgs.cuda()
        word_id = word_id.cuda()
        word_mask = word_mask.cuda()
        bbox = bbox.cuda()
        image = Variable(imgs)
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        bbox = Variable(bbox)
        bbox = torch.clamp(bbox,min=0,max=args.size-1)

        with torch.no_grad():
            ## Note LSTM does not use word_mask
            pred_anchor = model(image, word_id, word_mask)
        for ii in range(len(pred_anchor)):
            pred_anchor[ii] = pred_anchor[ii].view(   \
                    pred_anchor[ii].size(0),3,5,pred_anchor[ii].size(2),pred_anchor[ii].size(3))
        gt_param, target_gi, target_gj, best_n_list = build_target(bbox, pred_anchor)

        ## test: convert center+offset to box prediction
        pred_conf_list, gt_conf_list = [], []
        for ii in range(len(pred_anchor)):
            pred_conf_list.append(pred_anchor[ii][:,:,4,:,:].contiguous().view(1,-1))
            gt_conf_list.append(gt_param[ii][:,:,4,:,:].contiguous().view(1,-1))

        pred_conf = torch.cat(pred_conf_list, dim=1)
        gt_conf = torch.cat(gt_conf_list, dim=1)
        max_conf, max_loc = torch.max(pred_conf, dim=1)

        pred_bbox = torch.zeros(1,4)

        pred_gi, pred_gj, pred_best_n = [],[],[]
        for ii in range(1):
            if max_loc[ii] < 3*(args.size//32)**2:
                best_scale = 0
            elif max_loc[ii] < 3*(args.size//32)**2 + 3*(args.size//16)**2:
                best_scale = 1
            else:
                best_scale = 2

            grid, grid_size = args.size//(32//(2**best_scale)), 32//(2**best_scale)
            anchor_idxs = [x + 3*best_scale for x in [0,1,2]]
            anchors = [anchors_full[i] for i in anchor_idxs]
            scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
                x[1] / (args.anchor_imsize/grid)) for x in anchors]

            pred_conf = pred_conf_list[best_scale].view(1,3,grid,grid).data.cpu().numpy()
            max_conf_ii = max_conf.data.cpu().numpy()

            # print(max_conf[ii],max_loc[ii],pred_conf_list[best_scale][ii,max_loc[ii]-64])
            (best_n, gj, gi) = np.where(pred_conf[ii,:,:,:] == max_conf_ii[ii])
            best_n, gi, gj = int(best_n[0]), int(gi[0]), int(gj[0])
            pred_gi.append(gi)
            pred_gj.append(gj)
            pred_best_n.append(best_n+best_scale*3)

            pred_bbox[ii,0] = F.sigmoid(pred_anchor[best_scale][ii, best_n, 0, gj, gi]) + gi
            pred_bbox[ii,1] = F.sigmoid(pred_anchor[best_scale][ii, best_n, 1, gj, gi]) + gj
            pred_bbox[ii,2] = torch.exp(pred_anchor[best_scale][ii, best_n, 2, gj, gi]) * scaled_anchors[best_n][0]
            pred_bbox[ii,3] = torch.exp(pred_anchor[best_scale][ii, best_n, 3, gj, gi]) * scaled_anchors[best_n][1]
            pred_bbox[ii,:] = pred_bbox[ii,:] * grid_size
        pred_bbox = xywh2xyxy(pred_bbox)
        target_bbox = bbox.data.cpu()
        pred_bbox[:,0], pred_bbox[:,2] = (pred_bbox[:,0]-dw)/ratio, (pred_bbox[:,2]-dw)/ratio
        pred_bbox[:,1], pred_bbox[:,3] = (pred_bbox[:,1]-dh)/ratio, (pred_bbox[:,3]-dh)/ratio
        target_bbox[:,0], target_bbox[:,2] = (target_bbox[:,0]-dw)/ratio, (target_bbox[:,2]-dw)/ratio
        target_bbox[:,1], target_bbox[:,3] = (target_bbox[:,1]-dh)/ratio, (target_bbox[:,3]-dh)/ratio

        ## convert pred, gt box to original scale with meta-info
        top, bottom = round(float(dh[0]) - 0.1), args.size - round(float(dh[0]) + 0.1)
        left, right = round(float(dw[0]) - 0.1), args.size - round(float(dw[0]) + 0.1)
        img_np = imgs[0,:,top:bottom,left:right].data.cpu().numpy().transpose(1,2,0)

        ratio = float(ratio)
        new_shape = (round(img_np.shape[1] / ratio), round(img_np.shape[0] / ratio))
        ## also revert image for visualization
        img_np = cv2.resize(img_np, new_shape, interpolation=cv2.INTER_CUBIC)
        img_np = Variable(torch.from_numpy(img_np.transpose(2,0,1)).cuda().unsqueeze(0))

        pred_bbox[:,:2], pred_bbox[:,2], pred_bbox[:,3] = \
            torch.clamp(pred_bbox[:,:2], min=0), torch.clamp(pred_bbox[:,2], max=img_np.shape[3]), torch.clamp(pred_bbox[:,3], max=img_np.shape[2])
        target_bbox[:,:2], target_bbox[:,2], target_bbox[:,3] = \
            torch.clamp(target_bbox[:,:2], min=0), torch.clamp(target_bbox[:,2], max=img_np.shape[3]), torch.clamp(target_bbox[:,3], max=img_np.shape[2])

        iou = bbox_iou(pred_bbox, target_bbox, x1y1x2y2=True)
        accu_center = np.sum(np.array((target_gi == np.array(pred_gi)) * (target_gj == np.array(pred_gj)), dtype=float))/1
        accu = np.sum(np.array((iou.data.cpu().numpy()>0.5),dtype=float))/1

        acc.update(accu, imgs.size(0))
        acc_center.update(accu_center, imgs.size(0))
        miou.update(iou.data[0], imgs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.save_plot:
            if batch_idx%1==0:
                save_segmentation_map(pred_bbox,target_bbox,img_np,'test',batch_idx*imgs.size(0),\
                    save_path='./visulizations/%s/'%args.dataset)
        
        if batch_idx % args.print_freq == 0:
            print_str = '[{0}/{1}]\t' \
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                'Accu {acc.val:.4f} ({acc.avg:.4f})\t' \
                'Mean_iu {miou.val:.4f} ({miou.avg:.4f})\t' \
                'Accu_c {acc_c.val:.4f} ({acc_c.avg:.4f})\t' \
                .format( \
                    batch_idx, len(val_loader), batch_time=batch_time, \
                    data_time=data_time, \
                    acc=acc, acc_c=acc_center, miou=miou)
            print(print_str)
            logging.info(print_str)
    # print(best_n_list, pred_best_n)
    # print(np.array(target_gi), np.array(pred_gi))
    # print(np.array(target_gj), np.array(pred_gj),'-')
    # print(acc.avg, miou.avg,acc_center.avg)

    print_str = '[{0}/{1}]\t' \
        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
        'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
        'Accu {acc.val:.4f} ({acc.avg:.4f})\t' \
        'Mean_iu {miou.val:.4f} ({miou.avg:.4f})\t' \
        'Accu_c {acc_c.val:.4f} ({acc_c.avg:.4f})\t' \
        .format( \
            batch_idx, len(val_loader), batch_time=batch_time, \
            data_time=data_time, \
            acc=acc, acc_c=acc_center, miou=miou)
    print(print_str)

    logging.info("%f,%f,%f"%(acc.avg, float(miou.avg),acc_center.avg))
    return acc.avg


if __name__ == "__main__":
    main()
