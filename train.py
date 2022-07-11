from __future__ import division

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import argparse
import time
import cv2
import sys
import math
import numpy as np
from data.voc import VOCDetection
# from data.coco import COCODataset
from data.transforms import TrainTransforms, ValTransforms
# from evaluator.cocoapi_evaluator import COCOAPIEvaluator
from evaluator.vocapi_evaluator import VOCAPIEvaluator
from utils import distributed_utils
from utils.matcher import build_matcher
from utils.loss import build_criterion
from utils.com_flops_params import FLOPs_and_Params
from utils.misc import detection_collate, ModelEMA
from models.detr import DeTR


def parse_args():
    parser = argparse.ArgumentParser(description='DeTR Detection')
    # basic
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use cuda.')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size on each gpu')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='initial learning rate')
    parser.add_argument('--lr_backbone', default=1e-5, type=float,
                        help='lr for backbone')
    parser.add_argument('-size', '--img_size', default=512, type=int,
                        help='input size: [H, W].')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--max_epoch', type=int, default=150,
                        help='max epoch to train')
    parser.add_argument('--lr_drop', type=int, default=100,
                        help='lr decay epoch')
    parser.add_argument('--eval_epoch', type=int, default=10,
                        help='interval between evaluations')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='visualize target.')

    # Train
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch to train')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='keep training')
    parser.add_argument('--tfboard', action='store_true', default=True,
                        help='use tensorboard')

    # data loader
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading')

    # Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # Loss
    parser.add_argument('--aux_loss', action='store_true',
                        help="Use auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # Model
    parser.add_argument('-bk', '--backbone', default='r50', type=str,
                        help='backbone')
    parser.add_argument('--use_nms', action='store_true',
                        help="use nms to eval")

    # Transformer
    parser.add_argument('--num_encoders', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--num_decoders', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--mlp_dim', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--num_heads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true', default=False,
                        help="pre_norm")
    parser.add_argument('--batch_first', action='store_true', default=False,
                        help="batch first in MultiHeadAttention.")

    # dataset
    parser.add_argument('-root', '--data_root', default='/home/yifux/',
                        help='root to dataset')
    parser.add_argument('-d', '--dataset', default='voc',
                        help='voc or coco')

    # train trick
    parser.add_argument('--ema', action='store_true', default=False,
                        help='use ema training trick')
    parser.add_argument('--no_warmup', action='store_true', default=False,
                        help='do not use warmup')
    parser.add_argument('--wp_epoch', type=int, default=1,
                        help='wram-up epoch')

    # train DDP
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of GPUs.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local_rank')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--sybn', action='store_true', default=False,
                        help='use sybn.')

    return parser.parse_args()


def train():
    args = parse_args()
    distributed_utils.init_distributed_mode(args)
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # path to save model
    path_to_save = os.path.join('weights/', args.dataset)
    os.makedirs(path_to_save, exist_ok=True)

    # dataset and evaluator
    dataset, evaluator, num_classes = build_dataset(args, args.img_size, device)
    # dataloader
    dataloader = build_dataloader(args, dataset, detection_collate)

    # build model
    model = DeTR(args=args,
                 device=device,
                 img_size=args.img_size,
                 num_classes=num_classes,
                 trainable=True,
                 aux_loss=args.aux_loss,
                 use_nms=args.use_nms).to(device)
    model = model.to(device)
    # build matcher
    matcher = build_matcher(args)
    # build criterion
    criterion = build_criterion(args, device, matcher, num_classes)
    criterion.train()

    if distributed_utils.get_rank() == 0:
        model.trainable = False
        model.eval()
        # compute FLOPs and Params
        FLOPs_and_Params(model, args.img_size, device)
        model.trainable = True
        model.train()

    # DDP
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    model_without_ddp.train()

    # SyncBatchNorm
    if args.distributed:
        print('use SyncBatchNorm ...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # model params
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    # optimizer
    optimizer = torch.optim.AdamW(param_dicts,
                                  lr=args.lr,
                                  weight_decay=1e-4)
    # lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # keep training
    if args.resume is not None:
        print('keep training model: %s' % (args.resume))
        if args.distributed:
            model.module.load_state_dict(torch.load(args.resume, map_location=device))
        else:
            model.load_state_dict(torch.load(args.resume, map_location=device))

    # EMA
    ema = ModelEMA(model_without_ddp) if args.ema else None

    # use tfboard
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        log_path = os.path.join('log/', args.dataset, c_time)
        os.makedirs(log_path, exist_ok=True)

        tblogger = SummaryWriter(log_path)

    # basic
    world_size = distributed_utils.get_world_size()
    epoch_size = len(dataset) // (args.batch_size * world_size)
    base_lr = args.lr
    tmp_lr = base_lr
    warmup = not args.no_warmup

    best_map = 0.
    t0 = time.time()

    print("----------------------------------------------------------")
    print('Training model on:', args.dataset)
    print('The dataset size:', len(dataset))
    print("----------------------------------------------------------")
    # start train
    for epoch in range(args.start_epoch, args.max_epoch):
        # set epoch if DDP
        if args.distributed:
            dataloader.sampler.set_epoch(epoch)

        # # close mosaic and mixup
        # if args.max_epoch - epoch < 15:
        #     if args.mosaic:
        #         dataloader.dataset.mosaic = False
        #     if args.mixup:
        #         dataloader.dataset.mixup = False

        # train one epoch
        for iter_i, (images, targets) in enumerate(dataloader):
            ni = iter_i + epoch * epoch_size
            # warmup
            if epoch < args.wp_epoch and warmup:
                nw = args.wp_epoch * epoch_size
                tmp_lr = base_lr * pow(ni / nw, 4)
                set_lr(optimizer, tmp_lr)

            elif epoch == args.wp_epoch and iter_i == 0 and warmup:
                # warmup is over
                warmup = False
                tmp_lr = base_lr
                set_lr(optimizer, tmp_lr)

            # visualize target
            if args.vis:
                vis_data(images, targets, args.img_size)
                continue

            # to device
            images = images.to(device)
            targets = [{"labels": v["labels"].to(device),
                        "boxes": v["boxes"].to(device)} for v in targets]

            # forward
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                          for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            optimizer.step()

            # ema update
            if args.ema:
                ema.update(model)

            # display
            if iter_i % 10 == 0:
                if args.tfboard:
                    # viz loss
                    for k in loss_dict_reduced_unscaled.keys():
                        tblogger.add_scalar(k, loss_dict_reduced_unscaled[k].item(), ni)

                t1 = time.time()
                out_stream = '>>>>>>>>>> Training Stage [Epoch %d/%d] <<<<<<<<<< \n' % (
                    epoch + 1, args.max_epoch)
                out_stream += '[Iter %d/%d][lr %.6f][size: %d] \n' % (
                    iter_i,
                    epoch_size,
                    tmp_lr,
                    args.img_size)
                for k in loss_dict_reduced_unscaled.keys():
                    v = loss_dict_reduced_unscaled[k].item()
                    out_stream += '[' + str(k) + ': ' + str(round(v, 3)) + '] \n'
                out_stream += '[time: ' + str(round(t1 - t0, 3)) + ']'
                print(out_stream, flush=True)

                t0 = time.time()

        lr_scheduler.step()

        # evaluation
        if (epoch + 1) % args.eval_epoch == 0 or epoch + 1 == args.max_epoch:
            if evaluator is None:
                print('No evaluator ... go on training.')
                print('Saving state, epoch:', epoch + 1)
                # torch.save(model_eval.state_dict(), os.path.join(path_to_save,
                #                                                  'DeTR_' + repr(epoch + 1) + '.pth'))
                torch.save(model_eval.state_dict(), os.path.join(path_to_save,
                                                                 args.backbone + '_DeTR_' + repr(epoch + 1) + '.pth'))
            else:
                print('eval ...')
                # check ema
                if args.ema:
                    model_eval = ema.ema
                else:
                    model_eval = model.module if args.distributed else model

                # set eval mode
                model_eval.trainable = False
                model_eval.eval()

                if distributed_utils.get_rank() == 0:
                    # evaluate
                    evaluator.evaluate(model_eval)

                    cur_map = evaluator.map
                    if cur_map > best_map:
                        # update best-map
                        best_map = cur_map
                        # save model
                        print('Saving state, epoch:', epoch + 1)
                        # torch.save(model_eval.state_dict(), os.path.join(path_to_save,
                        #                                                  'DeTR_' + repr(epoch + 1) + '_' + str(
                        #                                                      round(best_map * 100., 2)) + '.pth'))
                        torch.save(model_eval.state_dict(), os.path.join(path_to_save, args.backbone +
                                                                         '_detr_' + repr(epoch + 1) + '_mAP_' + str(
                            round(best_map * 100., 2)) + '.pth'))

                    if args.tfboard:
                        if args.dataset == 'voc':
                            tblogger.add_scalar('07test/mAP', evaluator.map, epoch)
                        elif args.dataset == 'coco':
                            tblogger.add_scalar('val/AP50_95', evaluator.ap50_95, epoch)
                            tblogger.add_scalar('val/AP50', evaluator.ap50, epoch)

                if args.distributed:
                    # wait for all processes to synchronize
                    dist.barrier()

                # set train mode.
                model_eval.trainable = True
                model_eval.train()

    if args.tfboard:
        tblogger.close()


def build_dataset(args, img_size, device):
    if args.dataset == 'voc':
        data_dir = os.path.join(args.data_root, 'VOCdevkit')
        num_classes = 20
        dataset = VOCDetection(
            data_dir=data_dir,
            transform=TrainTransforms(img_size))

        evaluator = VOCAPIEvaluator(
            data_dir=data_dir,
            device=device,
            transform=ValTransforms(img_size))

    # elif args.dataset == 'coco':
    #     data_dir = os.path.join(args.data_root, 'COCO')
    #     num_classes = 80
    #     dataset = COCODataset(
    #         data_dir=data_dir,
    #         transform=TrainTransforms(img_size),
    #         image_set='train2017')
    #
    #     evaluator = COCOAPIEvaluator(
    #         data_dir=data_dir,
    #         device=device,
    #         transform=ValTransforms(img_size)
    #     )

    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)

    return dataset, evaluator, num_classes


def build_dataloader(args, dataset, collate_fn=None):
    # distributed
    if args.distributed and args.num_gpu > 1:
        # dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
            sampler=torch.utils.data.distributed.DistributedSampler(dataset)
        )

    else:
        # dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True
        )
    return dataloader


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def vis_data(images, targets, input_size):
    # vis data
    img_h = img_w = input_size
    mean = (0.406, 0.456, 0.485)
    std = (0.225, 0.224, 0.229)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    for bi in range(images.size(0)):
        # image
        img = images[bi].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
        img = ((img * std + mean) * 255).astype(np.uint8)
        cv2.imwrite('1.jpg', img)
        img_ = cv2.imread('1.jpg')
        # bboox
        bboxes = targets[bi]['boxes']
        for bbox in bboxes:
            cx, cy, bw, bh = bbox
            # print(x1, y1, x2, y2)
            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)
            x2 = int((cx + bw / 2) * img_w)
            y2 = int((cy + bh / 2) * img_h)
            cv2.rectangle(img_, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('img', img_)
    cv2.waitKey(0)


if __name__ == '__main__':
    train()
