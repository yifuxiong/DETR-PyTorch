import torch
import argparse
import os
from data.transforms import ValTransforms
from models.detr import DeTR
from evaluator.vocapi_evaluator import VOCAPIEvaluator
# from evaluator.bc_vocapi_evaluator import VOCAPIEvaluator
from evaluator.cocoapi_evaluator import COCOAPIEvaluator

parser = argparse.ArgumentParser(description='DeTR Detection')
# dataset
parser.add_argument('--version', default='backbone_detr', type=str,
                    help='resnet50 + detr')
parser.add_argument('-root', '--data_root', default='D:/',
                    help='root to dataset')
parser.add_argument('-d', '--dataset', default='voc',
                    help='voc or coco')
# basic
parser.add_argument('-size', '--img_size', default=512, type=int,
                    help='img_size')
parser.add_argument('--trained_model', default='weights/voc/mb2_detr_108_mAP_48.47.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use cuda.')
# model
parser.add_argument('-bk', '--backbone', default='mb2', type=str,
                    help='backbone')
parser.add_argument('--conf_thresh', default=0.5, type=float,
                    help='Confidence threshold')
parser.add_argument('--nms_thresh', default=0.5, type=float,
                    help='NMS threshold')
parser.add_argument('--use_nms', action='store_true', default=True,
                    help='use nms.')
parser.add_argument('--aux_loss', action='store_true',
                    help="Disables auxiliary decoding losses (loss at each layer)")
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
args = parser.parse_args()


def voc_test(model, data_dir, device, img_size):
    evaluator = VOCAPIEvaluator(
        data_dir=data_dir,
        device=device,
        transform=ValTransforms(img_size),
        display=True)

    # VOC evaluation
    evaluator.evaluate(model)


def coco_test(model, data_dir, device, img_size, test=False):
    if test:
        # test-dev
        print('test on test-dev 2017')
        evaluator = COCOAPIEvaluator(
            data_dir=data_dir,
            device=device,
            testset=True,
            transform=ValTransforms(img_size)
        )
    else:
        # eval
        evaluator = COCOAPIEvaluator(
            data_dir=data_dir,
            device=device,
            testset=False,
            transform=ValTransforms(img_size)
        )

    # COCO evaluation
    evaluator.evaluate(model)


if __name__ == '__main__':
    # dataset
    if args.dataset == 'voc':
        print('eval on voc ...')
        num_classes = 20
        data_dir = os.path.join(args.data_root, 'VOCdevkit')
    elif args.dataset == 'coco-val':
        print('eval on coco-val ...')
        num_classes = 80
        data_dir = os.path.join(args.data_root, 'COCO')
    elif args.dataset == 'coco-test':
        print('eval on coco-test-dev ...')
        num_classes = 80
        data_dir = os.path.join(args.data_root, 'COCO')
    else:
        print('unknow dataset !! we only support voc, coco-val, coco-test !!!')
        exit(0)

    # cuda
    if args.cuda:
        print('use cuda')
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # model
    model_name = args.version
    print('Model: ', model_name)

    # build model
    model = DeTR(args=args,
                 device=device,
                 img_size=args.img_size,
                 num_classes=num_classes,
                 trainable=False,
                 conf_thresh=args.conf_thresh,
                 nms_thresh=args.nms_thresh,
                 aux_loss=args.aux_loss,
                 use_nms=args.use_nms).to(device)

    # load weight
    model.load_state_dict(torch.load(args.trained_model, map_location=device), strict=False)
    model.to(device).eval()
    print('Finished loading model!')

    # evaluation
    with torch.no_grad():
        if args.dataset == 'voc':
            voc_test(model, data_dir, device, args.img_size)
        elif args.dataset == 'coco-val':
            coco_test(model, data_dir, device, args.img_size, test=False)
        elif args.dataset == 'coco-test':
            coco_test(model, data_dir, device, args.img_size, test=True)
