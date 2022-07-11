import torch
import argparse
from models.detr import DeTR


def parse_args():
    parser = argparse.ArgumentParser(description='DeTR Demo Detection')

    # basic
    parser.add_argument('--mode', default='image',
                        type=str, help='Use the data from image, video or camera')
    # swin: 256
    parser.add_argument('--img_size', default=512,
                        type=int, help='input image size')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use cuda')
    # model
    parser.add_argument('--version', default='backbone_detr', type=str,
                        help='backbone + detr')
    parser.add_argument('-tm', '--trained_model', default='', type=str,
                        help='model path')
    parser.add_argument('-bk', '--backbone', default='mb2', type=str,
                        help='backbone')
    parser.add_argument('--conf_thresh', default=0.5, type=float,
                        help='Confidence threshold')
    parser.add_argument('--nms_thresh', default=0.5, type=float,
                        help='NMS threshold')
    parser.add_argument('--use_nms', action='store_true', default=False,
                        help='use nms.')
    parser.add_argument('--aux_loss', action='store_true',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # Transformer
    parser.add_argument('--num_encoders', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--num_decoders', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    # swin: 1024
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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeTR(args=args,
                 device=device,
                 img_size=args.img_size,
                 num_classes=20,
                 trainable=False,
                 conf_thresh=args.conf_thresh,
                 nms_thresh=args.nms_thresh,
                 aux_loss=args.aux_loss,
                 use_nms=args.use_nms).to(device)

    input = torch.randn(1, 3, 512, 512).to(device)
    bboxes, scores, cls_inds = model(input)
    print('bboxes.shape:', bboxes.shape)
    print('scores.shape:', scores.shape)
    print('cls_inds.shape:', cls_inds.shape)
