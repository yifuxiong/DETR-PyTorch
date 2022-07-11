import torch
import argparse
from thop import profile  # thop==0.0.31.post2005241907
from models.detr import DeTR


def FLOPs_and_Params(model, size, device):
    x = torch.randn(1, 3, size, size).to(device)

    with torch.no_grad():
        flops, params = profile(model, inputs=(x,))
    print('- FLOPs : ', flops / 1e6, ' M')
    print('- Params : ', params / 1e6, ' M')


def parse_args():
    parser = argparse.ArgumentParser(description='DeTR Demo Detection')

    # basic
    parser.add_argument('--img_size', default=512,  # swin: 256
                        type=int, help='input image size')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use cuda')

    # model
    parser.add_argument('-tm', '--trained_model', default='../weights/voc/DeTR_143_47.56.pth', type=str,
                        help='model path')
    parser.add_argument('-bk', '--backbone', default='r50', type=str,
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
    parser.add_argument('--mlp_dim', default=2048, type=int,  # swin: 1024
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


if __name__ == "__main__":
    args = parse_args()
    size = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeTR(args=args,
                 device=device,
                 img_size=size,
                 num_classes=20,
                 trainable=False,
                 conf_thresh=0.5,
                 nms_thresh=0.5,
                 aux_loss=True,
                 use_nms=True).to(device)

    # load weight
    model.load_state_dict(torch.load(args.trained_model, map_location=device), strict=False)
    model.to(device).eval()

    FLOPs_and_Params(model, size, device)
