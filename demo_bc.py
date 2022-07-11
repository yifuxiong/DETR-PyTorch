import argparse
import os
import cv2
import time
import numpy as np
import torch

# from data.coco import coco_class_labels, coco_class_index

# voc_class_labels = [  # always index 0
#     'aeroplane', 'bicycle', 'bird', 'boat',
#     'bottle', 'bus', 'car', 'cat', 'chair',
#     'cow', 'diningtable', 'dog', 'horse',
#     'motorbike', 'person', 'pottedplant',
#     'sheep', 'sofa', 'train', 'tvmonitor'
# ]
# voc_class_index = np.arange(0, 20)

voc_class_labels = ['strain area']
voc_class_index = np.array([0])

from data.transforms import ValTransforms
from models.detr import DeTR


def parse_args():
    parser = argparse.ArgumentParser(description='DeTR Demo Detection')

    # basic
    parser.add_argument('--mode', default='image',
                        type=str, help='Use the data from image, video or camera')
    parser.add_argument('--img_size', default=512,  # swin: 256
                        type=int, help='input image size')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use cuda')
    parser.add_argument('--path_to_img', default='data/demo/images/',
                        type=str, help='The path to image files')
    parser.add_argument('--path_to_vid', default='data/demo/videos/traffic_accident.gif',
                        type=str, help='The path to video files')
    parser.add_argument('--path_to_save', default='data/',
                        type=str, help='The path to save the detection results video')
    parser.add_argument('--path_to_saveVid', default='data/video/result.avi',
                        type=str, help='The path to save the detection results video')
    parser.add_argument('-vs', '--visual_threshold', default=0.5,
                        type=float, help='visual threshold')

    # model
    parser.add_argument('--version', default='backbone_detr', type=str,
                        help='resnet50 + detr')
    parser.add_argument('-tm', '--trained_model', default='weights/voc/xxx.pth', type=str,
                        help='model path')
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


def plot_bbox_labels(img, bbox, label, cls_color, test_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
    # plot title bbox
    cv2.rectangle(img, (x1, y1 - t_size[1]), (int(x1 + t_size[0] * test_scale), y1), cls_color, -1)
    # put the test on the title bbox
    cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, test_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img


def visualize(img, bboxes, scores, cls_inds, class_colors, vis_thresh=0.3):
    ts = 0.4
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:
            cls_color = class_colors[int(cls_inds[i])]
            cls_id = voc_class_index[int(cls_inds[i])]
            mess = '%s: %.2f' % (voc_class_labels[cls_id], scores[i])
            img = plot_bbox_labels(img, bbox, mess, cls_color, test_scale=ts)

    return img


def detect(net,
           device,
           transform,
           vis_thresh,
           mode='image',
           path_to_img=None,
           path_to_vid=None,
           path_to_save=None):
    # class color
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(80)]
    save_path = os.path.join(path_to_save, mode)
    os.makedirs(save_path, exist_ok=True)

    # ------------------------- Camera ----------------------------
    if mode == 'camera':
        print('use camera !!!')
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while True:
            ret, frame = cap.read()
            if ret:
                if cv2.waitKey(1) == ord('q'):
                    break
                img_h, img_w = frame.shape[:2]
                scale = np.array([[img_w, img_h, img_w, img_h]])
                # prepare
                x = transform(frame)[0]
                x = x.unsqueeze(0).to(device)
                # inference
                t0 = time.time()
                bboxes, scores, cls_inds = net(x)
                t1 = time.time()
                print("detection time used ", t1 - t0, "s")

                # rescale
                bboxes *= scale

                frame_processed = visualize(img=frame,
                                            bboxes=bboxes,
                                            scores=scores,
                                            cls_inds=cls_inds,
                                            class_colors=class_colors,
                                            vis_thresh=vis_thresh)
                cv2.imshow('detection result', frame_processed)
                cv2.waitKey(1)
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

    # ------------------------- Image ----------------------------
    elif mode == 'image':
        for i, img_id in enumerate(os.listdir(path_to_img)):
            img = cv2.imread(path_to_img + '/' + img_id, cv2.IMREAD_COLOR)
            img_h, img_w = img.shape[:2]
            print('img_size:', img_h, img_w)
            scale = np.array([[img_w, img_h, img_w, img_h]])

            # prepare
            x = transform(img)[0]
            x = x.unsqueeze(0).to(device)
            # inference
            t0 = time.time()
            bboxes, scores, cls_inds = net(x)
            t1 = time.time()
            print("detection time used ", t1 - t0, "s")

            # rescale
            bboxes *= scale

            img_processed = visualize(img=img,
                                      bboxes=bboxes,
                                      scores=scores,
                                      cls_inds=cls_inds,
                                      class_colors=class_colors,
                                      vis_thresh=vis_thresh)

            cv2.imshow('detection', img_processed)
            cv2.imwrite(os.path.join(save_path, str(i).zfill(6) + '.jpg'), img_processed)
            cv2.waitKey(0)

    # ------------------------- Video ---------------------------
    elif mode == 'video':
        video = cv2.VideoCapture(path_to_vid)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        save_size = (640, 480)
        save_path = os.path.join(save_path, 'det.avi')
        fps = 15.0
        out = cv2.VideoWriter(save_path, fourcc, fps, save_size)

        while (True):
            ret, frame = video.read()

            if ret:
                # ------------------------- Detection ---------------------------
                img_h, img_w = frame.shape[:2]
                scale = np.array([[img_w, img_h, img_w, img_h]])
                # prepare
                x = transform(frame)[0]
                x = x.unsqueeze(0).to(device)
                # inference
                t0 = time.time()
                bboxes, scores, cls_inds = net(x)
                t1 = time.time()
                print("detection time used ", t1 - t0, "s")

                # rescale
                bboxes *= scale

                frame_processed = visualize(img=frame,
                                            bboxes=bboxes,
                                            scores=scores,
                                            cls_inds=cls_inds,
                                            class_colors=class_colors,
                                            vis_thresh=vis_thresh)

                frame_processed_resize = cv2.resize(frame_processed, save_size)
                out.write(frame_processed_resize)
                cv2.imshow('detection', frame_processed)
                cv2.waitKey(1)
            else:
                break
        video.release()
        out.release()
        cv2.destroyAllWindows()


def run():
    args = parse_args()

    # use cuda
    if args.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_name = args.version
    print('Model: ', model_name)

    # build model
    model = DeTR(args=args,
                 device=device,
                 img_size=args.img_size,
                 num_classes=20,
                 trainable=False,
                 conf_thresh=args.conf_thresh,
                 nms_thresh=args.nms_thresh,
                 aux_loss=args.aux_loss,
                 use_nms=args.use_nms).to(device)

    # input = torch.randn((1, 3, 512, 512))
    # bboxes, scores, cls_inds = model(input)
    # print(bboxes.shape)
    # print(scores.shape)
    # print(cls_inds.shape)

    # load weight
    model.load_state_dict(torch.load(args.trained_model, map_location=device), strict=False)
    model.to(device).eval()
    print('Finished loading model!')

    # run
    detect(net=model,
           device=device,
           transform=ValTransforms(args.img_size),
           mode=args.mode,
           vis_thresh=args.visual_threshold,
           path_to_img=args.path_to_img,
           path_to_vid=args.path_to_vid,
           path_to_save=args.path_to_save)


if __name__ == '__main__':
    run()
