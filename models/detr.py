import torch
import torch.nn as nn
import math
import numpy as np
from copy import deepcopy
from .backbone import build_backbone
from .transformer import build_transformer
# from .transformer_simple import build_transformer
from .mlp import MLP
import utils.box_ops as box_ops


class DeTR(nn.Module):
    def __init__(self,
                 args,
                 device,
                 img_size=512,
                 num_classes=20,
                 trainable=False,
                 conf_thresh=0.01,
                 nms_thresh=0.5,
                 aux_loss=False,
                 use_nms=False):
        super().__init__()
        self.device = device
        self.img_size = img_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.num_queries = args.num_queries
        self.aux_loss = aux_loss
        self.use_nms = use_nms

        # backbone
        self.backbone, feature_channels = build_backbone(pretrained=trainable, freeze_bn=trainable,
                                                         model=args.backbone)
        self.stride = 32

        # to compress channel of C5
        self.input_proj = nn.Conv2d(feature_channels, args.hidden_dim, kernel_size=1)

        # object query
        # 随机生成shape=(Nq, C=256)的向量矩阵
        self.query_embed = nn.Embedding(self.num_queries, args.hidden_dim)
        # position embedding
        self.pos_embed = self.position_embedding(num_pos_feats=args.hidden_dim // 2, normalize=True)

        # transformer
        self.transformer = build_transformer(args)

        # det
        # 分类
        self.cls_det = nn.Linear(args.hidden_dim, num_classes + 1)
        # 回归
        self.reg_det = MLP(args.hidden_dim, args.hidden_dim, 4, 3)

    # Position Embedding
    def position_embedding(self, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        hs = ws = self.img_size // self.stride

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        # generate xy coord mat
        # y_embed = [[0, 0, 0, ...], [1, 1, 1, ...]...]
        # x_embed = [[0, 1, 2, ...], [0, 1, 2, ...]...]
        y_embed, x_embed = torch.meshgrid([torch.arange(1, hs + 1, dtype=torch.float32),
                                           torch.arange(1, ws + 1, dtype=torch.float32)])
        if normalize:
            eps = 1e-6
            y_embed = y_embed / (hs + eps) * scale
            x_embed = x_embed / (ws + eps) * scale

        # [H, W] -> [1, H, W]
        y_embed = y_embed[None, :, :].to(self.device)
        x_embed = x_embed[None, :, :].to(self.device)

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=self.device)
        # torch.div(a, b, rounding_mode='floor') == (a // b)
        dim_t_ = torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats  # pytorch>=1.8.0
        dim_t = temperature ** (2 * dim_t_)

        pos_x = torch.div(x_embed[:, :, :, None], dim_t)
        pos_y = torch.div(y_embed[:, :, :, None], dim_t)
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # [B, d, H, W]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos

    @torch.jit.unused
    def set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def nms(self, dets, scores):
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0]  # xmin
        y1 = dets[:, 1]  # ymin
        x2 = dets[:, 2]  # xmax
        y2 = dets[:, 3]  # ymax

        areas = (x2 - x1) * (y2 - y1)  # the size of bbox
        order = scores.argsort()[::-1]  # sort bounding boxes by decreasing order

        keep = []  # store the final bounding boxes
        while order.size > 0:
            i = order[0]  # the index of the bbox with highest confidence
            keep.append(i)  # save it to keep
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            # Cross Area / (bbox + particular area - Cross Area)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]
        return keep

    def soft_nms(self, dets, scores, score_thresh=0., method=2, sigma2=0.5):
        '''
        :param dets: 对dets后添加对应的下标[0, 1, 2, ...]，dets.shape=(n, 5)
        :param scores: confidence
        :param nms_thresh: self.nms_thresh替代
        :param sigma2: gaussian sigma
        :param score_thresh: confidence thresh，detr里面设为0
        :param method:
        :return: index
        '''
        # dets_copy = deepcopy(dets)
        N = dets.shape[0]
        # index: (0, 1, 2, ..., n-1)
        indexes = np.array([np.arange(N)])
        # 将bboxes和对应的index拼接
        dets = np.concatenate((dets, indexes.T), axis=1)

        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        for i in range(N):
            # 找出i后面的最大score及其index
            pos = i + 1
            if i != N - 1:
                maxscore = np.max(scores[pos:], axis=0)
                maxpos = np.argmax(scores[pos:], axis=0)
            else:
                maxscore = scores[-1]
                maxpos = 0
            # 保证当前下标i的score最大
            if scores[i] < maxscore:
                dets[[i, maxpos + i + 1]] = dets[[maxpos + i + 1, i]]
                scores[[i, maxpos + i + 1]] = scores[[maxpos + i + 1, i]]
                areas[[i, maxpos + i + 1]] = areas[[maxpos + i + 1, i]]

            # IoU calculate
            xx1 = np.maximum(dets[i, 0], dets[pos:, 0])
            yy1 = np.maximum(dets[i, 1], dets[pos:, 1])
            xx2 = np.minimum(dets[i, 2], dets[pos:, 2])
            yy2 = np.minimum(dets[i, 3], dets[pos:, 3])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = w * h
            iou = intersection / (areas[i] + areas[pos:] - intersection)

            # Three methods: 1.linear 2.gaussian 3.original NMS
            if method == 1:
                weight = np.ones(iou.shape)
                weight[iou > self.nms_thresh] = weight[iou > self.nms_thresh] - iou[iou > self.nms_thresh]
            elif method == 2:
                weight = np.exp(-(iou * iou) / sigma2)
            else:
                weight = np.ones(iou.shape)
                weight[iou > self.nms_thresh] = 0
            scores[pos:] = weight * scores[pos:]

        # select the boxes and keep the corresponding indexes
        inds = dets[:, 4][scores > score_thresh]
        keep = inds.astype(int)
        # return dets_copy[keep]
        # keep = list(inds)
        return keep

    def forward(self, x):
        # backbone
        x = self.backbone(x)
        x = self.input_proj(x)

        # # transformer
        # h = self.transformer(x, self.query_embed.weight, self.pos_embed)[0]
        # # print('h.shape:', h.shape)
        #
        # # output: [M, B, N, C] where M = num_decoder since we use all intermediate outputs of decoder
        # outputs_class = self.cls_det(h)
        # outputs_coord = self.reg_det(h).sigmoid()
        # # print('cls_det.shape:', outputs_class.shape)
        # # print('reg_det.shape:', outputs_coord.shape)

        # # we only compute the loss of last output from decoder
        # outputs = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        # if self.aux_loss:
        #     outputs['aux_outputs'] = self.set_aux_loss(outputs_class, outputs_coord)

        h, reference = self.transformer(x, self.query_embed.weight, self.pos_embed)
        reference_before_sigmoid = inverse_sigmoid(reference)
        outputs_coords = []
        for lvl in range(h.shape[0]):
            tmp = self.reg_det(h[lvl])
            tmp[..., :2] += reference_before_sigmoid
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
        outputs_coord = torch.stack(outputs_coords)

        outputs_class = self.cls_det(h)
        outputs = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            outputs['aux_outputs'] = self.set_aux_loss(outputs_class, outputs_coord)

        # train
        if self.trainable:
            # The loss is computed in the external file
            return outputs

        # test
        else:
            with torch.no_grad():
                # batch_size = 1
                out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
                # [B, N, C] -> [N, C]
                prob = out_logits[0].softmax(-1)
                scores, labels = prob[..., :-1].max(-1)

                # convert to [x0, y0, x1, y1] format
                bboxes = box_ops.box_cxcywh_to_xyxy(out_bbox)[0]
                # bboxes = bboxes * self.img_size

                # intermediate outputs
                if 'aux_outputs' in outputs:
                    for i, aux_outputs in enumerate(outputs['aux_outputs']):
                        # batch_size = 1
                        out_logits_i, out_bbox_i = aux_outputs['pred_logits'], aux_outputs['pred_boxes']
                        # [B, N, C] -> [N, C]
                        prob_i = out_logits_i[0].softmax(-1)
                        scores_i, labels_i = prob_i[..., :-1].max(-1)

                        # convert to [x0, y0, x1, y1] format
                        bboxes_i = box_ops.box_cxcywh_to_xyxy(out_bbox_i)[0]
                        # bboxes_i = bboxes_i * self.img_size

                        scores = torch.cat([scores, scores_i], dim=0)
                        labels = torch.cat([labels, labels_i], dim=0)
                        bboxes = torch.cat([bboxes, bboxes_i], dim=0)

                # to cpu
                scores = scores.cpu().numpy()
                labels = labels.cpu().numpy()
                bboxes = bboxes.cpu().numpy()
                # threshold
                keep = np.where(scores >= self.conf_thresh)[0]

                if self.use_nms:
                    keep_nms = []
                    for i in range(self.num_classes):
                        inds = np.where(labels[keep] == i)[0]
                        if len(inds) == 0:
                            continue

                        c_bboxes = bboxes[inds]
                        c_scores = scores[inds]
                        # c_keep = self.nms(c_bboxes, c_scores)
                        c_keep = self.soft_nms(c_bboxes, c_scores)
                        keep_nms.extend(inds[c_keep])
                    keep = keep_nms

                scores = scores[keep]
                labels = labels[keep]
                bboxes = bboxes[keep]

                return bboxes, scores, labels


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
