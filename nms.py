import numpy as np
from copy import deepcopy


def nms(dets, scores, nms_thresh=0.5):
    """"
    Pure Python NMS baseline.
    """
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
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]
    return keep


def soft_nms(bboxes, Nt=0.3, sigma2=0.5, score_thresh=0.5, method=2):
    '''
    :param bboxes:
    :param Nt: iou_thresh
    :param sigma2: gaussian_sigma
    :param score_thresh: confidence
    :param method:
    :return:
    '''
    # 在 bboxes 之后添加对于的下标[0, 1, 2...], 最终 bboxes 的 shape 为 [n, 5], 前四个为坐标, 后一个为下标
    res_bboxes = deepcopy(bboxes)
    N = bboxes.shape[0]  # 总的 box 的数量
    indexes = np.array([np.arange(N)])  # 下标: 0, 1, 2, ..., n-1
    bboxes = np.concatenate((bboxes, indexes.T), axis=1)  # concatenate 之后, bboxes 的操作不会对外部变量产生影响
    # 计算每个 box 的面积
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    scores = bboxes[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # 找出 i 后面的最大 score 及其下标
        pos = i + 1
        if i != N - 1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        # 如果当前 i 的得分小于后面的最大 score, 则与之交换, 确保 i 上的 score 最大
        if scores[i] < maxscore:
            bboxes[[i, maxpos + i + 1]] = bboxes[[maxpos + i + 1, i]]
            scores[[i, maxpos + i + 1]] = scores[[maxpos + i + 1, i]]
            areas[[i, maxpos + i + 1]] = areas[[maxpos + i + 1, i]]
        # IoU calculate
        xx1 = np.maximum(bboxes[i, 0], bboxes[pos:, 0])
        yy1 = np.maximum(bboxes[i, 1], bboxes[pos:, 1])
        xx2 = np.minimum(bboxes[i, 2], bboxes[pos:, 2])
        yy2 = np.minimum(bboxes[i, 3], bboxes[pos:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = w * h
        iou = intersection / (areas[i] + areas[pos:] - intersection)
        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(iou.shape)
            weight[iou > Nt] = weight[iou > Nt] - iou[iou > Nt]
        elif method == 2:  # gaussian
            weight = np.exp(-(iou * iou) / sigma2)
        else:  # original NMS
            weight = np.ones(iou.shape)
            weight[iou > Nt] = 0
        scores[pos:] = weight * scores[pos:]
    # select the boxes and keep the corresponding indexes
    inds = bboxes[:, 5][scores > score_thresh]
    keep = inds.astype(int)
    return res_bboxes[keep]


def soft_nms_custom(bboxes, scores, iou_thresh=0.3, sigma2=0.5, score_thresh=0.5, method=2):
    '''
    :param bboxes:
    :param scores:
    :param iou_thresh:
    :param sigma2:
    :param score_thresh:
    :param method:
    :return: 操作时对bboxes后添加对应的下标[0, 1, 2, ...]，最终返回的bboxes的shape为[n, 5], 前四个为坐标，最后一个为下标
    '''
    bboxes_copy = deepcopy(bboxes)
    N = bboxes.shape[0]
    # index: (0, 1, 2, ..., n-1)
    indexes = np.array([np.arange(N)])
    # 将bboxes和对应的index拼接
    bboxes = np.concatenate((bboxes, indexes.T), axis=1)

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
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
        # 如果当前i的得分小于后面的最大score，则与之交换，确保i上的score最大
        if scores[i] < maxscore:
            bboxes[[i, maxpos + i + 1]] = bboxes[[maxpos + i + 1, i]]
            scores[[i, maxpos + i + 1]] = scores[[maxpos + i + 1, i]]
            areas[[i, maxpos + i + 1]] = areas[[maxpos + i + 1, i]]

        # IoU calculate
        xx1 = np.maximum(bboxes[i, 0], bboxes[pos:, 0])
        yy1 = np.maximum(bboxes[i, 1], bboxes[pos:, 1])
        xx2 = np.minimum(bboxes[i, 2], bboxes[pos:, 2])
        yy2 = np.minimum(bboxes[i, 3], bboxes[pos:, 3])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = w * h
        iou = intersection / (areas[i] + areas[pos:] - intersection)

        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:
            weight = np.ones(iou.shape)
            weight[iou > iou_thresh] = weight[iou > iou_thresh] - iou[iou > iou_thresh]
        elif method == 2:
            weight = np.exp(-(iou * iou) / sigma2)
        else:
            weight = np.ones(iou.shape)
            weight[iou > iou_thresh] = 0
        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    inds = bboxes[:, 4][scores > score_thresh]
    keep = inds.astype(int)
    return bboxes_copy[keep]


if __name__ == '__main__':
    bboxes = np.array([[10, 10, 110, 110], [150, 150, 250, 250], [120, 120, 220, 220], [120, 150, 220, 250]])
    scores = np.array([0.75, 0.56, 0.81, 0.46])
    # output = nms(bboxes, scores)
    output = soft_nms_custom(bboxes=bboxes, scores=scores)
    print(output)
