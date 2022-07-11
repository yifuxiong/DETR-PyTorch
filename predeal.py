import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

image_root = 'E:/learngit/bc-pytorch/BCdevkit/BC2021/JPEGImages/'
label_root = 'E:/learngit/bc-pytorch/BCdevkit/BC2021/Annotations/'
image_name = 'left_1-09-272092_2.png'
label_name = 'left_1-09-272092_2.xml'


def parse_xml(xml_path):
    in_file = open(xml_path, encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    pack = []
    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        if int(difficult) == 1:
            continue

        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        pack.append(b)
    return np.array(pack)


def vis(img_path, xml_path):
    img = cv2.imread(img_path)

    def get_thresh(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        pos = np.where(thresh > 0)
        print(pos)
        thresh[pos] -= 100
        thresh = np.clip(thresh, 0, 255)

        cv2.imshow('thresh', thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def gamma(img):
        img_norm = img / 255
        gamma = 1.0 / 1.4
        dst = np.power(img_norm, gamma)
        return dst

    img = gamma(img)
    pack = parse_xml(xml_path)
    for bbox in pack:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness=2)

    cv2.imshow('dst', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_path = os.path.join(image_root, image_name)
xml_path = os.path.join(label_root, label_name)
vis(image_path, xml_path)
