import os
import random

# root = 'D:/VOCdevkit/VOC2007/'
# root = '/home/yifux/VOCdevkit/VOC2007/'
root = '/home/yifux/BCdevkit/BC2021/'

image_root = root + '/JPEGImages/'

# save_root = './'
save_root = root + 'ImageSets/Main/'


def main(image_root, train_test_rate=0.9, train_val_rate=0.9):
    total_nums = 0
    image_list = []
    for file in os.listdir(image_root):
        # if not file.endswith('.jpg'):  # voc
        if not file.endswith('.png'):
            continue

        image_list.append(file.split('.')[0])
        total_nums += 1

    train_list = []
    train_nums = int(total_nums * train_test_rate)
    test_nums = total_nums - train_nums
    test_list = random.sample(image_list, k=test_nums)

    trainval_fp = open(save_root + 'trainval.txt', 'w', encoding='utf-8')
    test_fp = open(save_root + 'test.txt', 'w', encoding='utf-8')

    for file in image_list:
        if file in test_list:
            test_fp.write(file + '\n')
        else:
            train_list.append(file)
            trainval_fp.write(file + '\n')

    val_nums = int(train_nums * (1 - train_val_rate))
    val_list = random.sample(train_list, k=val_nums)
    train_fp = open(save_root + 'train.txt', 'w', encoding='utf-8')
    val_fp = open(save_root + 'val.txt', 'w', encoding='utf-8')

    for file in train_list:
        if file in val_list:
            val_fp.write(file + '\n')
        else:
            train_fp.write(file + '\n')

    print('total_nums:{}, train_nums:{}, val_nums:{}, test_nums:{}'.format(total_nums, train_nums, val_nums, test_nums))


if __name__ == '__main__':
    main(image_root)
