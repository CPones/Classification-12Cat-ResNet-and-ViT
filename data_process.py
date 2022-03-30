import cv2
import os
import random

#图像清洗
def img_resize(data_dir):
  print("文件路径下图像数量：", len(os.listdir(data_dir)))
  for img_name in os.listdir(data_dir):
    img_dir = os.path.join(data_dir, img_name)
    img = cv2.imread(img_dir)
    if img in None:
      os.remove(img_dir)#清除脏数据
    else:
      img_resize = cv2.resize(img, (224, 224))#重塑图像尺寸
      cv2.imwrite(img_dir, img_resize)

#数据集划分train：valid = 8 ：2
with open('data/dataset/image_list.txt', 'r') as f:
    image_list = f.readlines()
random.shuffle(image_list)
cut = int(len(image_list) * 0.8)
train_list = image_list[:cut]
valid_list = image_list[cut:]
      
#创建train_list.txt
with open('data/dataset/train_list.txt', 'w') as f:
    for path in train_list:
        img_paths = path.split()[0]
        img_label = path.split()[1]
        f.write(img_paths + ' ' + img_label + '\n')

#创建valid_list.txt
with open('data/dataset/valid_list.txt', 'w') as f:
    for path in valid_list:
        img_paths = path.split()[0]
        img_label = path.split()[1]
        f.write(img_paths + ' ' + img_label + '\n')
