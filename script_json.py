'''
将json文件转为yolo所需要的txt文件。将未转换的标注放入labels文件夹中，图片放入images文件夹中
json中[x1,y1,x2,y2]，（x1,y1)表示目标左上角坐标，（x2,y2)表示目标右下角坐标，图片左上角坐标为（0，0）
yolo的txt中[class,x_center,y_center,width,height](需要根据图片宽高进行归一化处理）
'''

import json
import os
from PIL import Image


def convert(img_size, box):  # 坐标转换
    dw = 1. / (img_size[0])
    dh = 1. / (img_size[1])
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh

    return x, y, w, h


def decode_json(json_path,img_name):
    txt_name = 'D:\work_dir\match\\traffic_light\dataset\yolo\lable\\' + img_name[0:-4] + '.txt'  # 生成txt文件存放的路径
    txt_file = open(txt_name, 'w')


    data = json.load(open(json_path, 'r', encoding='utf-8'))

    image_dir_path = r'D:\work_dir\match\traffic_light\dataset\train_images'
    image_path = image_dir_path + '\\' + img_name

    # 使用pillow读取图片，获取图片的宽和高
    img_pillow = Image.open(image_path)
    img_w = img_pillow.width  # 图片宽度
    img_h = img_pillow.height  # 图片高度

    for i in data['annotations']:
        if i["filename"] == "train_images\\" + img_name:
            for j in i["inbox"]:
                if j['color'] == "red":  # 目标的类别
                    x1 = i['bndbox']["xmin"]
                    y1 = i['bndbox']["ymin"]
                    x2 = i['bndbox']["xmax"]
                    y2 = i['bndbox']["ymax"]
                    bb = (x1, y1, x2, y2)
                    bbox = convert((img_w, img_h), bb)
                    txt_file.write('0' + " " + " ".join([str(a) for a in bbox]) + '\n')  # 此处将该目标类别记为“0”
                for j in i["inbox"]:
                    if j['color'] == "green":  # 目标的类别
                        x1 = i['bndbox']["xmin"]
                        y1 = i['bndbox']["ymin"]
                        x2 = i['bndbox']["xmax"]
                        y2 = i['bndbox']["ymax"]
                        bb = (x1, y1, x2, y2)
                        bbox = convert((img_w, img_h), bb)
                        txt_file.write('1' + " " + " ".join([str(a) for a in bbox]) + '\n')  # 此处将该目标类别记为“0”

if __name__ == "__main__":

    json_path = r'D:\work_dir\match\traffic_light\dataset\train.json'  # json文件的路径
    img_list = os.listdir(r'D:\work_dir\match\traffic_light\dataset\train_images')
    for i in img_list:
        decode_json(json_path,i)