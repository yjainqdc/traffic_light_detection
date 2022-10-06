import os
import json
from detector import Detector
import cv2
from draw import draw_bboxes

detector = Detector()

# capture = cv2.VideoCapture(r'C:\Users\qwer\Desktop\\tracking\all_video\val\val11.avi')
list = os.listdir(r'D:\work_dir\match\traffic_light\dataset\\test_images')
# file_handle = open('result_m_deepsort.txt', mode='w')
frame = 0

result = {"annotations": []}
for img in list:
    # 读取每帧图片
    im = cv2.imread('D:\work_dir\match\\traffic_light\dataset\\test_images\\' + img)
    if im is None:
        break

    # 缩小尺寸，4096*3000 -> 1080*720
    scale_x = 2704 / 1080
    scale_y = 1502 / 720
    im = cv2.resize(im, (1080, 720))

    bboxes = detector.detect(im)

    if len(bboxes) > 0:

        for x1, y1, x2, y2, lbl, conf in bboxes:
            frame_result = {"filename": "test_images\\" + img,
                            "conf":conf.item(),
                            "box": {
                                "xmin": x1,
                                "ymin": y1,
                                "xmax": x2,
                                "ymax": y2
                            },
                            "label": lbl
                            }
            result["annotations"].append(frame_result)
            print(frame_result)


        # 画框
        # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
        print('++++++++++++++++++++++++++++++++++++')
        print(bboxes)
        output_image_frame = draw_bboxes(im, bboxes, line_thickness=None)

        cv2.imshow('demo', output_image_frame)
        # cv2.imshow(output_image_frame)
        cv2.waitKey(1)

        frame += 1
with open(r"D:\work_dir\match\traffic_light\result.json","w") as f:
     json.dump(result,f)
     print("加载入文件完成...")