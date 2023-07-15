#!/usr/bin/env python


import math
import numpy as np
import cv2
import torch

from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression


class Custom_Inferer:
    def __init__(self, img_size, weights='yolov6_assets/yolov6lite_m.pt'):
        # Init model
        self.img_size = img_size
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = DetectBackend(weights, device=self.device)
        self.stride = self.model.stride
        self.class_names = ['Person', 'Chair', 'Table', 'Clothing', 'Whiteboard']
        self.img_size = self.check_img_size(self.img_size, s=self.stride)
        # Switch model to deploy status
        self.model_switch(self.model)

    def model_switch(self, model):
        ''' Model switch to deploy status '''
        from yolov6.layers.common import RepVGGBlock
        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()

        print("Switch model to deploy modality.")

    def infer(self, img, conf_thres=0.7, iou_thres=0.2, classes=None, agnostic_nms=False, max_det=1000, plot_box_and_label=False):
        ''' Model Inference and results visualization '''

        img, img_src = self.process_image(img, self.img_size, self.stride)
        img = img.to(self.device)
        if len(img.shape) == 3:
            img = img[None]
            # expand for batch dim
        pred_results = self.model(img)
        det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
        unique_object = set()

        if len(det):
            det[:, :4] = self.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
            if plot_box_and_label:
                for *xyxy, conf, cls in reversed(det):
                    class_num = int(cls)
                    # To hide unwanted class
                    # reference: {0:'Chair', 1:'Person', 2:'Table', 3:'Clothing', 4:'Whiteboard'} 
                    hidden_class_num = [3]  
                    if class_num in hidden_class_num:
                        continue
                    elif class_num < len(self.class_names):
                        class_detected = self.class_names[class_num]
                        unique_object.add(class_detected)
                        label = f'{class_detected} {conf:.2f}'
                        self.plot_box_and_label(img_src, max(round(sum(img_src.shape) / 2 * 0.003), 2), xyxy, label, color=self.generate_colors(class_num, True))
            img_src = np.asarray(img_src)
        
        return img_src, list(unique_object)

    @staticmethod
    def process_image(img_src, img_size, stride):
        image = letterbox(img_src, img_size, stride=stride)[0]
        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = torch.from_numpy(np.ascontiguousarray(image))
        image = image / 255.0  # 0 - 255 to 0.0 - 1.0

        return image, img_src

    @staticmethod
    def rescale(ori_shape, boxes, target_shape):
        '''Rescale the output to the original image shape'''
        ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
        padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio

        boxes[:, 0].clamp_(0, target_shape[1])  # x1
        boxes[:, 1].clamp_(0, target_shape[0])  # y1
        boxes[:, 2].clamp_(0, target_shape[1])  # x2
        boxes[:, 3].clamp_(0, target_shape[0])  # y2

        return boxes

    def check_img_size(self, img_size, s=32, floor=0):
        """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
        if isinstance(img_size, int):  # integer i.e. img_size=640
            new_size = max(self.make_divisible(img_size, int(s)), floor)
        elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
            new_size = [max(self.make_divisible(x, int(s)), floor) for x in img_size]
        else:
            raise Exception(f"Unsupported type of img_size: {type(img_size)}")

        if new_size != img_size:
            print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
        return new_size if isinstance(img_size,list) else [new_size]*2

    def make_divisible(self, x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor.
        return math.ceil(x / divisor) * divisor

    @staticmethod
    def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)

    @staticmethod
    def generate_colors(i, bgr=False):
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        palette = []
        for iter in hex:
            h = '#' + iter
            palette.append(tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4)))
        num = len(palette)
        color = palette[int(i) % num]
        return (color[2], color[1], color[0]) if bgr else color


