# -*- coding: utf-8 -*-
# @Time    : 2024/3/7 8:36
# @Author  : yblir
# @File    : glip_predict.py
# explain  : GLIP test code, running in very beginning to check setting

import warnings
warnings.filterwarnings("ignore")
from transformers import logging
logging.set_verbosity_error()
# pylab.rcParams['figure.figsize'] = 20, 12
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        """
        Initializes the Colors class with a palette derived from Ultralytics color scheme, converting hex codes to RGB.

        Colors derived from `hex = matplotlib.colors.TABLEAU_COLORS.values()`.
        """
        hexs = (
            "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86", "1A9334", "00D4BB",
            "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF", "520085", "CB38FF", "FF95C8", "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        """Returns color from palette by index `i`, in BGR format if `bgr=True`, else RGB; `i` is an integer index."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hexadecimal color `h` to an RGB tuple (PIL-compatible) with order (R, G, B)."""
        return tuple(int(h[1 + i: 1 + i + 2], 16) for i in (0, 2, 4))


def draw_images(image, boxes, classes, scores, colors, xyxy=True):
    """
    Boxing define for images
    Args:
        image: pillow and numpy, finally changed to pillow,h,w,c
        boxes: tensor or numpy, finally changed to numpy
        xyxy: xyxy type default
        classes: box class, list
        scores: predict score for boxes, list
        colors: box color
    Returns:
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image[:, :, ::-1])
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()

    font = ImageFont.truetype(font='configs/simhei.ttf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = max((image.size[0] + image.size[1]) // 300, 1)
    draw = ImageDraw.Draw(image)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        color = colors[i]

        label = '{}:{:.2f}'.format(classes[i], scores[i])
        tx1, ty1, tx2, ty2 = font.getbbox(label)
        tw, th = tx2 - tx1, ty2 - tx1

        text_origin = np.array([x1, y1 - th]) if y1 - th >= 0 else np.array([x1, y1 + 1])

        for j in range(thickness):
            draw.rectangle((x1 + j, y1 + j, x2 - j, y2 - j), outline=color)

        # Label
        draw.rectangle((text_origin[0], text_origin[1], text_origin[0] + tw, text_origin[1] + th), fill=color)
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)

    return image


config_file = "configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
weight_file = r'E:\premodel\glip_tiny_model_o365_goldg_cc_sbu.pth'

# update the config options with the config file
# manual override some options
cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

glip_demo = GLIPDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
    show_mask_heatmaps=False
)


def glip_inference(image_, caption_):
    # set colors for different classes
    colors_ = Colors()

    preds = glip_demo.compute_prediction(image_, caption_)
    top_preds = glip_demo._post_process(preds, threshold=0.5)

    # extract label, scores and boxes from results
    labels = top_preds.get_field("labels").tolist()
    scores = top_preds.get_field("scores").tolist()
    boxes = top_preds.bbox.detach().cpu().numpy()

    colors = [colors_(idx) for idx in labels]
    labels_names = glip_demo.get_label_names(labels)

    return boxes, scores, labels_names, colors


if __name__ == '__main__':
    # caption = 'bobble heads on top of the shelf'
    # caption = "Striped bed, white sofa, TV, carpet, person"
    # caption = "table on carpet"
    # caption = "Table, TV"
    caption = 'person'
    image = cv2.imread('docs/bus.jpg')

    boxes, scores, labels_names, colors = glip_inference(image, caption)

    print(labels_names, scores)
    print(boxes)

    image = draw_images(image=image, boxes=boxes, classes=labels_names, scores=scores, colors=colors)

    image.show()
    # image.save('bb.jpg')
