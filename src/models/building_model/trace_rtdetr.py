import os
import sys
import torch
import time
import torch.nn as nn
import torch_neuron
import json
from typing import NamedTuple




# class Output(NamedTuple):
#     pred_logits: torch.Tensor
#     pred_boxes: torch.Tensor
#
#
# current_dir = os.path.dirname(os.path.abspath(__file__))
#
#
# if current_dir not in sys.path:
#     sys.path.append(current_dir)
#
#
# # Model Tracing: ==========================
from src.core import YAMLConfig
from src.solver import TASKS
from src.core.yaml_utils import load_config
#
config_name = 'rtdetr_r18vd_6x_coco.yml'
config_path = os.path.join('configs', 'rtdetr', config_name)
# loaded_weights = torch.load("rtdetr_r18vd_5x_coco_objects365_from_paddle.pth")['ema']['module']
# #
# #
cfg = YAMLConfig(
    config_path
)
#
# rtdetr = cfg.model
# rtdetr.load_state_dict(loaded_weights)
# rtdetr.deploy()
#
# img = torch.rand(1, 3, 480, 480)
# #
# with torch.no_grad():
#     output = rtdetr(img)
#
# print(f'logits: \n {output.pred_logits}')
# print(f'bboxes: \n {output.pred_boxes}')
#
# print(f"pred_logits: {output.pred_logits}")
# print(f"pred_boxes: {output.pred_boxes}")
#
# print(f"pred_logits.shape: {output.pred_logits.shape}")
# print(f"pred_boxes.shape: {output.pred_boxes.shape}")



neuron_model_path = 'rtdetr480_neuron_test.pt'
#
# model_neuron = torch.neuron.trace(rtdetr, img, compiler_args = ['--neuroncore-pipeline-cores', '4', '--verbose', 'DEBUG'])
# # model_neuron = torch.neuron.trace(rtdetr, img)
# model_neuron.save(neuron_model_path)


# ---------------------------------------------------------------

# Model Benchmarking ==========================================
# os.environ['NEURON_RT_VISIBLE_CORES'] = '4'
# os.environ['NEURON_RT_NUM_CORES'] = '4'

os.environ['NEURON_RT_LOG_LEVEL'] = 'WARN'
rtdetr = torch.jit.load(neuron_model_path)
img = torch.rand(1, 3, 480, 480)
# Warm-up
output = rtdetr(img)

start_time = time.perf_counter()
output = rtdetr(img)
delta = time.perf_counter() - start_time
inference_time_ms = delta * 1000
print(f'inference time: {inference_time_ms:.3f} ms')
# print(output)
#
# TOP_PREDICTIONS = 100

# -----------------------------------------------------

# Post Processing ==================================================
classes = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}

# mscoco_category2label = {k: i for i, k in enumerate(classes.keys())}
# cls = {v: k for k, v in mscoco_category2label.items()}
# class_mapping = {k:classes[v] for k, v in cls.items()}
#
#
# file_name = 'classes.json'
#
# # Save the dictionary as a JSON file
# with open(file_name, 'w') as file:
#     json.dump(class_mapping, file, indent=4)
#
#
# logits, bbox_pred = output[0], output[1]
#
# scores = torch.sigmoid(logits)
# scores, index = torch.topk(scores.flatten(1), TOP_PREDICTIONS, axis=-1)
# labels = index % 80
# index = index // 80
# boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
# labels_num = torch.tensor([cls[int(x.item())] for x in labels.flatten()]).reshape(labels.shape)
# labels = [f'{class_mapping[int(x.item())]}\n' for x in labels.flatten()]
# labels_comp = [f'{classes[int(x.item())]}\n' for x in labels_num.flatten()]

# print(f'class_mapping: {class_mapping}')
# print(f'labels: {labels}')


# print(f'boxes: \n {boxes} \n')
# print(f'labels: \n {labels}')
# print(f'labels: \n {"".join(labels)} \n')
# print(f'labels: \n {"".join(labels_comp)} \n')

# print(f'mscoco_category2label:\n {mscoco_category2label}')
# print(f'cls: \n {cls}')
# ----------------------------------------------------------------