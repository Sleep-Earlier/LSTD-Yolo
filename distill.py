import warnings

warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.distill import DetectionDistiller

# from ultralytics.models.yolo.segment.distill import SegmentationDistiller
# from ultralytics.models.yolo.pose.distill import PoseDistiller
# from ultralytics.models.yolo.obb.distill import OBBDistiller

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': '/root/autodl-fs/MyYolo/tiny/prune/LSTD-yolo11s-prune/weights/prune.pt',
        'data': 'tinydata.yaml',
        'imgsz': 640,
        'epochs': 200,
        'batch': 8,
        'workers': 4,
        'cache': False,
        'optimizer': 'SGD',
        'device': '0',
        'close_mosaic': 10,
        'amp': False, # 如果蒸馏损失为nan，把amp设置为False
        'project': 'tiny/distill',
        'name': 'yolo11s-STFE-SPloss-bckd-mgd',


        # distill
        'prune_model': True,
        'teacher_weights': '/root/autodl-fs/MyYolo/tiny/train/LSTD-yolo11s/weights/best.pt',
        'teacher_cfg': '/root/autodl-fs/MyYolo/ultralytics/cfg/models/11/yolo11s-STFE.yaml',
        'kd_loss_type': 'all',
        'kd_loss_decay': 'cosine',

        'logical_loss_type': 'BCKD',
        'logical_loss_ratio': 0.8,

        'teacher_kd_layers': '16,18,20,22',
        'student_kd_layers': '16,18,20,22',
        'feature_loss_type': 'mgd',
        'feature_loss_ratio': 0.06
    }

    model = DetectionDistiller(overrides=param_dict)
    # model = SegmentationDistiller(overrides=param_dict)
    # model = PoseDistiller(overrides=param_dict)
    # model = OBBDistiller(overrides=param_dict)
    model.distill()