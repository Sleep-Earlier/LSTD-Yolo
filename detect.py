import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics import RTDETR



if __name__ == '__main__':
    model = YOLO('/root/autodl-fs/MyYolo/tiny/distill/yolo11s-STFE-SPloss-bckd-mgd/weights/best.pt')
    # model = RTDETR("/root/autodl-fs/MyYolo/tiny/train/rtdetr-l/weights/best.pt")
    model.predict(
        source='/root/autodl-fs/tinyV',
        imgsz=640,
        project='tiny/detect',
        name='pruned-LSTD-Yolo',
        save=True,
        line_width=1,
        show_labels=False,  # 不显示标签名称
        show_conf=True,     # 显示置信度 (添加此行)
    )