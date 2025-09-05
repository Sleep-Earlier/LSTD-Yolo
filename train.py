import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics import RTDETR

if __name__ == '__main__':
    model = YOLO(model=r'/autodl-fs/data/MyYolo/ultralytics/cfg/models/11/yolo11s-4.yaml')
    # model = YOLO(model=r'/autodl-fs/data/MyYolo/ultralytics/cfg/models/11/yolo11s-STFE_3.yaml')
    # model = YOLO(model=r'/root/autodl-fs/MyYolo/ultralytics/cfg/models/12/yolo12s.yaml')
    # model = YOLO(model=r'/root/autodl-fs/MyYolo/ultralytics/cfg/models/v8/yolov8s.yaml')
    # model = RTDETR("/root/autodl-fs/MyYolo/ultralytics/cfg/models/rt-detr/rtdetr-l.yaml")
    # result = model.train(
    #     pretrained=False,
    #     amp=False,
    #
    #     data='tinydata.yaml',
    #     imgsz=640,
    #     epochs=200,
    #     batch=8,
    #     workers=8,
    #     device='0',
    #
    #     optimizer='AdamW',
    #
    #     lr0=1e-4,
    #     warmup_epochs=4,
    #     lrf=0.1,
    #     weight_decay=0.0001,
    #     close_mosaic=10,
    #
    #     resume=False,
    #     project='tiny/train',
    #     name='rtdetr-l',
    #     single_cls=False,
    #     cache=False,
    # )

    result = model.train(
        pretrained=False,
        amp=False,

        # 指定数据集配置文件的路径
        data=r'data.yaml',
        # 输入图像的尺寸，设置为实验设定的 640x640 像素
        imgsz=640,
        # 训练的轮数，不同数据集有不同的训练轮数
        # 这里以 VisDrone 2019 - DET 数据集和 TinyPerson 数据集为例，设为 300
        epochs=200,
        # 每个批次的样本数量，设置为实验设定的 8
        batch=8,
        # 数据加载的工作进程数量，设为 0 表示使用主进程加载数据
        workers=8,
        # 训练使用的设备，为空字符串表示自动选择合适的设备
        device='0',
        # 优化器类型，使用实验设定的随机梯度下降（SGD）
        optimizer='SGD',
        # 优化器参数设置
        lr0=0.01,  # 初始学习率设置为 0.01
        momentum=0.937,  # 动量参数设置为 0.937
        weight_decay=0.0005,  # 权重衰减系数设置为 0.0005
        # 在训练的第 10 个 epoch 关闭 mosaic 数据增强
        close_mosaic=10,
        # 是否从上次中断的地方继续训练
        resume=False,
        # 训练结果保存的项目目录
        project='runs/train',
        # 训练结果保存的具体名称
        name='yolo11s-4',
        # 是否将所有类别视为单类别进行训练
        single_cls=False,
        # 是否缓存图像以加快训练速度
        cache=False,

        # seed=1993,
        # patience=10,
    )

