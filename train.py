import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO(model=r'/autodl-fs/data/MyYolo/ultralytics/cfg/models/11/yolo11-LSTDs.yaml')

    result = model.train(
        pretrained=False,
        amp=False,
        data=r'data.yaml',
        imgsz=640,
        epochs=200,
        batch=8,
        workers=8,
        device='0',
        optimizer='SGD',
        lr0=0.01,  
        momentum=0.937,  
        weight_decay=0.0005, 
        close_mosaic=10,
        resume=False,
        project='runs/train',
        name='yolo11-LSTD',

        single_cls=False,
        cache=False,
    )

