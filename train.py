import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO

if __name__ == '__main__':
    # 初始化模型（从 YAML 构建，也可以换成 'yolov8n.pt' 等预训练模型）
    model = YOLO('D:/yolov8改进/ultralytics-main/ultralytics/cfg/models/v8/yolov8-Modified_YOLOv8.yaml')

    # 开始训练
    model.train(
        data='D:/yolov12-main/data.yaml',
        imgsz=640,
        epochs=100,
        batch=32,
        optimizer='SGD',
        workers=8,
        close_mosaic=0,
        project='runs/train',
        name='exp',
        resume=True,
        device='cpu',
        # amp=False,
        patience=0,
    )
