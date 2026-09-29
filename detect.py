import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('D:/yolov8改进/ultralytics-main/runs/train/exp3/weights/best.pt') # select your model.pt path
    model.predict(source='C:/Users/24089/Desktop/run/1.png',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  show_conf=False,
                  show_labels=False,
                  # conf=0.2,
                  # iou=0.7,
                  # agnostic_nms=True,
                  # visualize=True, # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                )