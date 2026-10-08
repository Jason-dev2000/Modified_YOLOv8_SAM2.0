from ultralytics import YOLO, SAM
import numpy as np
import cv2
import os
import time



YOLO_WEIGHTS = r"D:/yolov8改进/ultralytics-main/runs/train/exp5/weights/best.pt"


SAM_WEIGHTS = r"D:/yolov12-main/ultralytics/sam2.1_b.pt"


INPUT_IMAGE_PATH = r"C:/Users/24089/Desktop/run/1.png"


OUTPUT_IMAGE_PATH = r"C:/Users/24089/Desktop/run/result.jpg"


CONFIDENCE = 0.2

IOU_THRESHOLD = 0.2


CONTOUR_THICKNESS = 2

DASH_LEN = 2


CROSS_SIZE = 5


DRAW_CENTER = True




def _draw_dashed_line(img, pt1, pt2, color, thickness=CONTOUR_THICKNESS, dash_len=DASH_LEN):
    """绘制虚线段"""
    x0, y0 = pt1
    x1, y1 = pt2
    dx = x1 - x0
    dy = y1 - y0
    length = np.hypot(dx, dy)
    if length == 0:
        return
    dx /= length
    dy /= length

    step = dash_len * 2
    n = int(np.ceil(length / step))

    for i in range(n):
        s = i * step
        e = s + dash_len
        if e > length:
            e = length
        xs = int(x0 + dx * s)
        ys = int(y0 + dy * s)
        xe = int(x0 + dx * e)
        ye = int(y0 + dy * e)
        cv2.line(img, (xs, ys), (xe, ye), color, thickness, cv2.LINE_AA)


def draw_dashed_contours(image, masks, color=(0, 255, 255), thickness=CONTOUR_THICKNESS, dash_len=DASH_LEN):

    for mask in masks:
        mask_img = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            cnt = cv2.approxPolyDP(cnt, 1.0, True)
            for i in range(len(cnt)):
                pt1 = tuple(cnt[i][0])
                pt2 = tuple(cnt[(i + 1) % len(cnt)][0])
                _draw_dashed_line(image, pt1, pt2, color, thickness, dash_len)

    return image


def draw_cross(img, center, size=CROSS_SIZE, color=(0, 255, 255), thickness=CONTOUR_THICKNESS):

    cx, cy = center
    half = size // 2


    cv2.line(img, (cx - half, cy), (cx + half, cy), color, thickness, cv2.LINE_AA)

    cv2.line(img, (cx, cy - half), (cx, cy + half), color, thickness, cv2.LINE_AA)



def draw_small_dot(img, center, color=(0, 255, 255)):

    cx, cy = center

    img[cy, cx] = color

    cv2.circle(img, (cx, cy), 1, color, -1)


def wait_for_file_path(file_path, check_interval=1):

    start_time = time.time()
    while not os.path.exists(file_path):
        elapsed = time.time() - start_time
        time.sleep(check_interval)





def main():

    wait_for_file_path(INPUT_IMAGE_PATH)


    yolo_model = YOLO(YOLO_WEIGHTS)


    sam_model = SAM(SAM_WEIGHTS)


    img = cv2.imread(INPUT_IMAGE_PATH)
    if img is None:
        img = cv2.imdecode(np.fromfile(INPUT_IMAGE_PATH, np.uint8), cv2.IMREAD_COLOR)




    results = yolo_model.predict(INPUT_IMAGE_PATH, conf=CONFIDENCE, iou=IOU_THRESHOLD)

    # 提取检测框
    bboxes = []
    for detection in results[0].boxes:
        x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
        bboxes.append([x1, y1, x2, y2])



    if len(bboxes) == 0:

        cv2.imwrite(OUTPUT_IMAGE_PATH, img)
        return



    sam_results = sam_model(source=INPUT_IMAGE_PATH, bboxes=bboxes, conf=CONFIDENCE, iou=IOU_THRESHOLD)
    masks = sam_results[0].masks.data.cpu().numpy()



    img = draw_dashed_contours(img, masks, color=(0, 255, 255), thickness=CONTOUR_THICKNESS, dash_len=DASH_LEN)


    if DRAW_CENTER:

        for mask in masks:
            mask_img = (mask * 255).astype(np.uint8)
            moments = cv2.moments(mask_img)

            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])


                draw_cross(img, (cx, cy), size=CROSS_SIZE, color=(0, 255, 255), thickness=CONTOUR_THICKNESS)




    cv2.imwrite(OUTPUT_IMAGE_PATH, img)



    cv2.imshow("Segmentation Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()