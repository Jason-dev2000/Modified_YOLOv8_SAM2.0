from ultralytics import YOLO, SAM
import numpy as np
import cv2
import os
import time

# ==================== 配置路径（请修改） ====================

# YOLO 权重路径
YOLO_WEIGHTS = r"D:/yolov8改进/ultralytics-main/runs/train/exp5/weights/best.pt"

# SAM2 权重路径
SAM_WEIGHTS = r"D:/yolov12-main/ultralytics/sam2.1_b.pt"

# 输入图像路径
INPUT_IMAGE_PATH = r"C:/Users/24089/Desktop/run/1.png"

# 输出图像路径
OUTPUT_IMAGE_PATH = r"C:/Users/24089/Desktop/run/result.jpg"

# 检测置信度
CONFIDENCE = 0.2

# IOU 阈值
IOU_THRESHOLD = 0.2

# 轮廓线宽（可调整：1, 2, 3... 数值越小线越细）
CONTOUR_THICKNESS = 2  # 修改这里：原来是2，现在改为1使线条更细

# 虚线线段长度（可调整）
DASH_LEN = 2

# 中心点标记大小（像素）- 改为更小的值
CROSS_SIZE = 5  # 原来是15，改为5使中心点更小

# 是否绘制中心点（True=绘制, False=不绘制）
DRAW_CENTER = True  # 如果不需要中心点，改为False/False


# ==================== 绘制函数 ====================

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
    """
    绘制黄色虚线轮廓
    color=(0, 255, 255) 是 OpenCV BGR 格式的黄色
    thickness: 轮廓线宽（1最细，2较细，3标准）
    dash_len: 虚线线段长度
    """
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
    """绘制十字中心点标记（小尺寸版本）"""
    cx, cy = center
    half = size // 2

    # 绘制水平线
    cv2.line(img, (cx - half, cy), (cx + half, cy), color, thickness, cv2.LINE_AA)
    # 绘制垂直线
    cv2.line(img, (cx, cy - half), (cx, cy + half), color, thickness, cv2.LINE_AA)

    # 可选：在中心绘制一个小点（增强可见性）
    # cv2.circle(img, (cx, cy), 1, color, -1)


def draw_small_dot(img, center, color=(0, 255, 255)):
    """绘制单个像素点（最小中心标记）"""
    cx, cy = center
    # 绘制一个像素点
    img[cy, cx] = color
    # 或者绘制小圆点
    cv2.circle(img, (cx, cy), 1, color, -1)


def wait_for_file_path(file_path, check_interval=1):
    """等待文件出现"""
    start_time = time.time()
    while not os.path.exists(file_path):
        elapsed = time.time() - start_time
        print(f"\r等待文件: {elapsed:.0f}s - {file_path}", end='', flush=True)
        time.sleep(check_interval)
    print(f"\n文件已找到: {file_path}")


# ==================== 主程序 ====================

def main():
    # 等待输入文件
    wait_for_file_path(INPUT_IMAGE_PATH)

    # 加载模型
    print("加载 YOLO 模型...")
    yolo_model = YOLO(YOLO_WEIGHTS)

    print("加载 SAM 模型...")
    sam_model = SAM(SAM_WEIGHTS)

    # 读取图像
    img = cv2.imread(INPUT_IMAGE_PATH)
    if img is None:
        img = cv2.imdecode(np.fromfile(INPUT_IMAGE_PATH, np.uint8), cv2.IMREAD_COLOR)

    print(f"图像尺寸: {img.shape}")

    # YOLO 检测
    print("YOLO 检测中...")
    results = yolo_model.predict(INPUT_IMAGE_PATH, conf=CONFIDENCE, iou=IOU_THRESHOLD)

    # 提取检测框
    bboxes = []
    for detection in results[0].boxes:
        x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
        bboxes.append([x1, y1, x2, y2])

    print(f"检测到 {len(bboxes)} 个细胞")

    if len(bboxes) == 0:
        print("未检测到细胞，保存原图")
        cv2.imwrite(OUTPUT_IMAGE_PATH, img)
        return

    # SAM 分割
    print("SAM 分割中...")
    sam_results = sam_model(source=INPUT_IMAGE_PATH, bboxes=bboxes, conf=CONFIDENCE, iou=IOU_THRESHOLD)
    masks = sam_results[0].masks.data.cpu().numpy()

    # 绘制黄色虚线轮廓（使用更细的线条）
    print(f"绘制轮廓（线宽={CONTOUR_THICKNESS}）...")
    img = draw_dashed_contours(img, masks, color=(0, 255, 255), thickness=CONTOUR_THICKNESS, dash_len=DASH_LEN)

    # 绘制中心点（小尺寸版本）
    if DRAW_CENTER:
        print(f"绘制中心点（大小={CROSS_SIZE}像素）...")
        for mask in masks:
            mask_img = (mask * 255).astype(np.uint8)
            moments = cv2.moments(mask_img)

            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])

                # 方法1：使用小十字（推荐，可见性好）
                draw_cross(img, (cx, cy), size=CROSS_SIZE, color=(0, 255, 255), thickness=CONTOUR_THICKNESS)

                # 方法2：如果CROSS_SIZE=0，可以使用小圆点（取消下面注释）
                # draw_small_dot(img, (cx, cy), color=(0, 255, 255))

    # 保存结果
    cv2.imwrite(OUTPUT_IMAGE_PATH, img)
    print(f"结果已保存: {OUTPUT_IMAGE_PATH}")

    # 显示结果
    cv2.imshow("Segmentation Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()