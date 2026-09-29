import yaml
import torch
import os
import cv2
import numpy as np
from ultralytics import YOLO
from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.memory_attention import MemoryAttentionLayer
from sam2.modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser
from ultralytics.models.sam2.sam2_image_predictor import SAM2ImagePredictor
from ultralytics.models.sam2.modeling.memory_encoder import CXBlock

def load_config(yaml_path):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"配置文件未找到: {yaml_path}")
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def build_sam_model(config_path, checkpoint_path, device='cpu'):
    config = load_config(config_path)
    model_config = config['model']

    image_encoder_config = model_config['image_encoder']
    trunk_config = image_encoder_config.get('trunk', {})
    trunk = Hiera(
        embed_dim=trunk_config.get('embed_dim', 144),
        num_heads=trunk_config.get('num_heads', 2),
        stages=trunk_config.get('stages', [2, 6, 36, 4]),
        global_att_blocks=trunk_config.get('global_att_blocks', [23, 33, 43]),
        window_pos_embed_bkg_spatial_size=trunk_config.get('window_pos_embed_bkg_spatial_size', [7, 7]),
        window_spec=trunk_config.get('window_spec', [8, 4, 16, 8])
    )

    position_encoding_config = image_encoder_config.get('position_encoding', {})
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=position_encoding_config.get('num_pos_feats', 256),
        temperature=position_encoding_config.get('temperature', 10000),
        normalize=position_encoding_config.get('normalize', True),
        scale=position_encoding_config.get('scale', None),
        warmup_cache=position_encoding_config.get('warmup_cache', True),
        image_size=position_encoding_config.get('image_size', 1024),
        strides=position_encoding_config.get('strides', (4, 8, 16, 32))
    )

    neck_config = image_encoder_config.get('neck', {})
    d_model = neck_config.get('d_model', 256)
    neck = FpnNeck(
        position_encoding=position_encoding,
        d_model=d_model,
        backbone_channel_list=neck_config.get('backbone_channel_list', [1152, 576, 288, 144]),
        fpn_top_down_levels=neck_config.get('fpn_top_down_levels', [2, 3]),
        fpn_interp_model=neck_config.get('fpn_interp_model', 'nearest')
    )

    image_encoder = ImageEncoder(trunk=trunk, neck=neck, scalp=image_encoder_config.get('scalp', 1))

    mask_downsampler = MaskDownSampler(embed_dim=256, kernel_size=4, stride=4, total_stride=16)
    fuser = Fuser(layer=CXBlock(dim=256, drop_path=0.1), num_layers=4, dim=256)

    memory_attention_config = model_config['memory_attention']
    memory_attention_layer_config = memory_attention_config['layer']
    memory_attention_layer = MemoryAttentionLayer(
        activation=memory_attention_layer_config.get('activation', 'relu'),
        dim_feedforward=memory_attention_layer_config.get('dim_feedforward', 2048),
        dropout=memory_attention_layer_config.get('dropout', 0.1),
        pos_enc_at_attn=memory_attention_layer_config.get('pos_enc_at_attn', False),
        self_attention=memory_attention_layer_config.get('self_attention', {}),
        cross_attention=memory_attention_layer_config.get('cross_attention', {}),
        d_model=d_model,
        pos_enc_at_cross_attn_keys=memory_attention_layer_config.get('pos_enc_at_cross_attn_keys', True),
        pos_enc_at_cross_attn_queries=memory_attention_layer_config.get('pos_enc_at_cross_attn_queries', False)
    )

    memory_encoder_config = model_config['memory_encoder']
    memory_encoder = MemoryEncoder(
        out_dim=memory_encoder_config.get('out_dim', 64),
        position_encoding=position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=fuser
    )

    sam_model = SAM2Base(
        image_encoder=image_encoder,
        memory_attention=memory_attention_layer,
        memory_encoder=memory_encoder
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    sam_model.load_state_dict(checkpoint, strict=False)
    sam_model.to(torch.device(device))

    return SAM2ImagePredictor(sam_model=sam_model)

def yolo_detect(model, image_path):
    image = cv2.imread(image_path)
    results = model.predict(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    return boxes, image

def sam_segment(sam_model, image, boxes):
    device = 'cpu'
    sam_model.set_image(image)
    box_tensor = torch.tensor(boxes, dtype=torch.float32).to(device)
    masks, _, _ = sam_model.predict(
        point_coords=None,
        point_labels=None,
        box=box_tensor,
        multimask_output=True,
        return_logits=True
    )
    print("Masks shape:", masks.shape)  # 检查掩码的形状
    return masks

def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image

import cv2
import numpy as np

import cv2
import numpy as np
import os

def save_segmentation_results(image, masks, boxes, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 先绘制检测框
    result_image_with_boxes = draw_boxes(image.copy(), boxes)
    result_image_with_overlay = image.copy()

    # 确保 masks 是 numpy 数组且形状是 (N, H, W)
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()

    if masks.ndim == 4:
        masks = masks[:, 0, :, :]  # 去掉冗余的维度 (N, 1, H, W)

    for i, mask in enumerate(masks):
        # 二值化掩码，将值大于0.5的地方设置为1，否则为0
        mask = (mask > 0.5).astype(np.uint8)

        # 创建一个与原图尺寸相同的空白图像（黑色背景）
        overlay = np.zeros_like(image, dtype=np.uint8)

        # 将掩码区域填充为红色（或绿色/蓝色等）
        overlay[mask == 1] = [0, 0, 255]  # 红色

        # 使用 cv2.addWeighted 进行图像叠加，透明度为0.5
        result_image_with_overlay = cv2.addWeighted(result_image_with_overlay, 1.0, overlay, 0.5, 0)

    # 保存结果
    result_path_with_boxes = os.path.join(output_dir, "result_with_boxes.jpg")
    result_path_with_overlay = os.path.join(output_dir, "result_with_overlay.jpg")

    cv2.imwrite(result_path_with_boxes, result_image_with_boxes)
    cv2.imwrite(result_path_with_overlay, result_image_with_overlay)

    print(f"✅ 带识别框的图像已保存到: {result_path_with_boxes}")
    print(f"✅ 带分割覆盖的图像已保存到: {result_path_with_overlay}")

def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image




def initialize_models(device='cpu'):
    yolo_model = YOLO(r"D:\yolov8改进\ultralytics-main\runs\train\exp\weights\best.pt")
    config_path = r'D:\yolov8改进\ultralytics-main\ultralytics\models\sam2\configs\sam2\sam2_hiera_b+.yaml'
    checkpoint_path = r'D:\yolov8改进\ultralytics-main\sam2.1_b.pt'
    sam_predictor = build_sam_model(config_path, checkpoint_path, device=device)
    return yolo_model, sam_predictor

def detect_and_segment(image_path, output_dir):
    yolo_model, sam_predictor = initialize_models(device='cpu')

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件未找到: {image_path}")

    boxes, image = yolo_detect(yolo_model, image_path)

    if boxes is None or len(boxes) == 0:
        print("没有检测到目标框")
        return

    masks = sam_segment(sam_predictor, image, boxes)
    save_segmentation_results(image, masks, boxes, output_dir)

if __name__ == "__main__":
    image_path = r'D:\yolov8改进\ultralytics-main\20250408_1.jpg'
    output_dir = r'D:\yolov8改进\ultralytics-main\runs\segmentation_output'
    detect_and_segment(image_path, output_dir)