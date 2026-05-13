# -*- coding: utf-8 -*-
"""
项目配置文件 - 羽毛球运动姿态识别系统
统一管理所有超参数、路径、类别信息
"""

import os
import numpy as np

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 原始视频数据集根目录（18个子文件夹，每个对应一个动作类别）
VIDEO_ROOT = r"/home/labq/deeplearning/VideoBadminton_Dataset/VideoBadminton_Dataset"

# 输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
KEYPOINTS_DIR = os.path.join(OUTPUT_DIR, "keypoints")       # 提取的关键点numpy
DATASETS_DIR = os.path.join(OUTPUT_DIR, "datasets")          # 构建好的LSTM数据集
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")              # 模型权重
CURVES_DIR = os.path.join(OUTPUT_DIR, "curves")              # 训练曲线图
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

for d in [OUTPUT_DIR, KEYPOINTS_DIR, DATASETS_DIR, MODELS_DIR, CURVES_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# YOLOv8-Pose 配置
# ============================================================
YOLO_MODEL = "yolov8n-pose.pt"        # 使用yolov8n-pose（nano模型，CPU友好）
YOLO_CONF = 0.50                       # 检测置信度阈值（过滤低质量检测）
YOLO_IOU = 0.45                        # NMS IoU阈值
YOLO_IMG_SIZE = 320                    # 推理图像尺寸（降低加速CPU推理）
PERSON_CONF = 0.60                     # 人物关键点最低置信度（过滤抖动点）
KEYPOINT_DIM = 17                      # COCO 17个关键点
KEYPOINT_VIS_DIM = 3                   # 每个关键点 (x, y, visibility)

# ============================================================
# 视频采样与降噪配置
# ============================================================
SAMPLE_INTERVAL = 1                    # 采样间隔（1=全采样，适配短视频数据集）
SMOOTH_WINDOW = 3                      # 滑动窗口中值滤波窗口大小（帧数少用小窗口）
ANOMALY_THRESHOLD = 50.0               # 相邻帧关键点位移异常阈值（像素）
INTERP_METHOD = "linear"               # 插值方法

# ============================================================
# 特征工程配置
# ============================================================
# 6组关节夹角定义：(顶点索引, 端点1索引, 端点2索引)
# 基于COCO 17关键点索引: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
#   5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
#   9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip,
#   13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
JOINT_ANGLES = {
    "left_shoulder_angle":  (5, 7, 11),   # 左肩角：肘-肩-髋
    "right_shoulder_angle": (6, 8, 12),   # 右肩角：肘-肩-髋
    "left_elbow_angle":     (7, 5, 9),    # 左肘角：肩-肘-腕
    "right_elbow_angle":    (8, 6, 10),   # 右肘角：肩-肘-腕
    "left_hip_angle":       (11, 5, 13),  # 左髋角：肩-髋-膝
    "right_hip_angle":      (12, 6, 14),  # 右髋角：肩-髋-膝
    "left_knee_angle":      (13, 11, 15), # 左膝角：髋-膝-踝
    "right_knee_angle":     (14, 12, 16), # 右膝角：髋-膝-踝
    "left_torso_angle":     (5, 11, 13),  # 左躯干倾斜角
    "right_torso_angle":    (6, 12, 14),  # 右躯干倾斜角
}

# 特征维度计算
COORD_FEATURES = KEYPOINT_DIM * 2        # 17个点的归一化x,y = 34
ANGLE_FEATURES = len(JOINT_ANGLES)        # 10个关节夹角
VELOCITY_FEATURES = KEYPOINT_DIM * 2      # 34维坐标速度（帧间差分）
ACCELERATION_FEATURES = KEYPOINT_DIM * 2  # 34维坐标加速度（二阶差分）
ANGULAR_VELOCITY_FEATURES = len(JOINT_ANGLES)  # 10维角速度
GEOMETRIC_FEATURES = 6                    # 髋肩扭转+站姿宽度+左右腕肩偏移+左右臂不对称
TOTAL_FEATURES = (COORD_FEATURES + ANGLE_FEATURES + VELOCITY_FEATURES
                  + ACCELERATION_FEATURES + ANGULAR_VELOCITY_FEATURES
                  + GEOMETRIC_FEATURES)  # 128维/帧

# ============================================================
# 滑动窗口与数据集配置
# ============================================================
SEQ_LENGTH = 15                          # 每个样本的帧数（增加时序信息）
SEQ_STRIDE = 3                           # 滑动窗口步长（更多重叠样本）
TRAIN_RATIO = 0.70                       # 训练集比例
VAL_RATIO = 0.15                         # 验证集比例
TEST_RATIO = 0.15                        # 测试集比例
RANDOM_SEED = 42                         # 随机种子

# ============================================================
# 数据增强配置
# ============================================================
AUG_KPT_SHIFT_RANGE = 5.0               # 关键点微小偏移范围（像素）
AUG_ANGLE_PERTURB_RANGE = 5.0           # 角度扰动范围（度）
AUG_PROB = 0.5                          # 数据增强概率
AUG_TEMPORAL_FLIP = True                # 时序翻转增强


# ============================================================
# 模型选择
# ============================================================
MODEL_TYPE = 'BiLSTM'                    # 模型类型: 'BiLSTM' 或 'Transformer'


# ============================================================
# BiLSTM 模型配置
# ============================================================
LSTM_HIDDEN_SIZE = 128                   # LSTM隐藏层大小
LSTM_NUM_LAYERS = 2                      # LSTM层数（双层，注意力池化）
FC_HIDDEN_SIZE = 128                     # 全连接层隐藏大小
DROPOUT_RATE = 0.3                       # Dropout比率
USE_BATCHNORM = True                     # 是否使用BatchNorm


# ============================================================
# Transformer 模型配置
# ============================================================
TRANSFORMER_MODEL_DIM = 128              # Transformer模型维度 (d_model)
TRANSFORMER_N_HEADS = 8                  # 多头注意力头数 (必须能整除MODEL_DIM)
TRANSFORMER_N_LAYERS = 4                 # Transformer编码器层数


# ============================================================
# 训练配置
# ============================================================
BATCH_SIZE = 64                          # 批次大小
LEARNING_RATE = 1e-3                     # 初始学习率
WEIGHT_DECAY = 1e-4                      # 权重衰减
NUM_EPOCHS = 120                         # 最大训练轮数
EARLY_STOP_PATIENCE = 25                 # 早停耐心值
WARMUP_EPOCHS = 8                        # 线性warmup轮数
LABEL_SMOOTHING = 0.1                    # 标签平滑系数
MIN_LR = 1e-6                            # 最小学习率
NUM_WORKERS = 0                          # DataLoader工作线程（CPU单线程更快）

# ============================================================
# 类别配置
# ============================================================
CLASS_NAMES = None  # 在运行时从文件夹动态加载
NUM_CLASSES = None  # 在运行时动态确定

def load_class_names():
    """从视频目录动态加载类别名称并编号"""
    global CLASS_NAMES, NUM_CLASSES
    dirs = sorted([d for d in os.listdir(VIDEO_ROOT)
                   if os.path.isdir(os.path.join(VIDEO_ROOT, d))])
    CLASS_NAMES = dirs
    NUM_CLASSES = len(dirs)
    return CLASS_NAMES

# ============================================================
# 球场检测配置
# ============================================================
COURT_HSV_LOWER = np.array([30, 40, 40])      # 绿色球场 HSV 下界
COURT_HSV_UPPER = np.array([85, 255, 255])    # 绿色球场 HSV 上界
COURT_CANNY_LOW = 50                           # Canny 边缘检测低阈值
COURT_CANNY_HIGH = 150                         # Canny 边缘检测高阈值
COURT_HOUGH_THRESHOLD = 80                     # Hough 线段投票阈值
COURT_HOUGH_MIN_LINE_LENGTH = 100              # 最短线段长度（像素）
COURT_HOUGH_MAX_LINE_GAP = 20                  # 最大线段间隙（像素）
COURT_RANSAC_THRESHOLD = 5.0                   # RANSAC 内点阈值（像素）
COURT_ANGLE_TOLERANCE = 15.0                   # 线段角度容差（度）

# 标准羽毛球双打场地尺寸（米）
COURT_LENGTH_M = 13.4                          # 长轴
COURT_WIDTH_M = 6.1                            # 短轴
NET_POSITION_M = 6.7                           # 球网距两端距离
COURT_CORNERS_REAL = np.array([                # 标准球场四角 (x, y) 米制
    [0, 0],                                     # 左上
    [COURT_LENGTH_M, 0],                        # 右上
    [COURT_LENGTH_M, COURT_WIDTH_M],            # 右下
    [0, COURT_WIDTH_M],                         # 左下
], dtype=np.float32)

# 俯视图画布尺寸（像素）
BIRDEYE_WIDTH = 600
BIRDEYE_HEIGHT = 280

# 热力图显示尺寸（像素）
HEATMAP_WIDTH = 400
HEATMAP_HEIGHT = 200

# 资源文件路径
COURT_TEMPLATE_PATH = '/home/labq/deeplearning/badminton_pose/court.jpg'

# ============================================================
# 羽毛球检测配置
# ============================================================
BALL_MODEL = "yolov8n.pt"                      # YOLOv8n 模型
BALL_CONF = 0.15                               # 低阈值检测小目标
BALL_IMG_SIZE = 640                            # 球检测输入尺寸
BALL_IOU = 0.45                                # NMS IoU 阈值
BALL_MAX_AREA = 800                            # 球最大面积（像素²）
BALL_TRACK_MAX_DISAPPEARED = 15                # 丢失目标最大保留帧数
BALL_TRACK_MAX_DISTANCE = 80                   # 关联匹配最大像素距离
KALMAN_PROCESS_NOISE = 0.03                    # 卡尔曼过程噪声
KALMAN_MEASUREMENT_NOISE = 10.0                # 卡尔曼观测噪声
TRAJECTORY_FIT_ORDER = 2                       # 轨迹多项式阶数（2=抛物线）
TRAJECTORY_MIN_POINTS = 4                      # 拟合最少点数
LANDING_PREDICTION_FRAMES = 30                 # 预测未来帧数

# ============================================================
# 热力图配置
# ============================================================
HEATMAP_KDE_BANDWIDTH = 0.3                    # KDE 带宽（米）
HEATMAP_RESOLUTION = 200                       # 网格分辨率
HEATMAP_BLUR_KERNEL = 31                       # 高斯模糊核（奇数）
HEATMAP_ALPHA = 0.6                            # 叠加透明度

# ============================================================
# 深度分析 GUI 配置
# ============================================================
DEEP_ANALYSIS_WINDOW_W = 1600
DEEP_ANALYSIS_WINDOW_H = 950
DEEP_ANALYSIS_CAMERA_INDEX = 0

# 预加载类别
try:
    load_class_names()
except Exception:
    pass