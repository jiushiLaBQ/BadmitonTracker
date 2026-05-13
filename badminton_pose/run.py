# -*- coding: utf-8 -*-
"""
主入口脚本 - 羽毛球运动姿态识别与动作分类系统
提供完整流程的命令行入口:
    1. 提取关键点
    2. 特征预处理
    3. 构建数据集
    4. 训练模型
    5. 评估模型
    6. GUI推理
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config


def step_extract_keypoints():
    """Step 1: 批量提取关键点"""
    print("\n" + "=" * 60)
    print("Step 1: YOLOv8-Pose 关键点批量提取")
    print("=" * 60)
    from modules.keypoint_extractor import KeypointExtractor
    extractor = KeypointExtractor()
    extractor.extract_batch()


def step_preprocess():
    """Step 2: 姿态降噪与特征工程"""
    print("\n" + "=" * 60)
    print("Step 2: 姿态降噪 + 关节角度计算 + 特征归一化")
    print("=" * 60)
    from modules.preprocessor import Preprocessor
    preprocessor = Preprocessor()
    preprocessor.process_batch()


def step_build_dataset():
    """Step 3: 构建滑动窗口数据集"""
    print("\n" + "=" * 60)
    print("Step 3: 滑动窗口切割 + 分层抽样划分数据集")
    print("=" * 60)
    from modules.dataset_builder import DatasetBuilder
    builder = DatasetBuilder()
    builder.build_from_processed()


def step_train():
    """Step 4: 训练BiLSTM模型"""
    print("\n" + "=" * 60)
    print("Step 4: BiLSTM模型训练")
    print("=" * 60)
    from train import main as train_main
    train_main()


def step_evaluate():
    """Step 5: 评估模型"""
    print("\n" + "=" * 60)
    print("Step 5: 模型评估 + 混淆矩阵")
    print("=" * 60)
    from evaluate import main as evaluate_main
    evaluate_main()


def step_gui():
    """Step 6: GUI推理界面"""
    print("\n" + "=" * 60)
    print("Step 6: 启动GUI推理界面")
    print("=" * 60)
    from gui_inference import main as gui_main
    gui_main()


def step_deep_gui():
    """Step 7: 深度分析GUI（4面板）"""
    print("\n" + "=" * 60)
    print("Step 7: 启动深度分析GUI")
    print("=" * 60)
    from gui_deep_analysis import main as deep_gui_main
    deep_gui_main()


def step_full_pipeline():
    """完整流水线：从关键点提取到训练完成"""
    step_extract_keypoints()
    step_preprocess()
    step_build_dataset()
    step_train()
    step_evaluate()


def main():
    parser = argparse.ArgumentParser(
        description="羽毛球运动姿态识别与动作分类系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run.py --step all          # 运行完整流水线
  python run.py --step extract      # 仅提取关键点
  python run.py --step preprocess   # 仅特征预处理
  python run.py --step build        # 仅构建数据集
  python run.py --step train        # 仅训练模型
  python run.py --step evaluate     # 仅评估模型
  python run.py --step gui          # 启动GUI推理
  python run.py --step deep_gui     # 启动深度分析GUI（4面板）
  python run.py --step pipeline     # 提取+预处理+构建+训练+评估
        """
    )
    parser.add_argument(
        '--step', type=str, default='all',
        choices=['all', 'extract', 'preprocess', 'build', 'train', 'evaluate', 'gui', 'deep_gui', 'pipeline'],
        help='执行步骤'
    )

    args = parser.parse_args()

    # 加载类别
    config.load_class_names()
    print(f"\n数据集路径: {config.VIDEO_ROOT}")
    print(f"检测到 {config.NUM_CLASSES} 个动作类别:")
    for i, name in enumerate(config.CLASS_NAMES):
        print(f"  [{i:2d}] {name}")

    step_map = {
        'all':      step_full_pipeline,
        'extract':  step_extract_keypoints,
        'preprocess': step_preprocess,
        'build':    step_build_dataset,
        'train':    step_train,
        'evaluate': step_evaluate,
        'gui':      step_gui,
        'deep_gui': step_deep_gui,
        'pipeline': step_full_pipeline,
    }

    func = step_map.get(args.step)
    if func:
        func()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
