# -*- coding: utf-8 -*-
"""
模块初始化（延迟导入，避免缺少依赖时整个包不可用）
"""

def __getattr__(name):
    if name == "KeypointExtractor":
        from .keypoint_extractor import KeypointExtractor
        return KeypointExtractor
    elif name == "Preprocessor":
        from .preprocessor import Preprocessor
        return Preprocessor
    elif name == "DatasetBuilder":
        from .dataset_builder import DatasetBuilder
        return DatasetBuilder
    elif name == "BiLSTMClassifier":
        from .model import BiLSTMClassifier
        return BiLSTMClassifier
    elif name == "CourtDetector":
        from .court_detector import CourtDetector
        return CourtDetector
    elif name == "BallDetector":
        from .ball_detector import BallDetector
        return BallDetector
    elif name == "CourtMapper":
        from .court_mapper import CourtMapper
        return CourtMapper
    elif name == "HeatmapGenerator":
        from .heatmap_generator import HeatmapGenerator
        return HeatmapGenerator
    raise AttributeError(f"module 'modules' has no attribute '{name}'")
