# -*- coding: utf-8 -*-
"""
训练建议生成器
根据视频分析数据（动作分布、落点、步伐）生成球员建议报告
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from .court_mapper import CourtMapper


# 动作分类
ATTACK_ACTIONS = {'Smash', 'Tap Smash', 'Drop Shot', 'Rush Shot', 'Cut'}
DEFEND_ACTIONS = {'Block', 'Lift', 'Defensive Clear', 'Defensive Drive'}
SERVE_ACTIONS = {'Short Serve', 'Long Serve'}
NET_ACTIONS = {'Push Shot', 'Flat Shot', 'Short Flat Shot'}
CLEAR_ACTIONS = {'Clear', 'Cross Court Flight', 'Transitional Slice', 'Rear Court Flat Drive'}


class Advisor:
    """训练建议生成器"""

    def generate_report(self, ball_heatmap, footwork_heatmap, class_names, all_predictions=None):
        """
        生成完整的训练建议报告

        Args:
            ball_heatmap: HeatmapGenerator 球落点
            footwork_heatmap: HeatmapGenerator 球员步伐
            class_names: list[str] 动作类别名
            all_predictions: list[(idx, conf)] 全部预测历史（可选）

        Returns:
            str 中文建议报告
        """
        sections = []

        # 1. 动作分布
        action_report = self._analyze_actions(ball_heatmap, class_names)
        if action_report:
            sections.append(action_report)

        # 2. 落点分布
        placement_report = self._analyze_placement(ball_heatmap)
        if placement_report:
            sections.append(placement_report)

        # 3. 步伐分析
        footwork_report = self._analyze_footwork(footwork_heatmap)
        if footwork_report:
            sections.append(footwork_report)

        # 4. 技术多样性
        diversity_report = self._analyze_diversity(ball_heatmap, class_names)
        if diversity_report:
            sections.append(diversity_report)

        if not sections:
            return "暂无足够数据生成建议，请确保视频中有足够的击球和移动数据。"

        return "\n\n".join(sections)

    def _analyze_actions(self, heatmap, class_names):
        """分析动作分布"""
        if not heatmap.class_points:
            return ""

        # 统计各动作出现次数
        action_counts = {}
        total = 0
        for idx, points in heatmap.class_points.items():
            if idx < len(class_names) and len(points) > 0:
                name = class_names[idx]
                action_counts[name] = len(points)
                total += len(points)

        if total == 0:
            return ""

        lines = ["【动作分布】"]

        # 排序显示
        sorted_actions = sorted(action_counts.items(), key=lambda x: -x[1])
        for name, count in sorted_actions[:8]:
            pct = count / total * 100
            bar = "█" * int(pct / 5)
            lines.append(f"  {name:<18} {count:>3}次 ({pct:4.1f}%) {bar}")

        # 计算进攻/防守占比
        attack_count = sum(action_counts.get(a, 0) for a in ATTACK_ACTIONS)
        defend_count = sum(action_counts.get(a, 0) for a in DEFEND_ACTIONS)
        attack_pct = attack_count / total * 100
        defend_pct = defend_count / total * 100

        suggestions = []
        if attack_pct < 30:
            suggestions.append("  → 进攻比例偏低({:.0f}%)，建议增加杀球、吊球等主动进攻".format(attack_pct))
        if defend_pct > 50:
            suggestions.append("  → 防守比例偏高({:.0f}%)，建议减少被动起高球，多争取主动".format(defend_pct))
        if attack_pct > 60:
            suggestions.append("  → 进攻比例较高({:.0f}%)，注意保持攻守平衡".format(attack_pct))

        if suggestions:
            lines.append("")
            lines.extend(suggestions)

        return "\n".join(lines)

    def _analyze_placement(self, heatmap):
        """分析球落点分布"""
        if len(heatmap.all_points) < 3:
            return ""

        points = np.array(heatmap.all_points)
        cm = CourtMapper()
        stats = cm.get_court_stats(points)

        total = stats['total']
        front = stats['front_count']
        back = stats['back_count']
        left = stats['left_count']
        right = stats['right_count']

        lines = ["【落点分布】"]
        lines.append(f"  前场: {front}次 ({front/total*100:.0f}%)  后场: {back}次 ({back/total*100:.0f}%)")
        lines.append(f"  左半场: {left}次 ({left/total*100:.0f}%)  右半场: {right}次 ({right/total*100:.0f}%)")

        # 萃点密度
        density = heatmap.get_point_density()
        std_x, std_y = density['std_position']
        lines.append(f"  落点分散度: 沿长轴±{std_x:.1f}m, 沿短轴±{std_y:.1f}m")

        suggestions = []
        if total > 0:
            if back / total > 0.7:
                suggestions.append("  → 落点过于集中在后场，建议多放网前小球")
            if front / total > 0.7:
                suggestions.append("  → 落点过于集中在前场，建议增加后场高远球和杀球")
            if left > 0 and right > 0:
                ratio = max(left, right) / min(left, right)
                if ratio > 2:
                    weak = "左" if left < right else "右"
                    suggestions.append(f"  → 落点严重偏向一侧，建议加强{weak}半场练习")
            if std_x < 2.0 and total > 10:
                suggestions.append("  → 萃点较为集中，容易被对手预判，建议增加落点变化")

        if suggestions:
            lines.append("")
            lines.extend(suggestions)

        return "\n".join(lines)

    def _analyze_footwork(self, heatmap):
        """分析步伐覆盖"""
        if len(heatmap.all_points) < 5:
            return ""

        points = np.array(heatmap.all_points)
        cm = CourtMapper()
        stats = cm.get_court_stats(points)

        total = stats['total']
        front = stats['front_count']
        back = stats['back_count']
        coverage = stats['coverage_area']
        mean_x, mean_y = stats['mean_position']

        lines = ["【步伐分析】"]
        lines.append(f"  移动范围: {coverage:.1f}m²")
        lines.append(f"  平均站位: x={mean_x:.1f}m (距发球线), y={mean_y:.1f}m")

        suggestions = []
        if coverage < 20 and total > 20:
            suggestions.append("  → 移动范围偏小，建议加强全场跑动训练")
        if mean_x > 9:
            suggestions.append("  → 平均站位偏后，注意前场跟进")
        elif mean_x < 4:
            suggestions.append("  → 平均站位偏前，注意后场防守覆盖")
        if mean_y > 4.5:
            suggestions.append("  → 站位偏右半场，注意左半场覆盖")
        elif mean_y < 1.5:
            suggestions.append("  → 站位偏左半场，注意右半场覆盖")

        if suggestions:
            lines.append("")
            lines.extend(suggestions)

        return "\n".join(lines)

    def _analyze_diversity(self, heatmap, class_names):
        """分析技术多样性"""
        used_actions = set()
        for idx in heatmap.class_points:
            if idx < len(class_names) and len(heatmap.class_points[idx]) > 0:
                used_actions.add(class_names[idx])

        total_types = len(used_actions)
        available = len(class_names)

        lines = ["【技术多样性】"]
        lines.append(f"  使用了 {total_types}/{available} 种技术动作")

        suggestions = []
        if total_types < 5:
            suggestions.append("  → 技术动作较为单一，建议丰富打法套路")
        elif total_types >= 10:
            suggestions.append("  → 技术动作丰富，继续保持")

        # 检查是否缺少重要技术
        missing_important = []
        for action in ['Smash', 'Drop Shot', 'Clear', 'Push Shot', 'Block']:
            if action not in used_actions:
                missing_important.append(action)
        if missing_important and total_types > 0:
            suggestions.append(f"  → 未检测到: {', '.join(missing_important)}，可针对性练习")

        if suggestions:
            lines.append("")
            lines.extend(suggestions)

        return "\n".join(lines)
