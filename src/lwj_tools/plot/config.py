#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""绘图配置 dataclass。

按从内到外分四层：

- :class:`FontConfig` / :class:`DataPointConfig` / :class:`ColorBarConfig` ——
  字体、数据标签、颜色条样式
- :class:`LegendConfig` / :class:`AxConfig` / :class:`TitleConfig` ——
  图例、坐标轴、标题
- :class:`SeriesConfig` —— 单个数据系列（一条折线 / 一组柱子 / 一组散点 / 一张热力图）
- :class:`SubplotConfig` —— 子图（包含多条 :class:`SeriesConfig`）
- :class:`PlotConfig` —— 整张画布（包含多个 :class:`SubplotConfig`）

所有 dataclass 都给出合理默认；只在需要微调时实例化对应对象传入。
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from ..common._typing import FilePath


class PlotType(Enum):
    """支持的绘图类型：折线、柱状、散点、热力图。"""

    LINE = 'Line'
    BAR = 'Bar'
    SCATTER = 'Scatter'
    HEATMAP = 'Heatmap'


@dataclass
class FontConfig:
    """字体配置

    Args:
        family: 字体族
        size: 字号(pt)
        weight: 字重，比如 bold
        style: 字体风格，比如 normal, italic
        color: css 颜色或者十六进制
    """
    family: str = 'SimHei'
    size: int = 12
    weight: str = 'normal'  # normal, bold, etc.
    style: str = 'normal'
    color: str = 'black'


@dataclass
class LegendConfig:
    """图例配置

    Args:
        enabled: 是否显示图例
        loc: 图例的位置
        font_config: 图例的字体配置
        frame_on: 图例边框
        shadow: 是否启用阴影
        border_pad: 图例边框内边距
        column_spacing: 多列图例时列间距
        handle_text_pad: 图例符号与文字间距
        n_col: 图例的列数
    """
    enabled: bool = True
    loc: str = 'best'
    font_config: FontConfig = field(default_factory=FontConfig)
    frame_on: bool = True
    shadow: bool = False
    border_pad: float = 0.4
    column_spacing: float = 2.0
    handle_text_pad: float = 0.8
    n_col: int = 1


@dataclass
class AxConfig:
    """坐标轴配置

    Args:
        ticks: 自定义刻度位置，比如 [0, 2, 4, 6]
        tick_labels: 刻度对应的文本标签，比如 ['一月', '二月', ...]
        tick_font: 刻度文字的字体配置
        tick_rotation: 刻度 label 的旋转度数
        grid_enabled: 是否启动网格线
        grid_alpha: 网格线透明度
        lim: 坐标轴范围
        label: 坐标轴标题，比如 销售额
        label_font: 坐标轴标题的字体配置
    """
    ticks: Optional[List[float]] = None
    tick_labels: Optional[List[str]] = None
    tick_font: FontConfig = field(default_factory=FontConfig)
    tick_rotation: int = 0
    grid_enabled: bool = True
    grid_alpha: float = 0.3
    lim: Optional[Tuple[float, float]] = None
    label: str = ""
    label_font: FontConfig = field(default_factory=FontConfig)


@dataclass
class DataPointConfig:
    """数据点配置

    Args:
        show_label: 是否在图上显示具体数值
        font: 数据标签的字体配置
        fmt_fn: 数值标签格式化方法
        offset_x: 数据标签相对于数据点的 x 轴偏移量
        offset_y: 数据标签相对于数据点的 y 轴偏移量
    """
    show_label: bool = False
    font: FontConfig = field(default_factory=FontConfig)
    color: str = "blue"
    fmt_fn: Callable[[float], str] = None
    offset_x: float = 0.0
    offset_y: float = 0.0

    def __post_init__(self):
        if self.fmt_fn is None:
            self.fmt_fn = lambda x: f'{x:.2f}'


@dataclass
class TitleConfig:
    """标题配置

    Args:
        text: 标题内容
        font: 标题字体
        pad: 标题与图表顶部的间距 (pt)
        loc: 对齐方式 left, center, right
    """
    text: str = ""
    font: FontConfig = field(default_factory=FontConfig)
    pad: float = 20.0
    loc: str = 'center'


@dataclass
class ColorBarConfig:
    """用于热力图

    Args:
        enabled: 是否显示颜色条
        label: 颜色条标题
        loc: 颜色条位置
        shrink: 颜色条缩放比例（<1 表示缩短）
        aspect: 颜色条宽高比（越大越细长）
    """
    enabled: bool = True
    label: str = ""
    loc: str = "right"
    shrink: float = 0.9
    aspect: float = 20


@dataclass
class SeriesConfig:
    """单个数据系列配置

    Args:
        x_data: 横轴数据
        y_data: 纵轴数据
        plot_type: 绘图类型
        label: 用于图例
        color: 柱子/折线的颜色
        alpha: 透明度 (0-1)
        width: 柱子的宽度/散点的大小/折线的粗细
        marker: 散点图/折线图标记的形状
        data_point_config: 数据标签配置
    """
    x_data: Union[List, np.ndarray]
    y_data: Union[List, np.ndarray]
    plot_type: PlotType
    label: str = ""
    color: str = "auto"
    alpha: float = 1.0
    width: float = 1.0
    marker: str = 'o'
    data_point_config: DataPointConfig = field(default_factory=DataPointConfig)
    color_bar_config: Optional[ColorBarConfig] = None


@dataclass
class SubplotConfig:
    """子图配置

    Args:
        series_list: 数据系列列表
        title_config: 子图标题配置
        x_axis_config: x轴配置
        y_axis_config: y轴配置
        legend_config: 图例配置
    """
    series_list: List[SeriesConfig]
    title_config: TitleConfig = field(default_factory=TitleConfig)
    x_axis_config: AxConfig = field(default_factory=AxConfig)
    y_axis_config: AxConfig = field(default_factory=AxConfig)
    legend_config: LegendConfig = field(default_factory=LegendConfig)


@dataclass
class PlotConfig:
    """综合绘图配置

    Args:
        subplots: 子图的配置
        layout: (rows, cols)
        figure_size:  图像大小 (英寸)
        dpi: 图像分辨率
        save_path: 图片保存路径
        show_plot: 是否调用 plt.show()
        main_title: 主标题
        main_title_config: 主标题的配置
    """
    subplots: List[SubplotConfig]
    layout: Tuple[int, int] = (1, 1)
    figure_size: Tuple[int, int] = (10, 6)
    dpi: int = 300
    save_path: Optional[FilePath] = None
    show_plot: bool = True
    main_title: Optional[str] = None
    main_title_config: TitleConfig = field(default_factory=TitleConfig)

    def __post_init__(self):
        max_capacity = self.layout[0] * self.layout[1]
        if len(self.subplots) > max_capacity:
            raise ValueError(f'子图数量（{len(self.subplots)}）超出了布局容量（{max_capacity}）')
