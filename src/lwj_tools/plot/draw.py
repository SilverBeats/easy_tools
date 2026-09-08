#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""matplotlib 之上的一层薄封装。

把 :mod:`lwj_tools.plot.config` 的 dataclass 转换为 :func:`matplotlib.pyplot`
调用，支持多子图布局、自动颜色分配、四种 plot_type（line/bar/scatter/heatmap）。

默认会设置中文字体优先级与 ``axes.unicode_minus = False`` 以保证中文标签不出现方块。

Example:
    >>> from lwj_tools.plot.config import (
    ...     PlotConfig, SubplotConfig, SeriesConfig, PlotType, AxConfig,
    ... )
    >>> from lwj_tools.plot.draw import EasyPlot
    >>> config = PlotConfig(
    ...     subplots=[
    ...         SubplotConfig(
    ...             series_list=[
    ...                 SeriesConfig(
    ...                     x_data=[1, 2, 3], y_data=[4, 5, 6],
    ...                     plot_type=PlotType.LINE, label="trend",
    ...                 ),
    ...             ],
    ...             x_axis_config=AxConfig(label="x"),
    ...             y_axis_config=AxConfig(label="y"),
    ...         ),
    ...     ],
    ...     layout=(1, 1),
    ...     show_plot=False,
    ... )
    >>> fig, axes = EasyPlot().draw(config)
"""


from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

from .config import FontConfig, PlotConfig, PlotType, SeriesConfig, SubplotConfig


class EasyPlot:
    """把 :class:`PlotConfig` 渲染为 matplotlib 图的薄封装。

    入口是 :meth:`draw`；构造时不接收参数，会在初始化时设置全局字体与
    ``unicode_minus`` 以保证中文显示。
    """

    def __init__(self):
        """初始化：设置中文字体优先级与 ``unicode_minus``。"""
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.sans-serif'] = [
            'SimHei', 'Microsoft YaHei', 'PingFang SC',
            'Heiti TC', 'WenQuanYi Micro Hei'
        ]
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.sans-serif'] = [
            'SimHei', 'Microsoft YaHei', 'PingFang SC',
            'Heiti TC', 'WenQuanYi Micro Hei'
        ]

    def _apply_font_config(self, font_config: FontConfig):
        """应用字体配置"""
        font_properties = FontProperties(
            family=font_config.family,
            size=font_config.size,
            weight=font_config.weight,
            style=font_config.style,
        )
        return font_properties

    def _get_default_colors(self, n: int) -> List[str]:
        """获取默认颜色列表"""
        if n <= 10:
            return plt.cm.tab10(np.linspace(0, 1, n)).tolist()
        else:
            return plt.cm.Set3(np.linspace(0, 1, n)).tolist()

    def _draw_single_bar(
        self, series: SeriesConfig, ax: plt.Axes
    ):
        bars = ax.bar(
            x=series.x_data,
            height=series.y_data,
            label=series.label,
            color=series.color,
            width=series.width,
            alpha=series.alpha,
        )

        if not series.data_point_config.show_label:
            return

        offset_x = series.data_point_config.offset_x
        offset_y = series.data_point_config.offset_y
        fmt_fn = series.data_point_config.fmt_fn
        font = series.data_point_config.font

        for bar in bars:
            height = bar.get_height()
            ax.text(
                x=bar.get_x() + bar.get_width() / 2 + offset_x,
                y=height + offset_y,
                s=fmt_fn(height),
                ha='center',
                va='bottom',
                fontproperties=self._apply_font_config(font),
                color=font.color,
            )

    def _draw_single_line(
        self, series: SeriesConfig, ax: plt.Axes
    ):
        line = ax.plot(
            series.x_data, series.y_data,
            label=series.label,
            color=series.color,
            linewidth=series.width,
            alpha=series.alpha,
            marker=series.marker,
        )[0]

        if not series.data_point_config.show_label:
            return

        offset_x = series.data_point_config.offset_x
        offset_y = series.data_point_config.offset_y
        fmt_fn = series.data_point_config.fmt_fn
        font = series.data_point_config.font

        for xi, yi in zip(series.x_data, series.y_data):
            ax.text(
                x=xi + offset_x,
                y=yi + offset_y,
                s=fmt_fn(yi),
                ha='center',
                va='bottom',
                fontproperties=self._apply_font_config(font),
                color=font.color,
            )

    def _draw_single_scatter(
        self, series: SeriesConfig, ax: plt.Axes
    ):
        scatter = ax.scatter(
            series.x_data,
            series.y_data,
            label=series.label,
            s=series.width,
            alpha=series.alpha,
            marker=series.marker,
        )

        if not series.data_point_config.show_label:
            return

        offset_x = series.data_point_config.offset_x
        offset_y = series.data_point_config.offset_y
        fmt_fn = series.data_point_config.fmt_fn
        font = series.data_point_config.font

        for xi, yi in zip(series.x_data, series.y_data):
            ax.text(
                x=xi + offset_x,
                y=yi + offset_y,
                s=fmt_fn(yi),
                ha='center',
                va='bottom',
                fontproperties=self._apply_font_config(font),
                color=font.color,
            )

    def _draw_single_heatmap(
        self, series: SeriesConfig, ax: plt.Axes
    ):
        im = ax.imshow(series.y_data, cmap='viridis', aspect='auto')

        if series.color_bar_config and series.color_bar_config.enabled:
            color_bar_config = series.color_bar_config
            cbar = ax.figure.colorbar(
                im,
                ax=ax,
                shrink=color_bar_config.shrink,
                aspect=color_bar_config.aspect,
            )
            if color_bar_config.label:
                cbar.set_label(color_bar_config.label)

    def _draw_single_series(
        self,
        series: SeriesConfig,
        ax: plt.Axes,
    ):
        if series.plot_type == PlotType.BAR:
            self._draw_single_bar(series, ax)
        elif series.plot_type == PlotType.LINE:
            self._draw_single_line(series, ax)
        elif series.plot_type == PlotType.SCATTER:
            self._draw_single_scatter(series, ax)
        elif series.plot_type == PlotType.HEATMAP:
            self._draw_single_heatmap(series, ax)
        else:
            raise ValueError(f'未知 PlotType: {series.plot_type}')

    def _configure_axes(self, subplot_config: SubplotConfig, ax: plt.Axes):
        # 设置子图标题
        if subplot_config.title_config.text:
            ax.set_title(
                subplot_config.title_config.text,
                fontproperties=self._apply_font_config(subplot_config.title_config.font),
                pad=subplot_config.title_config.pad,
                loc=subplot_config.title_config.loc,
                color=subplot_config.title_config.font.color,
            )

        # 设置X轴
        if subplot_config.x_axis_config.ticks is not None:
            ax.set_xticks(subplot_config.x_axis_config.ticks)
        if subplot_config.x_axis_config.tick_labels is not None:
            ax.set_xticklabels(subplot_config.x_axis_config.tick_labels)
        if subplot_config.x_axis_config.lim is not None:
            ax.set_xlim(subplot_config.x_axis_config.lim)
        if subplot_config.x_axis_config.label:
            ax.set_xlabel(
                subplot_config.x_axis_config.label,
                fontproperties=self._apply_font_config(subplot_config.x_axis_config.label_font),
            )

        # 设置Y轴
        if subplot_config.y_axis_config.ticks is not None:
            ax.set_yticks(subplot_config.y_axis_config.ticks)
        if subplot_config.y_axis_config.tick_labels is not None:
            ax.set_yticklabels(subplot_config.y_axis_config.tick_labels)
        if subplot_config.y_axis_config.lim is not None:
            ax.set_ylim(subplot_config.y_axis_config.lim)
        if subplot_config.y_axis_config.label:
            ax.set_ylabel(
                subplot_config.y_axis_config.label,
                fontproperties=self._apply_font_config(subplot_config.y_axis_config.label_font),
            )

        # 设置网格
        if subplot_config.x_axis_config.grid_enabled and subplot_config.y_axis_config.grid_enabled:
            ax.grid(alpha=subplot_config.x_axis_config.grid_alpha)
        elif subplot_config.x_axis_config.grid_enabled:
            ax.grid(axis='x', alpha=subplot_config.x_axis_config.grid_alpha)
        elif subplot_config.y_axis_config.grid_enabled:
            ax.grid(axis='y', alpha=subplot_config.y_axis_config.grid_alpha)

        # 应用坐标轴刻度字体配置
        ax.tick_params(
            axis='x', labelsize=subplot_config.x_axis_config.tick_font.size,
            labelcolor=subplot_config.x_axis_config.tick_font.color,
        )
        ax.tick_params(
            axis='y', labelsize=subplot_config.y_axis_config.tick_font.size,
            labelcolor=subplot_config.y_axis_config.tick_font.color,
        )

        # 设置图例
        if subplot_config.legend_config.enabled and any(s.label for s in subplot_config.series_list):
            ax.legend(
                loc=subplot_config.legend_config.loc,
                prop=self._apply_font_config(subplot_config.legend_config.font_config),
                frameon=subplot_config.legend_config.frame_on,
                shadow=subplot_config.legend_config.shadow,
                borderaxespad=subplot_config.legend_config.border_pad,
                columnspacing=subplot_config.legend_config.column_spacing,
                handletextpad=subplot_config.legend_config.handle_text_pad,
                ncol=subplot_config.legend_config.n_col,
            )

    def draw(self, config: PlotConfig):
        """根据 :class:`PlotConfig` 创建 figure / axes 并绘制。

        Args:
            config: 完整绘图配置。

        Returns:
            ``(fig, axes)`` 元组。``axes`` 在 ``layout == (1, 1)`` 时是长度为 1 的列表。
        """
        rows, cols = config.layout
        fig, axes = plt.subplots(
            nrows=rows,
            ncols=cols,
            figsize=config.figure_size,
            dpi=config.dpi,
        )

        if rows == cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # 设置主标题
        if config.main_title:
            fig.suptitle(
                t=config.main_title,
                fontproperties=self._apply_font_config(config.main_title_config.font),
                color=config.main_title_config.font.color,
            )

        # 获取所有需要自动分配颜色的系列
        auto_series_count = 0
        for subplot_config in config.subplots:
            for series in subplot_config.series_list:
                if series.color == 'auto':
                    auto_series_count += 1

        # 分配默认颜色
        default_colors = self._get_default_colors(auto_series_count)
        auto_color_idx = 0

        # 绘制每个子图
        for i, (subplot_config, ax) in enumerate(zip(config.subplots, axes)):

            # 为 auto 部分的 series 分配颜色
            for series in subplot_config.series_list:
                if series.color == 'auto':
                    series.color = default_colors[auto_color_idx]
                    auto_color_idx += 1

            # 绘制该子图的所有系列
            for series in subplot_config.series_list:
                self._draw_single_series(series, ax)

            # 配置坐标轴
            self._configure_axes(subplot_config, ax)

        # 隐藏未使用的子图
        for i in range(len(config.subplots), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if config.save_path:
            plt.savefig(config.save_path, dpi=config.dpi, bbox_inches='tight')

        if config.show_plot:
            plt.show()

        return fig, axes
