# -*- coding: utf-8 -*-
"""
数字图像处理工具 - 主程序 v2
- QGraphicsView 画布（缩放/平移/取色）
- 动态参数面板（按算法切换滑块）
- 空格键临时查看原图
- 四宫格算法对比、直方图联动阈值、导出报告图
"""
import sys
import time
import os
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QDockWidget, QToolBar, QAction, QFileDialog,
    QScrollArea, QGroupBox, QGridLayout, QMessageBox, QStatusBar,
    QGraphicsView, QGraphicsScene
)
from PyQt5.QtCore import Qt, QSize, QRectF
from PyQt5.QtGui import QImage, QPixmap

import algorithms as alg

# matplotlib 中文字体配置（避免标题乱码）
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# 算法注册表：名称 -> (函数, 默认参数dict, 分组)
# 参数 dict: key -> (label, lo, hi, default, step)
ALGORITHMS = {
    # 点处理
    'gray':       ('灰度化',          lambda im, p: alg.to_gray(im),            {},                          '点处理'),
    'invert':     ('反色/负片',       lambda im, p: alg.invert(im),             {},                          '点处理'),
    'binary':     ('二值化',          lambda im, p: alg.threshold_binary(im, p['th']),
                   {'th': ('阈值', 0, 255, 128, 1)},                                       '点处理'),
    'log':        ('对数变换',        lambda im, p: alg.log_transform(im),       {},                          '点处理'),
    'equalize':   ('直方图均衡化',    lambda im, p: alg.hist_equalize(im),       {},                          '点处理'),
    'stretch':    ('灰度拉伸',        lambda im, p: alg.stretch_gray(im, p['lo'], p['hi']),
                   {'lo': ('低切点', 0, 128, 20, 1), 'hi': ('高切点', 128, 255, 235, 1)},   '点处理'),
    # 滤波
    'mean':       ('均值滤波',       lambda im, p: alg.mean_filter(im, p['k']),
                   {'k': ('核大小', 3, 15, 3, 2)},                                        '滤波'),
    'gaussian':   ('高斯滤波',       lambda im, p: alg.gaussian_filter(im, p['k'], p['sigma']),
                   {'k': ('核大小', 3, 15, 5, 2), 'sigma': ('σ', 0.1, 5.0, 1.0, 0.1)},     '滤波'),
    'median':     ('中值滤波',       lambda im, p: alg.median_filter(im, p['k']),
                   {'k': ('核大小', 3, 15, 3, 2)},                                        '滤波'),
    # 边缘
    'sobel':      ('Sobel 边缘',      lambda im, p: alg.sobel_edge(im),         {},                          '边缘检测'),
    'prewitt':    ('Prewitt 边缘',   lambda im, p: alg.prewitt_edge(im),       {},                          '边缘检测'),
    'roberts':    ('Roberts 边缘',   lambda im, p: alg.roberts_edge(im),       {},                          '边缘检测'),
    'laplace':    ('Laplace 边缘',   lambda im, p: alg.laplace_edge(im),       {},                          '边缘检测'),
    'canny':      ('Canny 边缘',     lambda im, p: alg.canny_edge(im, p['low'], p['high']),
                   {'low': ('低阈值', 0, 255, 50, 1), 'high': ('高阈值', 0, 255, 150, 1)},  '边缘检测'),
    # 增强
    'lap_sharp':  ('Laplace 锐化',   lambda im, p: alg.laplace_sharpen(im),    {},                          '图像增强'),
    # 形态学
    'erode':      ('腐蚀',           lambda im, p: alg.erode(im, p['k'], p['iter']),
                   {'k': ('核大小', 3, 15, 3, 2), 'iter': ('迭代次数', 1, 10, 1, 1)},     '形态学'),
    'dilate':     ('膨胀',           lambda im, p: alg.dilate(im, p['k'], p['iter']),
                   {'k': ('核大小', 3, 15, 3, 2), 'iter': ('迭代次数', 1, 10, 1, 1)},     '形态学'),
    'open':       ('开运算',         lambda im, p: alg.morph_open(im, p['k'], p['iter']),
                   {'k': ('核大小', 3, 15, 3, 2), 'iter': ('迭代次数', 1, 10, 1, 1)},     '形态学'),
    'close':      ('闭运算',         lambda im, p: alg.morph_close(im, p['k'], p['iter']),
                   {'k': ('核大小', 3, 15, 3, 2), 'iter': ('迭代次数', 1, 10, 1, 1)},     '形态学'),
    # 伪彩色
    'pseudo_jet':    ('伪彩色·Jet',    lambda im, p: alg.pseudocolor(im, 'Jet'),    {},          '伪彩色'),
    'pseudo_hot':    ('伪彩色·Hot',    lambda im, p: alg.pseudocolor(im, 'Hot'),    {},          '伪彩色'),
    'pseudo_cool':   ('伪彩色·Cool',   lambda im, p: alg.pseudocolor(im, 'Cool'),   {},          '伪彩色'),
    'pseudo_rainbow':('伪彩色·Rainbow',lambda im, p: alg.pseudocolor(im, 'Rainbow'),{},          '伪彩色'),
    # 几何
    'flip_h':     ('水平翻转',       lambda im, p: alg.flip_h(im),             {},                          '几何变换'),
    'flip_v':     ('垂直翻转',       lambda im, p: alg.flip_v(im),             {},                          '几何变换'),
    'rot90':      ('旋转 90°',       lambda im, p: alg.rotate90(im),           {},                          '几何变换'),
    'rot180':     ('旋转 180°',      lambda im, p: alg.rotate180(im),          {},                          '几何变换'),
}


def ndarray_to_qpixmap(img):
    if img is None:
        return QPixmap()
    if img.ndim == 2:
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0],
                      QImage.Format_Grayscale8)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                       QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


# ============================================================
# 画布：QGraphicsView + 缩放/平移/取色
# ============================================================
class ImageView(QGraphicsView):
    def __init__(self, main_window):
        self.scene = QGraphicsScene()
        super().__init__(self.scene)
        self.main = main_window
        self.setRenderHint(0)  # SmoothPixmapTransform
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setAlignment(Qt.AlignCenter)
        self.pix_item = None
        self.current_img = None
        self.zoom_factor = 1.0

    def set_image(self, img):
        self.current_img = img
        pix = ndarray_to_qpixmap(img)
        self.scene.clear()
        self.pix_item = self.scene.addPixmap(pix)
        self.scene.setSceneRect(QRectF(pix.rect()))

    def wheelEvent(self, e):
        factor = 1.2 if e.angleDelta().y() > 0 else 1 / 1.2
        self.zoom_factor *= factor
        self.scale(factor, factor)

    def mouseMoveEvent(self, e):
        super().mouseMoveEvent(e)
        if self.current_img is None:
            return
        pos = self.mapToScene(e.pos())
        x, y = int(pos.x()), int(pos.y())
        h, w = self.current_img.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            px = self.current_img[y, x]
            if self.current_img.ndim == 3:
                b, g, r = int(px[0]), int(px[1]), int(px[2])
                self.main.show_pixel_info(x, y, f"RGB=({r},{g},{b})")
            else:
                v = int(px)
                self.main.show_pixel_info(x, y, f"灰度={v}")

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Space:
            self.main.show_original_temp()
        else:
            super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Space:
            self.main.restore_current()
        else:
            super().keyReleaseEvent(e)


# ============================================================
# 直方图分析窗口（可拖拽阈值线）
# ============================================================
class HistogramWindow:
    def __init__(self, img, on_thresh_change=None):
        self.plt = plt
        self.on_thresh_change = on_thresh_change
        self.fig, self.ax = plt.subplots(figsize=(8, 4.5))
        h = alg.hist_gray(img)
        self.ax.plot(h, color='k', lw=1)
        self.ax.set_title("灰度直方图 —— 拖动红线调整二值化阈值")
        self.ax.set_xlabel("灰度值")
        self.ax.set_ylabel("像素数")
        self.th = 128
        self.line = self.ax.axvline(self.th, color='r', lw=2)
        self.text = self.ax.text(0.02, 0.95, f"阈值 = {self.th}",
                                 transform=self.ax.transAxes, color='r', fontsize=12)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_move)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.dragging = False
        plt.tight_layout()
        plt.show()

    def _on_click(self, e):
        if e.inaxes == self.ax and abs(e.xdata - self.th) < 5:
            self.dragging = True

    def _on_move(self, e):
        if not self.dragging or e.inaxes != self.ax:
            return
        self.th = int(np.clip(e.xdata, 0, 255))
        self.line.set_xdata([self.th, self.th])
        self.text.set_text(f"阈值 = {self.th}")
        self.fig.canvas.draw_idle()
        if self.on_thresh_change:
            self.on_thresh_change(self.th)


# ============================================================
# 主窗口
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("数字图像处理实验室")
        self.resize(1500, 900)

        self.original = None
        self.current = None
        self.current_alg = 'gray'
        self.params = {}
        self.global_bright = 0
        self.global_contrast = 1.0
        self.global_gamma = 1.0
        self.history = []           # [(label, ndarray), ...]
        self.history_index = -1
        self._busy = False

        self._build_menu()
        self._build_toolbar()
        self._build_central()
        self._build_right_dock()
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("就绪 —— 先打开一张图片")

    # ---------- UI ----------
    def _build_menu(self):
        mb = self.menuBar()
        m = mb.addMenu("文件(&F)")
        self._add(m, "打开", self.open_image, "Ctrl+O")
        self._add(m, "保存结果", self.save_image, "Ctrl+S")
        self._add(m, "导出报告图", self.export_report, "Ctrl+E")
        m.addSeparator()
        self._add(m, "退出", self.close, "Ctrl+Q")

        m = mb.addMenu("分析(&A)")
        self._add(m, "直方图分析", self.show_histogram)
        self._add(m, "四宫格对比", self.show_compare)

        m = mb.addMenu("帮助(&H)")
        self._add(m, "关于", self.about)

    def _add(self, menu, text, slot, shortcut=None):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(shortcut)
        a.triggered.connect(slot)
        menu.addAction(a)

    def _build_toolbar(self):
        tb = QToolBar("主工具栏", self)
        tb.setIconSize(QSize(22, 22))
        self.addToolBar(Qt.TopToolBarArea, tb)
        for label, slot in [
            ("打开", self.open_image),
            ("保存", self.save_image),
            ("撤销", self.prev_step),
            ("重做", self.next_step),
            ("直方图", self.show_histogram),
            ("四宫格", self.show_compare),
            ("导出报告图", self.export_report),
        ]:
            a = QAction(label, self)
            a.triggered.connect(slot)
            tb.addAction(a)

    def _build_central(self):
        splitter = QSplitter(Qt.Horizontal)

        # 左侧：分组算法按钮
        left = QScrollArea()
        left.setWidgetResizable(True)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(QLabel("<b>算法列表</b>"))
        groups = {}
        for key, (name, func, params, group) in ALGORITHMS.items():
            if group not in groups:
                box = QGroupBox(group)
                bl = QVBoxLayout(box)
                groups[group] = bl
                left_layout.addWidget(box)
            btn = QPushButton(name)
            btn.setMinimumHeight(30)
            btn.clicked.connect(lambda _=False, k=key: self.select_algorithm(k))
            groups[group].addWidget(btn)
        left_layout.addStretch(1)
        left.setWidget(left_widget)
        left.setMaximumWidth(220)
        splitter.addWidget(left)

        # 中间画布
        self.view = ImageView(self)
        self.view.set_image(np.full((400, 600, 3), 40, dtype=np.uint8))
        splitter.addWidget(self.view)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

    def _build_right_dock(self):
        dock = QDockWidget("参数面板", self)
        dock.setAllowedAreas(Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        wrap = QWidget()
        wl = QVBoxLayout(wrap)

        # 全局调整（始终在顶部）
        g = QGroupBox("全局调整（叠加在所有算法上）")
        f = QGridLayout(g)
        self.sl_bright = self._make_slider(f, 0, "亮度", -100, 100, 0, self._on_global_change)
        self.sl_contrast = self._make_slider(f, 1, "对比度%", 0, 300, 100, self._on_global_change)
        self.sl_gamma = self._make_slider(f, 2, "伽马×100", 20, 400, 100, self._on_global_change)
        wl.addWidget(g)

        # 当前算法参数（动态重建）
        self.alg_group = QGroupBox("当前算法参数")
        self.alg_layout = QGridLayout(self.alg_group)
        self.alg_sliders = {}
        wl.addWidget(self.alg_group)

        # 历史
        g = QGroupBox("历史记录")
        self.history_layout = QVBoxLayout(g)
        wl.addWidget(g)

        wl.addStretch(1)
        dock.setWidget(wrap)
        dock.setMinimumWidth(280)

    def _make_slider(self, grid, row, name, lo, hi, val, slot):
        lab = QLabel(name)
        s = QSlider(Qt.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        vlab = QLabel(str(val))
        s.valueChanged.connect(lambda v: vlab.setText(str(v)))
        s.valueChanged.connect(slot)
        grid.addWidget(lab, row, 0)
        grid.addWidget(s, row, 1)
        grid.addWidget(vlab, row, 2)
        return s

    # ---------- 算法选择 ----------
    def select_algorithm(self, key):
        if self.original is None:
            QMessageBox.information(self, "提示", "请先打开一张图片")
            return
        self.current_alg = key
        # 初始化参数
        _, _, pdefs, _ = ALGORITHMS[key]
        self.params = {k: defaults[3] for k, defaults in pdefs.items()}
        self._rebuild_alg_panel()
        self.apply_current()
        # 把新算法结果压入历史
        label = ALGORITHMS[key][0]
        self._push_history(label, self.current)

    def _rebuild_alg_panel(self):
        # 清空旧滑块
        while self.alg_layout.count():
            it = self.alg_layout.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()
        self.alg_sliders = {}
        if self.current_alg == 'original':
            return  # 原图无参数
        _, _, pdefs, _ = ALGORITHMS[self.current_alg]
        for i, (key, (label, lo, hi, default, step)) in enumerate(pdefs.items()):
            lab = QLabel(label)
            s = QSlider(Qt.Horizontal)
            s.setRange(int(lo * 100), int(hi * 100)) if step < 1 else s.setRange(int(lo), int(hi))
            s.setValue(int(default * 100)) if step < 1 else s.setValue(int(default))
            vlab = QLabel(str(default))
            s.valueChanged.connect(lambda v, k=key, l=vlab, st=step: (
                l.setText(str(v / 100 if st < 1 else v)),
                self._on_alg_param_change(k, v, st)
            ))
            self.alg_layout.addWidget(lab, i, 0)
            self.alg_layout.addWidget(s, i, 1)
            self.alg_layout.addWidget(vlab, i, 2)
            self.alg_sliders[key] = (s, step)

    def _on_alg_param_change(self, key, value, step):
        self.params[key] = value / 100 if step < 1 else value
        self.apply_current(record_time=True, push_history=False)

    def _on_global_change(self, _v):
        self.global_bright = self.sl_bright.value()
        self.global_contrast = self.sl_contrast.value() / 100.0
        self.global_gamma = self.sl_gamma.value() / 100.0
        self.apply_current(record_time=True, push_history=False)

    # ---------- 应用算法 ----------
    def apply_current(self, record_time=False, push_history=False):
        if self.original is None:
            return
        t0 = time.time()
        try:
            base = alg.adjust_brightness(self.original, self.global_bright)
            base = alg.adjust_contrast(base, self.global_contrast)
            base = alg.gamma_transform(base, self.global_gamma)
            if self.current_alg == 'original':
                out = base
            else:
                func = ALGORITHMS[self.current_alg][1]
                out = func(base, self.params)
        except Exception as e:
            self.status.showMessage(f"算法错误: {e}")
            return
        dt = (time.time() - t0) * 1000
        self.current = out
        self.view.set_image(out)
        # PSNR（和原图比）
        if out.shape == self.original.shape:
            p = alg.psnr(self.original, out)
            psnr_str = f"PSNR={p:.2f}dB" if np.isfinite(p) else "PSNR=∞"
        else:
            psnr_str = "PSNR=N/A"
        alg_name = "原图" if self.current_alg == 'original' else ALGORITHMS[self.current_alg][0]
        self.status.showMessage(
            f"{alg_name}  耗时 {dt:.1f}ms  {psnr_str}  "
            f"尺寸 {out.shape[1]}×{out.shape[0]}"
        )
        if push_history:
            self._push_history(ALGORITHMS[self.current_alg][0], out)

    # ---------- 历史 ----------
    def _push_history(self, label, img):
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        self.history.append((label, img.copy()))
        if len(self.history) > 15:
            self.history.pop(0)
        self.history_index = len(self.history) - 1
        self._refresh_history_bar()

    def _refresh_history_bar(self):
        while self.history_layout.count():
            it = self.history_layout.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()
        for i, (label, _) in enumerate(self.history):
            b = QPushButton(f"{i+1}. {label}")
            if i == self.history_index:
                b.setStyleSheet("border:2px solid #e05050;")
            b.clicked.connect(lambda _=False, idx=i: self.goto_history(idx))
            self.history_layout.addWidget(b)

    def goto_history(self, idx):
        if 0 <= idx < len(self.history):
            self.history_index = idx
            label, img = self.history[idx]
            self.current = img
            self.view.set_image(img)
            self._refresh_history_bar()

    def prev_step(self):
        if self.history_index > 0:
            self.goto_history(self.history_index - 1)

    def next_step(self):
        if self.history_index < len(self.history) - 1:
            self.goto_history(self.history_index + 1)

    # ---------- 临时原图（空格） ----------
    def show_original_temp(self):
        if self.original is not None:
            self.view.set_image(self.original)

    def restore_current(self):
        if self.current is not None:
            self.view.set_image(self.current)

    # ---------- 状态栏 ----------
    def show_pixel_info(self, x, y, info):
        self.status.showMessage(f"({x},{y})  {info}")

    # ---------- 文件 ----------
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开图片", "", "图片 (*.png *.jpg *.jpeg *.bmp *.tif)"
        )
        if not path:
            return
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.warning(self, "错误", "无法读取图片")
            return
        self.original = img
        self.history = []
        self.history_index = -1
        self._push_history("原图", img)
        # 默认显示彩色原图，不自动应用灰度化
        self.current_alg = 'original'
        self.params = {}
        self._rebuild_alg_panel()
        self.current = img.copy()
        self.view.set_image(img)
        self.status.showMessage(f"已打开 {os.path.basename(path)}  {img.shape[1]}×{img.shape[0]}")

    def save_image(self):
        if self.current is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存", "result.png", "PNG (*.png);;JPG (*.jpg)")
        if path:
            cv2.imencode(os.path.splitext(path)[1], self.current)[1].tofile(path)
            self.status.showMessage(f"已保存 {path}")

    # ---------- 直方图分析 ----------
    def show_histogram(self):
        if self.current is None:
            return
        HistogramWindow(self.current, on_thresh_change=self._on_hist_thresh)

    def _on_hist_thresh(self, th):
        # 拖动阈值线时，切到二值化算法并更新
        if self.current_alg != 'binary':
            self.current_alg = 'binary'
            self.params = {'th': th}
            self._rebuild_alg_panel()
        else:
            self.params['th'] = th
            # 同步滑块
            if 'th' in self.alg_sliders:
                s, _ = self.alg_sliders['th']
                s.blockSignals(True)
                s.setValue(th)
                s.blockSignals(False)
        self.apply_current(record_time=True, push_history=False)

    # ---------- 四宫格对比 ----------
    def show_compare(self):
        if self.original is None:
            QMessageBox.information(self, "提示", "请先打开图片")
            return
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        panels = [
            ("原图", self.original),
            ("灰度化", alg.to_gray(self.original)),
            ("Sobel 边缘", alg.sobel_edge(self.original)),
            (f"当前：{ALGORITHMS[self.current_alg][0]}", self.current),
        ]
        for ax, (title, im) in zip(axes.ravel(), panels):
            if im.ndim == 2:
                ax.imshow(im, cmap='gray')
            else:
                ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    # ---------- 导出报告图 ----------
    def export_report(self):
        if self.original is None or self.current is None:
            return
        fig = plt.figure(figsize=(12, 7))
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])

        for ax, im, title in [
            (ax1, self.original, "原图"),
            (ax2, self.current, f"处理结果：{ALGORITHMS[self.current_alg][0]}"),
        ]:
            if im.ndim == 2:
                ax.imshow(im, cmap='gray')
            else:
                ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            ax.set_title(title)
            ax.axis('off')

        ax3.plot(alg.hist_gray(self.current), color='k', lw=1)
        ax3.set_title("结果图灰度直方图")
        ax3.set_xlabel("灰度值")

        plt.tight_layout()
        path, _ = QFileDialog.getSaveFileName(self, "导出报告图", "report.png", "PNG (*.png)")
        if path:
            plt.savefig(path, dpi=150, bbox_inches='tight')
            self.status.showMessage(f"报告图已导出：{path}")
        plt.close(fig)

    # ---------- 关于 ----------
    def about(self):
        QMessageBox.about(
            self, "关于",
            "<b>数字图像处理实验室</b><br>"
            "PyQt5 + NumPy + OpenCV<br>"
            "点处理 / 滤波 / 边缘 / 形态学 / 伪彩色 / 几何变换<br>"
            "快捷键：Ctrl+O 打开，Ctrl+S 保存，Ctrl+E 导出报告图，空格 临时看原图"
        )


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
