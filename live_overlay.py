"""One-view live camera eraser with visible hand tracking and green fingertip trail."""

from __future__ import annotations

import math
import time

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .camera import CameraWorker


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
GREEN = (40, 235, 135)
HAND_YELLOW = (255, 205, 55)


class LiveEraserWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Finger Eraser | Live")
        self.setMinimumSize(980, 660)
        self.resize(1440, 900)
        self.latest_frame: np.ndarray | None = None
        self.removal_mask: np.ndarray | None = None
        self.removal_bounds: tuple[int, int, int, int] | None = None
        self.trace: list[tuple[int, int]] = []
        self.trace_active = False
        self.tracking_gap_active = False
        self.last_draw_time = 0.0
        self.awaiting_release = False
        self.last_point: tuple[int, int] | None = None
        self.cursor: tuple[int, int] | None = None
        self.hand_points: list[tuple[float, float]] = []
        self.drawing = False
        self.brush_size = 12
        self.mask_history: list[np.ndarray] = []
        self.camera_worker: CameraWorker | None = None
        self._build_ui()
        self._apply_style()
        self._start_camera()

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(12)

        header = QHBoxLayout()
        title_block = QVBoxLayout()
        title = QLabel("LIVE ERASER")
        title.setObjectName("title")
        subtitle = QLabel("Keep one finger extended. Circle an object to remove it from the live view.")
        subtitle.setObjectName("muted")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header.addLayout(title_block)
        header.addStretch(1)

        self.camera_combo = QComboBox()
        for index in range(4):
            self.camera_combo.addItem(f"Camera {index}", index)
        self.camera_combo.currentIndexChanged.connect(self._restart_camera)
        header.addWidget(self.camera_combo)

        self.brush_value = QLabel("12 px")
        self.brush_value.setObjectName("muted")
        header.addWidget(self.brush_value)
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(4, 36)
        self.brush_slider.setValue(self.brush_size)
        self.brush_slider.setFixedWidth(110)
        self.brush_slider.valueChanged.connect(self._set_brush_size)
        self.brush_slider.setToolTip("Change the green trail thickness")
        header.addWidget(self.brush_slider)

        self.undo_button = QPushButton("Undo removal")
        self.undo_button.clicked.connect(self.undo_removal)
        self.undo_button.setEnabled(False)
        header.addWidget(self.undo_button)
        self.reset_button = QPushButton("Reset all")
        self.reset_button.clicked.connect(self.reset_removals)
        self.reset_button.setEnabled(False)
        header.addWidget(self.reset_button)
        layout.addLayout(header)

        self.live_view = QLabel("Waiting for camera...")
        self.live_view.setObjectName("liveView")
        self.live_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.live_view.setMinimumSize(720, 520)
        self.live_view.setSizePolicy(
            self.live_view.sizePolicy().Policy.Expanding,
            self.live_view.sizePolicy().Policy.Expanding,
        )
        layout.addWidget(self.live_view, 1)

        footer = QHBoxLayout()
        self.status_label = QLabel("Opening camera and hand tracker...")
        self.status_label.setObjectName("status")
        footer.addWidget(self.status_label)
        footer.addStretch(1)
        hint = QLabel("Green line: fingertip path  |  Close the loop: remove  |  Mouse: draw")
        hint.setObjectName("muted")
        footer.addWidget(hint)
        layout.addLayout(footer)
        self.setCentralWidget(root)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget#root { background: #11191b; }
            QWidget { color: #ecf4f1; font-family: 'Noto Sans'; font-size: 13px; }
            QLabel#title { font-size: 24px; font-weight: 800; }
            QLabel#muted { color: #9caaa8; }
            QLabel#status { color: #45d6b3; }
            QLabel#liveView { background: #080d0e; border: 1px solid #354548; border-radius: 8px; color: #9caaa8; }
            QComboBox { background: #1b2628; border: 1px solid #3b4b4d; border-radius: 5px; padding: 7px; min-width: 90px; }
            QPushButton { background: #1b2628; border: 1px solid #3b4b4d; border-radius: 5px; padding: 8px 11px; }
            QPushButton:hover { border-color: #45d6b3; }
            QPushButton:disabled { color: #667573; }
            QSlider::groove:horizontal { height: 5px; background: #3b4b4d; border-radius: 2px; }
            QSlider::handle:horizontal { background: #45d6b3; width: 15px; margin: -5px 0; border-radius: 7px; }
            """
        )

    def _start_camera(self) -> None:
        self.camera_worker = CameraWorker(self.camera_combo.currentData(), 640, 480)
        self.camera_worker.frame_ready.connect(self._on_frame)
        self.camera_worker.hand_update.connect(self._on_hand_update)
        self.camera_worker.landmarks_ready.connect(self._on_landmarks)
        self.camera_worker.status_changed.connect(self._set_status)
        self.camera_worker.start()

    def _stop_camera(self) -> None:
        worker = self.camera_worker
        self.camera_worker = None
        if worker is not None and worker.isRunning():
            worker.stop()
            worker.deleteLater()

    def _restart_camera(self, unused_index: int = 0) -> None:
        self._stop_camera()
        self.latest_frame = None
        self.removal_mask = None
        self.removal_bounds = None
        self.trace.clear()
        self.mask_history.clear()
        self.trace_active = False
        self.tracking_gap_active = False
        self.awaiting_release = False
        self.undo_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.live_view.setText("Reconnecting camera...")
        self._start_camera()

    def _on_frame(self, frame: np.ndarray) -> None:
        self.latest_frame = frame
        if self.removal_mask is None or self.removal_mask.shape != frame.shape[:2]:
            self.removal_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            self.removal_bounds = None
        self._render_live_view()

    def _on_landmarks(self, points: object) -> None:
        self.hand_points = points if isinstance(points, list) else []

    def _on_hand_update(self, x: float, y: float, gesture: str) -> None:
        if self.latest_frame is None:
            return
        now = time.monotonic()
        height, width = self.latest_frame.shape[:2]
        if x < 0 or y < 0:
            self.cursor = None
            self.drawing = False
            self._handle_tracking_gap(now)
            return

        point = (
            min(max(round(x * (width - 1)), 0), width - 1),
            min(max(round(y * (height - 1)), 0), height - 1),
        )
        self.cursor = point
        if gesture == "DRAW":
            self.last_draw_time = now
            if self.awaiting_release:
                return
            if not self.trace_active:
                self.trace_active = True
                self.trace = [point]
                self.last_point = point
                self.tracking_gap_active = False
            elif self.tracking_gap_active:
                self.trace.append(point)
                self.last_point = point
                self.tracking_gap_active = False
            elif self.last_point is not None and math.dist(point, self.last_point) >= 1.0:
                self.trace.append(point)
                self.last_point = point
                if self._loop_is_closed():
                    self._commit_loop()
                    self.awaiting_release = True
            self.drawing = True
            return

        self.drawing = False
        if self.awaiting_release:
            self.awaiting_release = False
        else:
            self._handle_tracking_gap(now)

    def _handle_tracking_gap(self, now: float) -> None:
        if self.awaiting_release:
            self.awaiting_release = False
            return
        if not self.trace_active:
            return
        if now - self.last_draw_time > 0.75:
            self._end_trace()
        else:
            self.tracking_gap_active = True
            self.last_point = None

    def _loop_is_closed(self) -> bool:
        if self.latest_frame is None or len(self.trace) < 10:
            return False
        xs = [point[0] for point in self.trace]
        ys = [point[1] for point in self.trace]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        minimum_span = max(30, round(min(self.latest_frame.shape[:2]) * 0.055))
        if min(width, height) < minimum_span:
            return False
        aspect = width / max(1, height)
        if not 0.35 <= aspect <= 2.8:
            return False
        path_length = sum(
            math.dist(self.trace[index - 1], self.trace[index])
            for index in range(1, len(self.trace))
        )
        if path_length < max(width, height) * 1.4:
            return False
        area = abs(cv2.contourArea(np.asarray(self.trace, dtype=np.int32)))
        if area < min(width, height) ** 2 * 0.12:
            return False
        closure_limit = max(14, min(width, height) * 0.2)
        return math.dist(self.trace[0], self.trace[-1]) <= closure_limit

    def _end_trace(self) -> None:
        if self.trace_active and self._loop_is_closed():
            self._commit_loop()
        self.trace_active = False
        self.tracking_gap_active = False
        self.trace.clear()
        self.last_point = None
        self.last_draw_time = 0.0

    def _commit_loop(self) -> None:
        if self.removal_mask is None or len(self.trace) < 3:
            return
        self.mask_history.append(self.removal_mask.copy())
        if len(self.mask_history) > 12:
            self.mask_history.pop(0)
        polygon = np.asarray(self.trace, dtype=np.int32)
        cv2.fillPoly(self.removal_mask, [polygon], 255, cv2.LINE_AA)
        x, y, width, height = cv2.boundingRect(polygon)
        if self.removal_bounds is not None:
            old_x, old_y, old_width, old_height = self.removal_bounds
            right = max(x + width, old_x + old_width)
            bottom = max(y + height, old_y + old_height)
            x = min(x, old_x)
            y = min(y, old_y)
            width = right - x
            height = bottom - y
        self.removal_bounds = (x, y, width, height)
        self.undo_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.status_label.setText("Marked circle removed from live view")
        self.trace_active = False
        self.tracking_gap_active = False
        self.trace.clear()
        self.last_point = None
        self.last_draw_time = 0.0
        self.awaiting_release = True

    def _render_live_view(self) -> None:
        if self.latest_frame is None:
            return
        frame = self.latest_frame.copy()
        if self.removal_mask is not None and self.removal_bounds is not None:
            x, y, width, height = self.removal_bounds
            padding = 12
            left = max(0, x - padding)
            top = max(0, y - padding)
            right = min(frame.shape[1], x + width + padding)
            bottom = min(frame.shape[0], y + height + padding)
            frame[top:bottom, left:right] = cv2.inpaint(
                frame[top:bottom, left:right],
                self.removal_mask[top:bottom, left:right],
                9,
                cv2.INPAINT_TELEA,
            )

        if self.hand_points:
            height, width = frame.shape[:2]
            points = [
                (round(x * (width - 1)), round(y * (height - 1)))
                for x, y in self.hand_points
            ]
            for first, second in HAND_CONNECTIONS:
                cv2.line(frame, points[first], points[second], HAND_YELLOW, 2, cv2.LINE_AA)
            for point in points:
                cv2.circle(frame, point, 3, HAND_YELLOW, -1, cv2.LINE_AA)

        if len(self.trace) > 1:
            trace_points = np.asarray(self.trace, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [trace_points], False, GREEN, max(3, self.brush_size), cv2.LINE_AA)
        if self.cursor is not None and self.drawing:
            cv2.circle(frame, self.cursor, self.brush_size + 5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, self.cursor, max(4, self.brush_size // 3), GREEN, -1, cv2.LINE_AA)

        height, width = frame.shape[:2]
        image = QImage(
            frame.data,
            width,
            height,
            frame.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.live_view.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.live_view.setPixmap(pixmap)

    def _set_brush_size(self, size: int) -> None:
        self.brush_size = size
        self.brush_value.setText(f"{size} px")

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def undo_removal(self) -> None:
        if not self.mask_history or self.latest_frame is None:
            return
        self.removal_mask = self.mask_history.pop()
        self._rebuild_removal_bounds()
        self.undo_button.setEnabled(bool(self.mask_history))
        self.reset_button.setEnabled(bool(np.any(self.removal_mask)))
        self.status_label.setText("Last removal undone")
        self._render_live_view()

    def reset_removals(self) -> None:
        if self.removal_mask is None:
            return
        self.removal_mask.fill(0)
        self.removal_bounds = None
        self.mask_history.clear()
        self.undo_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.status_label.setText("All removals reset")
        self._render_live_view()

    def _rebuild_removal_bounds(self) -> None:
        if self.removal_mask is None:
            self.removal_bounds = None
            return
        points = cv2.findNonZero(self.removal_mask)
        self.removal_bounds = cv2.boundingRect(points) if points is not None else None

    def closeEvent(self, event: object) -> None:
        self._stop_camera()
        event.accept()
