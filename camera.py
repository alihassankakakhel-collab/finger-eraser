"""Read webcam frames and recognize basic hand poses with MediaPipe."""

from __future__ import annotations

import math
from typing import Any

import cv2
import mediapipe as mp
from PySide6.QtCore import QThread, Signal


def index_finger_is_extended(points: Any) -> bool:
    """Detect an extended index finger without assuming it points upward."""
    metacarpal, joint, fingertip = points[5], points[6], points[8]
    joint_length = math.hypot(joint.x - metacarpal.x, joint.y - metacarpal.y)
    finger_length = math.hypot(fingertip.x - metacarpal.x, fingertip.y - metacarpal.y)
    return finger_length > joint_length * 1.3


class CameraWorker(QThread):
    frame_ready = Signal(object)
    landmarks_ready = Signal(object)
    hand_update = Signal(float, float, str)
    status_changed = Signal(str)

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480) -> None:
        super().__init__()
        self.camera_index = camera_index
        self.width = width
        self.height = height

    def run(self) -> None:
        camera = cv2.VideoCapture(self.camera_index)
        tracker = None
        if not camera.isOpened():
            self.status_changed.emit(f"Camera {self.camera_index} could not be opened")
            camera.release()
            return

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        try:
            tracker = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=0,
                min_detection_confidence=0.65,
                min_tracking_confidence=0.6,
            )
            self.status_changed.emit(f"Camera {self.camera_index} ready")
            smoothed_point: tuple[float, float] | None = None

            while not self.isInterruptionRequested():
                success, frame = camera.read()
                if not success:
                    self.status_changed.emit("Camera stopped returning frames")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = tracker.process(rgb_frame)
                if results.multi_hand_landmarks:
                    points = results.multi_hand_landmarks[0].landmark
                    self.landmarks_ready.emit(
                        [(float(point.x), float(point.y)) for point in points]
                    )
                    point = (float(points[8].x), float(points[8].y))
                    if smoothed_point is None:
                        smoothed_point = point
                    else:
                        smoothed_point = (
                            smoothed_point[0] * 0.25 + point[0] * 0.75,
                            smoothed_point[1] * 0.25 + point[1] * 0.75,
                        )

                    if index_finger_is_extended(points):
                        gesture = "DRAW"
                    else:
                        gesture = "HOVER"
                    self.hand_update.emit(*smoothed_point, gesture)
                else:
                    smoothed_point = None
                    self.landmarks_ready.emit([])
                    self.hand_update.emit(-1.0, -1.0, "NO_HAND")

                self.frame_ready.emit(rgb_frame.copy())
                
        except Exception as error:
            self.status_changed.emit(f"Camera or hand tracking error: {error}")
        finally:
            if tracker is not None:
                tracker.close()
            camera.release()
            self.status_changed.emit("Camera stopped")

    def stop(self) -> None:
        self.requestInterruption()
        self.wait(5000)
