import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")
import threading
import time
import os
import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QScrollArea, QTextEdit, QFileDialog, QMessageBox, QGridLayout, QSizePolicy)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PIL import Image, ImageDraw
import cv2
import yaml
import queue

# YOLOv8
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
    print("⚠ Bạn chưa cài 'ultralytics'. Chạy: pip install ultralytics")

class StartScreen(QMainWindow):
    def __init__(self, on_start):
        super().__init__()
        self.setWindowTitle("Chào Mừng - Nhận Diện Biển Báo Giao Thông")
        self.setFixedSize(500, 300)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        title_label = QLabel("Nhận Diện Biển Báo Giao Thông")
        title_label.setStyleSheet("""
            font-size: 20px; 
            font-weight: bold; 
            color: #2c3e50;
            padding: 10px;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        desc_label = QLabel("Ứng dụng sử dụng YOLOv8 để nhận diện biển báo giao thông từ video hoặc webcam.")
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setStyleSheet("""
            font-size: 12px; 
            color: #34495e; 
            padding: 5px;
        """)
        layout.addWidget(desc_label)

        start_button = QPushButton("Bắt Đầu")
        start_button.setStyleSheet("""
            QPushButton {
                font-size: 14px; 
                padding: 10px 20px; 
                background-color: #3498db; 
                color: white; 
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #20638f;
            }
        """)
        start_button.clicked.connect(on_start)
        layout.addWidget(start_button)
        layout.addStretch()

        self.center()

    def center(self):
        screen = QApplication.primaryScreen().geometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2, (screen.height() - size.height()) // 2)

class YOLOGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nhận diện biển báo giao thông - YOLOv8")
        self.setGeometry(100, 100, 1000, 850)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ecf0f1;
            }
        """)

        self.model = None
        self.class_names = []
        self.capture = None
        self.video_thread = None
        self.running = False
        self.frame = None
        self.processed_frame = None
        self.model_path = ""
        self.samples_path = "samples"
        self.result_queue = queue.Queue()
        self.default_image = None
        self.detected_labels_all = set()
        self.label_widgets = {}
        self.fps = 0.0
        self.start_time = 0
        self.frame_count = 0
        self.recording = False
        self.video_writer = None
        self.output_video_path = ""
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.last_alert_time = 0
        self.alert_cooldown = 1.0

        self._build_ui()

        alert_file = "alert.wav"
        if os.path.exists(alert_file):
            self.player.setSource(QUrl.fromLocalFile(alert_file))
        else:
            self._log(f"Không tìm thấy tệp {alert_file}. Cảnh báo âm thanh sẽ không hoạt động.")

        self.sign_desc = {
            "DP.135": {"desc": "Hết tất cả các lệnh cấm", "suggestion": "Tiếp tục lái xe bình thường, tuân thủ các biển báo khác nếu có."},
            "P.102": {"desc": "Cấm đi ngược chiều", "suggestion": "Không đi ngược chiều, tuân thủ hướng đi được phép."},
            "P.103a": {"desc": "Cấm xe ô tô", "suggestion": "Xe ô tô không được phép đi vào, tìm lộ trình khác."},
            "P.103b": {"desc": "Cấm xe ô tô rẽ phải", "suggestion": "Không rẽ phải nếu đang lái ô tô, tiếp tục đi thẳng hoặc rẽ trái nếu được phép."},
            "P.103c": {"desc": "Cấm xe ô tô rẽ trái", "suggestion": "Không rẽ trái nếu đang lái ô tô, tiếp tục đi thẳng hoặc rẽ phải nếu được phép."},
            "P.104": {"desc": "Cấm xe máy", "suggestion": "Xe máy không được phép đi vào, tìm lộ trình khác."},
            "P.106a": {"desc": "Cấm xe ô tô tải", "suggestion": "Xe tải không được phép đi vào, tìm lộ trình khác."},
            "P.106b": {"desc": "Cấm ô tô tải có khối lượng chuyên chở lớn hơn giới hạn", "suggestion": "Kiểm tra trọng tải xe, tìm lộ trình phù hợp nếu vượt giới hạn."},
            "P.107a": {"desc": "Cấm xe ô tô khách", "suggestion": "Xe khách không được phép đi vào, tìm lộ trình khác."},
            "P.112": {"desc": "Cấm người đi bộ", "suggestion": "Người đi bộ không được phép đi vào khu vực này."},
            "P.115": {"desc": "Hạn chế trọng tải toàn bộ xe", "suggestion": "Kiểm tra trọng tải xe, không đi vào nếu vượt giới hạn cho phép."},
            "P.117": {"desc": "Hạn chế chiều cao xe", "suggestion": "Kiểm tra chiều cao xe, không đi vào nếu vượt giới hạn."},
            "P.123a": {"desc": "Cấm rẽ trái", "suggestion": "Không rẽ trái, tiếp tục đi thẳng hoặc rẽ phải nếu được phép."},
            "P.123b": {"desc": "Cấm rẽ phải", "suggestion": "Không rẽ phải, tiếp tục đi thẳng hoặc rẽ trái nếu được phép."},
            "P.124a": {"desc": "Cấm quay đầu xe", "suggestion": "Không quay đầu xe, tìm vị trí khác để quay đầu nếu cần."},
            "P.124b": {"desc": "Cấm ô tô quay đầu xe", "suggestion": "Ô tô không được quay đầu, tìm lộ trình khác."},
            "P.124c": {"desc": "Cấm rẽ trái và quay đầu xe", "suggestion": "Không rẽ trái hoặc quay đầu, đi thẳng hoặc rẽ phải nếu được phép."},
            "P.125": {"desc": "Cấm vượt", "suggestion": "Không vượt xe khác, giữ tốc độ và khoảng cách an toàn."},
            "P.127": {"desc": "Tốc độ tối đa cho phép", "suggestion": "Không vượt quá tốc độ ghi trên biển, điều chỉnh tốc độ phù hợp."},
            "P.128": {"desc": "Cấm sử dụng còi", "suggestion": "Không sử dụng còi, sử dụng tín hiệu khác nếu cần."},
            "P.130": {"desc": "Cấm dừng xe và đỗ xe", "suggestion": "Không dừng hoặc đỗ xe tại khu vực này."},
            "P.131a": {"desc": "Cấm đỗ xe", "suggestion": "Không đỗ xe tại khu vực này, tìm nơi đỗ hợp pháp."},
            "P.137": {"desc": "Cấm rẽ trái và rẽ phải", "suggestion": "Chỉ được đi thẳng, không rẽ trái hoặc phải."},
            "R.301c": {"desc": "Các xe chỉ được rẽ trái", "suggestion": "Rẽ trái tại giao lộ, không đi thẳng hoặc rẽ phải."},
            "R.301d": {"desc": "Các xe chỉ được rẽ phải", "suggestion": "Rẽ phải tại giao lộ, không đi thẳng hoặc rẽ trái."},
            "R.301e": {"desc": "Các xe chỉ được rẽ trái", "suggestion": "Rẽ trái tại giao lộ, không đi thẳng hoặc rẽ phải."},
            "R.302a": {"desc": "Phải đi vòng sang bên phải", "suggestion": "Đi vòng sang phải theo hướng dẫn của biển."},
            "R.302b": {"desc": "Phải đi vòng sang bên trái", "suggestion": "Đi vòng sang trái theo hướng dẫn của biển."},
            "R.303": {"desc": "Nơi giao nhau chạy theo vòng xuyến", "suggestion": "Lái xe theo vòng xuyến, nhường đường cho xe từ bên trái."},
            "R.407a": {"desc": "Đường một chiều", "suggestion": "Chỉ đi theo một chiều, tuân thủ hướng dẫn."},
            "R.409": {"desc": "Chỗ quay xe", "suggestion": "Sử dụng khu vực này để quay đầu xe nếu cần."},
            "R.425": {"desc": "Bệnh viện", "suggestion": "Giảm tốc độ, chú ý người đi bộ gần bệnh viện."},
            "R.434": {"desc": "Bến xe", "suggestion": "Giảm tốc độ, chú ý xe ra vào bến."},
            "S.509a": {"desc": "Chiều cao an toàn", "suggestion": "Kiểm tra chiều cao xe, đảm bảo không vượt quá giới hạn."},
            "W.201a": {"desc": "Chỗ ngoặt nguy hiểm vòng bên trái", "suggestion": "Giảm tốc độ, cẩn thận khi vào khúc cua trái."},
            "W.201b": {"desc": "Chỗ ngoặt nguy hiểm vòng bên phải", "suggestion": "Giảm tốc độ, cẩn thận khi vào khúc cua phải."},
            "W.202a": {"desc": "Nhiều chỗ ngoặt nguy hiểm liên tiếp, chỗ đầu tiên sang trái", "suggestion": "Giảm tốc độ, chuẩn bị cho nhiều khúc cua bắt đầu từ trái."},
            "W.202b": {"desc": "Nhiều chỗ ngoặt nguy hiểm liên tiếp, chỗ đầu tiên sang phải", "suggestion": "Giảm tốc độ, chuẩn bị cho nhiều khúc cua bắt đầu từ phải."},
            "W.203b": {"desc": "Đường bị thu hẹp về phía trái", "suggestion": "Giảm tốc độ, nhường đường cho xe từ phía đối diện."},
            "W.203c": {"desc": "Đường bị thu hẹp về phía phải", "suggestion": "Giảm tốc độ, nhường đường cho xe từ phía đối diện."},
            "W.205a": {"desc": "Đường giao nhau (ngã tư)", "suggestion": "Giảm tốc độ, quan sát kỹ trước khi qua ngã tư."},
            "W.205b": {"desc": "Đường giao nhau (ngã ba bên trái)", "suggestion": "Giảm tốc độ, chú ý xe từ ngã ba bên trái."},
            "W.205d": {"desc": "Đường giao nhau (hình chữ T)", "suggestion": "Giảm tốc độ, chuẩn bị dừng hoặc rẽ tại ngã ba hình chữ T."},
            "W.207a": {"desc": "Giao nhau với đường không ưu tiên", "suggestion": "Ưu tiên xe trên đường chính, quan sát kỹ."},
            "W.207b": {"desc": "Giao nhau với đường không ưu tiên", "suggestion": "Ưu tiên xe trên đường chính, quan sát kỹ."},
            "W.207c": {"desc": "Giao nhau với đường không ưu tiên", "suggestion": "Ưu tiên xe trên đường chính, quan sát kỹ."},
            "W.208": {"desc": "Giao nhau với đường ưu tiên", "suggestion": "Nhường đường cho xe trên đường ưu tiên."},
            "W.209": {"desc": "Giao nhau có tín hiệu đèn giao thông", "suggestion": "Tuân thủ tín hiệu đèn giao thông."},
            "W.210": {"desc": "Giao nhau với đường sắt có rào chắn", "suggestion": "Dừng lại khi rào chắn đóng, quan sát tàu hỏa."},
            "W.219": {"desc": "Dốc xuống nguy hiểm", "suggestion": "Giảm tốc độ, kiểm tra phanh trước khi xuống dốc."},
            "W.221b": {"desc": "Đường có gồ giảm tốc", "suggestion": "Giảm tốc độ khi qua gồ giảm tốc để tránh hư xe."},
            "W.224": {"desc": "Đường người đi bộ cắt ngang", "suggestion": "Giảm tốc độ, nhường đường cho người đi bộ."},
            "W.225": {"desc": "Trẻ em", "suggestion": "Giảm tốc độ, chú ý trẻ em có thể chạy qua đường."},
            "W.227": {"desc": "Công trường", "suggestion": "Giảm tốc độ, chú ý công nhân và máy móc."},
            "W.233": {"desc": "Nguy hiểm khác", "suggestion": "Giảm tốc độ, quan sát kỹ các nguy cơ tiềm ẩn."},
            "W.235": {"desc": "Đường đôi", "suggestion": "Tuân thủ làn đường, không lấn làn."},
            "W.245a": {"desc": "Đi chậm", "suggestion": "Giảm tốc độ, lái xe cẩn thận."}
        }
        self.alert_signs = ["P.125", "P.127"]

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(30)

    def _build_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_widget.setStyleSheet("""
            QWidget {
                background-color: #34495e;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        buttons = [
            ("Load model", self.load_model, True),
            ("Load data.yaml", self.load_data_yaml, True),
            ("Chọn video", self.open_video, True),
            ("Mở webcam", self.open_webcam, True),
            ("Start", self.start, False),
            ("Stop", self.stop, False),
            ("Tiếp tục", self.resume, False),
            ("Snapshot", self.snapshot, True),
            ("Lưu lịch sử", self.save_detection_history, True),
            ("Bắt đầu lưu video", self.toggle_record_video, True)
        ]
        self.button_widgets = {}
        for text, command, enabled in buttons:
            btn = QPushButton(text)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 12px; 
                    padding: 8px 12px; 
                    background-color: #3498db; 
                    color: white; 
                    border-radius: 4px;
                    margin: 2px;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
                QPushButton:pressed {
                    background-color: #20638f;
                }
                QPushButton:disabled {
                    background-color: #95a5a6;
                }
            """)
            btn.clicked.connect(command)
            btn.setEnabled(enabled)
            top_layout.addWidget(btn)
            if text in ["Start", "Stop", "Tiếp tục", "Bắt đầu lưu video"]:
                self.button_widgets[text] = btn
        main_layout.addWidget(top_widget)

        self.lbl_info = QLabel("Model: (chưa tải) | Video: (chưa chọn) | FPS: 0.0")
        self.lbl_info.setStyleSheet("""
            font-size: 12px; 
            color: #2c3e50; 
            background-color: #dfe6e9; 
            padding: 8px; 
            border-radius: 4px;
        """)
        main_layout.addWidget(self.lbl_info)

        main_frame = QWidget()
        main_frame_layout = QHBoxLayout(main_frame)
        main_frame_layout.setContentsMargins(0, 0, 0, 0)
        main_frame_layout.setSpacing(10)

        self.video_panel = QLabel()
        self.video_panel.setStyleSheet("""
            background-color: black; 
            border: 2px solid #34495e; 
            border-radius: 5px;
        """)
        self.video_panel.setFixedSize(600, 450)  # Cố định kích thước video_panel
        self.video_panel.setAlignment(Qt.AlignCenter)  # Căn giữa nội dung trong video_panel
        main_frame_layout.addWidget(self.video_panel)

        right_widget = QWidget()
        right_widget.setFixedWidth(350)
        right_widget.setStyleSheet("""
            background-color: white; 
            border: 1px solid #bdc3c7; 
            border-radius: 5px;
            padding: 10px;
        """)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)

        recent_label = QLabel("Biển báo nhận diện gần nhất")
        recent_label.setStyleSheet("""
            font-size: 14px; 
            font-weight: bold; 
            color: #2c3e50;
            padding: 5px;
        """)
        recent_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(recent_label)

        self.sample_panel = QLabel()
        self.sample_panel.setStyleSheet("""
            background-color: #f7f9fb; 
            border: 1px solid #dfe6e9; 
            border-radius: 5px;
            max-height: 150px;
        """)
        self.sample_panel.setAlignment(Qt.AlignCenter)
        self.sample_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        right_layout.addWidget(self.sample_panel)

        self.sample_label = QLabel("(Chưa nhận diện)")
        self.sample_label.setWordWrap(True)
        self.sample_label.setAlignment(Qt.AlignCenter)
        self.sample_label.setStyleSheet("""
            font-size: 12px; 
            color: #34495e; 
            padding: 5px;
        """)
        right_layout.addWidget(self.sample_label)

        signs_label = QLabel("Các biển báo được nhận diện")
        signs_label.setStyleSheet("""
            font-size: 12px; 
            font-weight: bold; 
            color: #2c3e50;
            padding: 5px;
        """)
        signs_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(signs_label)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: white; 
                border: 1px solid #dfe6e9; 
                border-radius: 5px;
                min-height: 350px;  # Đảm bảo hiển thị ít nhất 3 biển báo
            }
            QScrollBar:vertical {
                border: none;
                background: #f7f9fb;
                width: 10px;
                margin: 0;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #3498db;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
        """)
        self.frame_signs = QWidget()
        self.frame_signs_layout = QGridLayout(self.frame_signs)
        self.frame_signs_layout.setAlignment(Qt.AlignTop)
        self.frame_signs_layout.setSpacing(8)
        self.frame_signs_layout.setContentsMargins(8, 8, 8, 8)
        self.scroll_area.setWidget(self.frame_signs)
        right_layout.addWidget(self.scroll_area)

        main_frame_layout.addWidget(right_widget)
        main_layout.addWidget(main_frame)

        self.txt_log = QTextEdit()
        self.txt_log.setFixedHeight(100)
        self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet("""
            font-size: 12px; 
            color: #2c3e50; 
            background-color: #f7f9fb; 
            border: 1px solid #dfe6e9; 
            border-radius: 5px;
            padding: 5px;
        """)
        main_layout.addWidget(self.txt_log)
        self._log("Ứng dụng sẵn sàng.")

    def _log(self, msg):
        t = time.strftime("%H:%M:%S")
        self.txt_log.append(f"[{t}] {msg}")

    def _on_sign_click(self, label):
        desc = self.sign_desc.get(label, {"desc": "(Chưa có mô tả)", "suggestion": "Không có gợi ý"})["desc"]
        suggestion = self.sign_desc.get(label, {"desc": "(Chưa có mô tả)", "suggestion": "Không có gợi ý"})["suggestion"]
        sample_path = os.path.join(self.samples_path, f"{label}.jpg")
        if os.path.exists(sample_path):
            img = Image.open(sample_path).resize((150, 150), Image.Resampling.LANCZOS).convert("RGB")
            img_array = np.array(img)
            qimg = QImage(img_array.data, img_array.shape[1], img_array.shape[0], img_array.strides[0], QImage.Format_RGB888)
            self.sample_panel.setPixmap(QPixmap.fromImage(qimg).scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            if self.default_image is None:
                default_img = Image.new("RGB", (150, 150), "gray")
                draw = ImageDraw.Draw(default_img)
                draw.text((10, 65), f"Không có ảnh {label}", fill="black")
                default_img_array = np.array(default_img)
                self.default_image = QPixmap.fromImage(QImage(default_img_array.data, 150, 150, default_img_array.strides[0], QImage.Format_RGB888))
            self.sample_panel.setPixmap(self.default_image)
        self.sample_label.setText(f"{label}: {desc}\nGợi ý: {suggestion}")
        self._log(f"Hiển thị thông tin biển báo: {label}")

    def _clear_detected_signs(self):
        self.detected_labels_all.clear()
        self.label_widgets.clear()
        for i in reversed(range(self.frame_signs_layout.count())):
            widget = self.frame_signs_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

    def load_model(self):
        if YOLO is None:
            QMessageBox.critical(self, "Lỗi", "Bạn chưa cài ultralytics. Chạy: pip install ultralytics")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Chọn model (.pt)", "", "YOLO model (*.pt)")
        if not path:
            return
        try:
            self.model = YOLO(path)
            self.model_path = path
            self._log(f"Tải model: {path}")
            self.lbl_info.setText(f"Model: {os.path.basename(path)} | Video: (chưa chọn) | FPS: 0.0")
            self.button_widgets["Start"].setEnabled(True)
        except Exception as e:
            self._log(f"Lỗi khi tải model: {e}")
            QMessageBox.critical(self, "Lỗi khi tải model", str(e))

    def load_data_yaml(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn data.yaml", "", "YAML (*.yaml)")
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            self.class_names = data.get("names", [])
            self._log(f"Đã tải {len(self.class_names)} nhãn từ {path}")
            desc_path = os.path.join(os.path.dirname(path), "sign_descriptions.yaml")
            if os.path.exists(desc_path):
                with open(desc_path, 'r', encoding='utf-8') as f:
                    self.sign_desc.update(yaml.safe_load(f))
                self._log("Đã tải mô tả biển báo từ sign_descriptions.yaml")
            else:
                self._log("⚠ Không tìm thấy sign_descriptions.yaml — dùng mô tả mặc định")
        except Exception as e:
            self._log(f"Lỗi khi tải data.yaml: {e}")
            QMessageBox.critical(self, "Lỗi", str(e))

    def open_video(self):
        if self.capture:
            self.capture.release()
            self.capture = None
        self._clear_detected_signs()
        path, _ = QFileDialog.getOpenFileName(self, "Chọn video", "", "Video files (*.mp4 *.avi *.mkv)")
        if path:
            try:
                self.capture = cv2.VideoCapture(path)
                if not self.capture.isOpened():
                    raise Exception("Không thể mở video")
                self.output_video_path = path
                self._log(f"Mở video: {path}")
                self.lbl_info.setText(f"Model: {os.path.basename(self.model_path) if self.model_path else '(chưa tải)'} | Video: {os.path.basename(path)} | FPS: 0.0")
                self.button_widgets["Start"].setEnabled(True)
                self.button_widgets["Tiếp tục"].setEnabled(False)
            except Exception as e:
                self._log(f"Lỗi khi mở video: {e}")
                QMessageBox.critical(self, "Lỗi", str(e))

    def open_webcam(self):
        if self.capture:
            self.capture.release()
            self.capture = None
        self._clear_detected_signs()
        try:
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                raise Exception("Không thể mở webcam")
            self.output_video_path = ""
            self._log("Mở webcam")
            self.lbl_info.setText(f"Model: {os.path.basename(self.model_path) if self.model_path else '(chưa tải)'} | Video: webcam | FPS: 0.0")
            self.button_widgets["Start"].setEnabled(True)
            self.button_widgets["Tiếp tục"].setEnabled(False)
        except Exception as e:
            self._log(f"Lỗi khi mở webcam: {e}")
            QMessageBox.critical(self, "Lỗi", str(e))

    def start(self):
        if not self.model or not self.capture:
            QMessageBox.warning(self, "Thiếu model hoặc video", "Hãy chọn model và video trước!")
            return
        self.running = True
        self.button_widgets["Start"].setEnabled(False)
        self.button_widgets["Stop"].setEnabled(True)
        self.button_widgets["Tiếp tục"].setEnabled(False)
        self.start_time = time.time()
        self.frame_count = 0
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()
        self._log("Bắt đầu nhận diện video...")

    def stop(self):
        self.running = False
        self.button_widgets["Start"].setEnabled(True)
        self.button_widgets["Stop"].setEnabled(False)
        self.button_widgets["Tiếp tục"].setEnabled(True)
        if self.recording and self.video_writer:
            self.recording = False
            self.video_writer.release()
            self.video_writer = None
            self.button_widgets["Bắt đầu lưu video"].setText("Bắt đầu lưu video")
            self._log(f"Đã lưu video: {os.path.abspath(self.output_video_path)}")
            QMessageBox.information(self, "Thành công", f"Video đã lưu: {os.path.abspath(self.output_video_path)}")
        self._log("Đã tạm dừng video.")

    def resume(self):
        if not self.model or not self.capture or not self.capture.isOpened():
            QMessageBox.warning(self, "Cảnh báo", "Không thể tiếp tục: Model hoặc video không sẵn sàng!")
            return
        self.running = True
        self.button_widgets["Start"].setEnabled(False)
        self.button_widgets["Stop"].setEnabled(True)
        self.button_widgets["Tiếp tục"].setEnabled(False)
        self.start_time = time.time()
        self.frame_count = 0
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()
        self._log("Tiếp tục nhận diện video...")

    def snapshot(self):
        if self.processed_frame is not None:
            fn = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(fn, self.processed_frame)
            self._log(f"Đã lưu snapshot (đã xử lý): {os.path.abspath(fn)}")
            QMessageBox.information(self, "Đã lưu", f"Ảnh đã lưu: {fn}")
        else:
            QMessageBox.warning(self, "Lỗi", "Không có khung hình đã xử lý để lưu! Hãy nhấn 'Start' trước.")

    def save_detection_history(self):
        if not self.detected_labels_all:
            QMessageBox.information(self, "Thông báo", "Chưa có biển báo nào được nhận diện!")
            return
        try:
            fn = f"detection_history_{int(time.time())}.csv"
            with open(fn, "w", encoding="utf-8") as f:
                f.write("Mã biển báo,Mô tả,Gợi ý\n")
                for label in sorted(self.detected_labels_all):
                    desc = self.sign_desc.get(label, {"desc": "Không có mô tả", "suggestion": "Không có gợi ý"})["desc"]
                    suggestion = self.sign_desc.get(label, {"desc": "Không có mô tả", "suggestion": "Không có gợi ý"})["suggestion"]
                    f.write(f"{label},{desc},{suggestion}\n")
            self._log(f"Đã lưu lịch sử nhận diện: {fn}")
            QMessageBox.information(self, "Đã lưu", f"Lịch sử nhận diện đã lưu: {fn}")
        except Exception as e:
            self._log(f"Lỗi khi lưu lịch sử: {e}")
            QMessageBox.critical(self, "Lỗi", str(e))

    def toggle_record_video(self):
        if not self.running:
            QMessageBox.warning(self, "Cảnh báo", "Hãy nhấn 'Start' hoặc 'Tiếp tục' để bắt đầu nhận diện video!")
            return
        if not self.recording:
            self.output_video_path = f"output_video_{int(time.time())}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = 30
            frame_size = (600, 450)  # Điều chỉnh kích thước video ghi thành 600x450
            try:
                self.video_writer = cv2.VideoWriter(self.output_video_path, fourcc, fps, frame_size)
                if not self.video_writer.isOpened():
                    raise Exception("Không thể khởi tạo VideoWriter")
                self.recording = True
                self.button_widgets["Bắt đầu lưu video"].setText("Dừng lưu video")
                self._log(f"Bắt đầu lưu video: {os.path.abspath(self.output_video_path)}")
            except Exception as e:
                self._log(f"Lỗi khởi tạo VideoWriter: {e}")
                QMessageBox.critical(self, "Lỗi", f"Không thể bắt đầu ghi video: {e}")
                return
        else:
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                self.button_widgets["Bắt đầu lưu video"].setText("Bắt đầu lưu video")
                self._log(f"Đã lưu video: {os.path.abspath(self.output_video_path)}")
                QMessageBox.information(self, "Thành công", f"Video đã lưu: {os.path.abspath(self.output_video_path)}")
            else:
                self._log("Lỗi: Không có VideoWriter để lưu")
                QMessageBox.warning(self, "Cảnh báo", "Không có video nào đang được ghi!")

    def _video_loop(self):
        while self.running and self.capture and self.capture.isOpened():
            try:
                ret, frame = self.capture.read()
                if not ret:
                    self._log("Video kết thúc hoặc không đọc được frame.")
                    break
                frame = cv2.resize(frame, (600, 450))  # Điều chỉnh kích thước frame thành 600x450
                self.frame = frame
                self.frame_count += 1

                elapsed_time = time.time() - self.start_time
                self.fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                video_source = "webcam" if self.capture.get(cv2.CAP_PROP_POS_FRAMES) == 0 else os.path.basename(self.output_video_path) if self.output_video_path else "video"
                model_name = os.path.basename(self.model_path) if self.model_path else "(chưa tải)"
                self.lbl_info.setText(f"Model: {model_name} | Video: {video_source} | FPS: {self.fps:.1f}")

                processed_frame = frame.copy()
                detected_labels = set()
                results = self.model(frame)
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    label = self.class_names[cls] if self.class_names and cls < len(self.class_names) else str(cls)
                    detected_labels.add(label)
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(processed_frame, f"{label} ({float(box.conf[0]):.2f})", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    if label in self.alert_signs and os.path.exists("alert.wav"):
                        current_time = time.time()
                        if current_time - self.last_alert_time >= self.alert_cooldown:
                            try:
                                self.player.play()
                                self.last_alert_time = current_time
                                self._log(f"Phát âm thanh cảnh báo cho biển {label}")
                            except Exception as e:
                                self._log(f"Lỗi phát âm thanh: {e}")
                self.result_queue.put((processed_frame, results))

                if self.recording and self.video_writer and self.video_writer.isOpened():
                    self.video_writer.write(processed_frame)

                self.processed_frame = processed_frame
                time.sleep(1 / 30)
            except Exception as e:
                self._log(f"Lỗi trong vòng lặp video: {e}")
                break
        if self.recording and self.video_writer:
            self.recording = False
            self.video_writer.release()
            self.video_writer = None
            self._log(f"Đã lưu video: {os.path.abspath(self.output_video_path)}")
            QTimer.singleShot(0, lambda: QMessageBox.information(self, "Thành công", f"Video đã lưu: {os.path.abspath(self.output_video_path)}"))
        self.stop()

    def update_gui(self):
        try:
            if self.result_queue.qsize() > 1:
                while not self.result_queue.empty():
                    self.result_queue.get_nowait()
            if not self.result_queue.empty():
                processed_frame, results = self.result_queue.get_nowait()
                detected_labels = set()

                if results:
                    for box in results[0].boxes:
                        cls = int(box.cls[0])
                        label = self.class_names[cls] if self.class_names and cls < len(self.class_names) else str(cls)
                        detected_labels.add(label)

                if detected_labels:
                    last_label = list(detected_labels)[-1]
                    desc = self.sign_desc.get(last_label, {"desc": "(Chưa có mô tả)", "suggestion": "Không có gợi ý"})["desc"]
                    suggestion = self.sign_desc.get(last_label, {"desc": "(Chưa có mô tả)", "suggestion": "Không có gợi ý"})["suggestion"]
                    sample_path = os.path.join(self.samples_path, f"{last_label}.jpg")
                    if os.path.exists(sample_path):
                        img = Image.open(sample_path).resize((150, 150), Image.Resampling.LANCZOS).convert("RGB")
                        img_array = np.array(img)
                        qimg = QImage(img_array.data, img_array.shape[1], img_array.shape[0], img_array.strides[0], QImage.Format_RGB888)
                        self.sample_panel.setPixmap(QPixmap.fromImage(qimg).scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    else:
                        if self.default_image is None:
                            default_img = Image.new("RGB", (150, 150), "gray")
                            draw = ImageDraw.Draw(default_img)
                            draw.text((10, 65), "Không có ảnh", fill="black")
                            default_img_array = np.array(default_img)
                            self.default_image = QPixmap.fromImage(QImage(default_img_array.data, 150, 150, default_img_array.strides[0], QImage.Format_RGB888))
                        self.sample_panel.setPixmap(self.default_image.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    self.sample_label.setText(f"{last_label}: {desc}\nGợi ý: {suggestion}")

                canvas_width = 310
                max_columns = 3
                padding = 8
                max_widget_width = (canvas_width - padding * (max_columns + 1)) // max_columns
                for label in detected_labels:
                    if label not in self.detected_labels_all:
                        self.detected_labels_all.add(label)
                        sample_path = os.path.join(self.samples_path, f"{label}.jpg")
                        widget = QWidget()
                        widget.setStyleSheet(f"""
                            QWidget {{
                                background-color: #f7f9fb;
                                border: 1px solid #dfe6e9;
                                border-radius: 4px;
                                padding: 5px;
                                max-width: {max_widget_width}px;
                            }}
                            QWidget:hover {{
                                background-color: #e8ecef;
                            }}
                        """)
                        widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
                        layout = QVBoxLayout(widget)
                        layout.setAlignment(Qt.AlignCenter)
                        layout.setSpacing(5)
                        w = max_widget_width - 10
                        h = w
                        if os.path.exists(sample_path):
                            img = Image.open(sample_path).resize((w, h), Image.Resampling.LANCZOS).convert("RGB")
                            img_array = np.array(img)
                            qimg = QImage(img_array.data, img_array.shape[1], img_array.shape[0], img_array.strides[0], QImage.Format_RGB888)
                            lbl_img = QLabel()
                            lbl_img.setPixmap(QPixmap.fromImage(qimg))
                        else:
                            if self.default_image is None:
                                default_img = Image.new("RGB", (w, h), "gray")
                                draw = ImageDraw.Draw(default_img)
                                draw.text((10, h//2 - 10), f"Không có ảnh {label}", fill="black")
                                default_img_array = np.array(default_img)
                                self.default_image = QPixmap.fromImage(QImage(default_img_array.data, w, h, default_img_array.strides[0], QImage.Format_RGB888))
                            lbl_img = QLabel()
                            lbl_img.setPixmap(self.default_image)
                        lbl_img.setAlignment(Qt.AlignCenter)
                        lbl_img.setMaximumSize(w, h)
                        lbl_img.mousePressEvent = lambda event, lbl=label: self._on_sign_click(lbl)
                        lbl_text = QLabel(label)
                        lbl_text.setStyleSheet("""
                            font-size: 10px; 
                            color: #2c3e50; 
                            font-weight: bold;
                        """)
                        lbl_text.setAlignment(Qt.AlignCenter)
                        lbl_text.setMaximumWidth(w)
                        layout.addWidget(lbl_img)
                        layout.addWidget(lbl_text)
                        index = len(self.label_widgets)
                        row = index // max_columns
                        col = index % max_columns
                        self.frame_signs_layout.addWidget(widget, row, col)
                        self.label_widgets[label] = widget

                rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
                scaled_pixmap = QPixmap.fromImage(qimg).scaled(600, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.video_panel.setPixmap(scaled_pixmap)

        except Exception as e:
            self._log(f"Lỗi update GUI: {e}")

def main():
    print("Thư mục làm việc hiện tại:", os.getcwd())
    app = QApplication(sys.argv)
    def start_app():
        start_screen.close()
        main_window = YOLOGUI()
        main_window.show()
        def cleanup():
            main_window.stop()
            if main_window.capture:
                main_window.capture.release()
            main_window.close()
        main_window.closeEvent = lambda event: cleanup()
        return main_window
    start_screen = StartScreen(start_app)
    start_screen.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()