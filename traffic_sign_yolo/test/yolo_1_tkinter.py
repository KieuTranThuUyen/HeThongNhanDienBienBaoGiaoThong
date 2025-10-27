import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")
import threading
import time
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import yaml
import queue

# Thêm pygame để hỗ trợ âm thanh
try:
    from pygame import mixer
except ImportError:
    mixer = None
    print("⚠ Bạn chưa cài 'pygame'. Chạy: pip install pygame để hỗ trợ cảnh báo âm thanh")

# YOLOv8
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
    print("⚠ Bạn chưa cài 'ultralytics'. Chạy: pip install ultralytics")

class StartScreen:
    def __init__(self, root, on_start):
        self.root = root
        self.root.title("Chào Mừng - Nhận Diện Biển Báo Giao Thông")
        self.root.geometry("500x300")
        self.root.resizable(False, False)
        self.root.eval('tk::PlaceWindow . center')

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(expand=True, fill="both", padx=20, pady=20)

        tk.Label(
            self.main_frame,
            text="Nhận Diện Biển Báo Giao Thông",
            font=("Arial", 18, "bold")
        ).pack(pady=20)

        tk.Label(
            self.main_frame,
            text="Ứng dụng sử dụng YOLOv8 để nhận diện biển báo giao thông từ video hoặc webcam.",
            font=("Arial", 10),
            wraplength=400,
            justify="center"
        ).pack(pady=10)

        tk.Button(
            self.main_frame,
            text="Bắt Đầu",
            command=on_start,
            font=("Arial", 12),
            padx=20,
            pady=10
        ).pack(pady=20)

class YOLOGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận diện biển báo giao thông - YOLOv8")
        self.root.geometry("1300x850")

        # Biến trạng thái
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

        # Danh sách mô tả và gợi ý mặc định
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

        self._build_ui()
        self.root.after(30, self.update_gui)

    def _build_ui(self):
        frm_top = tk.Frame(self.root)
        frm_top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        tk.Button(frm_top, text="Load model", command=self.load_model).pack(side=tk.LEFT, padx=4)
        tk.Button(frm_top, text="Load data.yaml", command=self.load_data_yaml).pack(side=tk.LEFT, padx=4)
        tk.Button(frm_top, text="Chọn video", command=self.open_video).pack(side=tk.LEFT, padx=4)
        tk.Button(frm_top, text="Mở webcam", command=self.open_webcam).pack(side=tk.LEFT, padx=4)
        self.btn_start = tk.Button(frm_top, text="Start", state=tk.DISABLED, command=self.start)
        self.btn_start.pack(side=tk.LEFT, padx=4)
        self.btn_stop = tk.Button(frm_top, text="Stop", state=tk.DISABLED, command=self.stop)
        self.btn_stop.pack(side=tk.LEFT, padx=4)
        self.btn_resume = tk.Button(frm_top, text="Tiếp tục", state=tk.DISABLED, command=self.resume)
        self.btn_resume.pack(side=tk.LEFT, padx=4)
        tk.Button(frm_top, text="Snapshot", command=self.snapshot).pack(side=tk.LEFT, padx=4)
        tk.Button(frm_top, text="Lưu lịch sử", command=self.save_detection_history).pack(side=tk.LEFT, padx=4)
        self.btn_record = tk.Button(frm_top, text="Bắt đầu lưu video", command=self.toggle_record_video)
        self.btn_record.pack(side=tk.LEFT, padx=4)

        self.lbl_info = tk.Label(self.root, text="Model: (chưa tải) | Video: (chưa chọn) | FPS: 0.0")
        self.lbl_info.pack(side=tk.TOP, anchor="w", padx=6)

        frm_main = tk.Frame(self.root)
        frm_main.pack(fill=tk.BOTH, expand=True)

        self.video_panel = tk.Label(frm_main, bg="black")
        self.video_panel.pack(side=tk.LEFT, padx=6, pady=6, fill=tk.BOTH, expand=True)

        frm_right = tk.Frame(frm_main, width=350, bg="white")
        frm_right.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        frm_right.pack_propagate(False)

        tk.Label(frm_right, text="Biển báo nhận diện gần nhất", font=("Arial", 14, "bold"), bg="white").pack(pady=10)
        self.sample_panel = tk.Label(frm_right, bg="white")
        self.sample_panel.pack(pady=10)
        self.sample_label = tk.Label(frm_right, text="(Chưa nhận diện)", font=("Arial", 12),
                                     wraplength=300, justify="center", bg="white")
        self.sample_label.pack(pady=10)

        tk.Label(frm_right, text="Các biển báo được nhận diện", bg="white", font=("Arial", 12, "bold")).pack(pady=(10, 4))
        canvas_width = 310
        self.canvas = tk.Canvas(frm_right, bg="white", width=canvas_width, height=250)
        scrollbar = tk.Scrollbar(frm_right, orient="vertical", command=self.canvas.yview)
        self.frame_signs = tk.Frame(self.canvas, bg="white")
        self.frame_signs.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.frame_signs, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.txt_log = tk.Text(self.root, height=5)
        self.txt_log.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=6)
        self._log("Ứng dụng sẵn sàng.")

    def _log(self, msg):
        t = time.strftime("%H:%M:%S")
        self.txt_log.insert(tk.END, f"[{t}] {msg}\n")
        self.txt_log.see(tk.END)

    def _on_sign_click(self, label):
        desc = self.sign_desc.get(label, {"desc": "(Chưa có mô tả)", "suggestion": "Không có gợi ý"})["desc"]
        suggestion = self.sign_desc.get(label, {"desc": "(Chưa có mô tả)", "suggestion": "Không có gợi ý"})["suggestion"]
        sample_path = os.path.join(self.samples_path, f"{label}.jpg")
        if os.path.exists(sample_path):
            img = Image.open(sample_path).resize((200, 200))
            tkimg = ImageTk.PhotoImage(img)
            self.sample_panel.imgtk = tkimg
            self.sample_panel.config(image=tkimg)
        else:
            if self.default_image is None:
                from PIL import ImageDraw
                default_img = Image.new("RGB", (200, 200), "gray")
                draw = ImageDraw.Draw(default_img)
                draw.text((10, 90), f"Không có ảnh {label}", fill="black")
                self.default_image = ImageTk.PhotoImage(default_img)
            self.sample_panel.config(image=self.default_image)
        self.sample_label.config(text=f"{label}: {desc}\nGợi ý: {suggestion}")
        self._log(f"Hiển thị thông tin biển báo: {label}")

    def _clear_detected_signs(self):
        self.detected_labels_all.clear()
        self.label_widgets.clear()
        for widget in self.frame_signs.winfo_children():
            widget.destroy()

    def load_model(self):
        if YOLO is None:
            messagebox.showerror("Lỗi", "Bạn chưa cài ultralytics. Chạy: pip install ultralytics")
            return
        path = filedialog.askopenfilename(title="Chọn model (.pt)", filetypes=[("YOLO model", "*.pt")])
        if not path:
            return
        try:
            self.model = YOLO(path)
            self.model_path = path
            self._log(f"Tải model: {path}")
            self.lbl_info.config(text=f"Model: {os.path.basename(path)} | Video: (chưa chọn) | FPS: 0.0")
            self.btn_start.config(state=tk.NORMAL)
        except Exception as e:
            self._log(f"Lỗi khi tải model: {e}")
            messagebox.showerror("Lỗi khi tải model", str(e))

    def load_data_yaml(self):
        path = filedialog.askopenfilename(title="Chọn data.yaml", filetypes=[("YAML", "*.yaml")])
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
            messagebox.showerror("Lỗi", str(e))

    def open_video(self):
        if self.capture:
            self.capture.release()
            self.capture = None
        self._clear_detected_signs()
        path = filedialog.askopenfilename(title="Chọn video", filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
        if path:
            try:
                self.capture = cv2.VideoCapture(path)
                if not self.capture.isOpened():
                    raise Exception("Không thể mở video")
                self._log(f"Mở video: {path}")
                self.lbl_info.config(text=f"Model: {os.path.basename(self.model_path)} | Video: {os.path.basename(path)} | FPS: 0.0")
                self.btn_start.config(state=tk.NORMAL)
                self.btn_resume.config(state=tk.DISABLED)
            except Exception as e:
                self._log(f"Lỗi khi mở video: {e}")
                messagebox.showerror("Lỗi", str(e))

    def open_webcam(self):
        if self.capture:
            self.capture.release()
            self.capture = None
        self._clear_detected_signs()
        try:
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                raise Exception("Không thể mở webcam")
            self._log("Mở webcam")
            self.lbl_info.config(text=f"Model: {os.path.basename(self.model_path)} | Video: webcam | FPS: 0.0")
            self.btn_start.config(state=tk.NORMAL)
            self.btn_resume.config(state=tk.DISABLED)
        except Exception as e:
            self._log(f"Lỗi khi mở webcam: {e}")
            messagebox.showerror("Lỗi", str(e))

    def start(self):
        if not self.model or not self.capture:
            messagebox.showwarning("Thiếu model hoặc video", "Hãy chọn model và video trước!")
            return
        self.running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_resume.config(state=tk.DISABLED)
        self.start_time = time.time()
        self.frame_count = 0
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()
        self._log("Bắt đầu nhận diện video...")
        if mixer:
            mixer.init()

    def stop(self):
        self.running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_resume.config(state=tk.NORMAL)
        if self.recording and self.video_writer:
            self.recording = False
            self.video_writer.release()
            self.video_writer = None
            self.btn_record.config(text="Bắt đầu lưu video")
            self._log(f"Đã lưu video: {os.path.abspath(self.output_video_path)}")
            messagebox.showinfo("Thành công", f"Video đã lưu: {os.path.abspath(self.output_video_path)}")
        self._log("Đã tạm dừng video.")
        if mixer:
            mixer.quit()

    def resume(self):
        if not self.model or not self.capture or not self.capture.isOpened():
            messagebox.showwarning("Cảnh báo", "Không thể tiếp tục: Model hoặc video không sẵn sàng!")
            return
        self.running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_resume.config(state=tk.DISABLED)
        self.start_time = time.time()  # Reset thời gian để tính FPS chính xác
        self.frame_count = 0
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()
        self._log("Tiếp tục nhận diện video...")
        if mixer:
            mixer.init()

    def snapshot(self):
        if self.processed_frame is not None:
            fn = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(fn, self.processed_frame)
            self._log(f"Đã lưu snapshot (đã xử lý): {os.path.abspath(fn)}")
            messagebox.showinfo("Đã lưu", f"Ảnh đã lưu: {os.path.abspath(fn)}")
        else:
            messagebox.showwarning("Lỗi", "Không có khung hình đã xử lý để lưu! Hãy nhấn 'Start' trước.")

    def save_detection_history(self):
        if not self.detected_labels_all:
            messagebox.showinfo("Thông báo", "Chưa có biển báo nào được nhận diện!")
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
            messagebox.showinfo("Đã lưu", f"Lịch sử nhận diện đã lưu: {fn}")
        except Exception as e:
            self._log(f"Lỗi khi lưu lịch sử: {e}")
            messagebox.showerror("Lỗi", str(e))

    def toggle_record_video(self):
        if not self.running:
            messagebox.showwarning("Cảnh báo", "Hãy nhấn 'Start' hoặc 'Tiếp tục' để bắt đầu nhận diện video!")
            return
        if not self.recording:
            self.output_video_path = f"output_video_{int(time.time())}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = 30
            frame_size = (800, 600)
            try:
                self.video_writer = cv2.VideoWriter(self.output_video_path, fourcc, fps, frame_size)
                if not self.video_writer.isOpened():
                    raise Exception("Không thể khởi tạo VideoWriter")
                self.recording = True
                self.btn_record.config(text="Dừng lưu video")
                self._log(f"Bắt đầu lưu video: {os.path.abspath(self.output_video_path)}")
            except Exception as e:
                self._log(f"Lỗi khởi tạo VideoWriter: {e}")
                messagebox.showerror("Lỗi", f"Không thể bắt đầu ghi video: {e}")
                return
        else:
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                self.btn_record.config(text="Bắt đầu lưu video")
                self._log(f"Đã lưu video: {os.path.abspath(self.output_video_path)}")
                messagebox.showinfo("Thành công", f"Video đã lưu: {os.path.abspath(self.output_video_path)}")
            else:
                self._log("Lỗi: Không có VideoWriter để lưu")
                messagebox.showwarning("Cảnh báo", "Không có video nào đang được ghi!")

    def _video_loop(self):
        while self.running and self.capture and self.capture.isOpened():
            try:
                ret, frame = self.capture.read()
                if not ret:
                    self._log("Video kết thúc hoặc không đọc được frame.")
                    break
                frame = cv2.resize(frame, (800, 600))
                self.frame = frame
                self.frame_count += 1

                elapsed_time = time.time() - self.start_time
                self.fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                video_source = "webcam" if self.capture.get(cv2.CAP_PROP_POS_FRAMES) == 0 else os.path.basename(self.output_video_path or self.model_path)
                self.lbl_info.config(text=f"Model: {os.path.basename(self.model_path)} | Video: {video_source} | FPS: {self.fps:.1f}")

                processed_frame = frame.copy()
                detected_labels = set()
                if self.frame_count % 3 == 0:
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
                        if mixer and label in self.alert_signs:
                            try:
                                mixer.Sound("alert.wav").play()
                            except Exception as e:
                                self._log(f"Lỗi phát âm thanh: {e}")
                    self.result_queue.put((processed_frame, results))
                else:
                    self.result_queue.put((processed_frame, None))

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
            self.root.after(0, lambda: messagebox.showinfo("Thành công", f"Video đã lưu: {os.path.abspath(self.output_video_path)}"))
        self.stop()

    def update_gui(self):
        try:
            while not self.result_queue.empty():
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
                        img = Image.open(sample_path).resize((200, 200))
                        tkimg = ImageTk.PhotoImage(img)
                        self.sample_panel.imgtk = tkimg
                        self.sample_panel.config(image=tkimg)
                    else:
                        if self.default_image is None:
                            from PIL import ImageDraw
                            default_img = Image.new("RGB", (200, 200), "gray")
                            draw = ImageDraw.Draw(default_img)
                            draw.text((10, 90), "Không có ảnh", fill="black")
                            self.default_image = ImageTk.PhotoImage(default_img)
                        self.sample_panel.config(image=self.default_image)
                    self.sample_label.config(text=f"{last_label}: {desc}\nGợi ý: {suggestion}")

                canvas_width = 310
                max_columns = 4
                padding = 4
                current_widgets = len(self.frame_signs.winfo_children()) // 2
                for label in detected_labels:
                    if label not in self.detected_labels_all:
                        self.detected_labels_all.add(label)
                        sample_path = os.path.join(self.samples_path, f"{label}.jpg")
                        if os.path.exists(sample_path):
                            img = Image.open(sample_path)
                            w = (canvas_width - padding * (max_columns + 1)) // max_columns
                            h = int(w * img.height / img.width)
                            img = img.resize((w, h))
                            tkimg = ImageTk.PhotoImage(img)
                            lbl_img = tk.Label(self.frame_signs, image=tkimg, bg="#f5f5f5")
                            lbl_img.image = tkimg
                        else:
                            if self.default_image is None:
                                from PIL import ImageDraw
                                default_img = Image.new("RGB", (200, 200), "gray")
                                draw = ImageDraw.Draw(default_img)
                                draw.text((10, 90), f"Không có ảnh {label}", fill="black")
                                self.default_image = ImageTk.PhotoImage(default_img.resize((w, h)))
                            tkimg = self.default_image
                            lbl_img = tk.Label(self.frame_signs, image=tkimg, bg="#f5f5f5")
                            lbl_img.image = tkimg
                        lbl_img.bind("<Button-1>", lambda event, lbl=label: self._on_sign_click(lbl))
                        lbl_text = tk.Label(self.frame_signs, text=label, bg="#f5f5f5", font=("Arial", 9))
                        index = current_widgets
                        row = (index // max_columns) * 2
                        col = index % max_columns
                        lbl_img.grid(row=row, column=col, padx=padding, pady=padding)
                        lbl_text.grid(row=row + 1, column=col)
                        self.label_widgets[label] = (lbl_img, lbl_text)
                        current_widgets += 1

                rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
                self.video_panel.imgtk = imgtk
                self.video_panel.config(image=imgtk)

        except Exception as e:
            self._log(f"Lỗi update GUI: {e}")

        self.root.after(30, self.update_gui)

def main():
    print("Thư mục làm việc hiện tại:", os.getcwd())
    root = tk.Tk()
    def start_app():
        root.destroy()
        main_root = tk.Tk()
        app = YOLOGUI(main_root)
        main_root.protocol("WM_DELETE_WINDOW", lambda: (app.stop(), app.capture.release() if app.capture else None, main_root.destroy()))
        main_root.mainloop()

    start_screen = StartScreen(root, start_app)
    root.mainloop()

if __name__ == "__main__":
    main()