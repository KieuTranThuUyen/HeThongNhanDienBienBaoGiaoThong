import threading
import time
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import yaml
import queue

# YOLOv8
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
    print("⚠ Bạn chưa cài 'ultralytics'. Chạy: pip install ultralytics")


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
        self.model_path = ""
        self.samples_path = "samples"  # Thư mục chứa ảnh mẫu biển báo
        self.result_queue = queue.Queue()
        self.default_image = None  # Hình ảnh mặc định khi không tìm thấy file

        # Danh sách mô tả mặc định
        self.sign_desc = {
            "DP.135": "Hết tất cả các lệnh cấm",
            "P.102": "Cấm đi ngược chiều",
            "P.103a": "Cấm xe ô tô",
            "P.103b": "Cấm xe ô tô rẽ phải",
            "P.103c": "Cấm xe ô tô rẽ trái",
            "P.104": "Cấm xe máy",
            "P.106a": "Cấm xe ô tô tải",
            "P.106b": "Cấm ô tô tải có khối lượng chuyên chở lớn hơn giới hạn",
            "P.107a": "Cấm xe ô tô khách",
            "P.112": "Cấm người đi bộ",
            "P.115": "Hạn chế trọng tải toàn bộ xe",
            "P.117": "Hạn chế chiều cao xe",
            "P.123a": "Cấm rẽ trái",
            "P.123b": "Cấm rẽ phải",
            "P.124a": "Cấm quay đầu xe",
            "P.124b": "Cấm ô tô quay đầu xe",
            "P.124c": "Cấm rẽ trái và quay đầu xe",
            "P.125": "Cấm vượt",
            "P.127": "Tốc độ tối đa cho phép",
            "P.128": "Cấm sử dụng còi",
            "P.130": "Cấm dừng xe và đỗ xe",
            "P.131a": "Cấm đỗ xe",
            "P.137": "Cấm rẽ trái và rẽ phải",
            "R.301c": "Các xe chỉ được rẽ trái",
            "R.301d": "Các xe chỉ được rẽ phải",
            "R.301e": "Các xe chỉ được rẽ trái",
            "R.302a": "Phải đi vòng sang bên phải",
            "R.302b": "Phải đi vòng sang bên trái",
            "R.303": "Nơi giao nhau chạy theo vòng xuyến",
            "R.407a": "Đường một chiều",
            "R.409": "Chỗ quay xe",
            "R.425": "Bệnh viện",
            "R.434": "Bến xe",
            "S.509a": "Chiều cao an toàn",
            "W.201a": "Chỗ ngoặt nguy hiểm vòng bên trái",
            "W.201b": "Chỗ ngoặt nguy hiểm vòng bên phải",
            "W.202a": "Nhiều chỗ ngoặt nguy hiểm liên tiếp, chỗ đầu tiên sang trái",
            "W.202b": "Nhiều chỗ ngoặt nguy hiểm liên tiếp, chỗ đầu tiên sang phải",
            "W.203b": "Đường bị thu hẹp về phía trái",
            "W.203c": "Đường bị thu hẹp về phía phải",
            "W.205a": "Đường giao nhau (ngã tư)",
            "W.205b": "Đường giao nhau (ngã ba bên trái)",
            "W.205d": "Đường giao nhau (hình chữ T)",
            "W.207a": "Giao nhau với đường không ưu tiên",
            "W.207b": "Giao nhau với đường không ưu tiên",
            "W.207c": "Giao nhau với đường không ưu tiên",
            "W.208": "Giao nhau với đường ưu tiên",
            "W.209": "Giao nhau có tín hiệu đèn giao thông",
            "W.210": "Giao nhau với đường sắt có rào chắn",
            "W.219": "Dốc xuống nguy hiểm",
            "W.221b": "Đường có gồ giảm tốc",
            "W.224": "Đường người đi bộ cắt ngang",
            "W.225": "Trẻ em",
            "W.227": "Công trường",
            "W.233": "Nguy hiểm khác",
            "W.235": "Đường đôi",
            "W.245a": "Đi chậm"
        }

        # Lưu tất cả biển báo đã nhận diện
        self.detected_labels_all = set()
        self.label_widgets = {}  # label: (image_label, text_label)

        self._build_ui()
        self.root.after(30, self.update_gui)  # chạy update GUI định kỳ

    # ========================= GIAO DIỆN ========================= #
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
        tk.Button(frm_top, text="Snapshot", command=self.snapshot).pack(side=tk.LEFT, padx=4)

        self.lbl_info = tk.Label(self.root, text="Model: (chưa tải) | Video: (chưa chọn) | FPS: 0.0")
        self.lbl_info.pack(side=tk.TOP, anchor="w", padx=6)

        frm_main = tk.Frame(self.root)
        frm_main.pack(fill=tk.BOTH, expand=True)

        # Vùng video
        self.video_panel = tk.Label(frm_main, bg="black")
        self.video_panel.pack(side=tk.LEFT, padx=6, pady=6, fill=tk.BOTH, expand=True)

        # Vùng thông tin bên phải
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
        canvas = tk.Canvas(frm_right, bg="white", width=330, height=250)
        scrollbar = tk.Scrollbar(frm_right, orient="vertical", command=canvas.yview)
        self.frame_signs = tk.Frame(canvas, bg="white")
        self.frame_signs.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.frame_signs, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.txt_log = tk.Text(self.root, height=5)
        self.txt_log.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=6)
        self._log("Ứng dụng sẵn sàng.")

    # ========================= HÀM PHỤ ========================= #
    def _log(self, msg):
        t = time.strftime("%H:%M:%S")
        self.txt_log.insert(tk.END, f"[{t}] {msg}\n")
        self.txt_log.see(tk.END)

    # Hàm xử lý khi nhấn vào biển báo
    def _on_sign_click(self, label):
        desc = self.sign_desc.get(label, "(Chưa có mô tả)")
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
        self.sample_label.config(text=f"{label}: {desc}")
        self._log(f"Hiển thị thông tin biển báo: {label}")

    # Hàm xóa các biển báo đã nhận diện
    def _clear_detected_signs(self):
        self.detected_labels_all.clear()
        self.label_widgets.clear()
        for widget in self.frame_signs.winfo_children():
            widget.destroy()

    # ========================= NÚT CHỨC NĂNG ========================= #
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
            self.lbl_info.config(text=f"Model: {os.path.basename(path)} | Video: (chưa chọn)")
            self.btn_start.config(state=tk.NORMAL)
        except Exception as e:
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
            messagebox.showerror("Lỗi", str(e))

    def open_video(self):
        self._clear_detected_signs()  # Xóa các biển báo cũ khi mở video mới
        path = filedialog.askopenfilename(title="Chọn video", filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
        if path:
            self.capture = cv2.VideoCapture(path)
            self._log(f"Mở video: {path}")
            self.lbl_info.config(text=f"Model: {os.path.basename(self.model_path)} | Video: {os.path.basename(path)}")

    def open_webcam(self):
        self._clear_detected_signs()  # Xóa các biển báo cũ khi mở webcam mới
        self.capture = cv2.VideoCapture(0)
        self._log("Mở webcam")
        self.lbl_info.config(text=f"Model: {os.path.basename(self.model_path)} | Video: webcam")

    def start(self):
        if not self.model or not self.capture:
            messagebox.showwarning("Thiếu model hoặc video", "Hãy chọn model và video trước!")
            return
        self.running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()
        self._log("Bắt đầu nhận diện video...")

    def stop(self):
        self.running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self._log("Đã dừng video.")

    def snapshot(self):
        if self.frame is not None:
            fn = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(fn, self.frame)
            self._log(f"Đã lưu snapshot: {fn}")
            messagebox.showinfo("Đã lưu", f"Ảnh đã lưu: {fn}")

    # ========================= VÒNG LẶP VIDEO ========================= #
    def _video_loop(self):
        while self.running and self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                self._log("Video kết thúc hoặc không đọc được frame.")
                break

            frame = cv2.resize(frame, (800, 600))

            # Chỉ xử lý YOLO 1 frame mỗi 3 frame để giảm tải
            if hasattr(self, "frame_count"):
                self.frame_count += 1
            else:
                self.frame_count = 0
            if self.frame_count % 3 == 0:
                results = self.model(frame)
                self.result_queue.put((frame.copy(), results))

            # Lưu frame để snapshot
            self.frame = frame

            time.sleep(1 / 30)

        self.stop()

    # ========================= CẬP NHẬT GUI AN TOÀN ========================= #
    def update_gui(self):
        try:
            while not self.result_queue.empty():
                frame, results = self.result_queue.get_nowait()
                r = results[0]
                detected_labels = set()

                for box in r.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = self.class_names[cls] if self.class_names and cls < len(self.class_names) else str(cls)
                    detected_labels.add(label)

                    x1, y1, x2, y2 = xyxy
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Biển báo gần nhất
                if detected_labels:
                    last_label = list(detected_labels)[-1]
                    desc = self.sign_desc.get(last_label, "(Chưa có mô tả)")
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
                    self.sample_label.config(text=f"{last_label}: {desc}")

                # Cập nhật tất cả biển báo
                canvas_width = 330
                max_columns = 4
                padding = 4

                # Sắp xếp lại lưới để tránh khoảng trống
                current_widgets = len(self.frame_signs.winfo_children()) // 2  # Số cặp (hình ảnh + nhãn)
                for label in detected_labels:
                    if label not in self.detected_labels_all:
                        self.detected_labels_all.add(label)
                        sample_path = os.path.join(self.samples_path, f"{label}.jpg")
                        if os.path.exists(sample_path):
                            img = Image.open(sample_path)
                            w = (canvas_width // max_columns) - 2 * padding
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
                        # Thêm sự kiện click vào hình ảnh biển báo
                        lbl_img.bind("<Button-1>", lambda event, lbl=label: self._on_sign_click(lbl))
                        lbl_text = tk.Label(self.frame_signs, text=label, bg="#f5f5f5", font=("Arial", 9))
                        index = current_widgets
                        row = (index // max_columns) * 2
                        col = index % max_columns
                        lbl_img.grid(row=row, column=col, padx=padding, pady=padding)
                        lbl_text.grid(row=row + 1, column=col)
                        self.label_widgets[label] = (lbl_img, lbl_text)
                        current_widgets += 1

                # Hiển thị video
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
                self.video_panel.imgtk = imgtk
                self.video_panel.config(image=imgtk)

        except Exception as e:
            self._log(f"Lỗi update GUI: {e}")

        self.root.after(30, self.update_gui)


def main():
    root = tk.Tk()
    app = YOLOGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()