# Hệ Thống Nhận Diện Biển Báo Giao Thông

Ứng dụng desktop (PySide6/Qt) sử dụng YOLOv8 (Ultralytics) và OpenCV để nhận diện biển báo giao thông từ video hoặc webcam, hiển thị thông tin, phát cảnh báo âm thanh và cho phép lưu ảnh/video, lịch sử nhận diện.

## Công nghệ sử dụng
- **Ngôn ngữ**: Python 3.10+
- **Nhận diện**: Ultralytics YOLOv8 (`ultralytics`), PyTorch (`torch`, `torchvision`)
- **Xử lý ảnh/video**: OpenCV (`opencv-python`), Pillow (`Pillow`)
- **Giao diện**: PySide6 (`PySide6`)


Thư mục chính:
- `TrafficSignYolo/main.py`: Ứng dụng GUI chính.
- `TrafficSignYolo/best.pt`: Model YOLOv8 đã huấn luyện.
- `TrafficSignYolo/data.yaml`: File cấu hình lớp/nhãn.
- `TrafficSignYolo/sign_descriptions.yaml`: Mô tả và gợi ý cho từng biển báo.
- `TrafficSignYolo/samples/`: Ảnh mẫu minh họa cho các biển báo.
- `TrafficSignYolo/alert.wav`: Âm thanh cảnh báo (tùy chọn).
- `TrafficSignYolo/traffic_video.mp4`: Video mẫu.

## Cài đặt (Windows)
1) Cài Python 3.10+ từ trang chủ.
2) Mở terminal git clone : https://github.com/KieuTranThuUyen/HeThongNhanDienBienBaoGiaoThong.git
3) pip install --upgrade pip
   pip install -r requirements.txt
```
## Chạy ứng dụng
(+) cd TrafficSignYolo
(+) python main.py
```
Trong ứng dụng:
- Nhấn "Load model" và chọn `TrafficSignYolo/best.pt` (hoặc `.pt` khác).
- Nhấn "Load data.yaml" và chọn `TrafficSignYolo/data.yaml` (ứng dụng tự tải `sign_descriptions.yaml` nếu có).
- Chọn nguồn video: "Chọn video" (có thể dùng `TrafficSignYolo/traffic_video.mp4`) hoặc "Mở webcam".
- Nhấn "Start" để bắt đầu nhận diện, "Stop" để tạm dừng, "Tiếp tục" để tiếp tục.
- "Snapshot" lưu ảnh khung hình đã xử lý; "Bắt đầu lưu video" để ghi video kết quả ra file `.mp4`.
- "Lưu lịch sử" xuất danh sách biển báo đã nhận diện ra file `.csv`.


