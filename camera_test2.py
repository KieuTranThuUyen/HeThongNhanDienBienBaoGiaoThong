import cv2
import time
from ultralytics import YOLO

MODEL_PATH = r"D:/uth-its/runs/detect/train/weights/best.pt"
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không mở được camera.")
    exit()

print("✅ Camera sẵn sàng. Nhấn Q để thoát.")

prev_time = 0
frame_count = 0
predict_every_n_frames = 2  # Dự đoán 1 frame / 2 frame

annotated_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Không đọc được khung hình.")
        break

    frame_count += 1
    current_time = time.time()

    # Resize ảnh về 416x416 để nhanh hơn
    resized_frame = cv2.resize(frame, (416, 416))

    if frame_count % predict_every_n_frames == 0:
        # Dự đoán trên ảnh resized
        results = model.predict(resized_frame, conf=0.5, verbose=False, imgsz=416)

        # Vẽ khung kết quả trên ảnh resized
        annotated_frame_resized = results[0].plot()

        # Phóng to lại khung kết quả về kích thước gốc (640x480 hoặc tương tự)
        annotated_frame = cv2.resize(annotated_frame_resized, (frame.shape[1], frame.shape[0]))

    # Nếu chưa có frame nhận diện thì dùng ảnh gốc
    if annotated_frame is None:
        annotated_frame = frame.copy()

    # Tính FPS trung bình
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # Hiển thị FPS
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("🚦 Traffic Sign Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
