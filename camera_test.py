import cv2
import time
from ultralytics import YOLO

# 🧠 Tải model đã huấn luyện
MODEL_PATH = r"D:/uth-its/runs/detect/train/weights/best.pt"
model = YOLO(MODEL_PATH)

# 📸 Mở webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không mở được camera.")
    exit()

print("✅ Camera sẵn sàng. Giơ biển báo trước webcam để kiểm tra. Nhấn Q để thoát.")

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Không đọc được khung hình.")
        break

    # Thời gian hiện tại để tính FPS
    current_time = time.time()

    # 🚦 Dự đoán đối tượng trong khung hình
    results = model.predict(frame, conf=0.5, verbose=False)

    # Vẽ khung kết quả
    annotated_frame = results[0].plot()

    # 🕒 Tính FPS
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # 📊 Hiển thị FPS
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 🖼️ Hiển thị cửa sổ camera
    cv2.imshow("🚦 Traffic Sign Detection", annotated_frame)

    # ⏹️ Thoát khi nhấn phím Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
