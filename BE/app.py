import cv2
from ultralytics import YOLO

# --- TẢI MODEL MỚI CỦA BẠN ---
# Chắc chắn rằng file 'best.pt' mới nhất nằm cùng thư mục
try:
    model = YOLO('best.pt')
except Exception as e:
    print(f"Lỗi khi tải model: {e}")
    exit()

# --- MỞ WEBCAM ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể mở webcam.")
    exit()

# --- VÒNG LẶP XỬ LÝ ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- ĐƯA KHUNG HÌNH VÀO MODEL ĐỂ NHẬN DIỆN ---
    results = model(frame)

    # --- TỰ ĐỘNG VẼ KẾT QUẢ LÊN KHUNG HÌNH ---
    # Hàm plot() sẽ tự động vẽ bounding box và tên class chi tiết
    annotated_frame = results[0].plot()

    # Hiển thị khung hình kết quả
    cv2.imshow('Nhan Dien Bien Bao Chi Tiet - An Q de thoat', annotated_frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- GIẢI PHÓNG TÀI NGUYÊN ---
cap.release()
cv2.destroyAllWindows()