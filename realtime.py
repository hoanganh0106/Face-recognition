import cv2
import numpy as np
import serial
import time
import tensorflow as tf

ARDUINO_PORT = 'COM4'
# Ngưỡng để nhận diện "me"
THRESHOLD = 0.1  # Score < 0.1 = "me"
MIN_CONFIDENCE = 95  # Độ tin cậy tối thiểu (%)
CONSECUTIVE_FRAMES = 5  # Số frame liên tiếp phải nhận diện đúng
SKIP_FRAMES = 2  # Chỉ xử lý 1 frame, bỏ qua 2 frames (tăng FPS)

print("[INFO] Dang nap mo hinh phat hien khuon mat...")
prototxtPath = "deploy.prototxt"
weightsPath = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
face_net = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] Dang nap mo hinh nhan dien khuon mat...")
recognizer_model = tf.keras.models.load_model(
    'face_recognizer.h5',
    safe_mode=False
)

try:
    ser = serial.Serial(ARDUINO_PORT, 9600, timeout=1)
    print(f"[SUCCESS] Da ket noi voi Arduino qua cong {ARDUINO_PORT}")
    time.sleep(2)
except Exception as e:
    print(f"[ERROR] Khong the ket noi voi Arduino: {e}")
    ser = None

print("[INFO] Dang khoi dong camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Khong the mo camera!")
    exit()
time.sleep(2.0)

print("[INFO] Bat dau nhan dien khuon mat. Nhan 'q' de thoat...")
print(f"[CONFIG] Nguong nhan dien: {THRESHOLD}, Do tin cay toi thieu: {MIN_CONFIDENCE}%")

# Bộ đếm frame liên tiếp để xác nhận
me_frame_count = 0
frame_count = 0  # Đếm frame để skip
found_me = False
last_detections = []  # Lưu kết quả detection cuối cùng

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Khong doc duoc frame tu camera!")
        break

    frame = cv2.flip(frame, 1)

    frame_count += 1
    (h, w) = frame.shape[:2]
    
# Chỉ xử lý face detection và recognition mỗi SKIP_FRAMES frame
    if frame_count % (SKIP_FRAMES + 1) == 0:
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        temp_found_me = False  # Reset tìm thấy trong frame này
        last_detections = []  # Lưu kết quả để vẽ lại các frame sau

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                face_roi = frame[startY:endY, startX:endX]
                if face_roi.size == 0:
                    continue
                
                try:
                    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    resized_face = cv2.resize(face_rgb, (224, 224))
                    # Chuẩn hóa giống như lúc training
                    normalized_face = (resized_face / 127.5) - 1
                    face_batch = np.expand_dims(normalized_face, axis=0)
                    
                    prediction = recognizer_model.predict(face_batch, verbose=0)
                    score = prediction[0][0]

                    # Score < THRESHOLD = "me", yêu cầu độ tin cậy cao
                    if score < THRESHOLD:
                        confidence_pct = (1 - score) * 100  # % tin cậy cho "Me"
                        if confidence_pct >= MIN_CONFIDENCE:
                            label = f"Me! ({confidence_pct:.1f}%)"
                            color = (0, 255, 0)
                            temp_found_me = True
                        else:
                            continue  # Không hiển thị nếu không đủ tin cậy
                    else:
                        confidence_pct = score * 100  # % tin cậy cho "Not Me"
                        label = f"Not Me ({confidence_pct:.1f}%)"
                        color = (0, 0, 255)

                    # Lưu kết quả để vẽ lại các frame sau
                    last_detections.append((startX, startY, endX, endY, label, color))

                except Exception as e:
                    print(f"[ERROR] Loi khi xu ly khuon mat: {e}")
                    continue

        # Cơ chế xác nhận bằng nhiều frame liên tiếp
        if temp_found_me:
            me_frame_count += 1
            if me_frame_count >= CONSECUTIVE_FRAMES:
                found_me = True
        else:
            me_frame_count = 0  # Reset nếu không tìm thấy
            found_me = False
    
    # Vẽ lại kết quả detection cuối cùng (mọi frame)
    for (startX, startY, endX, endY, label, color) in last_detections:
        cv2.putText(frame, label, (startX, startY - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    if ser:
        if found_me:
            ser.write(b'1')
        else:
            ser.write(b'0')

    cv2.imshow("He thong mo khoa bang khuon mat", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("[INFO] Dang dong ung dung...")
if ser:
    ser.write(b'0')
    ser.close()
    print("[INFO] Da dong ket noi Arduino")

cap.release()
cv2.destroyAllWindows()
print("[INFO] Hoan thanh!")