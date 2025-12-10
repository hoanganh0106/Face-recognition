# Face Recognition Training Pipeline (TensorFlow / Keras)

Dự án nghiên cứu & huấn luyện mô hình nhận diện khuôn mặt tự động bằng Transfer Learning (MobileNetV2) trên Python.

## Cấu trúc thư mục
- `thuthapdata.py`: Thu thập ảnh khuôn mặt từ webcam, tự động crop qua OpenCV DNN SSD Face Detector.
- `train_model.py`: Pipeline huấn luyện MobileNetV2 2 giai đoạn (Transfer Learning + Fine-tuning 20 layers).
- `realtime.py`: Chạy nhận diện khuôn mặt thời gian thực qua webcam.
- `deploy.prototxt`: Định nghĩa kiến trúc mạng SSD Face Detector.
- `requirements.txt`: Các thư viện phụ thuộc.

## Hướng dẫn sử dụng
1. Tạo môi trường ảo và cài đặt dependencies:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Thu thập dữ liệu:
   ```bash
   python thuthapdata.py
   ```

3. Huấn luyện mô hình:
   ```bash
   python train_model.py
   ```

4. Chạy nhận diện thời gian thực:
   ```bash
   python realtime.py
   ```
