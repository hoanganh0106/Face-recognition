import cv2
import os
import shutil
from datetime import datetime

def record_and_process_video():
    print("\nThu thập dữ liệu cho: 'me'")
    label = "me"
    output_folder = "data/me"
    
    print(f"\nCấu hình quay video")
    duration = int(input("Thời gian quay (giây): ") or "15")
    frame_skip = 5
    
    print(f"\nBắt đầu quay video...")
    print("Nhấn 'q' để dừng sớm\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"temp_video_{label}_{timestamp}.mp4"
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Lỗi: Không thể mở camera!")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    max_frames = duration * fps
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print("Lỗi: Không thể đọc frame từ camera")
            break
        
        out.write(frame)
        
        # Hiển thị preview với countdown
        remaining = max_frames - frame_count
        seconds_left = remaining // fps
        cv2.putText(frame, f"Recording: {seconds_left}s left", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Press 'q' to stop", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Recording Video', frame)
        
        frame_count += 1
        
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            print("\nDừng quay theo yêu cầu")
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"✓ Đã quay xong video: {video_path}")
    print(f"\nTách video thành frames...")
    
    temp_folder = f"temp_frames_{label}_{timestamp}"
    os.makedirs(temp_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            filename = f"{label}_{saved_count:03d}.jpg"
            filepath = os.path.join(temp_folder, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    
    print(f"✓ Đã tách {saved_count} frames từ {frame_count} frames video")
    print(f"\nDi chuyển frames vào {output_folder}...")
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Đếm số file hiện có để đánh số tiếp theo
    existing_files = [f for f in os.listdir(output_folder) if f.startswith(label) and f.endswith('.jpg')]
    start_number = len(existing_files) + 1
    
    # Di chuyển và đổi tên
    moved_count = 0
    for filename in sorted(os.listdir(temp_folder)):
        if filename.endswith('.jpg'):
            src = os.path.join(temp_folder, filename)
            new_filename = f"{label}_{start_number + moved_count:03d}.jpg"
            dst = os.path.join(output_folder, new_filename)
            shutil.move(src, dst)
            moved_count += 1
    
    print(f"✓ Đã di chuyển {moved_count} ảnh vào {output_folder}")
    
    # Dọn dẹp
    try:
        os.remove(video_path)
        os.rmdir(temp_folder)
        print("✓ Đã xóa file tạm")
    except:
        print("⚠ Không thể xóa một số file tạm, bạn có thể xóa thủ công")
    
    # Tổng kết
    print("\n" + "="*70)
    print("✓ HOÀN THÀNH!")
    print("="*70)
    print(f"Thư mục: {output_folder}")
    print(f"Số ảnh mới: {moved_count} ảnh")
    print(f"Tổng ảnh: {start_number + moved_count - 1} ảnh")
    print("="*70)

if __name__ == "__main__":
    try:
        record_and_process_video()
    except KeyboardInterrupt:
        print("\n\nĐã hủy bởi người dùng")
    except Exception as e:
        print(f"\n\nLỗi: {e}")
