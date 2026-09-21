# FER-CNN: Real-time Facial Emotion Recognition

Hệ thống nhận diện cảm xúc khuôn mặt theo thời gian thực bằng webcam, sử dụng CNN tùy biến với cơ chế attention và TensorFlow/Keras.

## 1. Giới thiệu

FER-CNN là project nghiên cứu và thử nghiệm facial emotion recognition. Ứng dụng phát hiện khuôn mặt từ webcam, cắt vùng khuôn mặt và phân loại thành bảy cảm xúc:

| Nhãn | Ý nghĩa |
| --- | --- |
| `Angry` | Tức giận |
| `Disgust` | Ghê tởm |
| `Fear` | Sợ hãi |
| `Happy` | Vui vẻ |
| `Sad` | Buồn |
| `Surprise` | Ngạc nhiên |
| `Neutral` | Bình thường |

Project gồm hai phần:

- **Huấn luyện và đánh giá:** thực hiện trong notebook trên FER2013.
- **Detect realtime:** thực hiện trong `realtime_emotion_test.py` bằng webcam.

Kết quả là dự đoán từ hình ảnh, không phải kết luận chắc chắn về trạng thái cảm xúc thật của một người.

## 2. Kết quả

Model tốt nhất được lưu tại `best_model.keras`. Kết quả dưới đây lấy từ classification report trên tập `PrivateTest` của FER2013, gồm **3.589 ảnh**.

| Chỉ số | Kết quả |
| --- | ---: |
| Accuracy | **0.70** |
| Macro precision | 0.69 |
| Macro recall | 0.68 |
| Macro F1-score | 0.69 |
| Weighted precision | 0.70 |
| Weighted recall | 0.70 |
| Weighted F1-score | 0.69 |

### Kết quả theo lớp

| Cảm xúc | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| Angry | 0.64 | 0.61 | 0.63 | 491 |
| Disgust | 0.76 | 0.69 | 0.72 | 55 |
| Fear | 0.60 | 0.50 | 0.54 | 528 |
| Happy | 0.90 | 0.87 | 0.89 | 879 |
| Sad | 0.57 | 0.53 | 0.55 | 594 |
| Surprise | 0.80 | 0.82 | 0.81 | 416 |
| Neutral | 0.60 | 0.76 | 0.67 | 626 |

Model nhận diện `Happy` và `Surprise` tốt hơn các lớp còn lại. `Fear` và `Sad` khó phân biệt hơn. Hiệu năng qua webcam còn phụ thuộc vào ánh sáng, góc mặt, khoảng cách, chất lượng camera và khác biệt giữa dữ liệu FER2013 với người dùng thực tế.

## 3. Dataset: FER2013

FER2013 là dataset phổ biến cho bài toán nhận diện cảm xúc khuôn mặt.

- Ảnh grayscale, kích thước `48 x 48` pixel.
- Input của model có shape `(48, 48, 1)`.
- Có bảy lớp: `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral`.
- Tổng số ảnh thường dùng: `35.887`.
- `Training`: `28.709` ảnh.
- `PublicTest`: `3.589` ảnh validation.
- `PrivateTest`: `3.589` ảnh test.

Notebook đọc dữ liệu từ CSV. Chuỗi gồm 2.304 pixel được reshape thành tensor `48 x 48 x 1`, sau đó chuẩn hóa về `[0, 1]` bằng phép chia cho `255.0`. Nhãn được chuyển thành one-hot vector.

Đường dẫn Kaggle mặc định:

```python
CSV_PATH = "/kaggle/input/datasets/deadskull7/fer2013/fer2013.csv"
```

Khi chạy local, cần tải dataset hợp lệ và cập nhật `CSV_PATH`.

## 4. Phương pháp đã áp dụng

### 4.1. Tiền xử lý

- Chuyển chuỗi pixel thành mảng NumPy.
- Reshape về `(48, 48, 1)`.
- Chuẩn hóa pixel bằng `pixel / 255.0`.
- One-hot encoding nhãn bằng `to_categorical`.

### 4.2. Data augmentation

Chỉ tập training sử dụng augmentation:

- Xoay tối đa `10` độ.
- Dịch ngang và dọc tối đa `5%`.
- Lật ngang.
- Zoom tối đa `5%`.
- Điền vùng khuyết bằng `nearest`.

Validation và test không dùng augmentation.

### 4.3. Kiến trúc CNN

Model trong `model_architecture.py` gồm:

1. Hai convolution block với `64` filters.
2. Hai convolution block với `128` filters.
3. Convolution `256` filters, residual block và dilation rate `2`.
4. Channel Attention và Spatial Attention theo ý tưởng CBAM.
5. Ba convolution layer với `512` filters.
6. Global Average Pooling thay cho Flatten.
7. Dense `384` units, Batch Normalization, Swish và Dropout.
8. Output `7` units với Softmax.

Residual block giúp duy trì thông tin từ tầng trước. Dilated convolution mở rộng vùng ngữ cảnh. Attention giúp model tập trung vào các đặc trưng quan trọng cho biểu cảm như mắt, lông mày và miệng.

### 4.4. Huấn luyện

| Thành phần | Thiết lập |
| --- | --- |
| Batch size | `64` |
| Epoch tối đa | `40` |
| Optimizer | Adam |
| Learning rate ban đầu | `1e-3` |
| Loss | `categorical_crossentropy` |
| Metric | `accuracy` |
| Checkpoint | Lưu model có `val_accuracy` cao nhất |
| Early stopping | Theo dõi `val_accuracy`, patience `8`, restore best weights |
| ReduceLROnPlateau | Factor `0.5`, patience `3` theo `val_loss` |
| Learning rate nhỏ nhất | `2e-5` |
| Class weights | Điều chỉnh nhẹ theo từng lớp |

## 5. Pipeline realtime

Phiên bản hiện tại dùng OpenCV Haar Cascade để phát hiện khuôn mặt lớn nhất:

```text
Webcam frame
    ↓
Lật ngang
    ↓
Grayscale → Haar Cascade
    ↓
Chọn khuôn mặt lớn nhất
    ↓
Crop hình vuông + margin
    ↓
Grayscale → resize 48x48 → chia 255
    ↓
CNN + attention
    ↓
Trung bình xác suất trong 7 frame gần nhất
    ↓
Cảm xúc có xác suất cao nhất
```

Detector chạy mỗi `5` frame để cải thiện FPS. Bounding box gần nhất được giữ lại giữa các lần detect. Model cảm xúc và face detector là hai bước độc lập: CNN chỉ nhận ảnh khuôn mặt đã được cắt.

## 6. Cấu trúc thư mục

```text
FER/
├── best_model.keras             # Model Keras tốt nhất
├── fer-80-acc-40-epoch.ipynb    # Chuẩn bị dữ liệu, huấn luyện và đánh giá
├── model_architecture.py        # CNN, residual block và SpatialAttention
├── realtime_emotion_test.py     # Nhận diện cảm xúc qua webcam
└── README.md                    # Tài liệu project
```

## 7. Công nghệ sử dụng

- Python `3.10+`.
- TensorFlow / Keras.
- OpenCV.
- NumPy.
- Pandas.
- Scikit-learn.
- Matplotlib, Seaborn và Plotly.
- Jupyter Notebook hoặc Kaggle Notebook.

## 8. Cài đặt

Mở PowerShell tại thư mục project:

```powershell
cd FER
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install tensorflow opencv-python numpy jupyter matplotlib pandas scikit-learn seaborn plotly
```

Nếu chỉ chạy webcam với model có sẵn:

```powershell
python -m pip install tensorflow opencv-python numpy
```

Kiểm tra package chính:

```powershell
python -c "import cv2, numpy, tensorflow; print('OpenCV:', cv2.__version__); print('NumPy:', numpy.__version__); print('TensorFlow:', tensorflow.__version__)"
```

## 9. Chạy ứng dụng realtime

Đảm bảo `best_model.keras`, `model_architecture.py` và `realtime_emotion_test.py` nằm cùng thư mục:

```powershell
python realtime_emotion_test.py
```

Nhấn `Q` hoặc `q` để thoát. Đổi camera bằng cách chỉnh `CAMERA_INDEX`:

```python
CAMERA_INDEX = 1
```

| Tham số | Giá trị | Ý nghĩa |
| --- | ---: | --- |
| `IMG_SIZE` | `48` | Kích thước input CNN |
| `CAMERA_INDEX` | `0` | Camera mặc định |
| `MARGIN_RATIO` | `0.18` | Margin quanh khuôn mặt |
| `DETECT_EVERY_N_FRAMES` | `5` | Chu kỳ face detection |
| `SMOOTHING_WINDOW` | `7` | Số prediction để làm mượt |

## 10. Huấn luyện và đánh giá lại

1. Chuẩn bị `fer2013.csv` và cập nhật `CSV_PATH`.
2. Chạy các cell trong `fer-80-acc-40-epoch.ipynb` theo thứ tự.
3. Kiểm tra biểu đồ loss và accuracy của training/validation.
4. Chạy classification report và confusion matrix trên `PrivateTest`.
5. Lưu model tốt nhất vào `best_model.keras`.
6. Chạy lại ứng dụng realtime để kiểm tra trên webcam.

Khi load model phải đăng ký custom layer `SpatialAttention`:

```python
model = load_model(
    "best_model.keras",
    custom_objects={"SpatialAttention": SpatialAttention},
    compile=False
)
```

Không đổi tên hoặc xóa `SpatialAttention` nếu chưa cập nhật logic lưu và load model.

## 11. Nâng cấp trong tương lai

### Giai đoạn 1: Cải thiện face detection và realtime

- [ ] Bổ sung face tracking để không phải chạy detector ở mọi chu kỳ frame.
- [ ] Hỗ trợ nhận diện và phân loại cảm xúc cho nhiều khuôn mặt trong cùng một frame.
- [ ] Cho phép lựa chọn detector và các tham số realtime từ command line hoặc file cấu hình.

### Giai đoạn 2: Cải thiện model và dữ liệu

- [ ] Bổ sung dữ liệu thực tế từ webcam với nhiều điều kiện ánh sáng, góc mặt, độ tuổi và thiết bị khác nhau.
- [ ] Xử lý mất cân bằng dữ liệu bằng class weighting được tối ưu có hệ thống, focal loss hoặc resampling.
- [ ] Thử các backbone nhẹ như MobileNetV3 hoặc EfficientNet-Lite để cân bằng giữa accuracy và tốc độ inference.
- [ ] Fine-tune từ mô hình pretrained phù hợp với bài toán ảnh khuôn mặt grayscale.
- [ ] Phân tích confusion matrix để cải thiện các cặp cảm xúc khó phân biệt như `Fear`/`Sad` và `Sad`/`Neutral`.
- [ ] Calibrate confidence để xác suất dự đoán phản ánh tốt hơn mức độ không chắc chắn của model.

### Giai đoạn 3: Tối ưu hiệu năng và chất lượng

- [ ] Đo benchmark riêng cho CPU, GPU và các độ phân giải webcam khác nhau.
- [ ] Chuyển model sang TensorFlow Lite hoặc ONNX để giảm thời gian inference và dung lượng phân phối.
- [ ] Bổ sung kiểm thử tự động cho preprocessing, bounding box, output shape và các trường hợp không phát hiện được khuôn mặt.
- [ ] Lưu metadata của mỗi lần huấn luyện gồm seed, phiên bản thư viện, số epoch thực tế và cấu hình model.
- [ ] Thêm logging, theo dõi FPS và thống kê confidence trong quá trình chạy.

### Giai đoạn 4: Hoàn thiện ứng dụng

- [ ] Tách face detection, emotion inference và giao diện hiển thị thành các module độc lập.
- [ ] Xây dựng giao diện trực quan để chọn camera, điều chỉnh tham số và xem lịch sử dự đoán.
- [ ] Đóng gói ứng dụng cho Windows, Linux và macOS.
- [ ] Viết tài liệu API và hướng dẫn đóng góp cho project.

## 12. Giới hạn

- Phiên bản realtime hiện tại chỉ xử lý khuôn mặt lớn nhất.
- FER2013 là ảnh grayscale đã crop, nên có khác biệt với dữ liệu webcam.
- Cảm xúc khuôn mặt phụ thuộc bối cảnh; prediction không phải sự thật tuyệt đối.