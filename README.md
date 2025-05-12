## 1. Preprocessing & Feature Engineering

Dữ liệu đầu vào bao gồm hai đặc trưng (features):

- `log_price_change`: log return giữa giá Close hiện tại và trước đó
- `log_vol_change`: thay đổi log của khối lượng giao dịch (Volume)

Công thức:
```python
log_price_change = log(Close_t / Close_{t-1})
log_vol_change   = log(Volume_t / Volume_{t-1})
```

Các giá trị này sau đó được:
- **clip** trong khoảng `[-3.0, 3.0]` để giảm nhiễu
- **normalize** (z-score normalization) với trung bình và độ lệch chuẩn được tính trên tập huấn luyện

Target cũng là `log_price_change`, được xử lý tương tự.

Dữ liệu được chia thành các chuỗi (`sequence`) độ dài `seq_len = 20` để làm đầu vào, và `tgt_len = 5` bước tiếp theo làm nhãn (multi-step forecasting). Sử dụng **stride=1** để tạo chuỗi overlapping.


## 2. Mô hình: Transformer kiến trúc encoder-only with ProbSparse Attention

Mô hình được xây dựng như một encoder-only Transformer đơn giản:

- **Input**: tensor `[batch_size, seq_len, 2]`
- **Positional Encoding**: learnable
- **Attention**: sử dụng **ProbSparse Attention** cho hiệu quả cao trên chuỗi dài
- **Encoder: 4 lớp encoder**
- **Output Layer**: hai tầng linear với ReLU ở giữa để sinh chuỗi log return dự đoán có độ dài tgt_len = 5  

## 3. Loss Function: Tanh Directional Loss

Loss chính là `TanhDirectionalLoss`, gồm 3 thành phần:

1. **MSE Loss**: giúp mô hình dự đoán chính xác biên độ lợi suất
2. **Directional Loss**: mô hình học **đúng hướng di chuyển thị trường** (up/down), dùng `BCEWithLogitsLoss` giữa `tanh(outputs)` và `sign(targets)`.
3. **Diversity Regularization**: khuyến khích mô hình tạo ra output có phương sai đủ lớn, tránh output phẳng (model thiên về dự đoán mean để cheat) và tránh vanishing gradient.


```python
loss = MSE + directional_weight * BCE + diversity_weight * -log(var)
```

Tanh giúp ổn định gradient và chuẩn hóa tín hiệu đầu ra thành khoảng `[-1, 1]`, phù hợp để biểu diễn vị thế (position).


## 4. Training & Scheduler & Gradient Stabilization

- Optimizer: `Adam`
- Scheduler: **ReduceLROnPlateau**


## 5. Autoregressive Testing

Ở bước test:
- Mỗi sample được dự đoán từng bước một trong `tgt_len=5` bước
- Mô hình autoregressively cập nhật input bằng chính prediction trước đó
- Dùng `use_ground_truth=True` để teacher-forcing (kiểm tra loss lý tưởng)

Log các metric:
- `Test Loss (MSE)`
- `Directional Accuracy (DA)`
- `Output Std`
- `Sign ratio` (số lượng tín hiệu dương)

## 6. Trading Backtest

Sử dụng signal từ dự đoán để giao dịch
Sau khi có dự đoán:

1. **Flatten** các chuỗi chồng lặp về chuỗi log return ban đầu (dài khoảng `N + tgt_len - 1`)
2. **Denormalize** để khôi phục log return thực tế
3. **Trading Logic**:
   - Nếu signal > threshold → **Buy**
   - Nếu signal < -threshold hoặc đã hold đủ `min_hold (hold tối thiếu 15 tiếng)` bước → **Sell**
4. Tính ROI bằng:
```python
np.exp(sum(position[t] * log_return[t]))
```

Mô hình cho thấy ROI cao hơn **chiến lược Buy & Hold** 
