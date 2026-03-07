# Phân tích Backend Development Guide

## 1. Tổng quan hệ thống

### Mục đích
Backend cung cấp **analytics theo thời gian thực** cho vận hành logistics, tập trung vào:
- **Phát hiện bottleneck**: Đoạn/ bước trong quy trình bị chậm bất thường
- **Đo lường rủi ro**: Dựa trên lịch sử thực thi (baseline thống kê), không dùng ML “hộp đen”

### Triết lý thiết kế
| Nguyên tắc | Ý nghĩa |
|------------|---------|
| **PostgreSQL-centric** | Phần lớn tính toán (baseline, anomaly) làm trong DB, Python chỉ orchestration |
| **Deterministic** | Kết quả lặp lại được, giải thích được |
| **No Black-Box ML (MVP)** | Chỉ dùng thống kê (mean, std, p95, z-score) |
| **Transparency** | Mọi metric có thể audit và hiểu được |

---

## 2. Kiến trúc & công nghệ

### Stack
- **Python + FastAPI**: API và orchestration
- **PostgreSQL**: Lưu trữ + engine tính toán (aggregations, percentiles)
- **asyncpg**: Driver async cho PostgreSQL
- **SQLAlchemy Core**: Query builder (không dùng ORM full)
- **Alembic**: Migrations
- **NumPy**: Tính toán nhẹ (nếu cần); **Pandas**: Chỉ cho đọc CSV

### Luồng dữ liệu
```
FastAPI (Python)  →  Gọi SQL / transaction  →  PostgreSQL (tính baseline, anomaly)
```
Backend **không** tính baseline/anomaly bằng Python; DB thực hiện qua các câu lệnh SQL đã mô tả trong spec.

---

## 3. Mô hình dữ liệu

### Bảng chính

| Bảng | Vai trò |
|------|--------|
| **processes** | Định danh một quy trình (vd: một lô hàng, đơn hàng) |
| **step_executions** | Từng bước thực thi: process_id, step_code, location, start/end, duration_minutes |
| **baselines** | “Chuẩn” thống kê theo (step_code, location): mean, std, p95, sample_size |
| **anomalies** | Các bước bị coi là bất thường: process_id, step_code, location, duration, z_score, risk_percent |

### Ràng buộc quan trọng
- `step_executions`: `duration_minutes > 0`, `end_time > start_time`
- **Baselines** được cập nhật **sau mỗi lần upload** (recompute toàn bộ từ `step_executions`)
- **Anomalies** được insert dựa trên so sánh với baselines (vượt p95 hoặc z_score ≥ 2)

---

## 4. Các tính năng chính

### 4.1 Data ingestion
- **Input**: CSV qua `multipart/form-data`
- **Cột bắt buộc**: `process_id`, `step_code`, `location`, `start_time`, `end_time`
- **Validation**: Thiếu cột → reject cả file; lỗi theo dòng (thời gian, duration) → báo lỗi nhưng vẫn xử lý các dòng hợp lệ
- **Output**: Số dòng xử lý, số dòng lỗi, danh sách validation_errors

### 4.2 Baseline computation
- Chạy **trong cùng transaction** với insert step_executions
- SQL dùng: `AVG`, `STDDEV_POP`, `PERCENTILE_CONT(0.95)`, `GROUP BY step_code, location`
- `ON CONFLICT (step_code, location) DO UPDATE` để upsert baselines

### 4.3 Bottleneck detection
- **Điều kiện** (chỉ cần 1 trong 2):
  1. `duration_minutes > baseline.p95`
  2. `z_score = (duration - mean) / std >= 2`
- **Risk percent**: `LEAST(100, (duration / p95) * 100)` (trong spec); ≥100% = vượt “worst case” lịch sử

### 4.4 Transaction
- Một upload = một transaction: insert processes → insert step_executions → recompute baselines → detect anomalies. Lỗi bất kỳ bước nào → rollback.

---

## 5. API Specification (tóm tắt)

| Endpoint | Mục đích |
|----------|----------|
| `POST /api/upload` | Upload CSV → validate, insert, baseline, anomaly |
| `GET /api/anomalies` | Danh sách bottleneck (có limit, min_risk) |
| `GET /api/process/{process_id}` | Chi tiết timeline + từng step có is_anomaly, baseline_mean, baseline_p95 |
| `GET /api/baselines` | Xem baseline theo step_code, location (transparency/debug) |

Spec còn đề cập **Schema Mapping** với endpoint `POST /api/schema/suggest` (xem mục 6).

---

## 6. Intelligent Schema Mapping – Phân tích & Lưu ý

### Mục đích
- Cho phép user upload CSV **không cần đúng tên cột** (vd: "Total Duration" thay vì "duration_minutes").
- Backend gợi ý ánh xạ cột CSV → schema nội bộ, có confidence; user confirm (qua frontend) rồi mới transform và đưa vào pipeline.

### ⚠️ Mâu thuẫn trong tài liệu
- Phần **Schema Mapping** trong guide đang dùng ví dụ schema **invoice**:
  - `invoice_id`, `vendor_name`, `amount`, `invoice_date`
- Trong khi toàn bộ backend thực tế dùng schema **logistics process**:
  - `process_id`, `step_code`, `location`, `start_time`, `end_time`

**Đề xuất**: Khi implement, **bỏ schema invoice**, dùng đúng schema process:

```json
{
  "process_id": "string",
  "step_code": "string",
  "location": "string",
  "start_time": "datetime (ISO 8601)",
  "end_time": "datetime (ISO 8601)"
}
```

Keyword dictionary nên là:

```python
FIELD_KEYWORDS = {
    "process_id": ["process", "id", "order", "shipment", "case", "ref", "reference"],
    "step_code": ["step", "code", "activity", "phase", "stage", "task"],
    "location": ["location", "warehouse", "facility", "site", "depot", "wh"],
    "start_time": ["start", "begin", "from", "time", "datetime"],
    "end_time": ["end", "finish", "to", "time", "datetime"]
}
```

Workflow vẫn giữ: Upload → Parse headers → Normalize → Match → Suggest → User confirm → Validate → Transform → Analytics.

---

## 7. So sánh với code hiện tại trong repo

| Thành phần | Trong Guide | Trong repo hiện tại |
|------------|-------------|----------------------|
| **Backend** | FastAPI + PostgreSQL, bottleneck + baseline + anomaly | Thư mục `Backend/` chỉ có README (rỗng) |
| **ML/api** | Không mô tả trong guide | FastAPI, load model driver/fleet/ops, predict risk (shipment-level) |

**Kết luận**: Guide mô tả một **backend mới** (analytics process, PostgreSQL); `ML/api` là service **khác** (AI risk prediction). Hai hệ thống có thể:
- Chạy tách (backend analytics + service ML riêng), hoặc
- Tích hợp sau (backend gọi ML service khi cần “risk prediction” nâng cao).

---

## 8. Điểm mạnh & Rủi ro

### Điểm mạnh
- **Rõ ràng**: Logic baseline/anomaly nằm trong SQL, dễ review và tái tạo.
- **Có thể mở rộng**: Có thể thêm bảng, index, hoặc batch job cho dataset lớn.
- **Phù hợp MVP**: Không phụ thuộc ML phức tạp, dễ bảo trì và giải thích cho operations.

### Rủi ro / Cần lưu ý
- **Schema mapping**: Cần thống nhất schema (process, không phải invoice) và validation kiểu dữ liệu (đặc biệt datetime).
- **Performance**: Với 50k dòng, recompute baseline + detect anomaly trong 1 transaction cần kiểm tra (index, lock). Guide đã đặt target &lt; 3s cho baseline.
- **Security**: Guide ghi rõ MVP chưa có auth/rate limit; cần bổ sung khi lên production.

---

## 9. Định nghĩa “Done” (theo guide)

Backend được coi production-ready khi:
- Data ingestion xử lý lỗi rõ ràng (validation, partial success).
- PostgreSQL thực hiện toàn bộ analytics (baseline, anomaly) hiệu quả.
- Bottleneck có thể reproduce và giải thích được.
- Output có confidence/risk metrics.
- API có tài liệu (OpenAPI/Swagger).
- Integration tests cover critical path.
- Đạt benchmark (upload size, latency) như đã nêu.

---

## 10. Thứ tự triển khai đề xuất

1. **Database**: Tạo schema (processes, step_executions, baselines, anomalies) + index bằng Alembic.
2. **Core API**: Implement `POST /api/upload` (parse CSV, validate, insert, gọi SQL baseline + anomaly trong transaction).
3. **Read API**: `GET /api/anomalies`, `GET /api/process/{id}`, `GET /api/baselines`.
4. **Schema mapping**: Module `schema_matcher` + `data_transformer`, endpoint `POST /api/schema/suggest` (dùng schema process, không dùng invoice).
5. **Integration tests**: Upload CSV hợp lệ/invalid, kiểm tra baselines và anomalies.
6. **Docs & observability**: OpenAPI, logging như mô tả trong guide.

---

## Tóm tắt

- Backend được thiết kế **PostgreSQL-centric**, **deterministic**, **transparent**, phù hợp analytics bottleneck logistics.
- Cần **sửa phần Schema Mapping** trong tài liệu/implementation: dùng schema **process** (process_id, step_code, location, start_time, end_time), không dùng schema invoice.
- Backend mới nên triển khai trong thư mục `Backend/`; `ML/api` giữ vai trò service AI riêng, tích hợp sau nếu cần.
