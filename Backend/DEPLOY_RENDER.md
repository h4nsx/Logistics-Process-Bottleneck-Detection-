# Deploy Backend lên Render — từng bước

## Chuẩn bị

- Repo đã push lên **GitHub/GitLab/Bitbucket** (Render kết nối qua đó).
- App dùng **PostgreSQL + asyncpg**; `app/config.py` tự chuẩn hoá mọi dạng URL (`postgres://`, `postgresql://`) → `postgresql+asyncpg://`.
- ML model files (`ML/model/process_models/`) đã được commit vào repo (xem `.gitignore`).

---

## Bước 1 — Tạo PostgreSQL trên Render

1. Vào [dashboard.render.com](https://dashboard.render.com) → **New +** → **PostgreSQL**.
2. Đặt tên (ví dụ `logistics-db`), chọn region gần bạn.
3. **Create Database**.
4. Sau khi tạo xong, vào database → copy **Internal Database URL**.

---

## Bước 2 — Tạo Web Service (Backend)

1. **New +** → **Web Service** → chọn repo.
2. Cấu hình:

| Mục | Giá trị |
|-----|--------|
| **Name** | `logistics-backend` |
| **Region** | Cùng region với PostgreSQL |
| **Branch** | `main` |
| **Root Directory** | `Backend` |
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 2` |

3. **Instance type**: Free hoặc Starter tùy nhu cầu.

> **Lưu ý Workers**: ML models được load vào bộ nhớ RAM khi khởi động. Nếu dùng `--workers 2`, mỗi worker load riêng (~30MB RAM/worker). Nếu RAM hạn chế, dùng `--workers 1`.

---

## Bước 3 — Biến môi trường

Trong Web Service → **Environment**:

| Key | Value |
|-----|--------|
| `DATABASE_URL` | **Link** → chọn PostgreSQL → property **Internal Database URL** |
| `PYTHON_VERSION` | `3.12.0` |
| `LOG_LEVEL` | `INFO` |

> `ML_MODEL_DIR` không cần set — code tự tính đường dẫn relative từ vị trí file source.

---

## Bước 4 — Health check

Trong **Settings** của Web Service:
- **Health Check Path**: `/health`

---

## Bước 5 — Deploy

1. **Save** → Render build + start.
2. Xem **Logs**: thấy dòng sau là OK:
   ```
   ML models ready: ['TRUCKING_DELIVERY_FLOW', 'IMPORT_CUSTOMS_CLEARANCE', 'WAREHOUSE_FULFILLMENT']
   Application startup complete.
   ```
3. Mở `https://<tên-service>.onrender.com/docs` để thử API.

---

## Dùng Blueprint (tạo DB + Web cùng lúc)

Trong repo đã có `Backend/render.yaml`:

1. **New +** → **Blueprint** → chọn repo → Render đọc `render.yaml`.
2. Render tự tạo DB + Web Service theo cấu hình.

---

## Free tier — lưu ý

- Service **sleep** sau ~15 phút không có request → cold start lần đầu ~30–60s.
- ML models load mất ~2–3s khi startup.
- PostgreSQL free: giới hạn storage 1GB, expire sau 90 ngày — xem docs Render.

---

## CORS (Frontend ở domain khác)

Sau khi có URL Frontend (Vercel/Netlify), sửa `app/main.py`:

```python
allow_origins=["https://your-frontend.vercel.app"]
```

---

## Tóm tắt

```
Build:   pip install -r requirements.txt
Start:   alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 2
Health:  /health
Root:    Backend
Python:  3.12.0
```
