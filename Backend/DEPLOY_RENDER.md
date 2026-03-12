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
| **Branch** | `backend` *(hoặc `main` nếu backend đã merge)* |
| **Root Directory** | **`Backend`** — **bắt buộc**. Nếu để trống, build sẽ lỗi `requirements.txt` not found vì file nằm trong `Backend/`. |
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
| `DATABASE_URL` | **Bắt buộc.** **Add from Database** → chọn PostgreSQL → **Internal Database URL**. Nếu thiếu, `alembic upgrade head` sẽ kết nối `localhost:5432` và báo `Connection refused`. |
| `LOG_LEVEL` | `INFO` |

> **Không set `PYTHON_VERSION` trên dashboard** nếu gặp lỗi version — dùng file `.python-version` ở repo root (đã có `3.11.9`).

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

Trong repo đã có **`render.yaml` ở root** (và bản sao trong `Backend/`):

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
Root:    Backend   ← bắt buộc (monorepo)
Python:  .python-version ở repo root (3.11.9)
```

---

## Troubleshooting

### `Could not open requirements file: requirements.txt`

- **Nguyên nhân**: Render đang build từ **gốc repo** trong khi `requirements.txt` chỉ có trong **`Backend/`**.
- **Cách xửa**: Vào service → **Settings** → **Root Directory** → nhập **`Backend`** → Save → Manual Deploy.

### Deploy từ branch `main` vs `backend`

- Nếu code backend đầy đủ nằm trên **`backend`**, nên chọn branch **`backend`** cho Web Service.
- Repo đã có **`render.yaml` ở root** — dùng **Blueprint** sẽ tự set `rootDir: Backend`.

### `PYTHON_VERSION must provide major.minor.patch`

- Xóa biến `PYTHON_VERSION` trên dashboard; để Render đọc `.python-version` ở root.
