# Deploy Backend lên Render — từng bước

## Chuẩn bị

- Repo đã push lên **GitHub/GitLab/Bitbucket** (Render kết nối qua đó).
- App dùng **PostgreSQL** + **psycopg async**; `app/config.py` tự đổi `postgres://` / `postgresql://` → `postgresql+psycopg://` (Render gắn DB hay cho URL dạng đó).

---

## Bước 1 — Tạo PostgreSQL trên Render

1. Vào [dashboard.render.com](https://dashboard.render.com) → **New +** → **PostgreSQL**.
2. Đặt tên (ví dụ `logistics-db`), chọn region gần bạn.
3. **Create Database**.
4. Sau khi tạo xong, vào database → copy **Internal Database URL** (hoặc External nếu app chạy ngoài Render — thường Web Service cùng region dùng Internal là đủ).

---

## Bước 2 — Tạo Web Service (Backend)

1. **New +** → **Web Service** → chọn repo.
2. Cấu hình:

| Mục | Giá trị |
|-----|--------|
| **Name** | `logistics-backend` (tuỳ bạn) |
| **Region** | Cùng region với PostgreSQL |
| **Branch** | `main` hoặc branch deploy |
| **Root Directory** | `Backend` *(nếu repo là monorepo; nếu repo chỉ có Backend thì để trống)* |
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | Xem bên dưới |

**Start Command** (chạy migration rồi mới bật app):

```bash
alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 2
```

3. **Instance type**: Free hoặc Starter tùy nhu cầu.

---

## Bước 3 — Biến môi trường

Trong Web Service → **Environment**:

| Key | Value |
|-----|--------|
| `DATABASE_URL` | **Link** database: Add Environment Variable → **Link** → chọn PostgreSQL vừa tạo → property **Internal Database URL** (Render tự inject). Không cần sửa tay thành `+psycopg` — code đã tự chuẩn hoá. |
| `PYTHON_VERSION` | `3.12.0` *(khuyến nghị; tránh 3.13 nếu build lỗi)* |

Không cần file `.env` trên Render — mọi thứ qua Environment.

---

## Bước 4 — Health check

Trong **Settings** của Web Service:

- **Health Check Path**: `/health`

Render sẽ gọi `GET https://<service>.onrender.com/health`.

---

## Bước 5 — Deploy

1. **Save** → Render build + start.
2. Xem **Logs**: thấy `Application startup complete` và không lỗi migration là OK.
3. Mở URL dạng `https://<tên-service>.onrender.com/docs` để thử API.

---

## Dùng Blueprint (một lần tạo DB + Web)

Trong repo đã có `Backend/render.yaml`. Trên Render:

1. **New +** → **Blueprint**.
2. Chọn repo → Render đọc `render.yaml`.
3. Chỉnh `rootDir` nếu cấu trúc repo khác (ví dụ không có thư mục `Backend` thì bỏ `rootDir` hoặc sửa lại).

---

## Free tier — lưu ý

- Service **sleep** sau một lúc không có request → lần đầu gọi có thể **cold start ~30s–1 phút**.
- PostgreSQL free có giới hạn storage / expire sau thời gian — xem docs Render.

---

## CORS (Frontend ở domain khác)

Sau khi có URL Frontend (Vercel/Netlify/…), sửa `app/main.py`:

```python
allow_origins=["https://your-frontend.vercel.app"]
```

Thay vì `["*"]` khi đã public production.

---

## Tóm tắt lệnh

```text
Build:  pip install -r requirements.txt
Start:  alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 2
Health: /health
```

Root directory: `Backend` nếu backend nằm trong subfolder repo.
