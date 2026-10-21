# 部署指南

## A. 前端 → GitHub Pages

1. 建仓库（例如 `jianpu-web`），把 `frontend/` 的内容放到 `docs/` 或根目录。
   本仓库已含一键工作流 `.github/workflows/pages.yml`（发布 `frontend/`）。
2. 编辑 `frontend/config.js`，把 `apiBase` 的占位符改成你的后端地址：
   ```js
   apiBase: "https://jianpu-api.onrender.com"
   ```
3. 仓库 **Settings → Pages → Source: GitHub Actions**。
4. push 到 `main`，几分钟后访问 `https://<user>.github.io/<repo>/`。

> 也支持临时覆盖：`https://<user>.github.io/<repo>/?api=https://your-backend`

## B. 后端 → 任选其一

后端需要：Python + **sumisora** 环境的 OMR 依赖 + **LilyPond** + 较长超时（多页 OMR 数分钟）。
GitHub Pages **不能**托管后端。

### B1. Render（简单，免费档够用）
- New → Web Service → 连接仓库，Root = `backend/`
- Build：`pip install -r requirements.txt`
- Start：`bash run.sh`
- 环境变量：
  - `SUMISORA_ROOT=/app/sumisora`
  - `ALLOWED_ORIGINS=https://<user>.github.io`
- 需自带可跑 OMR 的镜像：用 `backend/Dockerfile`（把里面的仓库地址换成你的 fork）。

### B2. Hugging Face Spaces (Docker)
- 新建 Docker Space，上传 `backend/`，SDK=Docker，端口 8000。
- Secrets 里设 `ALLOWED_ORIGINS`。

### B3. 自建 VPS / 内网
```bash
conda activate sumisora
pip install -r backend/requirements.txt
ALLOWED_ORIGINS="https://<user>.github.io" PORT=8000 bash backend/run.sh
# 前面加 nginx 反代 + TLS；OCR 很吃 CPU，建议限流 / 队列。
```

## C. CORS
后端只允许 `ALLOWED_ORIGINS` 中的来源。务必把 Pages 域名加进去，否则前端会被浏览器拦截。
预检（OPTIONS）由 `CORSMiddleware` 自动处理。

## D. 安全与合规
- **不要**把任何 GitHub token / 密钥写进前端或仓库；本仓库不含任何密钥。
- 后端对上传大小/类型做校验；生产建议再加：
  - 上传大小上限（nginx `client_max_body_size` / 反向代理）
  - 每 IP 限流（slowapi / nginx `limit_req`）
  - 任务队列（Redis + RQ / Celery），避免 OMR 挤爆进程
- SumisoraOMR 为 **AGPL-3.0-only**：通过网络对外提供服务时，须向用户提供对应源码。
