# 如何发布（你自己操作，不要把 token 发给我）

我已经把可发布的工程放在 `/liuyutian/jianpu_repro/webapp/`。
下面命令用**你自己的** GitHub 凭据执行。

```bash
cd /liuyutian/jianpu_repro/webapp
git init -b main
git add -A
git commit -m "jianpu web: GitHub Pages frontend + FastAPI backend"

# 用你新建的仓库 URL（不要再把 token 粘进聊天）
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git push -u origin main
```

## 配置前端指向后端
1. 部署后端（见 docs/DEPLOY.md，任选 Render / Fly / VPS / HF Spaces）。
2. 编辑 `frontend/config.js` 的 `apiBase` 为后端公网地址。
3. 在后端设置环境变量 `ALLOWED_ORIGINS=https://<用户名>.github.io`。
4. 仓库 **Settings → Pages → Source: GitHub Actions**，push 后自动发布。

> ⚠️ 你之前贴出的 PAT 已经泄露，请立刻到 https://github.com/settings/tokens 撤销。
> 以后只用 `gh auth login` 或 credential manager，不要把 token 贴到任何对话/代码里。
