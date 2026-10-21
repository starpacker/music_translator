# 五线谱 → 简谱 网页工具

上传一份五线谱 PDF，选择调号（如 `1=G`），得到：

1. **`*.jianpu.pdf`** — 标准简谱（真正移调后的音高）
2. **`*.insitu.pdf`** — 原位对照谱：每个简谱格精确排在对应五线谱小节下方

也可一次下载两个（zip）。

## 结构

```
frontend/          静态页面（部署到 GitHub Pages）
backend/app.py     FastAPI 服务：OMR → 移调 → 简谱 / 原位
backend/vendor/    梆笛排版工具 score_align.py（原位排版）
.github/workflows/ GitHub Pages 自动发布
```

## 本地运行

```bash
# 后端（必须在 sumisora conda 环境里，OMR 依赖都在那）
/opt/conda/envs/sumisora/bin/python backend/app.py
# 或
bash backend/run.sh

# 前端
cd frontend && python -m http.server 5173
# 打开 http://127.0.0.1:5173
```

环境变量：

| 变量 | 说明 | 默认 |
|---|---|---|
| `SUMISORA_ROOT` | SumisoraOMR 仓库根目录 | `/liuyutian/SumisoraOMR-main/SumisoraOMR-main` |
| `ALLOWED_ORIGINS` | 允许的前端来源（逗号分隔） | `http://localhost:5173` |
| `JIANPU_TONICS` | 覆盖可选调号 | 12 个大调 |

## API

`POST /api/convert` — multipart

| 字段 | 值 |
|---|---|
| `file` | PDF / PNG / JPG |
| `tonic` | `C` `Db` `D` … `B` |
| `mode` | `major` / `minor`（其余见 `/api/keys`） |
| `transpose` | `true` 真正移调；`false` 保留原调 |
| `format` | `jianpu` / `insitu` / `both` |

响应头带 `X-Detected-Key`、`X-Output-Key`。

## 原位版说明

原位排版基于 `梆笛排版工具`（`backend/vendor/score_align.py`）：从 PDF 矢量线中
识别五线谱系统与小节线，把简谱 PDF 的每一格按阅读顺序配到对应小节，再以透明
贴图叠到每个系统下方、页面相应加长。它使用**用户上传的原始 PDF**，因此 OMR 的
识别误差不会影响五线谱本身的排布。

- 只支持 PDF 输入（图片没有矢量分格）。
- 输出是高清页面图（约 288 dpi），页面会变长；打印时请选“实际大小/100%”。
- 简谱格数少于五线谱小节时会报错；多于时多余的排在最后。

## 支持的调式

`major` / `minor` 可用。七个教会调式（dorian、phrygian、lydian、mixolydian、
locrian、和声小调、旋律小调）已列出但暂未启用，需要简谱调式头渲染支持。
