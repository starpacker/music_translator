# 第三方组件与许可声明

本项目以 **AGPL-3.0** 发布（见 `LICENSE`）。它不重新分发下列组件，而是**调用本机安装或另行 clone 的上游仓库**；
此处列出它们的许可与用途，以满足署名与许可提示的要求。

## 运行期核心

| 组件 | 版本 | 许可 | 用途 |
| --- | --- | --- | --- |
| [SumisoraOMR](https://github.com/Tsukamotoshio/SumisoraOMR) | 仓库快照 | **AGPL-3.0** | OMR 引擎封装、MusicXML→简谱（music21 处理）、LilyPond 排版调度。本项目 `jianpu.py` 通过 `import` 方式调用。 |
| [Audiveris](https://github.com/Audiveris/audiveris) | 5.11.0 | **AGPL-3.0** | SumisoraOMR 内可选 OMR 后端。 |
| [Homr](https://github.com/liebharc/homr) | v0.7.0+ / fork | **AGPL-3.0** | 实际使用的端到端 OMR 模型（默认引擎）。内嵌 **oemer**（MIT, © BreezeWhite）与 **Polyphonic-TrOMR**（MIT, © NetEase）的模型与分词代码。 |
| [LilyPond](https://lilypond.org/) | 2.22 及以上（上游快照用 2.24.4） | **GPL-2.0** | 简谱 / 五线谱的矢量排版。以独立进程方式调用（`lilypond -dbackend=eps`），不链接。 |
| [jianpu-ly.py](https://github.com/ssb22/jianpu-ly) | v1.866 | **Apache-2.0** | 文本简谱 → LilyPond 源。见下方修改说明。 |
| [music21](https://github.com/cuthbertLab/music21) | 10.5.0 | BSD-3-Clause | MusicXML 解析、调号/时值/连音结构提取。 |
| [PyMuPDF](https://github.com/pymupdf/pymupdf) | 1.28.2 | **AGPL-3.0** 或商业双许可 | 五线谱 PDF 的矢量几何读取、页面合成、栅格化。本项目走 AGPL-3.0 分支。 |
| [Pillow](https://github.com/python-pillow/Pillow) | 12.3.0 | HPND | 图片预处理与 PNG 输出。 |
| [ReportLab](https://www.reportlab.com/) | 5.0.0 | BSD-3-Clause | 整页简谱 PDF 的容器生成。 |
| [NumPy](https://numpy.org/) | ≥1.24 | BSD-3-Clause | 数值与像素运算。 |

## 仅构建期（可选）

| 组件 | 许可 | 用途 |
| --- | --- | --- |
| [fonttools](https://github.com/fonttools/fonttools) | MIT | `tools/make_assets.py` 把字形轮廓导出为 SVG 路径，生成 `assets/logo.*`、`assets/social-preview.*`。 |

## jianpu-ly.py 的本地修改

仓库内 `scripts/jianpu-ly.py`（来自 SumisoraOMR 的 vendored 副本）相对官方 **v1.866** 有 4 处修改，
按 Apache-2.0 第 4(b) 条在文件头已声明：

1. 增加 `\override Voice.Rest #'staff-position = #0`，避免休止符被自动上移；
2. `Tie #'transparent` / `Tie #'staff-position` 改为现代语法 `Tie.transparent` / `Tie.staff-position`；
3. 修正 `sys.version_info()` 笔误为 `sys.version_info`；
4. 调整 Python 版本过低时的报错文案。

## 字体

Logo 与社交预览图中使用的字形来自系统字体，已通过 fonttools 转为**矢量轮廓路径**内嵌于 SVG，
成品 SVG/PNG 不含字体文件本身。
