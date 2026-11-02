<p align="center">
  <img src="assets/social-preview.png" alt="music_translator：五线谱 PDF 转简谱" width="100%">
</p>

<h1 align="center">music_translator</h1>

<p align="center">
  <em>五线谱 PDF → 简谱 —— 既能出整页简谱，也能把简谱<b>直接贴回原谱每一行五线谱的下方</b>。</em>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: AGPL-3.0" src="https://img.shields.io/badge/license-AGPL--3.0-2f6f4e.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776ab.svg">
  <img alt="LilyPond" src="https://img.shields.io/badge/LilyPond-2.22%2B-1a5c2a.svg">
  <img alt="PyMuPDF" src="https://img.shields.io/badge/PDF-PyMuPDF-d9534f.svg">
  <img alt="OMR" src="https://img.shields.io/badge/OMR-Homr%20%2F%20Audiveris-f0a500.svg">
</p>

<p align="center">
  <a href="#demo">效果</a> ·
  <a href="#modes">两种生成方式</a> ·
  <a href="#install">安装</a> ·
  <a href="#usage">用法</a> ·
  <a href="#how">原理</a> ·
  <a href="#tuning">可调参数</a> ·
  <a href="#limits">已知限制</a> ·
  <a href="#legacy">旧版实现</a>
</p>

---

## ✨ 这是什么

一句话：**五线谱 PDF 进去，简谱 PDF 出来**，而且有两种成品形态。

- **`jianpu` —— 整页简谱。** 干净的一页简谱，只有数字谱，适合打印、发谱、翻谱。
- **`insitu` —— 原位简谱。** 保留原版五线谱的每一行，把对应的简谱**贴在每一行五线谱下方**，一行对一行。适合对照学习、校对、教学。

整条流水线只有一个入口脚本：

```
五线谱 PDF ──OMR──► MusicXML ──┬──► 整页简谱 PDF   (mode=jianpu)
                              ├──► 原位简谱 PDF   (mode=insitu)
                              └──► 两者都要        (mode=both)
```

OMR 用的是 **Homr**（端到端光学乐谱识别，也可切到 Audiveris），
排版走 **music21 → jianpu-ly → LilyPond**，页面合成用 **PyMuPDF** 的矢量操作，
默认 `spaced` 和 `stamp` 使用矢量 PDF 合成；`strip` 引擎会栅格化页面。

---

<a id="demo"></a>

## 🖼 效果

### 原位简谱（`--mode insitu`）

每一行五线谱下方，就是这一行的简谱。半个字都不挪动原谱的位置。

下图取自梆笛分谱 `梆笛.pdf`（6 页 A4 矢量谱，51 行谱表）的实测输出：

<p align="center">
  <img src="docs/screenshots/insitu.png" alt="梆笛原位简谱：每行五线谱下方对齐一行简谱" width="88%">
</p>

### 整页简谱（`--mode jianpu`）

同一份谱子的纯简谱排版，A4 满页，附调号与小节线。

<p align="center">
  <img src="docs/screenshots/jianpu.png" alt="整页简谱：纯数字谱，含调号、八度点与减时线" width="88%">
</p>

---

<a id="modes"></a>

## 🧭 两种生成方式

`--mode` 决定出哪种成品，`--engine` 只影响 `insitu` 的实现方式。

| 方式 | 参数 | 产物 | 页面尺寸 | 典型用途 |
| --- | --- | --- | --- | --- |
| 整页简谱 | `--mode jianpu` | `<曲名>.jianpu.pdf` | A4 纵向 | 打印、发谱、只看简谱 |
| 原位简谱（加行距，默认） | `--mode insitu --engine spaced` | `<曲名>.insitu.pdf` | **页面加长** | 尽量避免简谱与原谱重叠 |
| 原位简谱（贴字） | `--mode insitu --engine stamp` | `<曲名>.insitu.pdf` | **与原谱完全一致** | 对照学习、校对、原版式交付 |
| 原位简谱（插条） | `--mode insitu --engine strip` | `<曲名>.insitu.pdf` | 每行谱表下方**加长** | 需要给简谱留出连续白位、页面可重排 |

### 三种原位排版引擎

三个引擎都做「五线谱一行 ↔ 简谱一行」的对齐，区别在于简谱放在哪里：

- **`spaced`（默认，矢量插入空间）**
  在谱表之间选取空白横线，避开单个字形和矢量路径的边界，插入足够的空间放置简谱，并相应加长页面。
  若找不到完全空白的横线，会尝试在无墨迹的列之间改变切线高度，并提示校对；
  无法安全切分原谱时，保留原谱，把该行简谱附在文件末尾，继续处理其他行；报告标记为不完整并记录附页位置。
  小节线可可靠检测时，简谱按原谱小节位置对齐，优先调整符号之间的空白；
  长弧线挡住空白时，会在能保护其他图形和文字的情况下连续调整弧线穿过的空隙，或安全分离曲线与数字后分别排版；不能安全处理时仍会缩放并在报告中标注。
  空间仍不足时仅压缩相应小节，无法确定分隔线时保留整行并提示校对。
  每次同时写入 `<输出前缀>.insitu.report.json`，记录输入和输出哈希、每行的小节范围、声部生成状态及逐小节伸缩情况；
  `incomplete` 表示有行或声部未生成，`needs_review` 表示有对齐疑点或回退。
  `rendered` 仅表示排版流程完成，不能证明识谱或原谱转换准确。

- **`stamp`（矢量贴字）**
  逐行把简谱渲染成一张透明小图，裁掉空白后，用 `page.show_pdf_page(..., overlay=True)` 贴到该行五线谱下方的空白处。
  **不新增页面、不改变页面尺寸**，原谱的版式、页码、裁剪框全部保持原样。
  实测：5 页源谱 → 5 页原位简谱，页面仍为 595×842 pt，`placed 40/40 lines`（40 行谱表全部贴到）。

- **`strip`（插条并加长）**
  检测每一组五线谱系统（system）的位置，在其下方插入一条透明「简谱条」，同时把页面**加高**相应高度。
  好处是简谱永远有连续、完整的一块地方，不会被原谱的歌词或标注挤到；代价是页面变长。
  实测：5 页源谱 → 5 页原位简谱，页面宽度不变、高度按该页行数加高（842 → 1316~1452 pt）。

> 需要原有页面尺寸时选 `stamp`；需要更多行间空间时选 `spaced` 或 `strip`。

---

<a id="install"></a>

## 📦 安装

### 1. 本项目 + Python 环境

```bash
git clone https://github.com/starpacker/music_translator.git
cd music_translator

python -m venv .venv && source .venv/bin/activate      # 建议 Python 3.10+
pip install -r requirements.txt
```

#### 本机环境记录（2026-11-02）

历史运行记录使用 `/opt/conda/envs/sumisora/bin/python` 与 `/usr/bin/lilypond`；
本次检查时二者均已不存在，不能直接照搬旧路径。本次在当前容器重建的环境为
`/tmp/music_translator_env/bin/python`（Python 3.13，按本仓库 `requirements.txt` 安装）；
`/tmp` 和系统安装可能随容器重置而消失。重建方式：

```bash
python -m venv /tmp/music_translator_env
/tmp/music_translator_env/bin/pip install -r requirements.txt
sudo apt-get update && sudo apt-get install -y --no-install-recommends lilypond
```

运行时可直接使用 `/tmp/music_translator_env/bin/python jianpu.py ...`；新环境首次识谱
可能需要联网下载 Homr 模型。项目克隆到其他机器时建议使用上面的 `.venv` 安装方式。
若环境变量 `TMPDIR` 指向不存在的目录，LilyPond 无法生成 PDF；可在命令前指定
`TMPDIR=/tmp`。例如：

```bash
TMPDIR=/tmp /tmp/music_translator_env/bin/python jianpu.py test.pdf --mode both -o output/test_baseline/test
```

### 2. LilyPond（必需，系统级）

```bash
# Debian / Ubuntu
sudo apt install lilypond        # 需 2.22 及以上
# macOS
brew install lilypond
```

脚本在 `PATH` 里找不到 `lilypond` 时，可以用 `LILYPOND` 环境变量指定：

```bash
export LILYPOND=/usr/local/bin/lilypond
```

### 3. 内置 OMR 与排版核心

仓库现在包含维护过的 SumisoraOMR `core/` 和 `scripts/jianpu-ly.py`，无需另行克隆上游。
`homr_shim.py` 适配官方 pip 版 Homr；首次识谱需要下载模型权重。
需要测试其他上游版本时，可以用 `SUMISORA_ROOT` 显式覆盖内置核心。

只想排版时，用 `--mxl` 传入 MusicXML，跳过 OMR。

### 4. NVIDIA GPU（可选）

默认安装由 Homr 解析 ONNX Runtime 依赖。CPU 环境可直接运行。
NVIDIA CUDA 环境需要兼容的 CUDA、cuDNN 运行库，以及 GPU 版 ONNX Runtime：

```bash
pip uninstall -y onnxruntime onnxruntime-directml
pip install onnxruntime-gpu
python -c "import onnxruntime as rt; print(rt.get_available_providers())"
```

输出应包含 `CUDAExecutionProvider`。同一环境只安装一种 ONNX Runtime 后端。
`--gpu auto` 同时检测 CUDA 和 Windows DirectML；`--gpu off` 强制使用 CPU。
GPU 推理初始化失败时，识谱流程会按现有逻辑尝试回退 CPU，实际设备以日志为准。

---

<a id="usage"></a>

## 🚀 用法

### 串行处理整个目录

```bash
python batch.py input/吹管 -o output --mode both --engine strip --gpu auto
```

逐份处理目录内 PDF，输出 `<曲名>.jianpu.pdf`、`<曲名>.insitu.pdf` 和逐份日志。
批处理和单文件命令默认使用 `spaced` 引擎。
单份失败后继续处理其他输入，最后汇总并返回非零退出码。
中间识谱结果保存在 `output/work/<曲名>/`，再次运行会复用缓存；需要重新识谱时指定新的 `--work` 目录。

### 最常用的一条命令

```bash
python jianpu.py 曲笛.pdf
```

默认 `--mode both`，一次产出两份：

```
曲笛.jianpu.pdf     # 整页简谱
曲笛.insitu.pdf     # 原位简谱（spaced 引擎）
```

### 只要其中一种

```bash
# 整页简谱
python jianpu.py 曲笛.pdf --mode jianpu

# 原位简谱：保持原页码、原页面尺寸
python jianpu.py 曲笛.pdf --mode insitu --engine stamp

# 原位简谱：插条并把页面加长
python jianpu.py 曲笛.pdf --mode insitu --engine strip
```

### 跳过 OMR（已有 MusicXML）

```bash
python jianpu.py 曲笛.pdf --mxl 曲笛.musicxml --mode both
```

OMR 结果默认缓存在 `<输出目录>/work/<曲名>.musicxml`，输入内容、设备选择、Homr 版本及识谱处理代码一致时才复用；没有缓存元数据的旧缓存会重新识谱：

```
[omr] 复用已有结果 .../work/曲笛.musicxml
```

对于 Homr 的结果，单文件流程会将时间上相邻、同声部同音高且具有唯一配对曲线的音符保守地识别为延音线；原始 OMR 文件仍保存在 `work/<曲名>.omr/`。这是自动推断，存在歧义时应对照原谱；若需要直接使用指定 MusicXML，可用 `--mxl`。
完整同音延音才会转换为延音横线。和弦中只有部分音延续时，可匹配的音会用独立弧线连接；原位谱跨行时会在两端显示续接弧线。部分延音、端点不匹配及复杂跨行情况仍会提示校对，并在原位报告中列出小节。单独从 MusicXML 导出的 MIDI 会逐音连接可确认的延音；重复音高或端点不匹配等有歧义的情况仍需校对试听。转换流程生成 MIDI 时目前会移除装饰音，简谱仍保留它们。
校对文本中的 `@PT...` 用于保存逐音延音关系；修改相关音符时请同时核对这些标记。

整页简谱默认尝试从 PDF 中恢复位于小节起点的矢量四分音符速度标记。程序要求该小节音符的数量与谱表位置能够和 MusicXML 对应，或能确认多小节休止线及其数字；无法确认的标记不会写入。恢复结果是推断，需对照原谱校对；已有 MusicXML 速度不会覆盖。原始 MusicXML 不会修改，派生文件及逐项依据保存在 `work/<输出前缀>.pdf-tempos.musicxml` 和 `.pdf-tempos.json`。如不需要推断，可加 `--no-vector-tempos`。

默认还会写入 `work/<输出前缀>.pdf-noteheads.json`，记录支持的 PDF 矢量符头与 MusicXML 的小节位置对照，以及可唯一定位的多小节休止数字。对字体身份可验证的符头，还会核对全音符、二分音符和实心符头类别；能连接两个符干的矢量符杠只作为符杠数量的下界。位于音符右侧且能唯一归属的音乐字体点可作为附点的部分证据；没有检测到点不能证明没有附点。符头尺寸与识别出的装饰音属性不一致时只提示复核，因为提示音也可能使用小符头。该报告只提供部分证据，不验证升降号、休止符附点、连音符、单独符尾、其他字体和全局小节对齐，也不能证明完整节奏或识谱准确；不支持的谱面和无法确认的项目会明确标出。多小节休止数量记录在报告中，不写入折叠后的 MusicXML。可用 `--no-pdf-audit` 跳过此报告。

对于特定的 C/G 调号变化，如果 PDF 音乐字体、符头、调号字形和相邻系统的调号相互印证，程序会尝试将 MusicXML 中延迟到下一系统的调号移回原谱位置。两种输出使用派生的 `work/<输出前缀>.pdf-keys.musicxml`，原始 MusicXML 保持不变；推断依据和未采用的候选保存在 `.pdf-keys.json`，仍需对照原谱。仅支持少量可确认的情形；可用 `--no-vector-keys` 禁用。

对于带嵌入式 `EngraverFontSet` 的矢量 PDF，整页简谱也会尝试恢复少数可确认的力度字形，并用整小节音符位置核对其对应拍点。已有 MusicXML 力度不会覆盖，无法唯一定位的标记只记录在 `work/<输出前缀>.pdf-dynamics.json`，不自动写入；派生 MusicXML 为 `.pdf-dynamics.musicxml`。这仍是推断，其他字体及扫描谱尚未覆盖；可加 `--no-vector-dynamics` 禁用。

两种简谱会保留 MusicXML 中已识别的常见音符记号，包括重音、断音、保持音、延长记号，以及常见颤音和波音；记号跟随原音符，包括装饰音、和弦和延音续音。普通、尖角和方形延长记号按 MusicXML 的形状与朝上或朝下方向显示（未指定方向时按照标准默认为朝上）。其他未支持的形状、装饰音附带的升降号及其他未支持的记号会提示校对；延长记号的精确位置、颤音延长线等复杂细节仍需人工核对。这些记号能否识别仍取决于输入 MusicXML 和识谱结果。

整页简谱会保留 MusicXML 中起止位置能准确对应简谱事件的渐强、渐弱线；无法准确排版的线会提示校对。对于原 PDF 中独立的双线渐强渐弱记号，还会尝试用音符位置和小节边界推断其端点，写入派生的 `.pdf-wedges.musicxml`，报告保存在 `work/<输出前缀>.pdf-wedges.json`。跨行、分离线段或端点不明确的情况不会自动补全；已有 MusicXML 力度线也不会被覆盖。推断结果需对照原谱，可加 `--no-vector-wedges` 禁用。原位版保留原 PDF 的记号，不重复添加这些推断线。

MusicXML 在小节边界改变调号时，整页简谱和原位简谱会同时更新首调参考与数字；小节中途变调目前会提示校对，并延至下一小节切换。
单升、单降、重升和重降唱名优先保留 MusicXML 的音名拼写，例如 C 调中的降 B 显示为 `b7`、重升 F 显示为 `##4`。三重及更多升降唱名用等音代替并提示校对；微分音尚不能准确表示，也会提示校对。
普通音、装饰音及和弦支持最多五个高低八度点；超出 MIDI 音域的输入会提示试听需要校对。
整页简谱会显示 MusicXML 中识别到的拍号及变化；原位简谱保留原五线谱上的拍号。为适应不完整小节而使用的内部排版拍长不会作为拍号印出。

原位简谱沿用原五线谱的小节编号，不额外打印每行重新计数的小节编号。

可用以下命令逐小节比较 MusicXML 第一声部和简谱提取出的音高、时值、装饰音、休止符时值及受支持的音符演奏记号，并检查普通旋律连音线端点和简谱数字、升降号、八度点对应的音高：

```bash
python scripts/audit_jianpu_events.py output/work/曲笛.musicxml
```

该检查输出 JSON，发现差异时退出码为 1。它不比较原五线谱，连音和多声部等情况
可能需要人工解释，不能作为 OMR 准确率的证明。

### 指定输出与前缀

`-o` 在 `mode=both` 时是**前缀**，在单模式下是目标文件名。
推荐只给一个不带后缀的名字，避免出现 `.insitu.insitu.pdf` 这种重叠后缀：

```bash
python jianpu.py 曲笛.pdf -o out/曲笛              # → out/曲笛.jianpu.pdf + out/曲笛.insitu.pdf
python jianpu.py 曲笛.pdf --mode jianpu -o out/曲笛-简谱.pdf
```

### 完整参数

| 参数 | 说明 |
| --- | --- |
| `score` | 源五线谱 PDF（**矢量 PDF 最佳**，扫描件需先 OMR 成矢量或提高分辨率） |
| `-o, --output` | 输出文件；`mode=both` 时作为前缀。默认与输入同名同目录 |
| `--mode {jianpu,insitu,both}` | 产物类型，默认 `both` |
| `--engine {spaced,stamp,strip}` | 原位简谱的实现方式，默认 `spaced` |
| `--mxl` | 已有的 MusicXML，给出后跳过 OMR |
| `--work` | OMR 中间结果目录，默认 `<输出目录>/work` |
| `--title` | 简谱标题，默认取输入文件名 |
| `--gpu {auto,on,off}` | OMR 是否用 GPU，默认 `auto` |

---

<a id="how"></a>

## ⚙️ 原理

```
┌──────────────┐   Homr / Audiveris    ┌────────────┐
│ 五线谱 PDF   │ ────────────────────► │ MusicXML   │
└──────────────┘                       └─────┬──────┘
                                             │ music21 解析调号 / 拍号 / 时值
                                             ▼
                                     ┌────────────────┐
                                     │ 简谱文本 (jianpu)│
                                     └───────┬────────┘
                       jianpu-ly.py ─────────┤───────── LilyPond
                                             ▼
                        ┌────────────────────┴────────────────────┐
                        │                                         │
                   整页简谱 PDF                            逐行简谱切片
                   (mode=jianpu)                          (insitu 引擎)
                                                                │
                     ┌──────────────────────────────────────────┴───────────┐
                     │                                                      │
              stamp：检测每行五线谱的                      strip：检测每行五线谱系统，
              位置，矢量贴字到其下方                        插入透明简谱条，页面加高
              （页码 / 尺寸不变）                           （页面变长）
```

三段关键逻辑：

1. **OMR + 缓存** — `jianpu.py:omr_to_musicxml()`。产物落在 `work/` 下，元数据匹配时复用。
2. **MusicXML → 简谱文本** — `music21` 取出调号（首调）、拍号、每个音的时值与八度，
   拼成 `jianpu-ly.py` 能吃的文本。和弦音需要 `OctavesAfter` 头，否则 jianpu-ly 会报
   `Ambiguous octave marks`；`\time` 则被故意丢掉，否则它会在每条五线谱下多印一个节拍记号字形。
3. **对齐** — OMR 给出的换行数常常和原谱对不上（实测 40 行谱表只标出 36 个换行）。
   `insitu/align.py` 用动态规划把「MusicXML 的小节序列」和「PDF 中真实的谱表几何」重新对齐，
   代价函数三项：
   - `_COST_MERGE = 1.0` — OMR 把两行并成了一行
   - `_COST_SPLIT = 1.2` — OMR 凭空造了一个换行
   - `_COST_COUNT = 0.4` — 该行小节数与预期不符

   `insitu/pdf_structure.py` 则从 MuseScore 导出的矢量 PDF 里读出每行谱表的上下沿与谱线，
   `insitu/vendor/score_align.py` 负责最终插入/贴字。

---

<a id="tuning"></a>

## 🔧 可调参数

### 原位贴字（`insitu/render_insitu.py`）

| 常量 | 默认 | 含义 |
| --- | --- | --- |
| `Y_OFFSET` | `20.0` | 简谱墨迹顶端距该行五线谱下沿的距离（pt）。调大 → 简谱整体下移 |
| `STAFF_SIZE` | `20` | jianpu-ly 字号，与原谱简谱字号保持一致 |
| `_LINE_WIDTH_PT` | `540` | 单行简谱渲染宽度（pt），超出会压缩换行 |

### 对齐代价（`insitu/align.py`）

| 常量 | 默认 | 含义 |
| --- | --- | --- |
| `_COST_MERGE` | `1.0` | OMR 把两行谱表合并成一行的代价 |
| `_COST_SPLIT` | `1.2` | OMR 多造出一个换行的代价 |
| `_COST_COUNT` | `0.4` | 每差一个小节的代价 |

调大某项 = 更不信任 OMR 在这方面的判断。

### 插条引擎（`insitu/vendor/score_align.py` 的 `Config`）

| 字段 | 默认 | 含义 |
| --- | --- | --- |
| `render_scale` | `4.0` | 简谱切片渲染倍率，越大越清晰、越慢 |
| `gap_above_pt` / `gap_below_pt` | `9.0` / `9.0` | 谱表与简谱条上下的留白 |
| `insert_height_pt` | `40.0` | 每条简谱条的基础高度 |
| `jianpu_first_left_pt` | `80.0` | 每页第一行简谱的左缩进（给行首符号留位） |
| `jianpu_other_left_pt` | `29.5` | 其余行简谱的左缩进 |
| `page_margin_pt` | `26.0` | 页面左右安全边距 |
| `jianpu_gap_pt` | `6.0` | 相邻两行简谱之间的间距 |
| `system_label_clearance_pt` | `31.5` | 避开行首系统标记（括号/谱号）所需空间 |
| `final_system_search_pt` | `46.0` | 最后一行系统的搜索半径 |
| `row_band_pt` | `30.0` | 谱表行高扫描带 |
| `cell_pad_pt` | `0.8` | 单元格左右内边距 |
| `cell_top_pad_pt` / `cell_bottom_pad_pt` | `2.5` / `21.5` | 单元格上下内边距（下边距含八度点与减时线） |

---

## 🧪 实测数据

以仓库自带的 `曲笛.pdf` 为例（5 页 A4 矢量谱，595×842 pt）：

| 项目 | 结果 |
| --- | --- |
| OMR 识别 | 193 小节，4/4 拍，调号 1=D |
| 整页简谱 | 3 页 A4（595.3×841.9），62 KB |
| 原位简谱 `stamp` | **5 页**（595×842，与原谱同尺寸），`placed 40/40 lines` |
| 原位简谱 `strip` | **5 页**（每页按行数加高到 1316–1452 pt） |
| 对齐 | 谱面 40 行谱表，OMR 只标出 36 个换行，DP 重排后 40 行全部贴上 |

---

<a id="limits"></a>

## ⚠️ 已知限制

- **主要面向单乐器旋律谱**。同一 MusicXML 分部中的显式多声部会分别排版，原位模式默认使用第一个乐器分部；合唱、钢琴大谱表及复杂复调仍需人工校对。
- **连音符**。支持常见的单层三连音及其他单层比例；嵌套连音符、跨小节分组及无法用排版网格精确表示的时值需要校对。
- **OMR 精度决定上限**。识别错音、错时值会原样带到简谱里；原位简谱能让你一眼看出错在哪一行，这正是它的用处。
- **`strip` 引擎的行检测是针对 MuseScore 矢量导出调的**。其它软件（Finale、Sibelius、Dorico）导出的 PDF 可能需要微调 `score_align.Config` 里的边距参数；`stamp` 引擎依赖 `pdf_structure.py` 读矢量谱线，纯扫描件请先转矢量。
- **跨页 / 跨行连音线**会在原位谱行首、行尾添加续接标记；部分和弦延音也会在可匹配的端点显示续接弧线。复杂跨行情况仍需核对。
- **调号**按首调（`1=D` 之类）标注；临时升降号会在音符前带 `#`/`b`。

---

## 🗂 目录结构

```
music_translator/
├── jianpu.py                 # 唯一入口：PDF → 整页简谱 / 原位简谱
├── batch.py                  # 目录级串行批处理，两种 PDF 与逐份日志
├── core/                     # 内置 SumisoraOMR 核心及本地修复
├── scripts/jianpu-ly.py       # 含合音八度与升降号修复的排版预处理器
├── tests/                    # 回归测试和固定输入样本
├── _env.py                   # 环境探测：SUMISORA_ROOT、lilypond、python
├── homr_shim.py              # SumisoraOMR homr_runner ↔ pip 版 homr 的适配层
├── requirements.txt          # 新管线依赖
├── insitu/
│   ├── align.py              # MusicXML 小节序列 ↔ PDF 谱表几何 的动态规划对齐
│   ├── pdf_structure.py      # 从矢量 PDF 读出每行谱表的位置、谱线、小节线
│   ├── render_insitu.py      # stamp 引擎：逐行渲染简谱并矢量贴字
│   └── vendor/score_align.py # strip 引擎：检测系统、插入简谱条、加长页面
├── tools/make_assets.py      # 生成 assets/logo 与 assets/social-preview
├── assets/                   # 图标与社交预览图（SVG + PNG）
├── docs/screenshots/         # README 用截图
├── 曲笛.pdf / 二胡.pdf        # 示例输入
└── prototype_cv/             # 旧版：纯 CV 模板匹配实现（见下文）
```

<a id="legacy"></a>

## 🧱 旧版实现：`prototype_cv/`

仓库里还留着第一代实现 —— 不依赖 OMR，直接用 OpenCV 在**图片**上做模板匹配：
去谱线 → 检测谱系 → 音符头模板匹配 → 符干/符杠追踪 → 估算时值 → 输出简谱文本或图片。

```bash
cd prototype_cv
pip install opencv-python numpy Pillow
python main.py ../input/erhu_p1.png
```

它的实测精度（`prototype_cv/evaluate.py`）：

| 测试曲目 | 小节 | 事件 | 音高 |
| --- | --- | --- | --- |
| 钢琴 K.331 | 20/20 (100%) | 111/111 (100%) | 213/213 (100%) |
| 二胡 8 页 | 93% | 97% | 99% |
| 曲笛 5 页 | 81% | 86% | 90% |

**为什么不继续用它：** 模板匹配对字体、扫描质量、倾斜非常敏感，换个出版社就得重做模板；
而且它输出的是文本/位图，无法把简谱精确地贴回原谱的每一行。
现在的 `jianpu.py` 走 OMR → MusicXML 路线，通用性和版式还原都更好。`prototype_cv/` 保留作对照与参考。

---

## 📄 许可

本项目以 **GNU Affero General Public License v3.0** 发布，见 [LICENSE](LICENSE)。
选 AGPL 是因为运行期依赖链里有 AGPL 组件（SumisoraOMR、Homr、PyMuPDF）。

第三方组件的版本、许可与本地修改说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

---

<p align="center">
  <sub>如果这个项目帮到了你，点个 ⭐ 让更多人看到。</sub>
</p>

## 本次识谱修复与验证

- 保留合音中每个音的八度和临时升降号，隐藏简谱辅助谱表的加线。
- 不因 OMR 拍号不可靠而截断实际音符。
- 多小节休止重排出现音符槽位数量歧义时抛错，避免猜测性补休止。
- 对 Maestro 矢量 PDF 中无调号、4/4、可明确判断的全音符合音，按谱面小节几何恢复音高和归属。其他节奏、字体、调号仍依赖 OMR；这不是全谱正确性的保证。

```bash
python -m unittest discover -s tests -v
```

测试使用 `tests/fixtures/` 中的固定样本，不依赖本机批处理缓存。
