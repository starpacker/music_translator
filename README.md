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
  <a href="#modes">生成方式</a> ·
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
- **`insitu` —— 原位简谱。** 保留原版五线谱的每一行，把对应的简谱**排在每一行五线谱下方**。适合对照学习、校对、教学。
  每页输出都与原谱**同尺寸**；行间放不下时增加页数，绝不把一页拉长。简谱可一行对一行（`--layout single`），也可保持原字号自动换行（`--layout wrap`）。

整条流水线只有一个入口脚本：

```
五线谱 PDF ──OMR──► MusicXML ──┬──► 整页简谱 PDF   (mode=jianpu)
                              ├──► 原位简谱 PDF   (mode=insitu)
                              └──► 两者都要        (mode=both)
```

OMR 用的是 **Homr**（端到端光学乐谱识别，也可切到 Audiveris），
排版走 **music21 → jianpu-ly → LilyPond**，页面合成用 **PyMuPDF** 的矢量操作，
全部输出都是矢量 PDF 合成；`strip` 已并入默认的 `spaced`，不再栅格化或加长页面。

---

<a id="demo"></a>

## 🖼 效果

### 原位简谱（`--mode insitu`）

每一行五线谱下方，就是这一行的简谱。原谱的每个字、每条线都原样保留；需要腾空间时整行谱表连同记号一起下移，横向位置不变。

下图取自梆笛分谱 `梆笛.pdf`（2 页 A4 矢量谱，18 行谱表）的实测输出：

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

## 🧭 生成方式与排版模式

所有模式集中在这里。三个开关互相独立：

- **`--mode`**：出哪种产物（整页简谱 / 原位简谱 / 两者）。
- **`--layout`**：原位简谱里，一行五线谱对应几行简谱（`single` 或 `wrap`）。
- **`--engine`**：原位简谱用哪种 PDF 合成方式（一般不用改）。

| 产物 | 参数 | 文件 | 页面 |
| --- | --- | --- | --- |
| 整页简谱 | `--mode jianpu` | `<曲名>.jianpu.pdf` | A4 纵向 |
| 原位简谱（分页重排，默认） | `--mode insitu --engine spaced` | `<曲名>.insitu.pdf` | **每页与原谱同尺寸**；放不下时**增加页数** |
| 原位简谱（旧版叠印，legacy） | `--mode insitu --engine stamp` | `<曲名>.insitu.pdf` | 与原谱逐页同尺寸、同页数；放不下的行在报告中标记，可能重叠 |
| 两者 | `--mode both` | 两个文件 | 同上 |

### 原位简谱的两种排版（`--layout`）

| 模式 | 行为 | 适用 |
| --- | --- | --- |
| `single`（默认） | 一行五线谱（及每个声部）对应**恰好一行**简谱。检测到原谱小节线时按小节逐格对齐：优先调整符号之间的空白，空间仍不足才压缩该小节，字形高度不变。 | 对照读谱、逐行校对 |
| `wrap` | 简谱保持原字号（staff size 20，与整页简谱一致），放不下就**按小节边界换行**；一行五线谱可对应多行简谱，不做任何横向压缩。 | 音符密集、不想压缩时 |

`wrap` 下若**单个小节**本身就宽于可用行宽，渲染会明确报错（`LineTooWideError`），该行在报告中标记为未完成；
不会悄悄裁掉、缩小字号或加宽页面。`single` 也不改变字形高度，只压缩横向空白/小节宽度。

### 页面尺寸：只加页，不加长

每个输出页的 MediaBox 都等于对应源页的尺寸。插入简谱使行间变高时：

1. 原谱每个谱表连同它的记号（歌词、力度、连音线、速度文字、装饰音）作为**完整对象整体平移**，
   绝不按水平切线切开原谱；
2. 后续谱表整体下移；
3. 当前页放不下时，在**同尺寸的新页**继续，简谱若换行也可续到新页；
4. 报告中记录 `pages`（源页 ↔ 输出页）、`output_source_bounds` 和 `continuation_pages`。

**同一尺寸的连续源页会连续排满，不会留白。** 旧行为是「一个源页 → 一整组输出页」，
若某源页的最后一行刚好填满，下一源页就会从新的一页开始，导致上一输出页大半空白。
现在 `paginate()` 会把**尺寸相同的相邻源页合并成一条连续流**：上一页排到哪，下一行的谱表
和简谱就接着排在哪，只有在真的放不下时才开新页。因此输出页数可能少于「源页数 ×」，
但每一输出页的 MediaBox 始终等于某个源页的尺寸（A4 仍是 A4）。

> 例：`梆笛.pdf` 2 个 A4 源页、18 行谱表，输出为 3 个 A4 页；
> `pages = [(1,[1]), (2,[1,2]), (3,[2])]` — 第 2 张输出页同时承载源页 1 的末尾和源页 2 的开头。

> **为什么以前有“无法安全切分 → 附录页 / incomplete”的限制？**
> 旧 `spaced` 引擎要在行间找一条空白水平线，把页面切成上下两段再插入空间。歌词、连音线、速度记号
> 一旦跨在这条线上，切线就会切穿原谱；找不到安全切线时只能保留原谱，把该行简谱放进文末附录。
> 新引擎（`insitu/reflow.py`）不再切线：它把原谱的绘制对象按谱表分组整体搬移，
> 因此“切不开”的情况不再存在，附录页和 `no_safe_source_cut` 已从输出和报告中移除。
> 只有当整页内容被包在**一个无法拆分的多谱表 Form XObject** 里时，才会退化为矢量轮廓重放（大纲化文字，
> 报告中 `source_text_outlined: true`），外观不变但文字不可再选取。

### 三种 `--engine`

- **`spaced`（默认）**：按对象重排并分页，页面尺寸不变。报告状态：
  `rendered` / `needs_review`（对齐回退、小节数不符、延音线待校对等）/ `incomplete`（有谱表或声部未生成）。
- **`stamp`**：旧版叠印路径，只在原页已有空白处贴简谱，**不移动原谱、不加页**。空白不足时该行标记
  `insufficient_overlay_space` 并计入 `incomplete`。它无法保证不重叠，**不是主路线**；日常请用 `spaced`。
- **`strip`**：旧命令的兼容别名，等同 `spaced`；不再加长页面，也不再栅格化。

两种 `--layout` 都可用于 `spaced` 与 `stamp`。原位简谱每行的最后一个终止线统一显示为普通单小节线 `|`，
不打印终止双线（重复记号仍保留）。

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

本机 Conda 安装在 `/opt/conda`，当前 `conda env list` 只列出 `base`，没有旧记录中的
`sumisora` 环境。`base` 使用 Python 3.13.5；LilyPond 位于 `/usr/bin/lilypond`，版本 2.22.1。
这台服务器已经有项目运行环境 `/tmp/music_translator_env`。它是用 Conda 的 Python 创建的 `venv`，
所以不会显示在 `conda env list` 中；其中已安装本项目依赖。登录服务器后这样激活并运行：

```bash
conda deactivate                  # 如果提示符当前显示 (base)
source /tmp/music_translator_env/bin/activate
which python                      # 应显示 /tmp/music_translator_env/bin/python
```

激活后，在项目根目录 `/liuyutian/music_translator_refresh` 执行下面的四条常用命令。
四个参数 `--mode both --engine spaced --layout single` 就是默认值，写出来是为了让命令自解释；
`TMPDIR=/tmp` 必须带，否则 LilyPond 可能找不到可用的临时目录。

**1. 固定调 · 单个 PDF**（C 恒为 `1`，按音名标升降号）

```bash
TMPDIR=/tmp python jianpu.py input/01_梆笛.pdf \
  -o output/01_梆笛_fixed \
  --mode both --engine spaced --layout single --pitch-mode fixed
```

产出 `output/01_梆笛_fixed.jianpu.pdf` 与 `output/01_梆笛_fixed.insitu.pdf`。
`-o` 在这里是**文件名前缀**，不是目录。

**2. 固定调 · 整个文件夹**

```bash
TMPDIR=/tmp python batch.py input -o output_fixed \
  --mode both --engine spaced --layout single --pitch-mode fixed \
  --zip output_fixed/insitu_fixed.zip
```

`-o` 在这里是**输出目录**；每份曲目得到 `<曲名>.jianpu.pdf`、`<曲名>.insitu.pdf`、
`<曲名>.insitu.report.json` 和 `<曲名>.log`。`--zip` 会把本次成功生成的
`insitu.pdf` 打成 `output_fixed/insitu_fixed.zip`（平铺，失败的不入包）。
识谱缓存放在 `output_fixed/work/`，重复运行可直接复用。

**3. 首调 · 单个 PDF**（默认 `1=D` 之类，随调号移动）

```bash
TMPDIR=/tmp python jianpu.py input/01_梆笛.pdf \
  -o output/01_梆笛 \
  --mode both --engine spaced --layout single --pitch-mode movable
```

`--pitch-mode movable` 也是默认值，可省略；省略时行为相同。

**4. 首调 · 整个文件夹**

```bash
TMPDIR=/tmp python batch.py input -o output \
  --mode both --engine spaced --layout single --pitch-mode movable \
  --zip output/insitu.zip
```

这是最常用的批量命令：`output/` 下得到全部曲目的两种 PDF、逐份报告与日志，
以及一个 `output/insitu.zip` 方便一次下载全部原位简谱。

该环境位于 `/tmp`，容器重置后可能需要重建。重建时在项目根目录执行：

```bash
/opt/conda/bin/python -m venv /tmp/music_translator_env
/tmp/music_translator_env/bin/pip install -r requirements.txt
```

首次识谱可能需要联网下载 Homr 模型。项目克隆到其他机器时建议使用上面的 `.venv` 安装方式。
若环境变量 `TMPDIR` 指向不存在的目录，LilyPond 无法生成 PDF；运行前指定 `TMPDIR=/tmp`。

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
python batch.py input/吹管 -o output --mode both --layout single --pitch-mode movable --gpu auto
```

逐份处理目录内 PDF，输出 `<曲名>.jianpu.pdf`、`<曲名>.insitu.pdf` 和逐份日志。
批处理和单文件命令默认使用 `spaced` 引擎、`single` 排版；加 `--layout wrap` 可改为原字号换行。
单份失败后继续处理其他输入，最后汇总并返回非零退出码。
中间识谱结果保存在 `output/work/<曲名>/`，再次运行会复用缓存；需要重新识谱时指定新的 `--work` 目录。

加 `--zip` 可以把本次所有成功生成的 `insitu.pdf` 打成一个压缩包，省去逐个下载：

```bash
python batch.py input -o output --zip output/insitu.zip
```

压缩包内文件为平铺的 `<曲名>.insitu.pdf`；失败或未生成原位简谱的曲子不会被打包，也不会写入空的占位文件。归档先写成 `.zip.part` 再原子改名，中途中断不会留下看似完整的半截包。

### 最常用的一条命令

```bash
python jianpu.py 曲笛.pdf
```

默认 `--mode both`，一次产出两份：

```
曲笛.jianpu.pdf     # 整页简谱
曲笛.insitu.pdf     # 原位简谱（默认：原页尺寸，放不下自动加页）
```

### 只要其中一种

```bash
# 整页简谱
python jianpu.py 曲笛.pdf --mode jianpu

# 原位简谱：原字号，允许一行五线谱对应多行简谱
python jianpu.py 曲笛.pdf --mode insitu --layout wrap

# 原位简谱：只在原页空白内叠印，不移动原谱、不加页
python jianpu.py 曲笛.pdf --mode insitu --engine stamp
```

### 跳过 OMR（已有 MusicXML）

```bash
python jianpu.py 曲笛.pdf --mxl 曲笛.musicxml --mode both
```

### 简谱唱名模式

默认使用首调（`--pitch-mode movable`）：主音唱作 `1`，调号变化时数字随调性变化，并在需要处标出 `1=G` 等调号提示。
固定调使用 `--pitch-mode fixed`：C 始终为 `1`、D 为 `2`，按音名拼写显示升降号；调号变化不会改变数字所对应的音名。
两种模式都适用于整页简谱和两种原位排版。

```bash
python jianpu.py 曲笛.pdf --pitch-mode fixed       # 固定调，两种 PDF 都生成
python jianpu.py 曲笛.pdf --mode insitu --pitch-mode movable  # 首调原位简谱
python batch.py input/吹管 -o output --pitch-mode fixed  # 批量生成固定调简谱
python batch.py input/ -o output --pitch-mode fixed
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
| `--engine {spaced,stamp,strip}` | 原位简谱合成方式，默认 `spaced`；`strip` 为兼容别名 |
| `--layout {single,wrap}` | 原位简谱排版，默认 `single`（一行对一行）；`wrap` 保原字号自动换行 |
| `--pitch-mode {movable,fixed}` | 首调（默认）或固定调（C=1） |
| `--mxl` | 已有的 MusicXML，给出后跳过 OMR |
| `--work` | OMR 中间结果目录，默认 `<输出目录>/work` |
| `--title` | 简谱标题，默认取输入文件名 |
| `--gpu {auto,on,off}` | OMR 是否用 GPU，默认 `auto` |
| `--no-vector-slurs` | 跳过「从 PDF 连奏线重建 slur」，改用 MusicXML 原有标记 |

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
              spaced（默认）：按谱表把原谱                  stamp：检测每行五线谱的
              绘制对象整体重排，在其下方                    位置，矢量贴字到其下方
              加简谱；放不下就加同尺寸新页                   （页码 / 尺寸不变）
```

三段关键逻辑：

1. **OMR + 缓存** — `jianpu.py:omr_to_musicxml()`。产物落在 `work/` 下，元数据匹配时复用。
2. **MusicXML → 简谱文本** — `music21` 取出调号（首调）、拍号、每个音的时值与八度，
   拼成 `jianpu-ly.py` 能吃的文本。和弦音需要 `OctavesAfter` 头，否则 jianpu-ly 会报
   `Ambiguous octave marks`；`\time` 则被故意丢掉，否则它会在每条五线谱下多印一个节拍记号字形。
3. **对齐** — OMR 给出的换行数常常和原谱对不上（实测 `梆笛.pdf` 有 18 行谱表，OMR 只标出 17 个换行）。
   `insitu/align.py` 用动态规划把「MusicXML 的小节序列」和「PDF 中真实的谱表几何」重新对齐，
   代价函数三项：
   - `_COST_MERGE = 1.0` — OMR 把两行并成了一行
   - `_COST_SPLIT = 1.2` — OMR 凭空造了一个换行
   - `_COST_COUNT = 0.4` — 该行小节数与预期不符

   `insitu/pdf_structure.py` 则从 MuseScore 导出的矢量 PDF 里读出每行谱表的上下沿与谱线；
   `insitu/reflow.py` 把每行谱表连同记号整体搬到同尺寸的新页，`insitu/line_layout.py` 负责
   一行简谱的 `single` 压缩对齐或 `wrap` 换行。

   **对齐是按源页逐页进行的。** 一个谱行不可能跨页，所以跨页分配小节在结构上就是错的：
   如果某一页的 OMR 少识别了几行，全局 DP 会从上一页“偷”小节来补齐，把上一页的音乐印到
   下一页的谱表下面。现在 `align_from_files()` 先按源页切分两个序列，再各自做 DP；
   某一页的识别缺口只留在该页，并在报告里记 `pdf_measures != musicxml_measures`。

   多页合并时还要显式补上页边界：Homr 的单页输出常常不给第一小节写 `<print>`，
   若合并时不留标记，上一页最后一行会和本页第一行粘成一行（`_merge_homr_musicxml_pages()`
   因此会在后续每一页的首小节写入 `new-page="yes"`）。

4. **连奏线（slur）** — Homr 只写 `<slur type="start">`，从不写 `stop`，所以简谱里的
   连奏线会一路延长到最后一个小节。`core/omr/vector_slurs.py` 改为**直接量取矢量 PDF 里的
   连奏线**：slur 在 PDF 里是 `type='fs'`（既填充又描边）的闭合曲线，恰好含 2 个三次贝塞尔项，
   端点就是 `items[0][1]` 与 `items[0][4]`；再把这些端点吸附到最近的符头，换算成
   MusicXML 的小节/音符索引，重写 `<slur>` 标记（原始 MusicXML 只在 `work/` 下改写副本）。
   报告 `*.pdf-slurs.json` 记录 `status`（`inferred` 全部解析 / `partial` 部分解析、残缺曲线已记入
   `problems` 但仍采用已解析的 / `unverified` 一条都没解析成 / `no_slurs` 原谱没有曲线）、`curves`
   数量以及 `fragment_count_mismatch` / `orphan_fragment` / `unresolved_endpoint` 等问题。
   若不想要这一步，加 `--no-vector-slurs`。

   > 有些 MuseScore 导出的音乐字体**没有字体名**（PDF 里报告空字符串），
   > 只按字体名匹配会漏掉全部符头，导致所有连奏线都 `unresolved_endpoint`。
   > 现在空字体名也会检查，因为那些专用于符头的码位（`œ ˙ w Ï ú`）不会出现在正文字体里。

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

### 分页重排（`insitu/reflow.py`）

| 常量 | 默认 | 含义 |
| --- | --- | --- |
| `paginate(..., gap=9.0)` | `9.0` | 谱表/简谱之间的最小竖向间距（pt） |
| `margin` | `min(30, 页高×6%)` | 页面上下安全边距 |
| `header_limit` | 首谱表上沿 − max(38, 行距×7) | 高于此线的对象视为页眉（标题、速度记号等） |
| `footer_limit` | max(末谱表下沿+38, 页高−60) | 低于此线的对象视为页脚（页码等） |

页眉只在源页对应的**第一张**输出页出现，页脚在每张输出页重复；每个源谱表及其记号是
不可分割的搬运单位。

### 分行与对齐（`insitu/line_layout.py`）

| 行为 | `single` | `wrap` |
| --- | --- | --- |
| 简谱行数 | 始终 1 行 | 按小节边界 1…N 行 |
| 字号 | 不变 | 不变 |
| 横向处理 | 只用水平压缩：先吃符号间空白，仍不够才把**单个小节**横向压窄；字形高度不变。逐格对齐的结果仍记 `aligned`，被压窄的小节另记在 `compressed_measure_count` / 逐小节 `placement` 里，不算回退 | 不压缩；单小节超宽直接报 `LineTooWideError` |
| 小节线对齐 | 检测原谱小节线后逐格对齐 | 不做小节对齐 |
| 放置结果 | `aligned`；无可用小节线时退为 `line`（自然宽度）或 `scaled`（整行等比压到行宽，属真正的回退） | `natural` |

---

## 🧪 实测数据

以仓库自带的 `02_曲笛1-2.pdf` 为例（2 页 A4 矢量谱，595×842 pt）：

| 项目 | 结果 |
| --- | --- |
| 原位简谱 `spaced`（默认） | **4 页 A4**（每页 595.3×841.9，与原谱同尺寸），17/17 行全部贴上 |
| 对齐 | 17 行谱表逐页 DP 对齐，`alignment_mismatch_count = 0`，`placement_fallback_count = 0` |

`梆笛.pdf`（2 页 A4，18 行谱表，110 小节）实测：

| 项目 | 结果 |
| --- | --- |
| 原位简谱 `spaced` | **3 页 A4**；`pages = [(1,[1]), (2,[1,2]), (3,[2])]`，连续排满不留白 |
| 对齐 | 18/18 行贴上，`alignment_mismatch_count = 0` |
| 连奏线 | 从 PDF 矢量曲线重建 15 条 slur，全部吸附到符头 |

---

<a id="limits"></a>

## ⚠️ 已知限制

- **主要面向单乐器旋律谱**。同一 MusicXML 分部中的显式多声部会分别排版，原位模式默认使用第一个乐器分部；合唱、钢琴大谱表及复杂复调仍需人工校对。
- **连音符**。支持常见的单层三连音及其他单层比例；嵌套连音符、跨小节分组及无法用排版网格精确表示的时值需要校对。
- **OMR 精度决定上限**。识别错音、错时值会原样带到简谱里；原位简谱能让你一眼看出错在哪一行，这正是它的用处。
  若 Homr 整段漏识（例如 `05_中音笙1-2.pdf` 第 2 页只输出上半页 4 行），缺失的音乐**不会**从别处挪来填补：
  该页剩余谱表按 `line`（自然宽度）放置并在报告中逐行记 `五线谱 N 格，识谱分配 M 小节`；
  若某行一个小节都没分到，该行标 `no_measures_assigned`、状态降为 `incomplete`，其余情况为
  `needs_review`。这是 OMR 的缺口，不是排版错误，也绝不会从别的页挪小节来填补。
- **原位简谱依赖矢量 PDF**：`pdf_structure.py` 要从矢量谱线读每行位置，纯扫描件请先 OMR 成矢量。被整体包进单个多谱表 Form XObject 的页面会退化为矢量轮廓（文字不可选）。
- **页数可能增加**：原位简谱保持每页原尺寸，简谱需要更多空间时新增同尺寸页面，因此输出页数可能多于原谱；页码与原谱的对应关系写在 `.insitu.report.json` 的 `pages` 中。
- **`single` 的横向压缩下限**：符号间空白用尽后只压缩该小节；若单行仍远宽于行宽，`wrap` 更合适。
- **跨页 / 跨行连音线**会在原位谱行首、行尾添加续接标记；部分和弦延音也会在可匹配的端点显示续接弧线。复杂跨行情况仍需核对。
- **连奏线来自 PDF 矢量层**，不是 OMR 的猜测：`*.pdf-slurs.json` 若为 `partial` 或带有
  `unresolved_endpoint` / `fragment_count_mismatch` / `orphan_fragment`，说明有曲线没能吸附到符头；
  已解析的连奏线照常采用，残缺的那条不会退化成 OMR 那种「有头无尾」的长线。只有 `unverified`
  （一条都没解析成）才整份退回 MusicXML 原有标记——而 Homr 的原标记常常只有 `start` 没有 `stop`。
  扫描件没有可用矢量曲线，也只能退回原标记。
- **调号**默认按首调标注（`1=D` 之类）；`--pitch-mode fixed` 固定 C 为 `1` 并按音名显示升降号。临时升降号会在音符前带 `#`/`b`。

---

## 🗂 目录结构

```
music_translator/
├── jianpu.py                 # 唯一入口：PDF → 整页简谱 / 原位简谱
├── batch.py                  # 目录级串行批处理，两种 PDF、逐份日志与可选 zip 打包
├── core/                     # 内置 SumisoraOMR 核心及本地修复
│   └── omr/vector_slurs.py   # 从 PDF 矢量连奏线重建 slur 端点
├── scripts/jianpu-ly.py       # 含合音八度与升降号修复的排版预处理器
├── tests/                    # 回归测试和固定输入样本
├── _env.py                   # 环境探测：SUMISORA_ROOT、lilypond、python
├── homr_shim.py              # SumisoraOMR homr_runner ↔ pip 版 homr 的适配层
├── requirements.txt          # 新管线依赖
├── insitu/
│   ├── align.py              # MusicXML 小节序列 ↔ PDF 谱表几何 的动态规划对齐
│   ├── pdf_structure.py      # 从矢量 PDF 读出每行谱表的位置、谱线、小节线
│   ├── render_insitu.py      # 原位简谱流程：逐谱表/声部渲染并合成报告
│   ├── line_layout.py        # single / wrap 分行与小节对齐
│   ├── reflow.py             # 原谱对象整体重排、同尺寸分页
│   └── vendor/score_align.py # 旧插条实现（已不由 CLI 调用）
├── tools/make_assets.py      # 生成 assets/logo 与 assets/social-preview
├── assets/                   # 图标与社交预览图（SVG + PNG）
├── docs/screenshots/         # README 用截图
├── input/                    # 示例输入（梆笛、曲笛、新笛、中音笙、唢呐、印象国乐胡琴…）
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
