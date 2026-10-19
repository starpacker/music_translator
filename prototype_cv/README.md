# Music Score to Jianpu Converter

基于 OpenCV 计算机视觉的五线谱转简谱工具。支持钢琴大谱表 (Grand Staff) 和单行谱 (二胡、笛子等)。
支持 PDF 直接输入、可指定调性 (`--key`，1=C / 1=G / …)、以及跨音符头的检测（x 符头）。

测试精度：
- **钢琴** Mozart Turkish March K.331（手写 ground truth）: **90%** 小节 / **95%** 事件 / **99%** 音高
- **二胡 + 曲笛 全部 13 页**（`run_tests.py`，对照 PDF 矢量精确 ground truth）:
  **38%** 小节 / **48%** 事件 / **56%** 音高 —— 这是当前 CV 管线的真实基线，
  后续每一轮修改都用 `run_tests.py` 复测。逐页数字见下文「评估与精度」。

---

## 目录

- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [Pipeline 总览](#pipeline-总览)
- [各阶段算法详解](#各阶段算法详解)
- [简谱记号约定](#简谱记号约定)
- [评估与精度](#评估与精度)
- [已知局限](#已知局限)

---

## 快速开始

### 依赖

```
Python >= 3.10
opencv-python (cv2)
numpy
Pillow
```

```bash
pip install opencv-python numpy Pillow
```

### 运行

```bash
cd prototype_cv

# 默认读取 ../input/piano_p1.png (钢琴)
python main.py

# 指定输入图片
python main.py ../input/erhu_p1.png

# 覆盖拍号 (如 3/4 拍 = 3.0)
python main.py score.png --bpm 3.0

# 精度评估
python evaluate.py
python evaluate.py ground_truth_2.md output/jianpu.txt

# 批量处理二胡 8 页
python batch_erhu.py
```

### 输出文件 (`output/` 目录)

| 文件 | 说明 |
|------|------|
| `jianpu.txt` | 简谱文本 (钢琴分高/低音，单行谱按行输出) |
| `jianpu_on_staff.png` | 五线谱原图 + 简谱标注对照图 |
| `jianpu_visual.png` | PIL 渲染的简谱排版图 (红色数字 + 减时线 + 八度点) |
| `jianpu_clean.png` | 纯简谱排版图 (仅钢琴模式) |
| `confidence.txt` | 逐小节置信度评分，低置信度标 `[!]` |

---

## 项目结构

```
prototype_cv/                          # 核心代码 (~8750 行)
├── main.py                            # 主 pipeline + 可视化 (2100 行)
├── config.py                          # 全局超参数配置 (268 行)
├── staff_removal.py                   # 谱线提取与去除 (43 行)
├── pitch_detection.py                 # 谱系检测 + 大谱表配对 + 布局检测 (348 行)
├── template_matching.py               # 音符头检测 (678 行)
├── symbol_detection.py                # 小节线/升降号/休止符/拍号/调号/连弧检测 (1280 行)
├── note_assignment.py                 # 音符分配到高/低音谱 (112 行)
├── stem_tracking.py                   # 符干追踪 (176 行)
├── note_unit.py                       # 和弦分组 + 符杠计数 + 时值估算 + 小节分割 (2257 行)
├── jianpu_formatter.py                # 简谱格式化 (含调号/升降号持续) (181 行)
├── jianpu_visual.py                   # PIL 简谱渲染 (357 行)
├── confidence.py                      # 置信度评分 (244 行)
├── evaluate.py                        # 自动评估 (361 行)
├── batch_erhu.py                      # 二胡多页批处理 (113 行)
├── ground_truth.md                    # 钢琴 ground truth
├── ground_truth_2.md                  # 二胡 ground truth
└── output/                            # 生成的输出文件 (gitignored)
```

---

## Pipeline 总览

```
输入图片
  │
  ▼
1. 谱线提取 (staff_removal)
  │  → staff_lines, music_symbols, binary
  ▼
2. 谱系检测 + 布局判断 (pitch_detection)
  │  → systems, dy, layout (grand_staff / single_staff)
  ▼
2a. 连弧检测与遮罩 (symbol_detection.detect_slur_arcs)
  │  → 从 music_symbols 中减去连线弧
  ▼
2b. 拍号检测 (symbol_detection.detect_time_signature)
  │  → beats_per_measure (支持中途变拍号)
  ▼
2c. 调号检测 (symbol_detection.detect_key_signature)
  │  → key_sig: {'type': '#'/'b', 'count': N, 'notes': [...]}
  ▼
3. 小节线检测 (symbol_detection.detect_barlines)
  │  → barlines_per_system (自适应阈值)
  ▼
4. 音符头检测 (template_matching)
  │  → all_notes (形态学 + 模板 + NMS，含空心符头)
  ▼
5. 音符分配 (note_assignment) [仅钢琴]
  │  → treble_notes, bass_notes
  ▼
6. 升降号检测 (symbol_detection)
  │  → accidentals_map
  ▼
7. 休止符 + 连音标记检测 (symbol_detection)
  │  → rests, tuplet_markers
  ▼
8. 符干追踪 (stem_tracking)
  │  → stem_dir, stem_tip, stem_length
  ▼
9. 和弦分组 + 时值估算 + 小节分割 (note_unit)
  │  含：符杠计数 (含厚带双beam检测)、比例间距法、
  │      连音覆盖、附点检测、多小节休止展开
  ▼
10. 简谱格式化 (jianpu_formatter)
  │  含：升降号持续、调号应用、还原号取消、和弦括号
  ▼
输出文本 + 可视化图片 → output/
```

---

## 各阶段算法详解

所有像素级参数均基于 `dy`（谱线间距）动态计算，无硬编码像素值。全部超参数集中在 `config.py`。

### 1. 谱线提取 (`staff_removal.py`)

1. 灰度化 → Otsu 二值化 → 反转（白前景/黑背景）
2. 水平形态学开运算（核长 = 图片宽度/30）→ 提取主谱线
3. 短水平核（宽度/100）→ 提取加线，减去过粗线段
4. 从 binary 中减去谱线 → `music_symbols`（纯符号图）

### 2. 谱系检测 (`pitch_detection.py`)

- 水平投影 → 峰值聚类 → 每 5 条等距线为一个谱系
- 布局检测：相邻谱系间隙判断大谱表 vs 单行谱
- 输出 `dy` = 平均谱线间距，全局参数基准

### 2a. 连弧检测 (`symbol_detection.detect_slur_arcs`)

- 在 music_symbols 上做连通域分析
- 筛选条件：宽 > 3dy、高 < 1.5dy、填充率 < 0.22、周长 > 1.2×宽
- 生成 arc_mask 从 music_symbols 中减去，防止下游误检

### 2b/2c. 拍号与调号检测

- **拍号**: 模板匹配数字 (2/4, 3/4, 4/4 等)，支持沿行中途变拍号
- **调号**: 在谱号右侧、拍号左侧区域检测升降号簇，输出受影响的音高列表
- 续页抑制：第 2 页起跳过谱号区域的假拍号检测

### 3. 小节线检测 (`symbol_detection.py`)

- 双策略：模板匹配 + 垂直形态学开运算
- 自适应阈值（每行谱独立计算）
- 间距过滤 + 音符头近邻惩罚

### 4. 音符头检测 (`template_matching.py`)

| 步骤 | 方法 | 作用 |
|------|------|------|
| Step 0 | 谱号/升降号排除 | 避免谱号边缘误检 |
| Step 1 | 椭圆形态学开运算 + 连通域 | 主检测，含和弦拆分 |
| Step 2 | 合成模板匹配（含加线） | 补充检测 |
| Step 3 | NMS + 去重 | 合并重复 |

### 5-7. 音符分配 / 升降号 / 休止符

- **分配**: 高低音谱中点分界 + 位置-得分联合过滤
- **升降号**: 多模板多尺度 (×0.7~1.35)，阈值 0.60，music_symbols 补充还原号
- **休止符**: 四分/八分模板 + 多级误检过滤

### 8. 符干追踪 (`stem_tracking.py`)

- 列密度扫描定位 `stem_x`
- 上下逐行追踪（允许间隙 0.3dy）
- 输出：方向 (up/down)、端点 y、长度

### 9. 和弦分组 + 时值 (`note_unit.py`)

**符杠计数** (`_count_beams`):
- 符干端点 ROI 内水平投影，屏蔽谱线位置
- 标准带：厚度 0.08~0.55dy → 单 beam
- 厚带检测：厚度 > max 且 ≤ 2×max → 两 beam 合并（十六分音符）
  - 需双侧 ink 验证 + 距离检查 + 音符头遮罩
- Staff-line 跨越合并 + gap 过滤 + music_symbols 恢复

**时值估算** (双策略):
1. 符杠法：beam_count → 基础时值
2. 比例间距法：x 间距按比例分配拍数

**附点检测**: 右侧小圆点 → 时值 ×1.5

**多小节休止**: 粗横线 + 数字 → 展开为 N 个空小节

### 10. 简谱格式化 (`jianpu_formatter.py`)

- 音高数字 + 八度后缀 (`'` / `,`)
- 升降号前缀 (`#` / `b`)，小节内持续
- 调号自动应用，还原号取消
- 和弦括号 `[1 3 5]`
- 时值后缀 `/2` `/4` `/3` `/6`

---

## 简谱记号约定

| 记号 | 含义 | 示例 |
|------|------|------|
| `1`-`7` | 基本音符 do-si (默认四分 = 1 拍) | `1` = do |
| `0` | 休止符 | `0` = 四分休止 |
| `'` | 高八度 (可叠加) | `1'` = 高音do |
| `,` | 低八度 (可叠加) | `1,` = 低音do |
| `#` | 升号前缀 | `#4` = 升fa |
| `b` | 降号前缀 | `b3` = 降mi |
| `[...]` | 和弦 | `[1 3 5]` = C 大三和弦 |
| `/2` | 八分音符 (半拍) | `1/2` |
| `/4` | 十六分音符 (1/4 拍) | `1/4` |
| `/3` | 八分三连音 (1/3 拍) | `1/3` |
| `/6` | 十六分三连音 (1/6 拍) | `1/6` |
| `.` | 附点 (×1.5) | `1.` = 附点四分 |
| `\|...\|` | 小节 | `\|1 2 3 4\|` |

---

## PDF 输入与矢量 Ground Truth

样例 PDF（`二胡.pdf` / `曲笛.pdf`）是 Sibelius 导出的**矢量** PDF：谱线、小节线、
符干、符杠均为矢量路径，符头/升降号/休止符/符尾均为内嵌 Opus 字体的字形。因此：

- `pdf_tools.py` — 把 PDF 光栅化（默认 288 DPI，与 `input/` 中 PNG 一致）供 CV 管线使用，结果缓存于 `output/pdf_cache/`；
- `pdf_truth.py` — 直接从矢量内容恢复精确乐谱内容（谱表、小节线、符头音高、
  升降号、附点、符干/符杠时值、休止符、多小节休止、中途变拍号、调号），
  不经过任何图像识别；已与人写 `ground_truth_2.md` 逐小节核对一致；
- `run_tests.py` — 每页跑 CV 管线并与矢量 truth 对比，输出小节/事件/音高精度和 diff。

字形解码表（渲染内嵌字形逐一验证）：`œ` 实心符头、`¿` x 符头、`w`/`˙` 全/二分符头、
`#`/`b` 升降号、`Œ`/`‰`/`Ó`/`∑` 四分/八分/二分/全休止、`J`/`j` 符尾、`™`/`˙` 附点、
`æ` 颤弓线、`>` 重音（忽略）。

## 调性 (1=X)

`jianpu_formatter` 输出的是「绝对音高以 1=C 记写」的字符串（调号/临时号都已解析成
`#`/`b` 前缀），因此 `jianpu_transpose.py` 可以在最后一步做**无损**重编号：

```
绝对 midi ──(指定主音)──> 新度数 + 八度点 + 升降号拼写
```

- `--key auto`（默认）: 由检测到的调号推主音（3 个升号 → 1=A）；
- `--key C|G|D|bB|#F|…`: 强制主音；无调号乐曲默认仍为 1=C，与历史 ground truth 兼容；
- 变化音拼写约定: `#1 #2 #4 b3 b6 b7`（常见简谱习惯）。

单元测试: `python test_jianpu_transpose.py`。

## 评估与精度

```bash
# 钢琴 (默认 ground truth)
python evaluate.py

# 二胡
python evaluate.py ground_truth_2.md output/jianpu.txt

# 全语料矢量 truth 回归
python run_tests.py
```

### 当前精度

手写 ground truth（`evaluate.py`）:

| 测试曲目 | 小节 | 事件 | 音高 |
|----------|------|------|------|
| 钢琴 K.331 | 18/20 (90%) | 106/111 (95%) | 208/211 (99%) |

矢量 ground truth（`run_tests.py`，2026-10 基线）:

| 页 | 小节 | 事件 | 音高 | 页 | 小节 | 事件 | 音高 |
|----|------|------|------|----|------|------|------|
| 二胡 1 | 79% | 88% | 91% | 曲笛 1 | 22% | 27% | 32% |
| 二胡 2 | 27% | 33% | 40% | 曲笛 2 | 19% | 35% | 39% |
| 二胡 3 | 53% | 60% | 69% | 曲笛 3 | 13% | 17% | 46% |
| 二胡 4 | 28% | 28% | 33% | 曲笛 4 | 37% | 46% | 59% |
| 二胡 5 | 0%* | 43% | 48% | 曲笛 5 | 38% | 49% | 49% |
| 二胡 6 | 77% | 86% | 91% | | | | |
| 二胡 7 | 62% | 74% | 77% | | | | |
| 二胡 8 | 33% | 57% | 59% | | | | |

\* 二胡 5 的低分主要是 CV 把单符杠组数成双符杠（时值 /2 → /4）以及调号升号
应用不一致；x 符头页面（曲笛 5、二胡 4）在加入 x 符头检测后已从 0% 提升。

### 已知的主要 CV 缺陷（下一轮修复目标）

1. **符杠计数偏多**: 部分页面单符杠被数成 2 条 → 八分音符变十六分音符
2. **调号/还原号误检**: 偶发假还原号取消调号升号
3. **小节分割**: 漏检小节线导致空小节 (`0 0`) 或事件错位
4. **附点节奏**: 附点八分+十六分组合偶尔丢附点

---

## 已知局限

1. **符杠计数**: 密集音符组的 beam 检测受谱线残留影响，厚带检测缓解但未完全解决
   （当前最高优先级缺陷，见上表）
2. **连音检测**: 仅支持 "3" 和 "6" 模板，无数字标记则无法检测
3. **休止符**: 仅四分/八分模板，全休止/十六分休止未覆盖
4. **图片要求**: 需 >= 150 DPI 清晰扫描件，倾斜 > 2-3° 会失败
5. **手写乐谱**: 不支持
6. **模板集**: 来源单一，不同出版社字体可能匹配失败
7. **x 符头**: 已支持 (`template_matching.find_x_noteheads`，含形状验证以排除
   符干×符杠交叉点与实心符头的误检)；带颤弓/重音叠加的 x 符头召回率仍偏低
