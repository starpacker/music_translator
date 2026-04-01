# Design: Note Unit Refactor — 逐音符识别架构

**Date:** 2026-04-01
**Status:** Approved

## 1. 动机

现有 pipeline 的时值识别和和弦分组是分离的后处理步骤：先检测音符头，再按 x 坐标分组和弦，再用间距比例法估算时值。这导致两个问题：

1. **和弦分组不可靠**：垂直对齐但属于不同声部的音符会被错误合并
2. **时值靠猜测**：符杠检测不一致时，依赖间距比例启发式后备，无物理依据

**核心改变**：从每个音符头出发追踪符干，用符干作为纽带，同时解决和弦分组（共享符干 = 和弦）和时值识别（符干端部形态 = 时值）。

## 2. 设计原则

- 每个音符的所有信息（音高、升降号、时值）在进入格式化之前就已完全确定
- 和弦由物理连接（共享符干）决定，不靠 x 坐标近似
- 时值由符干端部的实际形态决定，不靠间距猜测
- 在现有代码基础上重构，保留能用的模块

## 3. 模块变更总览

### 保留（不改或微调）

| 模块 | 作用 | 改动 |
|------|------|------|
| `staff_removal.py` | 谱线提取 | 不改 |
| `pitch_detection.py` | 谱系检测、大谱表配对、y→音高映射 | 不改 |
| `template_matching.py` | 音符头检测 | 不改 |
| `symbol_detection.py` | 小节线、升降号、休止符检测 | 删除 `detect_note_durations()` 和 `_count_beams_improved()` |
| `note_assignment.py` | 分配音符到高/低音谱 | 不改 |
| `jianpu_formatter.py` | 简谱格式化 | 简化：直接使用 NoteUnit 中的 pitch/accidental/duration |
| `evaluate.py` | 精度评估 | 不改 |

### 新增

| 模块 | 作用 |
|------|------|
| `stem_tracking.py` | 从音符头出发追踪符干，返回符干方向、端点、长度 |
| `note_unit.py` | 构建音符单元：基于符干的和弦分组 + 时值识别 + 小节分割 |

### 删除

| 模块 | 原因 |
|------|------|
| `chord_grouping.py` | 被 `note_unit.py` 替代 |
| `duration_estimation.py` | 被 `note_unit.py` 替代 |

## 4. 新数据流

```
1. staff_removal       → (staff_lines, music_symbols, binary)
2. pitch_detection     → systems, grand_staff_pairs
3. template_matching   → noteheads [(x, y, w, h, score)]
4. symbol_detection    → barline_xs, global_accidentals, rests
5. note_assignment     → treble_notes, bass_notes (含 clef/system/pair_idx)
6. stem_tracking (新)  → 每个音符的 StemInfo
7. note_unit (新)      → NoteUnit 列表 (含和弦分组 + 时值)
8. 升降号关联          → NoteUnit.notes[].accidental 填充
9. 小节分割            → 各小节的 [NoteUnit + Rest] 列表
10. jianpu_formatter   → 简谱文本 + 可视化
```

## 5. `stem_tracking.py` 详细设计

### 输入
- `music_symbols`: 去除谱线后的二值图（在此图上追踪，避免谱线干扰）
- `note`: 单个音符头 dict（含 x, y, w, h, y_center）
- `dy`: 谱线间距

### 算法

**Step 1 — 确定符干 x 坐标**：
- 以音符头中心 `cx = x + w//2` 为基准
- 在 `cx ± dy*0.3` 范围内的每一列，计算音符头上方/下方 `dy*2` 范围内的白色像素密度
- 取密度最高的列作为 `stem_x`

**Step 2 — 向上追踪**：
- 从音符头上边缘 `y` 向上扫描
- 搜索窗口宽度 = `max(3, int(dy * 0.15))` 像素
- 每行检查窗口内白色像素比例
- 当连续 `int(dy * 0.3)` 行密度低于阈值时，符干结束
- 记录上端点 `stem_top_y`

**Step 3 — 向下追踪**：
- 同上，从音符头下边缘 `y + h` 向下
- 记录下端点 `stem_bot_y`

**Step 4 — 判断方向**：
- 上方长度 = `y_center - stem_top_y`
- 下方长度 = `stem_bot_y - y_center`
- 较长方向 = 符干方向 (`'up'` 或 `'down'`)
- 最短符干长度阈值 = `dy * 1.5`，低于此 → `stem_dir = None`（无符干，全音符候选）

### 输出

```python
StemInfo = {
    'stem_x': int,                  # 符干 x 坐标
    'stem_dir': 'up' | 'down' | None,  # 方向
    'stem_tip_y': int,              # 符干端点 y（远离音符头的那端）
    'stem_length': float,           # 符干长度（像素）
}
```

### 追踪图选择

使用 `music_symbols`（已去除谱线），不使用 `binary`（含谱线）。原因：谱线穿过符干区域会干扰密度计算，在去线图上符干是干净的垂直线段。

## 6. `note_unit.py` 详细设计

### 6.1 和弦分组

**输入**：音符列表（已含 StemInfo）

**规则**：两个音符属于同一和弦，当且仅当同时满足：
1. `|stem_x_a - stem_x_b| < dy * 0.5`
2. `stem_dir` 相同（都 up 或都 down）
3. 在同一个 `system` 上

**流程**：
1. 按 `stem_x` 排序
2. 贪心扫描：当前音符与组内第一个音符满足上述 3 条件 → 加入组，否则新建组
3. 组内按 `y_center` 降序排列（低音在前，与现有格式一致）

**无符干音符**（`stem_dir = None`）：
- 单独成一个 NoteUnit，时值由符头空心/实心决定

### 6.2 时值识别

**空心符头判断**（保留现有逻辑）：
- 音符头中心区域 `dy * 0.2` 半径内的填充度
- 填充度 ≤ 0.4 → 空心

**无符干**：
- 空心 → 全音符 (4.0)
- 实心 → 异常情况，默认四分音符 (1.0)

**有符干，空心** → 二分音符 (2.0)

**有符干，实心** → 分析符干端部：

**符杠检测**（beamed notes）：
```
ROI 区域:
  - y 范围: stem_tip_y ± dy * 0.5（符干端点附近）
  - x 范围: stem_x ± dy * 1.5
在 music_symbols 图上取 ROI
水平投影分析:
  - 投影值归一化到 [0, 1]
  - 密度 > 0.4 且宽度 > dy * 0.8 的水平带 = 符杠
  - 符杠厚度验证: dy * 0.08 ~ dy * 0.45
计数:
  - 0 条符杠 → 检查符尾
  - 1 条 → 八分音符 (0.5)
  - 2 条 → 十六分音符 (0.25)
  - 3+ 条 → 三十二分音符 (0.125)
```

**符尾检测**（isolated flag，无符杠时）：
```
如果符杠数 = 0:
  检查符干端点右侧的像素密度
  ROI: stem_tip_y ± dy * 0.8, stem_x 到 stem_x + dy * 1.0
  如果右侧密度 > 阈值 → 有符尾
    符尾形态分析（多少个弯曲）→ 八分(0.5) 或十六分(0.25)
  否则 → 四分音符 (1.0)
```

**和弦内时值**：
- 和弦共享符干，因此共享时值
- 取和弦内任一音符（实际上取组的 stem_tip 分析即可，只需做一次）

### 6.3 NoteUnit 数据结构

```python
NoteUnit = {
    'notes': [                  # 和弦 = 多个，单音 = 1个
        {
            'pitch': "3'",      # 简谱音高（y_to_jianpu 的结果）
            'accidental': '#',  # '#' / 'b' / None（升降号关联后填充）
            'x': 1965,
            'y_center': 693,
            'clef': 'treble',
            'system': [...],
            'pair_idx': 0,
        },
    ],
    'duration': 0.5,            # 时值（拍数）
    'stem_dir': 'up',           # 'up' / 'down' / None
    'stem_x': 1970,             # 符干 x
    'x': 1965.0,                # 音符头平均 x（用于排序和小节分割）
}
```

### 6.4 小节分割

与现有逻辑相似：
1. 将 NoteUnit 列表和休止符列表合并为事件列表，按 x 排序
2. 用小节线 x 坐标切分为各小节
3. 复用现有的休止符清洗逻辑（去重、去除与和弦重叠的假休止符）

## 7. `jianpu_formatter.py` 简化

### 改动

- `format_note()`: 不再内部调用 `y_to_jianpu()`，直接读取 `note['pitch']` 和 `note['accidental']`
- `format_event()`: 不再对和弦内音符做时值投票，直接使用 `NoteUnit['duration']`
- `format_measure()`: 删除弱起小节检测逻辑（如需保留，移到 `note_unit.py`）
- 升降号小节内持续逻辑保留不变

## 8. `main.py` pipeline 改动

### 删除的步骤
- `detect_note_durations()` 调用
- `group_into_chords()` 调用
- `estimate_durations_by_spacing()` 调用

### 新步骤（替换位置）

```python
# Step 6: 符干追踪
from stem_tracking import track_stem
for note in treble_notes + bass_notes:
    note['stem'] = track_stem(music_symbols, note, dy)

# Step 7: 构建音符单元
from note_unit import build_note_units, segment_into_measures
treble_units = build_note_units(pair_treble, accidentals_map, dy)
bass_units = build_note_units(pair_bass, accidentals_map, dy)

# Step 8: 小节分割
treble_measures = segment_into_measures(treble_units, pair_t_rests, barlines, dy)
bass_measures = segment_into_measures(bass_units, pair_b_rests, barlines, dy)

# Step 9: 格式化（使用简化后的 formatter）
```

### 删除的 import
```python
# 删除:
from chord_grouping import group_into_chords, segment_into_measures
from duration_estimation import estimate_durations_by_spacing
```

## 9. 测试标准

- 重构后，`evaluate.py` 对第 1 行的精度必须保持 **10/10 (100%)**
- 所有 10 个小节（5 treble + 5 bass）exact match
- 如果新的符干追踪法在某些音符上给出与旧方法不同的时值，需逐一验证正确性

## 10. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 符干在 music_symbols 图上被部分擦除（去谱线时误伤） | 同时在 binary 图上做辅助验证 |
| 符杠连接多个和弦，端部形态分析区域重叠 | 符杠检测区域限制在 stem_tip 附近 ±dy*0.5，不会跨到相邻音符 |
| 空心音符的符干较细，追踪可能中断 | 放宽追踪阈值，或用 binary 图辅助 |
| 符尾（flag）形态复杂，检测困难 | 先实现符杠检测（覆盖大多数情况），符尾作为第二优先级 |
