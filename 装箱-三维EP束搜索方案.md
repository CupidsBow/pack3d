# 三维装箱：生产窗口 + EMS/EP + 分层束搜索

实现：`pack3d.cpp` + `rank.cpp`。几何默认 **EMS 与 EP 同时维护**，Expand 时把两类落点取并；可用 `ems` / `ep` 只开一种。搜索用 **按已装件数分层的 beam**。  
不是离线全局排列：一次最多看见缓冲里 12 件未装箱。流水线一次来一件；对当前窗口搜「再装 8 件」的最好状态并落实，再补槽。目标是装填率（已装体积），件数只作并列打破。

不维护高度图，不跑可采纳 A\*。

---

## 文件

| 文件 | 作用 |
|---|---|
| `pack3d.cpp` | 求解器（几何、窗口、束搜索） |
| `pack_state.hpp` | 状态 / 全局量，给求解器和 rank 共用 |
| `rank.hpp` / `rank.cpp` | `state_rank` + 28 维 `rank_features_vec` + 可选 JSONL dump |
| `train_rank.py` | 读 dump 做条数/特征校验（训练脚本入口） |
| `pack_viewer.html` | 三维查看器模板（占位 `__PACK_DATA__`） |
| `input.txt` | 算例：`W D H` 后每行一件 `w d h` |
| `快件高度 快件长度 快件宽度.txt` | 原始件尺寸（带表头） |

编译与运行（必须带 `-fopenmp`；数字是线程数，默认 1；可选 `both` / `ems` / `ep`，默认两者都开）：

```
g++ -O2 -std=c++17 -fopenmp -o pack3d pack3d.cpp rank.cpp
./pack3d 8 < input.txt > pack3d.out
./pack3d 8 --rank-dump rank.jsonl < input.txt > pack3d.out
python3 train_rank.py rank.jsonl
./pack3d 8 ep < input.txt > pack3d.out
./pack3d 8 ems < input.txt > pack3d.out
```

stdout：箱数 `N`，下一行各箱装填率（占 `W*D*H` 的百分比）。  
stderr：每箱的取件 / 每层搜索 / 落实状态，以及最后一行汇总。

---

## 设定

```
容器:     W × D × H，原点在左-后-下
件 i:     (w, d, h)，6 种轴对齐旋转
状态 u:   (eps, ems, placed, g)
          eps     = Extreme Point 列表，空箱 (0,0,0)
          ems     = 空最大长方体，空箱 = 整箱
          placed  = 已装位姿（碰撞、支撑、更新几何）
          g       = 已装体积
窗口:     g_open = 已到达、且未装进「上一箱」的件
          某状态的 pending = g_open \ 该状态已装
```

下一件的左后下角放在 **EMS 支承角 ∪ EP 点** 的去重并集上。同一落点只用 `residual_box` 打一次分，不把 EMS 盒尺寸和 EP 残腔混在一套 rank 里。

---

## 生产窗口（不是看见剩余全部 SKU）

流水线一次送来 1 件。缓冲最多 `BUFFER_CAP=12` 件未装箱，这也是可装窗口 `OPEN_CAP`。未装件数不会超过 12。  
补满缓冲后，对当前 12 件做 `CHUNK_K=8` 层束搜索（每层再装 1 件）；在搜出的状态里落实估价最好的那一个，丢掉其余 beam 分支，再按空槽补满缓冲，进入下一轮。

`g_typical_edge` = 已经到达件的 min-edge 中位数，至少 80。搭桥最小支承宽度用它；**不看未来件**。

每箱流程：

```
1. 填缓冲：pending < 12 且流未尽，一次取 1 件。
2. 流已尽：tight / 任意姿态一直装到装不动，封箱。
3. 否则对当前窗口束搜索最多 8 层（先 tight，没有再用任意姿态）。
4. 在达到本轮最多装件数的状态里，按 state_rank 落实一个，beam 收成这一份。
5. 本轮至少装成 1 件：回到 1 补槽。
6. 本轮一件都没装上：封箱。剩件带到下一空箱。
```

比空箱还大的件直接跳过。其余不丢弃。  
一箱墙钟 `TIME_LIMIT=90s`；超时后 `expand_state` 原样复制状态（看起来像「装不下」）。

---

## 合法放置

6 种旋转，但底面积小于最大面的 1/3 视为立不稳，该姿态不搜。位姿合法当且仅当：

- 整件在箱内
- 与已装件 AABB 不重叠
- 底面积 ≥ 最大面的 1/3（否则立不稳）
- 底面支承（`z=0` 视为地板；否则只认共面顶 `p.z+p.h == z`，不同高度不搭）。满足其一即可：
  - **坐实**：底面中心落在某一共面顶上，且重叠面积 ≥ 底面积的 60%
  - **搭桥**：至少两块互不重合的支承矩形；跨度主轴为 X 或 Y，底面中心两侧都有支承，且每一侧沿跨度方向的宽度 ≥ `max(40, g_typical_edge / 4)` mm。中心可以在缝里，不再要求两侧面积之和 ≥ 60% 底面积。若中心的 X 落在支承缝里，只认 X 向桥（Y 向同理），避免长边刮擦被当成另一轴搭桥。

---

## 更新几何

件 `k` 放在 `(x,y,z)`，尺寸 `(w,d,h)`，放置原点永远是最小角。

**EMS（默认）**：空箱只有一块 `[0,W)×[0,D)×[0,H)`。与 `k` 相交的每个 EMS 用 Lai–Chan difference process 最多裂成 6 块（左/右/后/前/下/上，允许重叠）。丢掉零体积，丢掉被另一块完全包含的。超过 `MAXEMS=1280` 时按最短边、再按体积留最大的那些。

落点不只钉左后下角：试该盒底面四个角（件贴齐后仍在盒内），以及底面 z 上已装件顶面与该盒的重叠角。极大空盒的最小角常常是悬空的（挡 `-z` 的货可能在底面另一侧）。件能放进去当且仅当三边都不超过该盒，再查支承。

与 EP 同时开启时，这些角和 EP 点取并后，**残余一律从落点做 `residual_box`**，否则同一坐标会因「整块 EMS」和「局部残盒」两套尺子对不上。

**EP（`ep`）**：

1. 丢掉落在 `k` 内部的旧点。
2. 加入 `k` 在 +X / +Y / +Z 卦限的 6 个角。
3. **Crainic**：把 `k` 的三个远顶点沿另外两轴投影到已装件的最近阻挡面（无阻挡则落到原点侧）。
4. **反向 stamp**：每个已装件与 `k` 的 XY / YZ / XZ 重叠矩形，拷到 `k` 的 +Z / +X / +Y 面上（四个角）。

去重后若超过 `MAXEP=640`，在按 `(z,y,x)` 排序的列表上均匀抽样。空腔用 `residual_box`：以 EP 为最小角，沿 +x/+y/+z 长到墙或已装件。

巨大空腔：至少两条轴 ≥ 件的 2 倍，且残余体积 ≥ 5× 件体积。

---

## 分数（不要混用）

当前 `state_rank` 在 `rank.cpp`：体积仍压过几何，同 \(g\) 的姿态用 compactness / 末件接触 / 空腔 / 末件残腔 / 窗口大件搁浅来拆开。`better_state` 仍只比装填率。

同一件的合法姿态不再单独打分：每个 `(SKU, 旋转, 落点)` 都生成后继，由 `state_rank` 在整层 `cand` 里截断。

tight 模式丢掉巨大空腔，以及浪费 > 2× 件体积的姿态。  
非 tight：只要存在非巨大姿态，就丢掉巨大姿态。

**状态分** `state_rank` —— 层内排序：

`rank_features_vec(st)` → 28 维无量纲 `phi[]`（训练 / 推理用）。`state_rank` 仍用手调线性，热路径不依赖 vec。

**28 维 `phi`（顺序固定，见 `rank_feature_name(i)`）**

| # | 名 | 含义 |
|---|---|---|
| 0–1 | fill, nplaced | 装填率、件数 |
| 2–8 | floor_cover … maxy_d | 地板覆盖、包络 |
| 9–10 | wall_contact, item_contact | 整箱墙/货接触 |
| 11–12 | nems, nep | 空腔点数 |
| 13–18 | max_cav_* … max_cav_h | EMS/EP 空腔体积与最大盒三边 |
| 19–22 | pending_* , unplaceable | 窗口 pending 与搁浅 |
| 23–27 | compactness, last_* | 紧凑度、末件姿态 |

**JSONL dump**（`--rank-dump rank.jsonl` 或 `PACK3D_RANK_DUMP`）

- `type=cand`：每层每个候选（最多 4096/层），含 `phi`, `rank`, `kept`, `g`, `nplaced`, `bin`, `layer`
- `type=pair`：同 `(g,nplaced)` 下 kept 最高 vs 未 kept 最高（最多 512/层，成对训练用）

**报优 / 装填率** `better_state`：先比 `g`，再比件数。

---

## 一层束搜索

```
buffer_round(cur, tight_only):
    对 cur 里每个状态 Expand（OpenMP 按状态切开）
    用 better_state 更新 best
    select_top → 留下 beam 个，作为下一层 cur
    有任一状态真正多装了一件则 progress=true
```

`Expand` 一次只多装 **一件**（每个候选 SKU 的全部合法姿态都打分）。  
每个 `(SKU, 旋转, 落点)` 都打分。两遍：第一遍堆留 top-`beam` 分得阈值；第二遍只物化 `rank >= 阈值` 的状态。  
单个父状态内：合法落点先收齐，再对落点列表并行 `score_pose`（外层已在 parallel 里时用 task，否则 parallel for）。  
pending 为空、超时或没有合法姿态：原样留下，progress 为假，外层可以封箱。

`select_top`：

1. 按 `state_rank` 排序
2. `geom_sig` 去重（已装 id + 前几个空腔；不同放置顺序走到同一几何只留一份）
3. 按 `|geom|/6` 分桶，每桶约 `beam/8`，溢出按名次补满 `beam`

层与层、箱与箱串行（下一箱依赖剩件）。

---

## 参数（与代码一致）

```
beam          480     每层活状态
TIME_LIMIT    90s     每箱墙钟
MAXEP         640     EP 点数上限
MAXEMS        1280    EMS 空盒上限
MAX_PLACED    256     单箱件数硬顶
SUPPORT_RATIO 0.60    坐实路径重叠面积门槛（搭桥不走这条）
BUFFER_CAP    12      未装件数上限
OPEN_CAP      12      可装窗口 = 缓冲
CHUNK_K       8       每轮从窗口搜这么多层，落实最好状态后再补槽
```

线程数由 argv 传入，扩状态和算 `state_rank` 可并行。8 核大约 2.7×，不是 8×（`residual_box` 偏内存带宽）。合并顺序可能改变并列名次，装填率会有很小抖动。

---

## 数据流

```
空箱 beam
    │
    ├─ pending<12 且流未尽 → 取 1 件直到满
    ├─ 对当前窗口束搜索最多 8 层（每层装 1 件）
    ├─ 落实估价最好的状态，丢掉其余分支
    ├─ 装成过 → 回到补槽
    └─ 本轮装不动（或流尽）→ 封箱，剩件进下一箱
            │
            每层：pending SKU × 旋转 × (EMS角 ∪ EP)
                  Feasible → residual_box（tight / 巨大腔过滤）
                  第一遍：score → 堆留 top beam 分 → 阈值
                  第二遍：score → rank>=阈值 才物化 State
                  select_top → 留下 beam
            （先 tight，没有再用任意姿态）
```

---

## 不做

- 高度图 / 长数组当状态
- 全局 \(f=g+h\) 优先队列当 A\*
- 离线看见尚未到达的件
- 对连续坐标穷举落点
- 件数优先或「按到达顺序必须先装」的开关（已去掉）
- 不同 `|placed|` 混在一层里取 top-k
