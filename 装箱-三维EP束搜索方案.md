# 三维装箱：生产窗口 + EP + 分层束搜索

实现：`pack3d.cpp`。几何用 **Extreme Point**，搜索用 **按已装件数分层的 beam**。  
不是离线全局排列：一次最多看见缓冲 12 + 传送带 10 = 22 件。目标是装填率（已装体积），件数只作并列打破。

不维护高度图，不用 EMS 当主状态，不跑可采纳 A\*。

---

## 文件

| 文件 | 作用 |
|---|---|
| `pack3d.cpp` | 求解器 |
| `pack_viewer.html` | 三维查看器模板（占位 `__PACK_DATA__`） |
| `input.txt` | 算例：`W D H` 后每行一件 `w d h` |
| `快件高度 快件长度 快件宽度.txt` | 原始件尺寸（带表头） |

编译与运行（必须带 `-fopenmp`；数字是线程数，默认 1）：

```
g++ -O2 -std=c++17 -fopenmp -o pack3d pack3d.cpp
./pack3d 8 < input.txt > pack3d.out
```

stdout：箱数 `N`，下一行各箱装填率（占 `W*D*H` 的百分比）。  
stderr：每箱 `packed / fill / leftover / seen`，最后一行汇总。  
运行结束会读模板写出 `pack3d.html`（可随时删，下次再生成）。

---

## 设定

```
容器:     W × D × H，原点在左-后-下
件 i:     (w, d, h)，6 种轴对齐旋转
状态 u:   (EPs, placed, g)
          EPs     = 候选角点，空箱只有 (0,0,0)
          placed  = 已装位姿（碰撞、支撑、投影）
          g       = 已装体积
窗口:     g_open = 已到达、且未装进「上一箱」的件
          某状态的 pending = g_open \ 该状态已装
```

下一件的左后下角只放在某个 EP 上。

---

## 生产窗口（不是看见剩余全部 SKU）

件按传送带每批 `BATCH_SIZE=10` 到达，这 10 件**不占**缓冲槽。  
缓冲最多停 `BUFFER_CAP=12` 件未装箱。下一批评入的条件是：当前所有未装件数 ≤ 12（本批传送带已装完或已停进缓冲）。一次可装窗口 `OPEN_CAP=22`。

`g_typical_edge` = 已经到达件的 min-edge 中位数，至少 80。只用来判断空腔是「还能塞」还是死缝，**不看未来件**。

每箱流程：

```
1. force_room(12)
   把 pending 里 min-edge 最小的 12 件当填料停着（FFD），
   其余必须本轮去装。直到每个 beam 状态 pending ≤ 12，
   再把下一批 10 件放进 g_open。

2. 若 1 失败：force_room(0)（填料也装）。
   若 pending 仍 > 12，封箱。
   本箱已装位姿留下；g_open 里没装上的件带到下一空箱。

3. 每批评入之后 tight-pack：
   任意 pending 件，只要落在非巨大空腔且浪费 ≤ 2× 件体积，就填缝。
   不走 FFD。

4. 流耗尽：force_room(0)，封箱。
```

比空箱还大的件直接跳过。其余不丢弃。  
一箱墙钟 `TIME_LIMIT=55s`；超时后 `expand_state` 原样复制状态（看起来像「装不下」）。

---

## 合法放置

6 种旋转。位姿合法当且仅当：

- 整件在箱内
- 与已装件 AABB 不重叠
- 底面支承面积 ≥ 底面积的 60%（`z=0` 视为地板）
- 底面中心落在地板或某件顶面上（中心不能架在缝上）

---

## 更新 EP

件 `k` 放在 `(x,y,z)`，尺寸 `(w,d,h)`，放置原点永远是最小角：

1. 丢掉落在 `k` 内部的旧点。
2. 加入 `k` 在 +X / +Y / +Z 卦限的 6 个角。
3. **Crainic**：把 `k` 的三个远顶点沿另外两轴投影到已装件的最近阻挡面（无阻挡则落到原点侧）。
4. **反向 stamp**：每个已装件与 `k` 的 XY / YZ / XZ 重叠矩形，拷到 `k` 的 +Z / +X / +Y 面上（四个角）。  
   经典 EP 只把**新件**投影到**旧件**上，悬挑后再塞短件时台面角不会出现；这一步补上，不针对某种布局写特例。

去重后若超过 `MAXEP=640`，在按 `(z,y,x)` 排序的列表上均匀抽样，高低角都留，不只留最低 z。

空腔用 `residual_box`：以 EP 为最小角，沿 +x/+y/+z 长到墙或已装件。碰到障碍时，在还能切的轴上选留下体积最大的切法（最多 12 轮）。这是 `collect_placements` 里最贵的一步。

巨大空腔：至少两条轴 ≥ 件的 2 倍，且残余体积 ≥ 5× 件体积。

---

## 分数（不要混用）

**落点分** `place_score` —— 只比较同一件的不同 EP / 旋转：

```
巨大空腔:  vol + contact/3
否则:      vol/10 − cavity_waste − 80·height_mismatch + contact/3
```

- `contact`：与箱壁、已装件共面的接触面积
- `cavity_waste`：残余盒体积 − 件体积
- `height_mismatch`：与 xy 间隙 ≤ 40 的邻件比顶面高度差；没有邻居则 0

tight 模式另外丢掉巨大空腔，以及浪费 > 2× 件体积的姿态。  
非 tight：只要存在非巨大姿态，就丢掉巨大姿态（小件不要去占给后面大件留的洞）。

**状态分** `state_rank` —— 层内排序：

```
rank = g·10000 − compactness·20 + cavity_tiebreak
```

- `compactness`：各件最小角之和 + 包围盒底面积/8 + 最高 z + 3·|EPs|（同等体积偏好更挤向原点）
- `cavity_tiebreak`：各 EP 上的残余盒。最短边 ≥ `g_typical_edge` 的算可用（奖最大洞 + 总和），否则算死缝（重罚）

**报优 / 装填率** `better_state`：先比 `g`，再比件数。

**大件搁浅惩罚**（只加在 `select_top` 的前 `max(beam, 3·beam)` 个上）：  
看 pending 里体积最大的 4 件，当前 EP 上已完全放不下的体积 × 20000 从 rank 里减掉。略空一点但还能放大件的布局，优于把大件卡死的贪心填满。

---

## 一层束搜索

```
buffer_round(cur, max_pending, tight_only):
    对 cur 里每个状态 Expand（OpenMP 按状态切开）
    用 better_state 更新 best
    select_top → 留下 beam 个，作为下一层 cur
    有任一状态真正多装了一件则 progress=true
```

`Expand` 一次只多装 **一件**（每个候选 SKU 独立试 `pos_keep` 个姿态）。  
`max_pending ≥ 0` 且 pending 已 ≤ 该值：原样留下（给 admit 腾槽）。  
装不出合法姿态：也原样留下，progress 为假，外层可以封箱。

`select_top`：

1. 按 `state_rank` 排序
2. 对头部 pool 再减大件搁浅惩罚，重排
3. `ep_sig` 去重（已装 id + 前几个 EP；不同放置顺序走到同一几何只留一份）
4. 按 `|EPs|/6` 分桶，每桶约 `beam/8`，溢出按名次补满 `beam`

层与层、箱与箱串行（下一箱依赖剩件）。

---

## 参数（与代码一致）

```
beam          480     每层活状态
pos_keep      3       同一 (状态, SKU) 保留的姿态数
TIME_LIMIT    55s     每箱墙钟
MAXEP         640
MAX_PLACED    256     单箱件数硬顶
SUPPORT_RATIO 0.60
BUFFER_CAP    12
BATCH_SIZE    10
OPEN_CAP      22
```

线程数由 argv 传入，扩状态和算 `state_rank` 可并行。8 核大约 2.7×，不是 8×（`residual_box` 偏内存带宽）。合并顺序可能改变并列名次，装填率会有很小抖动。

---

## 数据流

```
空箱 beam
    │
    ├─ force_room(12)：FFD 装到 pending≤12
    ├─ 放入下一批 10 件
    ├─ tight-pack 填缝
    └─ 装不动且 pending>12 → 封箱，剩件进下一箱
            │
            每层：pending SKU × 旋转 × EP
                  Feasible → place_score 留 pos_keep
                  UpdateEPs（含反向 stamp）
                  state_rank + 大件惩罚 + 去重分桶
```

---

## 不做

- 高度图 / 长数组当状态
- 全局 \(f=g+h\) 优先队列当 A\*
- 离线看见尚未到达的件
- 对连续坐标穷举落点
- 件数优先或「按到达顺序必须先装」的开关（已去掉）
- 不同 `|placed|` 混在一层里取 top-k
