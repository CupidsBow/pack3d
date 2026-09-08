/*
  3D container packing — EMS / EP + layered beam search

  ---------------------------------------------------------------------------
  What this program does
  ---------------------------------------------------------------------------
  Pack a stream of rectangular parcels into identical containers using a
  production schedule (not a global offline permutation of all items).
  Default geometry uses both Empty Maximal Spaces (EMS) and Extreme Points
  (EP): each expansion tries the union of EMS supported corners and EP
  candidates. Pass `ems` or `ep` to use only one. Search is a layered beam:
  each layer places one more item, then keeps the best `beam` partial packs.

  Input (stdin):
    W D H
    w d h          (one parcel per line, until EOF)

  Output (stdout):
    N
    r_1 r_2 ... r_N
    N = number of containers. r_i = fill rate of container i, percent of W*D*H.

  Also writes pack3d.html (template pack_viewer.html) with per-bin placements.

  Build and run (OpenMP is required; first number is threads, default 1;
  optional `both` / `ems` / `ep` selects geometry, default both):
    g++ -O2 -std=c++17 -fopenmp -o pack3d pack3d.cpp rank.cpp
    ./pack3d 8 < input.txt > pack3d.out
    ./pack3d 8 ep < input.txt > pack3d.out
    ./pack3d 8 ems < input.txt > pack3d.out

  ---------------------------------------------------------------------------
  Production window (not "see all remaining SKUs")
  ---------------------------------------------------------------------------
  The line feeds one parcel at a time. The staging buffer holds at most
  BUFFER_CAP (12) unpacked items; that is also the packable window (OPEN_CAP).
  Never more than 12 unpacked SKUs at once. Fill the buffer, then run CHUNK_K
  (8) beam layers on that frozen window (each layer places one more item). Among
  the resulting states pick the best evaluation and commit it as the only
  live packing; then refill the holes and search the next chunk of 8.

  g_open = all arrived items that are not yet packed in a *previous* container.
  A State's pending set is g_open minus whatever that State already packed.
  Packed items in the current box stay in g_open until the box is sealed.

  Per container loop:
    1. Fill the buffer: while pending < 12 and the stream is not empty, take 1.
    2. Beam-search up to CHUNK_K further placements from the current window.
    3. Commit the best of those states (most items this chunk, then state_rank).
    4. If the stream is exhausted: keep placing (tight, then any pose) until
       stuck, then seal.
    5. If the chunk packed at least one item, go back to 1.
    6. If nothing packed, seal. Leftover g_open items go to the next box.

  Items larger than the empty box are skipped. Nothing else is dropped.

  ---------------------------------------------------------------------------
  Geometry
  ---------------------------------------------------------------------------
  Origin is left-back-bottom. Six axis rotations, minus poses whose bottom
  area is < 1/3 of the parcel's largest face (unstable: a flat/narrow piece
  stood on edge). A placement is legal iff it is in-bin,
  AABB-disjoint from packed items, and well-supported: z=0 is the floor;
  otherwise a coplanar platform (center on some top and overlap >= 60%)
  or an X- or Y-axis two-sided bridge (center may sit in the gap; each
  side's span-width >= max(40, g_typical_edge/4) mm). Different-height
  piers are not support.

  Combined (default): keep both lists on the state. Expand unions EMS
  origins with EP points, then scores every distinct (x,y,z) with the same
  residual_box (do not mix EMS-box size and EP residual — they are not
  comparable).

  EMS: empty box is one maximal AABB. After placing k, every EMS that
  intersects k is split by the Lai–Chan difference process. Placement tries
  four bottom corners plus packed-top overlaps on that floor.

  EP: after placing k, points are updated by dropping interiors, adding k's
  six +X/+Y/+Z corners, Crainic projections, and reverse stamp of packed
  overlap onto k's three positive faces. If more than MAXEP points remain,
  subsample across sorted (z,y,x).

  ---------------------------------------------------------------------------
  Scoring
  ---------------------------------------------------------------------------
  Every legal (SKU, rot, origin) becomes a successor. Tight/huge filters may
  drop poses before that. Layer cut is state_rank in rank.cpp: volume
  dominates, then compactness / last-pose contact / cavity / last-fit /
  stranded pending. better_state / reported fill: packed volume, then count.

  ---------------------------------------------------------------------------
  Beam
  ---------------------------------------------------------------------------
  beam = 480 states per layer. TIME_LIMIT = 90s per container; after that
  expand_state copies the state unchanged (looks like "cannot pack").
  select_top: rank by fill, dedup similar geometry signatures, bucket by cavity
  count so one contour family cannot fill the whole beam.

  One buffer_round = try to place one more item on every live state, then
  keep top `beam`. Two passes: (1) score every pose, discard the State, keep
  only the rank; the beam-th highest rank is the cutoff. (2) score again and
  keep poses with rank >= cutoff, then materialize and dedup.
  Within each parent state, pose scoring uses OpenMP tasks when already inside
  the outer parallel team, else a parallel for over the pose list.
*/

#include <bits/stdc++.h>
#include <omp.h>
#include "pack_state.hpp"
#include "rank.hpp"
using namespace std;

static const double TIME_LIMIT = 90.0;  // wall seconds per container, then expand stops
static const double SUPPORT_RATIO = 0.60;
static const int CHUNK_K = 8;        // beam-pack this many, commit best, refill
int g_nthreads = 1;
bool g_use_ems = true;               // default both; `ems` / `ep` select one
bool g_use_ep = true;

int W, D, H, N;
vector<array<int, 3>> orig;   // input dimensions, index = item id
vector<long long> vol;
vector<int> min_edge;
vector<int> g_open;           // arrived, not packed in a previous container
int g_seen = 0;               // how far the arrival pointer has moved
int g_typical_edge = 80;      // median min-edge of arrived items; bridge width

// Only items that have already arrived may influence typical_edge (bridge).
void update_typical_edge() {
    if (g_seen <= 0) {
        g_typical_edge = 80;
        return;
    }
    vector<int> edges(min_edge.begin(), min_edge.begin() + g_seen);
    sort(edges.begin(), edges.end());
    g_typical_edge = max(80, edges[g_seen / 2]);
}
chrono::steady_clock::time_point t0;

double elapsed() {
    return chrono::duration<double>(chrono::steady_clock::now() - t0).count();
}
bool time_up() { return elapsed() > TIME_LIMIT; }

State empty_box() {
    State st;
    st.nep = 1;
    st.eps[0] = {0, 0, 0};
    st.nems = 1;
    st.ems[0] = {0, 0, 0, W, D, H};
    return st;
}

bool ems_intersects_place(const EmsBox& e, const Place& k) {
    return e.x < k.x + k.w && k.x < e.x + e.w && e.y < k.y + k.d && k.y < e.y + e.d &&
           e.z < k.z + k.h && k.z < e.z + e.h;
}

bool ems_contains(const EmsBox& a, const EmsBox& b) {
    return a.x <= b.x && a.y <= b.y && a.z <= b.z && a.x + a.w >= b.x + b.w &&
           a.y + a.d >= b.y + b.d && a.z + a.h >= b.z + b.h;
}

void ems_push(vector<EmsBox>& o, int x, int y, int z, int w, int d, int h) {
    if (w > 0 && d > 0 && h > 0) o.push_back({x, y, z, w, d, h});
}

// Lai–Chan difference process: one maximal empty box minus an intersecting
// packed AABB yields at most 6 remnants (they overlap on purpose).
void difference_split(const EmsBox& e, const Place& k, vector<EmsBox>& o) {
    int x1 = e.x, x2 = e.x + e.w;
    int y1 = e.y, y2 = e.y + e.d;
    int z1 = e.z, z2 = e.z + e.h;
    int x3 = k.x, x4 = k.x + k.w;
    int y3 = k.y, y4 = k.y + k.d;
    int z3 = k.z, z4 = k.z + k.h;
    ems_push(o, x1, y1, z1, x3 - x1, y2 - y1, z2 - z1);
    ems_push(o, x4, y1, z1, x2 - x4, y2 - y1, z2 - z1);
    ems_push(o, x1, y1, z1, x2 - x1, y3 - y1, z2 - z1);
    ems_push(o, x1, y4, z1, x2 - x1, y2 - y4, z2 - z1);
    ems_push(o, x1, y1, z1, x2 - x1, y2 - y1, z3 - z1);
    ems_push(o, x1, y1, z4, x2 - x1, y2 - y1, z2 - z4);
}

void commit_ems(State& st, vector<EmsBox>& pts) {
    if (pts.empty()) {
        st.nems = 1;
        st.ems[0] = {0, 0, 0, 0, 0, 0};
        return;
    }
    sort(pts.begin(), pts.end());
    pts.erase(unique(pts.begin(), pts.end()), pts.end());
    const int n = (int)pts.size();
    vector<char> keep(n, 1);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            if (!ems_contains(pts[j], pts[i])) continue;
            if (ems_contains(pts[i], pts[j])) {
                if (j < i) {
                    keep[i] = 0;
                    break;
                }
            } else {
                keep[i] = 0;
                break;
            }
        }
    }
    vector<EmsBox> maximal;
    maximal.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (keep[i]) maximal.push_back(pts[i]);
    }
    int m = (int)maximal.size();
    if (m > MAXEMS) {
        sort(maximal.begin(), maximal.end(), [](const EmsBox& a, const EmsBox& b) {
            int ma = min({a.w, a.d, a.h});
            int mb = min({b.w, b.d, b.h});
            if (ma != mb) return ma > mb;
            return 1LL * a.w * a.d * a.h > 1LL * b.w * b.d * b.h;
        });
        maximal.resize(MAXEMS);
        sort(maximal.begin(), maximal.end());
        m = MAXEMS;
    }
    st.nems = m;
    memcpy(st.ems, maximal.data(), (size_t)m * sizeof(EmsBox));
}

void update_ems(State& st, const Place& k) {
    vector<EmsBox> nxt;
    nxt.reserve((size_t)st.nems * 6 + 8);
    for (int i = 0; i < st.nems; ++i) {
        const EmsBox& e = st.ems[i];
        if (!ems_intersects_place(e, k)) {
            nxt.push_back(e);
            continue;
        }
        difference_split(e, k, nxt);
    }
    commit_ems(st, nxt);
}

// Item origins on the floor of EMS `e`. LBB of a maximal box is often floating
// (the -z block may sit on the opposite corner of the bottom face). Try all
// four bottom corners so a pack can seed on any wall, plus corners of packed
// tops that actually meet this floor — those are the supported ledges the
// min-corner misses.
static const int MAX_EMS_ORIGINS = 12;

void ems_try_origin(vector<Pt>& o, const EmsBox& e, int x, int y, int item_w, int item_d) {
    if (x < e.x || y < e.y) return;
    if (x + item_w > e.x + e.w || y + item_d > e.y + e.d) return;
    o.push_back({x, y, e.z});
}

void ems_placement_origins(const State& st, const EmsBox& e, int item_w, int item_d, vector<Pt>& o) {
    o.clear();
    if (item_w > e.w || item_d > e.d) return;
    int x0 = e.x, x1 = e.x + e.w - item_w;
    int y0 = e.y, y1 = e.y + e.d - item_d;
    ems_try_origin(o, e, x0, y0, item_w, item_d);
    ems_try_origin(o, e, x1, y0, item_w, item_d);
    ems_try_origin(o, e, x0, y1, item_w, item_d);
    ems_try_origin(o, e, x1, y1, item_w, item_d);
    if (e.z > 0) {
        for (int i = 0; i < st.nplaced; ++i) {
            const Place& p = st.placed[i];
            if (p.z + p.h != e.z) continue;
            int ox0 = max(p.x, e.x), ox1 = min(p.x + p.w, e.x + e.w);
            int oy0 = max(p.y, e.y), oy1 = min(p.y + p.d, e.y + e.d);
            if (ox1 <= ox0 || oy1 <= oy0) continue;
            // Corners of this top ∩ EMS floor, plus item-flush origins on that
            // top so the parcel need not sit at the (often floating) EMS LBB.
            ems_try_origin(o, e, ox0, oy0, item_w, item_d);
            ems_try_origin(o, e, ox1 - item_w, oy0, item_w, item_d);
            ems_try_origin(o, e, ox0, oy1 - item_d, item_w, item_d);
            ems_try_origin(o, e, ox1 - item_w, oy1 - item_d, item_w, item_d);
            int fx0 = max(e.x, p.x), fx1 = min(e.x + e.w - item_w, p.x + p.w - item_w);
            int fy0 = max(e.y, p.y), fy1 = min(e.y + e.d - item_d, p.y + p.d - item_d);
            if (fx0 <= fx1 && fy0 <= fy1) {
                ems_try_origin(o, e, fx0, fy0, item_w, item_d);
                ems_try_origin(o, e, fx1, fy0, item_w, item_d);
                ems_try_origin(o, e, fx0, fy1, item_w, item_d);
                ems_try_origin(o, e, fx1, fy1, item_w, item_d);
            }
            if ((int)o.size() >= MAX_EMS_ORIGINS * 3) break;
        }
    }
    sort(o.begin(), o.end());
    o.erase(unique(o.begin(), o.end()), o.end());
    if ((int)o.size() > MAX_EMS_ORIGINS) o.resize(MAX_EMS_ORIGINS);
}

// Union of EMS supported corners and EP points, then unique. Scoring later
// uses residual_box at each origin so the two sources share one scale.
void collect_candidate_origins(const State& st, int item_w, int item_d, int item_h, vector<Pt>& o) {
    o.clear();
    if (g_use_ems) {
        vector<Pt> local;
        for (int e = 0; e < st.nems; ++e) {
            const EmsBox& g = st.ems[e];
            if (item_w > g.w || item_d > g.d || item_h > g.h) continue;
            ems_placement_origins(st, g, item_w, item_d, local);
            o.insert(o.end(), local.begin(), local.end());
        }
    }
    if (g_use_ep) {
        for (int e = 0; e < st.nep; ++e) o.push_back(st.eps[e]);
    }
    if (o.empty()) return;
    sort(o.begin(), o.end());
    o.erase(unique(o.begin(), o.end()), o.end());
}

void rotated_size(int id, int rot, int& w, int& d, int& h) {
    w = orig[id][ROT_PERM[rot][0]];
    d = orig[id][ROT_PERM[rot][1]];
    h = orig[id][ROT_PERM[rot][2]];
}

// Bottom area must be at least 1/3 of the largest face; otherwise this
// rotation stands a flat/narrow parcel on edge. Skip those poses.
bool base_stable(int w, int d, int h) {
    long long bot = 1LL * w * d;
    long long largest = max({bot, 1LL * w * h, 1LL * d * h});
    return bot * 3 >= largest;
}

bool in_bin(int x, int y, int z, int w, int d, int h) {
    return x >= 0 && y >= 0 && z >= 0 && x + w <= W && y + d <= D && z + h <= H;
}

bool overlap(const Place& a, int x, int y, int z, int w, int d, int h) {
    return x < a.x + a.w && a.x < x + w && y < a.y + a.d && a.y < y + d &&
           z < a.z + a.h && a.z < z + h;
}

// Bottom must rest on z=0, or on coplanar packed tops via either:
//   platform: center on some top and overlap >= SUPPORT_RATIO of w*d, or
//   bridge:   >= 2 support rects; the span axis (X or Y) has bearing on
//             both sides of the bottom center, each side's span-width
//             >= max(40, g_typical_edge/4). Center may sit in the gap.
//             If the center's X (resp. Y) sits in a contact gap, only
//             that axis counts — a long-Y scrape cannot pass as a
//             Y-bridge. Tops at different heights are not support.
bool well_supported(const State& st, int x, int y, int z, int w, int d) {
    if (z == 0) return true;
    long long bottom = 1LL * w * d;
    if (bottom <= 0) return false;
    int cx = x + w / 2, cy = y + d / 2;
    long long sup = 0;
    bool center_ok = false;
    int n_rects = 0;
    int x_left = 0, x_right = 0;
    int y_lo = 0, y_hi = 0;
    bool cover_cx = false, cover_cy = false;
    for (int i = 0; i < st.nplaced; ++i) {
        const Place& p = st.placed[i];
        if (p.z + p.h != z) continue;
        int sx0 = max(x, p.x), sx1 = min(x + w, p.x + p.w);
        int sy0 = max(y, p.y), sy1 = min(y + d, p.y + p.d);
        int ox = sx1 - sx0, oy = sy1 - sy0;
        if (ox <= 0 || oy <= 0) continue;
        ++n_rects;
        sup += 1LL * ox * oy;
        if (p.x <= cx && cx < p.x + p.w && p.y <= cy && cy < p.y + p.d) center_ok = true;
        if (sx0 <= cx && cx < sx1) cover_cx = true;
        if (sy0 <= cy && cy < sy1) cover_cy = true;
        x_left += max(0, min(sx1, cx) - max(sx0, x));
        x_right += max(0, min(sx1, x + w) - max(sx0, cx));
        y_lo += max(0, min(sy1, cy) - max(sy0, y));
        y_hi += max(0, min(sy1, y + d) - max(sy0, cy));
    }
    if (center_ok && sup * 10000 >= bottom * (long long)llround(SUPPORT_RATIO * 10000))
        return true;
    int min_bear = max(40, g_typical_edge / 4);
    if (n_rects < 2) return false;
    bool x_ok = x_left >= min_bear && x_right >= min_bear;
    bool y_ok = y_lo >= min_bear && y_hi >= min_bear;
    if (!cover_cx && cover_cy) return x_ok;
    if (!cover_cy && cover_cx) return y_ok;
    return x_ok || y_ok;
}

bool feasible(const State& st, int x, int y, int z, int w, int d, int h) {
    if (!base_stable(w, d, h)) return false;
    if (!in_bin(x, y, z, w, d, h)) return false;
    for (int i = 0; i < st.nplaced; ++i) {
        if (overlap(st.placed[i], x, y, z, w, d, h)) return false;
    }
    return well_supported(st, x, y, z, w, d);
}

// Largest empty AABB with min-corner at (x,y,z) growing +x/+y/+z until a
// packed item or the container wall. An intersecting AABB is resolved by
// cutting on the axis that leaves the largest remaining volume (up to 12
// passes). Used for the tight/huge pose filter. O(placed) per origin, so this
// is the hot cost inside collect_placements.
Res residual_box(const State& st, int x, int y, int z) {
    int rx = max(0, W - x), ry = max(0, D - y), rz = max(0, H - z);
    for (int it = 0; it < 12; ++it) {
        bool cut = false;
        for (int i = 0; i < st.nplaced; ++i) {
            const Place& p = st.placed[i];
            if (!(x < p.x + p.w && p.x < x + rx && y < p.y + p.d && p.y < y + ry &&
                  z < p.z + p.h && p.z < z + rz)) {
                continue;
            }
            // Distance from residual origin to this obstacle on each axis
            // (negative = the obstacle already covers that origin face).
            int cx = (p.x >= x) ? p.x - x : -1;
            int cy = (p.y >= y) ? p.y - y : -1;
            int cz = (p.z >= z) ? p.z - z : -1;
            long long bestv = -1;
            int nx = rx, ny = ry, nz = rz;
            auto try_cut = [&](int nrx, int nry, int nrz) {
                if (nrx < 0 || nry < 0 || nrz < 0) return;
                long long v = 1LL * nrx * nry * nrz;
                if (v > bestv) {
                    bestv = v;
                    nx = nrx;
                    ny = nry;
                    nz = nrz;
                }
            };
            if (cx >= 0) try_cut(cx, ry, rz);
            if (cy >= 0) try_cut(rx, cy, rz);
            if (cz >= 0) try_cut(rx, ry, cz);
            if (bestv < 0) return {0, 0, 0};
            if (nx != rx || ny != ry || nz != rz) {
                rx = nx;
                ry = ny;
                rz = nz;
                cut = true;
            }
        }
        if (!cut) break;
    }
    return {rx, ry, rz};
}

// Unused volume if (w,d,h) sits in residual r. Tight-pack rejects waste > 2x vol.
long long cavity_waste(const Res& r, int w, int d, int h) {
    long long cav = 1LL * r.x * r.y * r.z;
    long long item = 1LL * w * d * h;
    return max(0LL, cav - item);
}

// True if this cavity is a poor fit: at least two axes twice the item, and
// residual volume at least 5x the item. Prefer a tighter hole when one exists.
bool cavity_too_big(const Res& r, int w, int d, int h) {
    int loose = (r.x >= w * 2) + (r.y >= d * 2) + (r.z >= h * 2);
    return loose >= 2 && 1LL * r.x * r.y * r.z >= 5LL * w * d * h;
}

// Dedup EPs and enforce MAXEP. Truncation strides through the z-sorted list
// so high corners are not discarded first (early code kept only lowest z).
void commit_eps(State& st, vector<Pt>& pts) {
    if (pts.empty()) {
        st.nep = 1;
        st.eps[0] = {0, 0, 0};
        return;
    }
    sort(pts.begin(), pts.end());
    pts.erase(unique(pts.begin(), pts.end()), pts.end());
    int n = (int)pts.size();
    if (n > MAXEP) {
        vector<Pt> keep;
        keep.reserve(MAXEP);
        for (int i = 0; i < MAXEP; ++i) {
            keep.push_back(pts[(int)((long long)i * (n - 1) / (MAXEP - 1))]);
        }
        sort(keep.begin(), keep.end());
        keep.erase(unique(keep.begin(), keep.end()), keep.end());
        pts.swap(keep);
        n = (int)pts.size();
    }
    st.nep = n;
    memcpy(st.eps, pts.data(), (size_t)n * sizeof(Pt));
}

// Rebuild the EP set after packing item k.
// 1) Drop points now inside k.
// 2) Add k's six corners on the +X / +Y / +Z octant (placement origin is
//    always a min-corner, so these are the new empty-space corners).
// 3) Crainic: project k's vertices onto existing items (nearest blocking
//    face along the two unused axes).
// 4) Reverse stamp: every packed item's AABB overlap with k, copied onto
//    k's +X, +Y, +Z faces. Classic EP only projects the *new* item onto
//    *old* items, so a short piece tucked beside an overhang never created
//    the ledge corner. This step does, without special-casing that layout.
void update_eps(State& st, const Place& k) {
    vector<Pt> pts;
    pts.reserve((size_t)st.nep + (size_t)st.nplaced * 12 + 16);
    for (int i = 0; i < st.nep; ++i) {
        const Pt& p = st.eps[i];
        if (p.x >= k.x && p.x < k.x + k.w && p.y >= k.y && p.y < k.y + k.d &&
            p.z >= k.z && p.z < k.z + k.h) {
            continue;
        }
        pts.push_back(p);
    }

    auto add = [&](int x, int y, int z) {
        if (x < 0 || y < 0 || z < 0 || x >= W || y >= D || z >= H) return;
        pts.push_back({x, y, z});
    };

    add(k.x + k.w, k.y, k.z);
    add(k.x, k.y + k.d, k.z);
    add(k.x, k.y, k.z + k.h);
    add(k.x + k.w, k.y + k.d, k.z);
    add(k.x + k.w, k.y, k.z + k.h);
    add(k.x, k.y + k.d, k.z + k.h);

    // Crainic: for each of k's three "far" vertices, walk back along the two
    // unused axes to the nearest packed face (or the container origin).
    int yx_x = 0, xy_y = 0, zx_x = 0, zy_y = 0, yz_z = 0, xz_z = 0;
    auto consider = [&](const Place& i) {
        if (i.y <= k.y + k.d && k.y + k.d <= i.y + i.d && i.z <= k.z && k.z <= i.z + i.h &&
            i.x + i.w <= k.x + k.w) {
            yx_x = max(yx_x, i.x + i.w);
        }
        if (i.x <= k.x && k.x <= i.x + i.w && i.y <= k.y + k.d && k.y + k.d <= i.y + i.d &&
            i.z + i.h <= k.z + k.h) {
            yz_z = max(yz_z, i.z + i.h);
        }
        if (i.x <= k.x + k.w && k.x + k.w <= i.x + i.w && i.z <= k.z && k.z <= i.z + i.h &&
            i.y + i.d <= k.y + k.d) {
            xy_y = max(xy_y, i.y + i.d);
        }
        if (i.x <= k.x + k.w && k.x + k.w <= i.x + i.w && i.y <= k.y && k.y <= i.y + i.d &&
            i.z + i.h <= k.z + k.h) {
            xz_z = max(xz_z, i.z + i.h);
        }
        if (i.y <= k.y && k.y <= i.y + i.d && i.z <= k.z + k.h && k.z + k.h <= i.z + i.h &&
            i.x + i.w <= k.x + k.w) {
            zx_x = max(zx_x, i.x + i.w);
        }
        if (i.x <= k.x && k.x <= i.x + i.w && i.z <= k.z + k.h && k.z + k.h <= i.z + i.h &&
            i.y + i.d <= k.y + k.d) {
            zy_y = max(zy_y, i.y + i.d);
        }
    };
    for (int i = 0; i < st.nplaced; ++i) consider(st.placed[i]);
    add(yx_x, k.y + k.d, k.z);
    add(k.x, k.y + k.d, yz_z);
    add(k.x + k.w, xy_y, k.z);
    add(k.x + k.w, k.y, xz_z);
    add(zx_x, k.y, k.z + k.h);
    add(k.x, zy_y, k.z + k.h);

    // Reverse stamp: XY overlap of each packed item onto k's +Z face, YZ
    // onto +X, XZ onto +Y. Four corners of each overlap rectangle.
    for (int t = 0; t < st.nplaced; ++t) {
        const Place& i = st.placed[t];
        int x0 = max(i.x, k.x), x1 = min(i.x + i.w, k.x + k.w);
        int y0 = max(i.y, k.y), y1 = min(i.y + i.d, k.y + k.d);
        int z0 = max(i.z, k.z), z1 = min(i.z + i.h, k.z + k.h);
        if (x0 <= x1 && y0 <= y1) {
            int z = k.z + k.h;
            add(x0, y0, z);
            add(x1, y0, z);
            add(x0, y1, z);
            add(x1, y1, z);
        }
        if (y0 <= y1 && z0 <= z1) {
            int x = k.x + k.w;
            add(x, y0, z0);
            add(x, y1, z0);
            add(x, y0, z1);
            add(x, y1, z1);
        }
        if (x0 <= x1 && z0 <= z1) {
            int y = k.y + k.d;
            add(x0, y, z0);
            add(x1, y, z0);
            add(x0, y, z1);
            add(x1, y, z1);
        }
    }

    commit_eps(st, pts);
}

// Commit a pose: append Place, add volume, update EMS and/or EPs.
void apply_place(State& st, int id, int rot, int x, int y, int z, int w, int d, int h) {
    if (st.nplaced >= MAX_PLACED) return;
    Place p{id, x, y, z, w, d, h, rot};
    st.placed[st.nplaced++] = p;
    st.g += vol[id];
    st.last_waste = 0;
    st.last_huge = 0;
    if (g_use_ems) update_ems(st, p);
    if (g_use_ep) update_eps(st, p);
}

bool is_packed(const State& st, int id) {
    for (int i = 0; i < st.nplaced; ++i) {
        if (st.placed[i].id == id) return true;
    }
    return false;
}

// True if some rotation of `id` fits an empty box. Oversized SKUs are skipped.
bool item_fits_empty(int id) {
    int lw = -1, ld = -1, lh = -1;
    for (int r = 0; r < 6; ++r) {
        int w, d, h;
        rotated_size(id, r, w, d, h);
        if (w == lw && d == ld && h == lh) continue;
        lw = w;
        ld = d;
        lh = h;
        if (!base_stable(w, d, h)) continue;
        if (w <= W && d <= D && h <= H) return true;
    }
    return false;
}

// Reported / "best packing" comparison: volume first, then how many items.
bool better_state(const State& a, const State& b) {
    if (a.g != b.g) return a.g > b.g;
    return a.nplaced > b.nplaced;
}

// Fingerprint of a packing for beam dedup (ids + a few EPs/EMS). Same geometry
// reached by different placement orders collapses to one beam slot.
uint64_t geom_sig(const State& st) {
    uint64_t h = st.g ^ (uint64_t)st.nplaced * 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < st.nplaced; ++i) {
        h ^= (uint64_t)(st.placed[i].id + 1) * 1000003ULL;
        h = (h << 7) | (h >> 57);
    }
    int take_ep = min(st.nep, 8);
    for (int i = 0; i < take_ep; ++i) {
        const Pt& p = st.eps[i];
        h ^= (uint64_t)(p.x + 1) * 1000003ULL;
        h ^= (uint64_t)(p.y + 1) * 1000033ULL;
        h ^= (uint64_t)(p.z + 1) * 1000037ULL;
        h = (h << 7) | (h >> 57);
    }
    int take_ems = min(st.nems, 8);
    for (int i = 0; i < take_ems; ++i) {
        const EmsBox& g = st.ems[i];
        h ^= (uint64_t)(g.x + 1) * 1000039ULL;
        h ^= (uint64_t)(g.y + 1) * 1000081ULL;
        h ^= (uint64_t)(g.z + 1) * 1000099ULL;
        h ^= (uint64_t)(g.w + 1) * 1000117ULL;
        h ^= (uint64_t)(g.d + 1) * 1000133ULL;
        h ^= (uint64_t)(g.h + 1) * 1000151ULL;
        h = (h << 7) | (h >> 57);
    }
    return h;
}

struct Keep {
    int id = 0, rot = 0, x = 0, y = 0, z = 0, w = 0, d = 0, h = 0;
    long long waste = 0;
    bool huge = false;
};

// Rank heap helpers (pass 1 keeps only top `beam` scores in memory).
void rank_heap_offer(vector<long long>& h, int cap, long long r) {
    if (cap <= 0) return;
    if ((int)h.size() < cap) {
        h.push_back(r);
        push_heap(h.begin(), h.end(), greater<long long>());
        return;
    }
    if (r <= h.front()) return;
    pop_heap(h.begin(), h.end(), greater<long long>());
    h.back() = r;
    push_heap(h.begin(), h.end(), greater<long long>());
}

long long rank_threshold(const vector<long long>& h, int beam) {
    if (h.empty()) return LLONG_MIN;
    if ((int)h.size() < beam) return LLONG_MIN;
    return h.front();
}

void merge_rank_heaps(vector<long long>& dst, const vector<long long>& src, int cap) {
    for (long long r : src) rank_heap_offer(dst, cap, r);
}

// Enumerate 6 rotations × all EMS/EP candidates. Every remaining legal pose
// becomes a successor; ranking happens later in select_top.
// tight_only: only non-huge cavities with waste <= 2x item vol (sliver fill).
// Otherwise, if any non-huge pose exists, drop huge ones so small SKUs do not
// occupy a cavern meant for later larges.
void collect_placements(const State& u, int id, vector<Keep>& tops, bool tight_only) {
    tops.clear();
    struct Row {
        Keep k;
        long long waste;
        bool huge;
    };
    vector<Row> rows;
    int lastw = -1, lastd = -1, lasth = -1;
    for (int r = 0; r < 6; ++r) {
        int w, d, h;
        rotated_size(id, r, w, d, h);
        if (w == lastw && d == lastd && h == lasth) continue;
        lastw = w;
        lastd = d;
        lasth = h;
        if (!base_stable(w, d, h)) continue;
        if (w > W || d > D || h > H) continue;
        vector<Pt> origins;
        collect_candidate_origins(u, w, d, h, origins);
        for (const Pt& p : origins) {
            int x = p.x, y = p.y, z = p.z;
            if (!feasible(u, x, y, z, w, d, h)) continue;
            Res rs = residual_box(u, x, y, z);
            Keep k{id, r, x, y, z, w, d, h, cavity_waste(rs, w, d, h),
                   cavity_too_big(rs, w, d, h)};
            rows.push_back(Row{k, k.waste, k.huge});
        }
    }
    if (rows.empty()) return;
    if (tight_only) {
        // Sliver pass: ignore caverns and poses that leave >2x waste.
        vector<Row> kept;
        for (const Row& row : rows) {
            if (!row.huge && row.waste <= 2LL * vol[id]) kept.push_back(row);
        }
        rows.swap(kept);
        if (rows.empty()) return;
    } else {
        bool any_tight = false;
        for (const Row& row : rows) {
            if (!row.huge) {
                any_tight = true;
                break;
            }
        }
        if (any_tight) {
            vector<Row> kept;
            for (const Row& row : rows) {
                if (!row.huge) kept.push_back(row);
            }
            rows.swap(kept);
        }
    }
    for (const Row& row : rows) tops.push_back(row.k);
}

void list_pending(const State& u, vector<int>& ids);
void select_top(vector<State>& cand, int beam, vector<State>& nxt);

// Score one pose; return rank and optionally materialize if rank >= thresh.
long long score_pose(const State& u, const Keep& k, long long thresh, vector<State>* out,
                     State& best) {
    State v = u;
    apply_place(v, k.id, k.rot, k.x, k.y, k.z, k.w, k.d, k.h);
    v.last_waste = k.waste;
    v.last_huge = k.huge ? 1 : 0;
    long long r = state_rank(v);
    if (better_state(v, best)) best = v;
    if (out && r >= thresh) out->push_back(std::move(v));
    return r;
}

static const int POSE_PARALLEL_MIN = 24;  // min legal poses before inner parallel

void gather_pending_poses(const State& u, const vector<int>& ids, bool tight_only,
                          vector<Keep>& tops) {
    tops.clear();
    if (u.nplaced >= MAX_PLACED) return;
    vector<Keep> part;
    for (int id : ids) {
        collect_placements(u, id, part, tight_only);
        tops.insert(tops.end(), make_move_iterator(part.begin()), make_move_iterator(part.end()));
    }
}

void parallel_pass1_poses(const State& u, const vector<Keep>& tops, vector<long long>& heap,
                          int cap, State& best) {
    const int n = (int)tops.size();
    if (n == 0) return;
    const int nt = max(1, g_nthreads);

    if (n < POSE_PARALLEL_MIN || nt == 1) {
        for (int i = 0; i < n; ++i) {
            rank_heap_offer(heap, cap, score_pose(u, tops[i], LLONG_MIN, nullptr, best));
        }
        return;
    }

    vector<vector<long long>> local((size_t)nt);
    vector<State> tbest((size_t)nt, best);
    for (auto& h : local) h.reserve((size_t)cap);

    if (omp_in_parallel()) {
        const int chunk = max(POSE_PARALLEL_MIN, (n + nt - 1) / nt);
#pragma omp taskgroup
        for (int lo = 0; lo < n; lo += chunk) {
            int hi = min(n, lo + chunk);
#pragma omp task shared(local, tbest, tops, u, cap) firstprivate(lo, hi)
            {
                int tid = omp_get_thread_num();
                for (int i = lo; i < hi; ++i) {
                    rank_heap_offer(local[tid], cap,
                                    score_pose(u, tops[i], LLONG_MIN, nullptr, tbest[tid]));
                }
            }
        }
    } else {
#pragma omp parallel num_threads(nt)
        {
            int tid = omp_get_thread_num();
#pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) {
                rank_heap_offer(local[tid], cap,
                                score_pose(u, tops[i], LLONG_MIN, nullptr, tbest[tid]));
            }
        }
    }

    for (int t = 0; t < nt; ++t) {
        merge_rank_heaps(heap, local[t], cap);
        if (better_state(tbest[t], best)) best = tbest[t];
    }
}

void parallel_pass2_poses(const State& u, const vector<Keep>& tops, long long thresh,
                          vector<State>& out, State& best) {
    const int n = (int)tops.size();
    if (n == 0) return;
    const int nt = max(1, g_nthreads);

    if (n < POSE_PARALLEL_MIN || nt == 1) {
        for (int i = 0; i < n; ++i) score_pose(u, tops[i], thresh, &out, best);
        return;
    }

    vector<vector<State>> local((size_t)nt);
    vector<State> tbest((size_t)nt, best);

    if (omp_in_parallel()) {
        const int chunk = max(POSE_PARALLEL_MIN, (n + nt - 1) / nt);
#pragma omp taskgroup
        for (int lo = 0; lo < n; lo += chunk) {
            int hi = min(n, lo + chunk);
#pragma omp task shared(local, tbest, tops, u, thresh) firstprivate(lo, hi)
            {
                int tid = omp_get_thread_num();
                for (int i = lo; i < hi; ++i) {
                    score_pose(u, tops[i], thresh, &local[tid], tbest[tid]);
                }
            }
        }
    } else {
#pragma omp parallel num_threads(nt)
        {
            int tid = omp_get_thread_num();
#pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) {
                score_pose(u, tops[i], thresh, &local[tid], tbest[tid]);
            }
        }
    }

    size_t total = 0;
    for (int t = 0; t < nt; ++t) {
        total += local[t].size();
        if (better_state(tbest[t], best)) best = tbest[t];
    }
    out.reserve(out.size() + total);
    for (int t = 0; t < nt; ++t) {
        out.insert(out.end(), make_move_iterator(local[t].begin()),
                   make_move_iterator(local[t].end()));
    }
}

void pass1_pending(const State& u, const vector<int>& ids, vector<long long>& heap, int cap,
                   bool tight_only, State& best, bool* placed_any) {
    vector<Keep> tops;
    gather_pending_poses(u, ids, tight_only, tops);
    if (tops.empty()) return;
    if (placed_any) *placed_any = true;
    parallel_pass1_poses(u, tops, heap, cap, best);
}

void materialize_pending(const State& u, const vector<int>& ids, long long thresh,
                         vector<State>& out, bool tight_only, State& best, bool* placed_any) {
    vector<Keep> tops;
    gather_pending_poses(u, ids, tight_only, tops);
    if (tops.empty()) return;
    if (placed_any) *placed_any = true;
    parallel_pass2_poses(u, tops, thresh, out, best);
}

void pass1_state(const State& u, vector<long long>& heap, int cap, bool tight_only,
                 bool* progress, State& best) {
    if (time_up()) {
        rank_heap_offer(heap, cap, state_rank(u));
        return;
    }
    vector<int> ids;
    ids.reserve(OPEN_CAP);
    list_pending(u, ids);
    if (ids.empty()) {
        rank_heap_offer(heap, cap, state_rank(u));
        return;
    }
    bool placed_any = false;
    pass1_pending(u, ids, heap, cap, tight_only, best, &placed_any);
    if (!placed_any) rank_heap_offer(heap, cap, state_rank(u));
    else if (progress) *progress = true;
}

void pass2_state(const State& u, long long thresh, vector<State>& out, bool tight_only,
                 bool* progress, State& best) {
    if (time_up()) {
        if (state_rank(u) >= thresh) out.push_back(u);
        return;
    }
    vector<int> ids;
    ids.reserve(OPEN_CAP);
    list_pending(u, ids);
    if (ids.empty()) {
        if (state_rank(u) >= thresh) out.push_back(u);
        return;
    }
    bool placed_any = false;
    materialize_pending(u, ids, thresh, out, tight_only, best, &placed_any);
    if (!placed_any && state_rank(u) >= thresh) out.push_back(u);
    else if (placed_any && progress) *progress = true;
}

// One beam layer: pass 1 finds the beam-th highest rank; pass 2 materializes
// every pose with rank >= that cutoff, then select_top keeps `beam` states.
bool buffer_round(vector<State>& cur, vector<State>& cand, vector<State>& nxt, int beam,
                  State& best, bool tight_only) {
    bool progress = false;
    const int ns = (int)cur.size();
    const int nt = max(1, g_nthreads);
    vector<long long> rank_heap;
    rank_heap.reserve((size_t)beam);

    if (nt == 1 || ns <= 1) {
        for (int i = 0; i < ns; ++i) {
            pass1_state(cur[i], rank_heap, beam, tight_only, &progress, best);
        }
    } else {
        vector<vector<long long>> local((size_t)nt);
        vector<State> tbest((size_t)nt, best);
        vector<unsigned char> prog((size_t)nt, 0);
        for (auto& h : local) h.reserve((size_t)beam);
#pragma omp parallel num_threads(nt)
        {
            int tid = omp_get_thread_num();
            bool local_prog = false;
#pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < ns; ++i) {
                pass1_state(cur[i], local[tid], beam, tight_only, &local_prog, tbest[tid]);
            }
            if (local_prog) prog[tid] = 1;
        }
        for (int t = 0; t < nt; ++t) {
            if (prog[t]) progress = true;
            if (better_state(tbest[t], best)) best = tbest[t];
            merge_rank_heaps(rank_heap, local[t], beam);
        }
    }

    const long long thresh = rank_threshold(rank_heap, beam);

    cand.clear();
    if (nt == 1 || ns <= 1) {
        for (int i = 0; i < ns; ++i) {
            pass2_state(cur[i], thresh, cand, tight_only, &progress, best);
        }
    } else {
        vector<vector<State>> local((size_t)nt);
        vector<State> tbest((size_t)nt, best);
        vector<unsigned char> prog((size_t)nt, 0);
#pragma omp parallel num_threads(nt)
        {
            int tid = omp_get_thread_num();
            bool local_prog = false;
#pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < ns; ++i) {
                pass2_state(cur[i], thresh, local[tid], tight_only, &local_prog, tbest[tid]);
            }
            if (local_prog) prog[tid] = 1;
        }
        size_t total = 0;
        for (int t = 0; t < nt; ++t) {
            total += local[t].size();
            if (prog[t]) progress = true;
            if (better_state(tbest[t], best)) best = tbest[t];
        }
        cand.reserve(total);
        for (int t = 0; t < nt; ++t) {
            cand.insert(cand.end(), make_move_iterator(local[t].begin()),
                        make_move_iterator(local[t].end()));
        }
    }

    if (cand.empty()) return false;
    select_top(cand, beam, nxt);
    cand.clear();
    cand.shrink_to_fit();
    if (nxt.empty()) return false;
    cur.swap(nxt);
    return progress;
}
// Unpacked members of g_open for this State (window minus already placed).
int pending_count(const State& u) {
    int c = 0;
    for (int id : g_open) {
        if (!is_packed(u, id)) ++c;
    }
    return c;
}

// Same set as pending_count, in g_open arrival order.
void list_pending(const State& u, vector<int>& ids) {
    ids.clear();
    for (int id : g_open) {
        if (!is_packed(u, id)) ids.push_back(id);
    }
}

// True if any live beam state still has more unpacked items than `max_pending`.
bool any_pending_over(const vector<State>& cur, int max_pending) {
    for (const State& u : cur) {
        if (pending_count(u) > max_pending) return true;
    }
    return false;
}

// Keep `beam` successors: sort by state_rank, dedup by geom_sig, bucket by cavity
// count so one contour family cannot occupy every slot.
void select_top(vector<State>& cand, int beam, vector<State>& nxt) {
    int m = (int)cand.size();
    vector<int> idx(m);
    iota(idx.begin(), idx.end(), 0);
    vector<long long> rank(m);
    const int nt = max(1, g_nthreads);
    if (nt > 1 && m > 8) {
#pragma omp parallel for num_threads(nt) schedule(static)
        for (int i = 0; i < m; ++i) rank[i] = state_rank(cand[i]);
    } else {
        for (int i = 0; i < m; ++i) rank[i] = state_rank(cand[i]);
    }
    sort(idx.begin(), idx.end(), [&](int a, int b) { return rank[a] > rank[b]; });

    nxt.clear();
    unordered_set<uint64_t> seen;
    seen.reserve((size_t)beam * 2);
    // Bucket by cavity count / 6: similar "how fragmented is empty space" layouts
    // share a bucket. Cap ~beam/8 each, overflow fills remaining slots by rank.
    int bucket_used[32] = {};
    int cap_per_bucket = max(2, beam / 8);
    vector<int> overflow;
    vector<char> kept(m, 0);
    for (int id : idx) {
        uint64_t sig = geom_sig(cand[id]);
        if (!seen.insert(sig).second) continue;
        int b = min(31, (cand[id].nep + cand[id].nems) / 6);
        if ((int)nxt.size() < beam && bucket_used[b] < cap_per_bucket) {
            nxt.push_back(cand[id]);
            kept[id] = 1;
            ++bucket_used[b];
        } else {
            overflow.push_back(id);
        }
    }
    for (int id : overflow) {
        if ((int)nxt.size() >= beam) break;
        nxt.push_back(cand[id]);
        kept[id] = 1;
    }
    if (rank_dump_enabled()) rank_dump_layer(cand, rank, idx, kept, m);
}

string load_text(const string& path) {
    ifstream in(path, ios::binary);
    if (!in) return {};
    return string((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
}

string dirname_of(const string& p) {
    auto s = p.find_last_of("/\\");
    if (s == string::npos) return ".";
    return p.substr(0, s);
}

struct BinDump {
    double fill = 0;
    long long packedVol = 0;
    int leftover = 0;
    int seen = 0;
    vector<Place> placed;
};

string pack_json(const vector<BinDump>& bins) {
    ostringstream o;
    o << fixed << setprecision(4);
    o << "{\"W\":" << W << ",\"D\":" << D << ",\"H\":" << H << ",\"n\":" << N << ",\"bins\":[";
    for (size_t b = 0; b < bins.size(); ++b) {
        if (b) o << ',';
        const BinDump& u = bins[b];
        o << "{\"fill\":" << u.fill << ",\"vol\":" << u.packedVol
          << ",\"left\":" << u.leftover << ",\"seen\":" << u.seen << ",\"placed\":[";
        for (size_t i = 0; i < u.placed.size(); ++i) {
            const Place& p = u.placed[i];
            if (i) o << ',';
            o << "{\"id\":" << (p.id + 1) << ",\"x\":" << p.x << ",\"y\":" << p.y << ",\"z\":" << p.z
              << ",\"w\":" << p.w << ",\"d\":" << p.d << ",\"h\":" << p.h << ",\"rot\":" << p.rot << '}';
        }
        o << "]}";
    }
    o << "]}";
    return o.str();
}

void write_html(const vector<BinDump>& bins, const string& argv0) {
    string tmpl;
    const string names[] = {"pack_viewer.html", dirname_of(argv0) + "/pack_viewer.html"};
    for (const string& path : names) {
        tmpl = load_text(path);
        if (tmpl.find("__PACK_DATA__") != string::npos) break;
        tmpl.clear();
    }
    if (tmpl.empty()) {
        cerr << "skip html: pack_viewer.html not found\n";
        return;
    }
    const string needle = "__PACK_DATA__";
    tmpl.replace(tmpl.find(needle), needle.size(), pack_json(bins));
    ofstream("pack3d.html") << tmpl;
    cerr << "wrote pack3d.html\n";
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    auto t_start = chrono::steady_clock::now();
    t0 = t_start;

    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        bool digits = !a.empty() && all_of(a.begin(), a.end(), [](unsigned char c) { return isdigit(c); });
        if (digits) {
            g_nthreads = max(1, stoi(a));
            continue;
        }
        if (a == "ems") {
            g_use_ems = true;
            g_use_ep = false;
            continue;
        }
        if (a == "ep") {
            g_use_ems = false;
            g_use_ep = true;
            continue;
        }
        if (a == "both") {
            g_use_ems = true;
            g_use_ep = true;
            continue;
        }
        if (a == "--rank-dump" && i + 1 < argc) {
            rank_dump_open(argv[++i]);
            continue;
        }
        cerr << "unknown arg '" << a
             << "', expected thread count, both/ems/ep, or --rank-dump <file.jsonl>\n";
        return 1;
    }
    if (!rank_dump_enabled()) {
        const char* env = getenv("PACK3D_RANK_DUMP");
        if (env && env[0]) rank_dump_open(env);
    }
    omp_set_dynamic(0);
    omp_set_num_threads(g_nthreads);
    cerr << fixed << setprecision(4);
    const char* geom = (g_use_ems && g_use_ep) ? "both" : (g_use_ems ? "ems" : "ep");
    cerr << "threads " << g_nthreads << "  geom " << geom << '\n';

    if (!(cin >> W >> D >> H)) return 0;
    {
        int w, d, h;
        while (cin >> w >> d >> h) {
            orig.push_back({w, d, h});
            vol.push_back(1LL * w * d * h);
            min_edge.push_back(min({w, d, h}));
        }
    }
    N = (int)orig.size();

    int beam = 480;     // live partial packs per layer
    long long box_vol = 1LL * W * D * H;
    vector<double> rates;
    vector<BinDump> dumps;
    int next_item = 0;
    int skipped = 0;
    int packed_items = 0;

    // Drop SKUs that cannot enter an empty box; they never block admit/leftover.
    auto skip_unfittable_open = [&]() {
        vector<int> keep;
        keep.reserve(g_open.size());
        for (int id : g_open) {
            if (item_fits_empty(id)) keep.push_back(id);
            else ++skipped;
        }
        g_open.swap(keep);
    };

    // One container: reset the 90s clock. Leftover from the previous box
    // is already in g_open (buffer).
    while (next_item < N || !g_open.empty()) {
        t0 = chrono::steady_clock::now();
        skip_unfittable_open();
        if (g_open.empty() && next_item >= N) break;

        if (rank_dump_enabled()) rank_dump_begin_bin((int)rates.size() + 1);

        vector<State> cur(1, empty_box()), nxt, cand;
        State best;

        // True when some beam state already holds BUFFER_CAP unpacked items.
        auto buffer_full = [&]() -> bool {
            return any_pending_over(cur, BUFFER_CAP - 1);
        };

        // Pull one-at-a-time until the buffer is full or the stream ends.
        auto fill_buffer = [&]() {
            while (next_item < N && !buffer_full()) {
                int id = next_item++;
                g_seen = next_item;
                if (item_fits_empty(id)) g_open.push_back(id);
                else ++skipped;
                update_typical_edge();  // cavity vs sliver uses arrived SKUs only
            }
        };

        // Tight poses first, then any legal pose. False = nobody packed this layer.
        auto pack_one_layer = [&]() -> bool {
            if (buffer_round(cur, cand, nxt, beam, best, true)) return true;
            return buffer_round(cur, cand, nxt, beam, best, false);
        };

        // Fill window → search up to CHUNK_K layers → commit one state.
        // Seal when a chunk packs nothing, or after draining the stream.
        while (true) {
            fill_buffer();

            if (next_item >= N) {
                int guard = 0;
                while (guard++ < 10000 && pack_one_layer()) {
                }
                break;
            }

            int n0 = 0;
            for (const State& u : cur) n0 = max(n0, u.nplaced);
            n0 = max(n0, best.nplaced);

            int got = 0;
            int guard = 0;
            while (got < CHUNK_K && guard++ < 10000) {
                if (!pack_one_layer()) break;
                ++got;
            }
            if (got == 0) {
                break;
            }

            int need = n0;
            for (const State& u : cur) need = max(need, u.nplaced);
            need = min(need, n0 + CHUNK_K);
            const State* pick = nullptr;
            long long pick_rank = 0;
            for (const State& u : cur) {
                if (u.nplaced < need) continue;
                long long r = state_rank(u);
                if (!pick || r > pick_rank || (r == pick_rank && better_state(u, *pick))) {
                    pick = &u;
                    pick_rank = r;
                }
            }
            if (!pick) {
                break;
            }
            State chosen = *pick;
            cur.clear();
            cur.push_back(chosen);
            best = chosen;
        }
        for (const State& u : cur) {
            if (better_state(u, best)) best = u;
        }

        // Empty best: leftover items that still do not fit (should be rare after
        // skip_unfittable_open). Drop them rather than spinning forever.
        if (best.nplaced == 0) {
            if (g_open.empty()) continue;
            skipped += (int)g_open.size();
            g_open.clear();
            continue;
        }

        double rate = box_vol ? 100.0 * best.g / box_vol : 0.0;
        rates.push_back(rate);
        packed_items += best.nplaced;

        // Carry unpacked g_open items to the next container (not discarded).
        vector<int> leftover;
        leftover.reserve(g_open.size());
        for (int id : g_open) {
            if (!is_packed(best, id)) leftover.push_back(id);
        }
        cerr << "bin " << rates.size() << "  packed " << best.nplaced
             << "  fill " << rate << "%  leftover " << leftover.size()
             << "  seen " << g_seen << "/" << N << '\n';
        BinDump dump;
        dump.fill = rate;
        dump.packedVol = best.g;
        dump.leftover = (int)leftover.size();
        dump.seen = g_seen;
        dump.placed.assign(best.placed, best.placed + best.nplaced);
        dumps.push_back(std::move(dump));
        g_open.swap(leftover);
    }

    cout << rates.size() << '\n';
    cout << fixed << setprecision(4);
    for (size_t i = 0; i < rates.size(); ++i) {
        if (i) cout << ' ';
        cout << rates[i];
    }
    cout << '\n';

    write_html(dumps, argc > 0 ? argv[0] : "");

    double wall = chrono::duration<double>(chrono::steady_clock::now() - t_start).count();
    cerr << "containers " << rates.size()
         << "  items " << packed_items << "/" << N
         << "  skipped " << skipped
         << "  mean_fill "
         << (rates.empty() ? 0.0 : accumulate(rates.begin(), rates.end(), 0.0) / rates.size())
         << "%  time " << wall << "s\n";
    rank_dump_close();
    return 0;
}
