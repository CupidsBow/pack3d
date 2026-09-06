/*
  3D container packing — Extreme Points + layered beam search

  ---------------------------------------------------------------------------
  What this program does
  ---------------------------------------------------------------------------
  Pack a stream of rectangular parcels into identical containers using a
  production schedule (not a global offline permutation of all items).
  Geometry is Extreme Points (EP): the left-back-bottom corner of the next
  item is only tried at current corner candidates. Search is a layered beam:
  each layer places one more item, then keeps the best `beam` partial packs.

  Input (stdin):
    W D H
    w d h          (one parcel per line, until EOF)

  Output (stdout):
    N
    r_1 r_2 ... r_N
    N = number of containers. r_i = fill rate of container i, percent of W*D*H.

  Also writes pack3d.html (template pack_viewer.html) with per-bin placements.

  Build and run (OpenMP is required; argv is the thread count, default 1):
    g++ -O2 -std=c++17 -fopenmp -o pack3d pack3d.cpp
    ./pack3d 8 < input.txt > pack3d.out

  ---------------------------------------------------------------------------
  Production window (not "see all remaining SKUs")
  ---------------------------------------------------------------------------
  Items arrive in batches of BATCH_SIZE (10) on a conveyor. Those 10 do not
  occupy the staging buffer. The buffer holds at most BUFFER_CAP (12) unpacked
  items. The next conveyor batch is admitted only after every unpacked item
  fits in the buffer — i.e. pending <= 12 — meaning the current conveyor has
  been packed or parked. At most OPEN_CAP = 22 SKUs are packable at once.

  g_open = all arrived items that are not yet packed in a *previous* container.
  A State's pending set is g_open minus whatever that State already packed.
  Packed items in the current box stay in g_open until the box is sealed.

  Per container loop:
    1. force_room(12): keep the 12 smallest-min-edge SKUs as fillers; try to
       pack the rest (FFD). Repeat until every beam state has pending <= 12,
       then admit the next 10 onto g_open.
    2. If that fails, force_room(0): pack anyone who still fits. If pending
       is still > 12, seal the box. Leftover (unpacked members of g_open)
       go to the next empty container. Packed placements stay in this box.
    3. After each admit, tight-pack: place any pending item that sits in a
       non-huge cavity with waste <= 2x item volume (fill slivers).
    4. When the stream is exhausted, force_room(0) and seal.

  Items larger than the empty box are skipped. Nothing else is dropped.

  ---------------------------------------------------------------------------
  Geometry
  ---------------------------------------------------------------------------
  Origin is left-back-bottom. Six axis rotations. A placement is legal iff
  it is in-bin, AABB-disjoint from packed items, bottom-face support >= 60%,
  and the bottom-face center lies on the floor or on another item's top.

  After placing k, EPs are updated by:
    - dropping points inside k
    - adding k's six +X/+Y/+Z corners
    - Crainic projections of k's vertices onto already packed items
    - reverse: every packed item's XY/YZ/XZ overlap stamped onto k's three
      positive faces (so a ledge packed *after* an overhang still gets EPs)
  If more than MAXEP points remain, subsample across sorted (z,y,x) so both
  floor and ceiling candidates survive (do not keep only the lowest z).

  ---------------------------------------------------------------------------
  Scoring (fill-rate first; no count-first / order-online switch)
  ---------------------------------------------------------------------------
  place_score (same item, different EP/rot):
    huge cavity  ->  vol + contact/3
    else         ->  vol/10 - cavity_waste - 80*height_mismatch + contact/3
  Height matching uses nearby packed tops (xy-gap <= 40). Huge = at least two
  residual axes >= 2x the item, and residual volume >= 5x item volume.

  better_state / reported fill: packed volume, then item count.
  state_rank for beam: g*10000 - compactness*20 + cavity_tiebreak.
  cavity_tiebreak uses residual boxes at EPs vs g_typical_edge (median min-edge
  of items that have already arrived, at least 80): reward usable cavities,
  penalize dead slivers. Top beam candidates also lose 20000 * volume of the
  largest pending SKUs that no longer fit any EP (avoid stranding larges).

  ---------------------------------------------------------------------------
  Beam
  ---------------------------------------------------------------------------
  beam = 480 states per layer, pos_keep = 3 placements per (state, item).
  TIME_LIMIT = 55s per container; after that expand_state copies the state
  unchanged (looks like "cannot pack"). select_top: rank, penalize unplaceable
  larges on the top pool, dedup similar EP signatures, bucket by nep so one
  contour family cannot fill the whole beam.

  One buffer_round = try to place one more item on every live state, then
  keep top `beam`. Expanding those states is OpenMP-parallel; ranking too.
  Layers and containers stay serial (next box depends on leftover).
*/

#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

static const int MAX_PLACED = 256;   // hard cap of items in one container
static const int MAXEP = 640;        // EP list cap; extras are z-strided, not lowest-z only
static const double TIME_LIMIT = 55.0;  // wall seconds per container, then expand stops
static const double SUPPORT_RATIO = 0.60;
static const int BATCH_SIZE = 10;    // conveyor batch; does not consume buffer slots
static const int BUFFER_CAP = 12;    // max unpacked items that may be parked
static const int OPEN_CAP = BUFFER_CAP + BATCH_SIZE;  // packable window
// 6 distinct axis-aligned orientations of (w,d,h).
static const int ROT_PERM[6][3] = {
    {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}
};

struct Pt {
    int x, y, z;
    bool operator<(const Pt& o) const {
        if (z != o.z) return z < o.z;
        if (y != o.y) return y < o.y;
        return x < o.x;
    }
    bool operator==(const Pt& o) const { return x == o.x && y == o.y && z == o.z; }
};

// One packed item: original index `id`, AABB (x,y,z)+(w,d,h), rotation index.
struct Place {
    int id, x, y, z, w, d, h, rot;
};

// Partial packing of the current container. `g` is packed volume. Empty box
// starts with a single EP at the origin. Beam search copies whole States.
struct State {
    long long g = 0;
    int nplaced = 0;
    int nep = 1;
    Place placed[MAX_PLACED];
    Pt eps[MAXEP];
    State() { eps[0] = {0, 0, 0}; }
};

int W, D, H, N;
vector<array<int, 3>> orig;   // input dimensions, index = item id
vector<long long> vol;
vector<int> min_edge;
vector<int> g_open;           // arrived, not packed in a previous container
int g_seen = 0;               // how far the arrival pointer has moved
int g_typical_edge = 80;      // median min-edge of arrived items; cavity vs sliver
int g_nthreads = 1;

// Only items that have already arrived may influence "is this leftover cavity
// still useful". Using the full stream would leak future SKU sizes.
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

void rotated_size(int id, int rot, int& w, int& d, int& h) {
    w = orig[id][ROT_PERM[rot][0]];
    d = orig[id][ROT_PERM[rot][1]];
    h = orig[id][ROT_PERM[rot][2]];
}

bool in_bin(int x, int y, int z, int w, int d, int h) {
    return x >= 0 && y >= 0 && z >= 0 && x + w <= W && y + d <= D && z + h <= H;
}

bool overlap(const Place& a, int x, int y, int z, int w, int d, int h) {
    return x < a.x + a.w && a.x < x + w && y < a.y + a.d && a.y < y + d &&
           z < a.z + a.h && a.z < z + h;
}

int face_overlap(int a0, int a1, int b0, int b1) {
    int lo = max(a0, b0), hi = min(a1, b1);
    return max(0, hi - lo);
}

// Bottom must rest on z=0 or on packed tops covering >= SUPPORT_RATIO of w*d,
// and the bottom-face center must sit on some supporting top (no bridging
// with the center in a gap).
bool well_supported(const State& st, int x, int y, int z, int w, int d) {
    if (z == 0) return true;
    long long bottom = 1LL * w * d;
    if (bottom <= 0) return false;
    long long sup = 0;
    int cx = x + w / 2, cy = y + d / 2;
    bool center_ok = false;
    for (int i = 0; i < st.nplaced; ++i) {
        const Place& p = st.placed[i];
        if (p.z + p.h != z) continue;
        int ox = face_overlap(x, x + w, p.x, p.x + p.w);
        int oy = face_overlap(y, y + d, p.y, p.y + p.d);
        if (ox <= 0 || oy <= 0) continue;
        sup += 1LL * ox * oy;
        if (p.x <= cx && cx < p.x + p.w && p.y <= cy && cy < p.y + p.d) center_ok = true;
    }
    return center_ok && sup * 10000 >= bottom * (long long)llround(SUPPORT_RATIO * 10000);
}

bool feasible(const State& st, int x, int y, int z, int w, int d, int h) {
    if (!in_bin(x, y, z, w, d, h)) return false;
    for (int i = 0; i < st.nplaced; ++i) {
        if (overlap(st.placed[i], x, y, z, w, d, h)) return false;
    }
    return well_supported(st, x, y, z, w, d);
}

struct Res {
    int x, y, z;
};

// Largest empty AABB with min-corner at (x,y,z) growing +x/+y/+z until a
// packed item or the container wall. An intersecting AABB is resolved by
// cutting on the axis that leaves the largest remaining volume (up to 12
// passes). Used for best-fit, "cavity too big", and waste. O(placed) per EP,
// so this is the hot cost inside collect_placements.
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

int xy_gap(int a0, int a1, int b0, int b1) {
    if (a1 <= b0) return b0 - a1;
    if (b1 <= a0) return a0 - b1;
    return 0;
}

// |new top - neighbour top| for packed items that share an xy-edge within 40.
// Zero if there is no such neighbour (do not punish isolated first items).
int height_mismatch(const State& st, int x, int y, int z, int w, int d, int h) {
    int top = z + h;
    int best = INT_MAX / 4;
    for (int i = 0; i < st.nplaced; ++i) {
        const Place& p = st.placed[i];
        int ox = face_overlap(x, x + w, p.x, p.x + p.w);
        int oy = face_overlap(y, y + d, p.y, p.y + p.d);
        int gx = xy_gap(x, x + w, p.x, p.x + p.w);
        int gy = xy_gap(y, y + d, p.y, p.y + p.d);
        bool neigh = (ox > 0 && gy <= 40) || (oy > 0 && gx <= 40);
        if (!neigh) continue;
        best = min(best, abs(top - (p.z + p.h)));
    }
    if (best > INT_MAX / 8) return 0;
    return best;
}

// Beam tie-break from leftover cavities at current EPs. A residual shorter
// than g_typical_edge is treated as a dead sliver (future arrived SKUs are
// unlikely to fit). Reward one large usable pocket; penalize fragmented waste.
long long cavity_tiebreak(const State& st) {
    long long max_cav = 0, sum_cav = 0, dead = 0;
    int seen = min(st.nep, MAXEP);
    for (int e = 0; e < seen; ++e) {
        Res r = residual_box(st, st.eps[e].x, st.eps[e].y, st.eps[e].z);
        int me = min({r.x, r.y, r.z});
        long long v = 1LL * r.x * r.y * r.z;
        if (me >= g_typical_edge) {
            max_cav = max(max_cav, v);
            sum_cav += v;
        } else {
            dead += v;
        }
    }
    return max_cav / 1000 + sum_cav / 8000 - dead * 4;
}

// Contact with container walls plus coplanar faces of packed items. Favours
// placements that stack flush rather than floating in the middle of a void.
long long contact_area(const State& st, int x, int y, int z, int w, int d, int h) {
    long long c = 0;
    if (x == 0) c += 1LL * d * h;
    if (y == 0) c += 1LL * w * h;
    if (z == 0) c += 1LL * w * d;
    if (x + w == W) c += 1LL * d * h;
    if (y + d == D) c += 1LL * w * h;
    if (z + h == H) c += 1LL * w * d;
    int x2 = x + w, y2 = y + d, z2 = z + h;
    for (int i = 0; i < st.nplaced; ++i) {
        const Place& p = st.placed[i];
        int px2 = p.x + p.w, py2 = p.y + p.d, pz2 = p.z + p.h;
        if (x2 == p.x || px2 == x) {
            c += 1LL * face_overlap(y, y2, p.y, py2) * face_overlap(z, z2, p.z, pz2);
        }
        if (y2 == p.y || py2 == y) {
            c += 1LL * face_overlap(x, x2, p.x, px2) * face_overlap(z, z2, p.z, pz2);
        }
        if (z2 == p.z || pz2 == z) {
            c += 1LL * face_overlap(x, x2, p.x, px2) * face_overlap(y, y2, p.y, py2);
        }
    }
    return c;
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

// Commit a pose: append Place, add volume, rebuild EPs from the new AABB.
void apply_place(State& st, int id, int rot, int x, int y, int z, int w, int d, int h) {
    if (st.nplaced >= MAX_PLACED) return;
    Place p{id, x, y, z, w, d, h, rot};
    st.placed[st.nplaced++] = p;
    st.g += vol[id];
    update_eps(st, p);
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
        if (w <= W && d <= D && h <= H) return true;
    }
    return false;
}

// Spread penalty: sum of min-corners + bounding-box footprint + EP count.
// Subtracted in state_rank so two equal-volume packs prefer the denser one
// (items pulled toward origin, fewer leftover corners).
long long compactness(const State& st) {
    long long c = 0;
    int maxx = 0, maxy = 0, maxz = 0;
    for (int i = 0; i < st.nplaced; ++i) {
        const Place& p = st.placed[i];
        c += p.x + p.y + p.z;
        maxx = max(maxx, p.x + p.w);
        maxy = max(maxy, p.y + p.d);
        maxz = max(maxz, p.z + p.h);
    }
    c += 1LL * maxx * maxy / 8 + maxz;
    c += 3LL * st.nep;
    return c;
}

// Any EP + rotation that is currently feasible. Used only to detect stranded
// large pending SKUs (not to pick a pose).
bool item_fits_somewhere(const State& st, int id) {
    int lw = -1, ld = -1, lh = -1;
    for (int r = 0; r < 6; ++r) {
        int w, d, h;
        rotated_size(id, r, w, d, h);
        if (w == lw && d == ld && h == lh) continue;
        lw = w;
        ld = d;
        lh = h;
        if (w > W || d > D || h > H) continue;
        for (int e = 0; e < st.nep; ++e) {
            if (feasible(st, st.eps[e].x, st.eps[e].y, st.eps[e].z, w, d, h)) return true;
        }
    }
    return false;
}

// Penalise layouts that already cannot place the largest pending SKUs.
// Only the 4 largest unpacked items in the window are checked (EP scan).
long long pending_unplaceable_vol(const State& st) {
    int ids[OPEN_CAP], nids = 0;
    for (int id : g_open) {
        if (!is_packed(st, id)) {
            if (nids < OPEN_CAP) ids[nids++] = id;
        }
    }
    if (nids == 0) return 0;
    sort(ids, ids + nids, [](int a, int b) { return vol[a] > vol[b]; });
    int take = min(4, nids);
    long long pen = 0;
    for (int t = 0; t < take; ++t) {
        if (!item_fits_somewhere(st, ids[t])) pen += vol[ids[t]];
    }
    return pen;
}

// Same SKU, different pose: prefer tight cavities, height-aligned stacks,
// and wall/item contact. `r` is the residual already computed at this EP.
long long place_score(const State& st, const Res& r, int x, int y, int z, int w, int d, int h,
                      int id) {
    long long contact = contact_area(st, x, y, z, w, d, h);
    if (cavity_too_big(r, w, d, h)) return vol[id] + contact / 3;
    return vol[id] / 10 - cavity_waste(r, w, d, h) - 80LL * height_mismatch(st, x, y, z, w, d, h) +
           contact / 3;
}

// Layer ranking: packed volume dominates; compactness and leftover cavities
// only break ties among similar fills.
long long state_rank(const State& st) {
    return st.g * 10000LL - compactness(st) * 20 + cavity_tiebreak(st);
}

// Reported / "best packing" comparison: volume first, then how many items.
bool better_state(const State& a, const State& b) {
    if (a.g != b.g) return a.g > b.g;
    return a.nplaced > b.nplaced;
}

// Fingerprint of a packing for beam dedup (ids + a few EPs). Same geometry
// reached by different placement orders collapses to one beam slot.
uint64_t ep_sig(const State& st) {
    uint64_t h = st.g ^ (uint64_t)st.nplaced * 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < st.nplaced; ++i) {
        h ^= (uint64_t)(st.placed[i].id + 1) * 1000003ULL;
        h = (h << 7) | (h >> 57);
    }
    int take = min(st.nep, 12);
    for (int i = 0; i < take; ++i) {
        h ^= (uint64_t)(st.eps[i].x + 1) * 1000003ULL;
        h ^= (uint64_t)(st.eps[i].y + 1) * 1000033ULL;
        h ^= (uint64_t)(st.eps[i].z + 1) * 1000037ULL;
        h = (h << 7) | (h >> 57);
    }
    return h;
}

struct Keep {
    long long ps;
    int id, rot, x, y, z, w, d, h;
};

// Enumerate 6 rotations × all EPs, score legal poses, keep pos_keep best.
// tight_only: only non-huge cavities with waste <= 2x item vol (post-admit
// sliver fill). Otherwise, if any non-huge pose exists, drop huge ones so
// small SKUs do not occupy a cavern meant for later larges.
void collect_placements(const State& u, int id, int pos_keep, vector<Keep>& tops, bool tight_only) {
    tops.clear();
    struct Row {
        Keep k;
        Res r;
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
        if (w > W || d > D || h > H) continue;
        for (int e = 0; e < u.nep; ++e) {
            int x = u.eps[e].x, y = u.eps[e].y, z = u.eps[e].z;
            if (!feasible(u, x, y, z, w, d, h)) continue;
            Res rs = residual_box(u, x, y, z);
            Keep k{place_score(u, rs, x, y, z, w, d, h, id), id, r, x, y, z, w, d, h};
            rows.push_back(Row{k, rs, cavity_waste(rs, w, d, h), cavity_too_big(rs, w, d, h)});
        }
    }
    if (rows.empty()) return;
    if (tight_only) {
        // Post-admit sliver pass: ignore caverns and poses that leave >2x waste.
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
            // Normal FFD: a small SKU must not sit in a huge leftover cavity
            // when a tighter pose exists (keep the cavern for a later large).
            vector<Row> kept;
            for (const Row& row : rows) {
                if (!row.huge) kept.push_back(row);
            }
            rows.swap(kept);
        }
    }
    sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.k.ps > b.k.ps; });
    int take = min(pos_keep, (int)rows.size());
    for (int i = 0; i < take; ++i) tops.push_back(rows[i].k);
}

// Clone `u` once per kept pose and commit that placement into the candidate list.
void push_keeps(const State& u, const vector<Keep>& tops, vector<State>& cand) {
    for (const Keep& k : tops) {
        State v = u;
        apply_place(v, k.id, k.rot, k.x, k.y, k.z, k.w, k.d, k.h);
        cand.push_back(v);
    }
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

// FFD order: park by smallest min-edge first (then volume). Those stay in buffer.
bool smaller_filler(int a, int b) {
    if (min_edge[a] != min_edge[b]) return min_edge[a] < min_edge[b];
    return vol[a] < vol[b];
}

// FFD leave-set: park the `max_pending` smallest-min-edge items in the
// buffer; everyone else must be packed this round. max_pending 0 / <0 =
// attempt every pending SKU (seal squeeze).
void must_leave_pending(vector<int> ids, int max_pending, vector<int>& leave) {
    leave.clear();
    if (ids.empty()) return;
    if (max_pending < 0 || max_pending == 0) {
        leave = std::move(ids);
        return;
    }
    sort(ids.begin(), ids.end(), smaller_filler);
    if ((int)ids.size() <= max_pending) return;
    leave.assign(ids.begin() + max_pending, ids.end());
}

void select_top(vector<State>& cand, int beam, vector<State>& nxt);

// Try each SKU in `ids` independently (one extra item per successor). Does not
// pack two items in the same expansion; the next buffer_round does the next one.
void expand_pending(const State& u, const vector<int>& ids, int pos_keep, vector<State>& cand,
                    bool tight_only) {
    if (u.nplaced >= MAX_PLACED) return;
    vector<Keep> tops;
    for (int id : ids) {
        collect_placements(u, id, pos_keep, tops, tight_only);
        push_keeps(u, tops, cand);
    }
}

// One successor generation from state `u`.
//   time_up / already under max_pending / nothing to pack -> copy `u` through.
//   tight_only: every pending SKU is a candidate (sliver fill after admit).
//   else: FFD leave-set from must_leave_pending (pack larges, park fillers).
// If no legal pose exists, copy `u` unchanged so the beam does not die;
// progress stays false and the outer loop can seal.
void expand_state(const State& u, vector<State>& out, int pos_keep, int max_pending, bool tight_only,
                 bool* progress) {
    if (time_up()) {
        out.push_back(u);
        return;
    }
    vector<int> ids;
    ids.reserve(OPEN_CAP);
    list_pending(u, ids);
    int pc = (int)ids.size();
    if (pc == 0 || (max_pending >= 0 && pc <= max_pending)) {
        out.push_back(u);
        return;
    }
    vector<int> leave;
    if (tight_only) leave = ids;
    else must_leave_pending(ids, max_pending, leave);
    if (leave.empty()) {
        out.push_back(u);
        return;
    }
    size_t n0 = out.size();
    expand_pending(u, leave, pos_keep, out, tight_only);
    if (out.size() == n0) out.push_back(u);
    else if (progress) *progress = true;
}

// One beam layer: each live state tries to pack one more pending item.
// max_pending >= 0 stops expanding once pending is small enough (admit).
// tight_only uses the sliver filter. States are independent; OpenMP splits
// `cur` with per-thread candidate lists, then ranks and truncates to `beam`.
bool buffer_round(vector<State>& cur, vector<State>& cand, vector<State>& nxt, int beam,
                  int pos_keep, State& best, int max_pending, bool tight_only) {
    cand.clear();
    bool progress = false;
    const int ns = (int)cur.size();
    const int nt = max(1, g_nthreads);
    if (nt == 1 || ns <= 1) {
        for (const State& u : cur) {
            expand_state(u, cand, pos_keep, max_pending, tight_only, &progress);
        }
    } else {
        // Independent expand_state per beam member. dynamic,1 because residual
        // cost varies with nep × placed. Merge thread-local lists after the barrier.
        vector<vector<State>> local((size_t)nt);
        vector<unsigned char> prog((size_t)nt, 0);
        for (auto& v : local) v.reserve((cand.capacity() / (size_t)nt) + 8);
#pragma omp parallel num_threads(nt)
        {
            int tid = omp_get_thread_num();
            bool local_prog = false;
#pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < ns; ++i) {
                expand_state(cur[i], local[tid], pos_keep, max_pending, tight_only, &local_prog);
            }
            if (local_prog) prog[tid] = 1;
        }
        size_t total = 0;
        for (int t = 0; t < nt; ++t) {
            total += local[t].size();
            if (prog[t]) progress = true;
        }
        cand.reserve(total);
        for (int t = 0; t < nt; ++t) {
            cand.insert(cand.end(), make_move_iterator(local[t].begin()),
                        make_move_iterator(local[t].end()));
        }
    }
    if (cand.empty()) return false;
    for (const State& v : cand) {
        if (better_state(v, best)) best = v;
    }
    select_top(cand, beam, nxt);
    if (nxt.empty()) return false;
    cur.swap(nxt);
    return progress;
}

// True if any live beam state still has more unpacked items than the admit cap.
bool any_pending_over(const vector<State>& cur, int max_pending) {
    for (const State& u : cur) {
        if (pending_count(u) > max_pending) return true;
    }
    return false;
}

// Keep `beam` successors: sort by state_rank, then on the top pool subtract
// unplaceable large-SKU volume so a slightly emptier layout that can still
// take a big pending item beats a greedy fill that traps it. Dedup by ep_sig
// and bucket by EP count so one contour family cannot occupy every slot.
void select_top(vector<State>& cand, int beam, vector<State>& nxt) {
    int m = (int)cand.size();
    vector<int> idx(m);
    iota(idx.begin(), idx.end(), 0);
    vector<long long> rank(m);
    const int nt = max(1, g_nthreads);
    if (nt > 1 && m > 8) {
        // residual_box inside cavity_tiebreak is the expensive part of rank.
#pragma omp parallel for num_threads(nt) schedule(static)
        for (int i = 0; i < m; ++i) rank[i] = state_rank(cand[i]);
    } else {
        for (int i = 0; i < m; ++i) rank[i] = state_rank(cand[i]);
    }
    sort(idx.begin(), idx.end(), [&](int a, int b) { return rank[a] > rank[b]; });
    int pool = min(m, max(beam * 3, beam));
    if (nt > 1 && pool > 8) {
#pragma omp parallel for num_threads(nt) schedule(static)
        for (int i = 0; i < pool; ++i) {
            rank[idx[i]] -= pending_unplaceable_vol(cand[idx[i]]) * 20000LL;
        }
    } else {
        for (int i = 0; i < pool; ++i) {
            rank[idx[i]] -= pending_unplaceable_vol(cand[idx[i]]) * 20000LL;
        }
    }
    sort(idx.begin(), idx.begin() + pool, [&](int a, int b) { return rank[a] > rank[b]; });

    nxt.clear();
    unordered_set<uint64_t> seen;
    seen.reserve((size_t)beam * 2);
    // Bucket by EP count / 6: similar "how fragmented is empty space" layouts
    // share a bucket. Cap ~beam/8 each, overflow fills remaining slots by rank.
    int bucket_used[32] = {};
    int cap_per_bucket = max(2, beam / 8);
    vector<int> overflow;
    for (int id : idx) {
        uint64_t sig = ep_sig(cand[id]);
        if (!seen.insert(sig).second) continue;
        int b = min(31, cand[id].nep / 6);
        if ((int)nxt.size() < beam && bucket_used[b] < cap_per_bucket) {
            nxt.push_back(cand[id]);
            ++bucket_used[b];
        } else {
            overflow.push_back(id);
        }
    }
    for (int id : overflow) {
        if ((int)nxt.size() >= beam) break;
        nxt.push_back(cand[id]);
    }
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
        cerr << "unknown arg '" << a << "', expected a thread count\n";
        return 1;
    }
    omp_set_dynamic(0);
    omp_set_num_threads(g_nthreads);
    cerr << "threads " << g_nthreads << '\n';

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
    int pos_keep = 3;   // poses kept per (state, SKU) after scoring EPs
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

    // One container: reset the 55s clock. Leftover from the previous box
    // is already in g_open (buffer / unfinished conveyor).
    while (next_item < N || !g_open.empty()) {
        t0 = chrono::steady_clock::now();
        skip_unfittable_open();
        if (g_open.empty() && next_item >= N) break;

        vector<State> cur(1), nxt, cand;
        cand.reserve(beam * (size_t)OPEN_CAP * (size_t)pos_keep + 8);
        State best;

        // Pack until every beam member has pending <= max_pend (FFD if
        // max_pend > 0). False means some state is stuck above the cap.
        auto force_room = [&](int max_pend) -> bool {
            int guard = 0;
            while (any_pending_over(cur, max_pend) && guard++ < 10000) {
                if (!buffer_round(cur, cand, nxt, beam, pos_keep, best, max_pend, false)) {
                    break;
                }
            }
            return !any_pending_over(cur, max_pend);
        };

        while (true) {
            int incoming = min(BATCH_SIZE, N - next_item);
            if (incoming > 0) {
                // Admit next conveyor batch only when unpacked count <= 12.
                int need = BUFFER_CAP;
                if (!force_room(need)) {
                    force_room(0);  // squeeze fillers too
                    if (any_pending_over(cur, need)) break;  // seal this box
                }
                for (int i = 0; i < incoming; ++i) {
                    int id = next_item++;
                    g_seen = next_item;
                    if (item_fits_empty(id)) g_open.push_back(id);
                    else ++skipped;
                }
                update_typical_edge();  // cavity vs sliver uses arrived SKUs only
            }

            // After a batch arrives, pack any pending SKU that fits a tight
            // cavity (does not require FFD). Stops when nothing tight remains.
            int guard = 0;
            while (guard++ < 10000) {
                if (!buffer_round(cur, cand, nxt, beam, pos_keep, best, -1, true)) break;
            }

            if (incoming == 0) {
                force_room(0);  // stream done: pack whatever still fits
                break;
            }
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
    return 0;
}
