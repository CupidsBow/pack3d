#include "rank.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <utility>

using namespace std;

static const long long W_G = 10000;
static const long long W_N = 1;
static const long long W_COMPACT = 20;
static const long long W_HM = 80;
static const long long W_UNPLACE = 20000;

static const char* kFeatNames[RANK_DIM] = {
    "fill",           "nplaced",        "floor_cover",    "maxz_h",
    "origin_used",    "minx_w",         "miny_d",         "maxx_w",
    "maxy_d",         "wall_contact",   "item_contact",   "nems",
    "nep",            "max_cav_vol",    "sum_usable_cav", "dead_cav",
    "max_cav_w",      "max_cav_d",      "max_cav_h",      "pending_n",
    "pending_top4_v", "pending_top4_e", "unplaceable",    "compactness",
    "last_lbb",       "last_waste",     "last_hm",        "last_walls",
};

const char* rank_feature_name(int i) {
    if (i < 0 || i >= RANK_DIM) return "?";
    return kFeatNames[i];
}

static int face_overlap(int a0, int a1, int b0, int b1) {
    int lo = max(a0, b0), hi = min(a1, b1);
    return max(0, hi - lo);
}

static int xy_gap(int a0, int a1, int b0, int b1) {
    if (a1 <= b0) return b0 - a1;
    if (b1 <= a0) return a0 - b1;
    return 0;
}

static long long box_volume() { return 1LL * W * D * H; }

static long long wall_surface() {
    return 2LL * (1LL * W * D + 1LL * W * H + 1LL * D * H);
}

static void contact_last(const State& st, long long& wall, long long& item) {
    wall = item = 0;
    if (st.nplaced <= 0) return;
    const Place& a = st.placed[st.nplaced - 1];
    int x = a.x, y = a.y, z = a.z, w = a.w, d = a.d, h = a.h;
    if (x == 0) wall += 1LL * d * h;
    if (y == 0) wall += 1LL * w * h;
    if (z == 0) wall += 1LL * w * d;
    if (x + w == W) wall += 1LL * d * h;
    if (y + d == D) wall += 1LL * w * h;
    if (z + h == H) wall += 1LL * w * d;
    int x2 = x + w, y2 = y + d, z2 = z + h;
    for (int i = 0; i < st.nplaced - 1; ++i) {
        const Place& p = st.placed[i];
        int px2 = p.x + p.w, py2 = p.y + p.d, pz2 = p.z + p.h;
        if (x2 == p.x || px2 == x) {
            item += 1LL * face_overlap(y, y2, p.y, py2) * face_overlap(z, z2, p.z, pz2);
        }
        if (y2 == p.y || py2 == y) {
            item += 1LL * face_overlap(x, x2, p.x, px2) * face_overlap(z, z2, p.z, pz2);
        }
        if (z2 == p.z || pz2 == z) {
            item += 1LL * face_overlap(x, x2, p.x, px2) * face_overlap(y, y2, p.y, py2);
        }
    }
}

static void contact_all(const State& st, long long& wall, long long& item) {
    wall = item = 0;
    for (int i = 0; i < st.nplaced; ++i) {
        const Place& a = st.placed[i];
        int x = a.x, y = a.y, z = a.z, w = a.w, d = a.d, h = a.h;
        if (x == 0) wall += 1LL * d * h;
        if (y == 0) wall += 1LL * w * h;
        if (z == 0) wall += 1LL * w * d;
        if (x + w == W) wall += 1LL * d * h;
        if (y + d == D) wall += 1LL * w * h;
        if (z + h == H) wall += 1LL * w * d;
        int x2 = x + w, y2 = y + d, z2 = z + h;
        for (int j = 0; j < i; ++j) {
            const Place& p = st.placed[j];
            int px2 = p.x + p.w, py2 = p.y + p.d, pz2 = p.z + p.h;
            if (x2 == p.x || px2 == x) {
                item += 1LL * face_overlap(y, y2, p.y, py2) * face_overlap(z, z2, p.z, pz2);
            }
            if (y2 == p.y || py2 == y) {
                item += 1LL * face_overlap(x, x2, p.x, px2) * face_overlap(z, z2, p.z, pz2);
            }
            if (z2 == p.z || pz2 == z) {
                item += 1LL * face_overlap(x, x2, p.x, px2) * face_overlap(y, y2, p.y, py2);
            }
        }
    }
}

static int height_mismatch_last(const State& st) {
    if (st.nplaced <= 0) return 0;
    const Place& a = st.placed[st.nplaced - 1];
    int top = a.z + a.h;
    int best = INT_MAX / 4;
    for (int i = 0; i < st.nplaced - 1; ++i) {
        const Place& p = st.placed[i];
        int ox = face_overlap(a.x, a.x + a.w, p.x, p.x + p.w);
        int oy = face_overlap(a.y, a.y + a.d, p.y, p.y + p.d);
        int gx = xy_gap(a.x, a.x + a.w, p.x, p.x + p.w);
        int gy = xy_gap(a.y, a.y + a.d, p.y, p.y + p.d);
        bool neigh = (ox > 0 && gy <= 40) || (oy > 0 && gx <= 40);
        if (!neigh) continue;
        best = min(best, abs(top - (p.z + p.h)));
    }
    if (best > INT_MAX / 8) return 0;
    return best;
}

static int wall_count_last(const State& st) {
    if (st.nplaced <= 0) return 0;
    const Place& a = st.placed[st.nplaced - 1];
    int n = 0;
    if (a.x == 0) ++n;
    if (a.y == 0) ++n;
    if (a.z == 0) ++n;
    if (a.x + a.w == W) ++n;
    if (a.y + a.d == D) ++n;
    if (a.z + a.h == H) ++n;
    return n;
}

static long long compactness_raw(const State& st, int& maxx, int& maxy, int& maxz, int& minx,
                                 int& miny) {
    long long c = 0;
    maxx = maxy = maxz = 0;
    minx = W;
    miny = D;
    for (int i = 0; i < st.nplaced; ++i) {
        const Place& p = st.placed[i];
        c += p.x + p.y + p.z;
        maxx = max(maxx, p.x + p.w);
        maxy = max(maxy, p.y + p.d);
        maxz = max(maxz, p.z + p.h);
        minx = min(minx, p.x);
        miny = min(miny, p.y);
    }
    if (st.nplaced == 0) {
        minx = miny = 0;
    }
    c += 1LL * maxx * maxy / 8 + maxz;
    c += 3LL * (g_use_ep ? st.nep : 0) + 3LL * (g_use_ems ? st.nems : 0);
    return c;
}

struct CavStats {
    long long max_vol = 0;
    long long sum_usable = 0;
    long long dead = 0;
    int max_w = 0, max_d = 0, max_h = 0;
};

static CavStats cavity_stats(const State& st) {
    CavStats s;
    if (g_use_ems) {
        int seen = min(st.nems, MAXEMS);
        for (int e = 0; e < seen; ++e) {
            const EmsBox& g = st.ems[e];
            int me = min({g.w, g.d, g.h});
            long long v = 1LL * g.w * g.d * g.h;
            if (me >= g_typical_edge) {
                if (v > s.max_vol) {
                    s.max_vol = v;
                    s.max_w = g.w;
                    s.max_d = g.d;
                    s.max_h = g.h;
                }
                s.sum_usable += v;
            } else {
                s.dead += v;
            }
        }
    } else {
        int seen = min(st.nep, MAXEP);
        for (int e = 0; e < seen; ++e) {
            Res r = residual_box(st, st.eps[e].x, st.eps[e].y, st.eps[e].z);
            int me = min({r.x, r.y, r.z});
            long long v = 1LL * r.x * r.y * r.z;
            if (me >= g_typical_edge) {
                if (v > s.max_vol) {
                    s.max_vol = v;
                    s.max_w = r.x;
                    s.max_d = r.y;
                    s.max_h = r.z;
                }
                s.sum_usable += v;
            } else {
                s.dead += v;
            }
        }
    }
    return s;
}

static long long cavity_tiebreak(const State& st) {
    CavStats s = cavity_stats(st);
    return s.max_vol / 1000 + s.sum_usable / 8000 - s.dead * 4;
}

static bool sku_fits_some_ems(const State& st, int id) {
    int lw = -1, ld = -1, lh = -1;
    for (int r = 0; r < 6; ++r) {
        int w, d, h;
        rotated_size(id, r, w, d, h);
        if (w == lw && d == ld && h == lh) continue;
        lw = w;
        ld = d;
        lh = h;
        if (!base_stable(w, d, h)) continue;
        for (int e = 0; e < st.nems; ++e) {
            const EmsBox& g = st.ems[e];
            if (w <= g.w && d <= g.d && h <= g.h) return true;
        }
    }
    return false;
}

static long long pending_unplaceable_vol(const State& st) {
    if (!g_use_ems) return 0;
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
        if (!sku_fits_some_ems(st, ids[t])) pen += vol[ids[t]];
    }
    return pen;
}

static void pending_top4(const State& st, int& cnt, long long& top4vol, float& top4edge) {
    cnt = 0;
    top4vol = 0;
    top4edge = 0;
    int ids[OPEN_CAP], nids = 0;
    for (int id : g_open) {
        if (!is_packed(st, id)) {
            if (nids < OPEN_CAP) ids[nids++] = id;
        }
    }
    cnt = nids;
    if (nids == 0) return;
    sort(ids, ids + nids, [](int a, int b) { return vol[a] > vol[b]; });
    int take = min(4, nids);
    long long esum = 0;
    for (int t = 0; t < take; ++t) {
        top4vol += vol[ids[t]];
        esum += min_edge[ids[t]];
    }
    top4edge = (float)esum / take;
}

static float clamp01(float x) {
    if (x < 0.f) return 0.f;
    if (x > 1.f) return 1.f;
    return x;
}

static bool origin_occupied(const State& st) {
    for (int i = 0; i < st.nplaced; ++i) {
        const Place& p = st.placed[i];
        if (p.x <= 0 && p.y <= 0 && p.z <= 0 && p.x + p.w > 0 && p.y + p.d > 0 && p.z + p.h > 0) {
            return true;
        }
    }
    return false;
}

static float floor_cover_ratio(const State& st) {
    long long floor = 0;
    for (int i = 0; i < st.nplaced; ++i) {
        const Place& p = st.placed[i];
        if (p.z == 0) floor += 1LL * p.w * p.d;
    }
    long long cap = 1LL * W * D;
    if (cap <= 0) return 0.f;
    return clamp01((float)floor / (float)cap);
}

RankVec rank_features_vec(const State& st) {
    RankVec out;
    const long long bvol = max(1LL, box_volume());
    const long long wsurf = max(1LL, wall_surface());
    const float typ = (float)max(1, g_typical_edge);
    const float wh = (float)max(1, W + D + H);

    int maxx, maxy, maxz, minx, miny;
    long long compact = compactness_raw(st, maxx, maxy, maxz, minx, miny);
    CavStats cav = cavity_stats(st);
    long long wall_c, item_c;
    contact_all(st, wall_c, item_c);

    int pend_n = 0;
    long long top4v = 0;
    float top4e = 0;
    pending_top4(st, pend_n, top4v, top4e);
    long long unpl = pending_unplaceable_vol(st);

    float last_lbb = 0.f;
    int last_walls = 0;
    if (st.nplaced > 0) {
        const Place& a = st.placed[st.nplaced - 1];
        last_lbb = (float)(a.x + a.y + a.z) / wh;
        last_walls = wall_count_last(st);
    }

    float compact_norm = (float)compact / (wh * (float)max(1, st.nplaced) + (float)bvol / 1000.f);

    out.v[0] = clamp01((float)st.g / (float)bvol);
    out.v[1] = clamp01((float)st.nplaced / (float)max(1, MAX_PLACED));
    out.v[2] = floor_cover_ratio(st);
    out.v[3] = clamp01((float)maxz / (float)max(1, H));
    out.v[4] = origin_occupied(st) ? 1.f : 0.f;
    out.v[5] = clamp01((float)minx / (float)max(1, W));
    out.v[6] = clamp01((float)miny / (float)max(1, D));
    out.v[7] = clamp01((float)maxx / (float)max(1, W));
    out.v[8] = clamp01((float)maxy / (float)max(1, D));
    out.v[9] = clamp01((float)wall_c / (float)wsurf);
    out.v[10] = clamp01((float)item_c / (float)wsurf);
    out.v[11] = clamp01((float)st.nems / (float)MAXEMS);
    out.v[12] = clamp01((float)st.nep / (float)MAXEP);
    out.v[13] = clamp01((float)cav.max_vol / (float)bvol);
    out.v[14] = clamp01((float)cav.sum_usable / (float)bvol);
    out.v[15] = clamp01((float)cav.dead / (float)bvol);
    out.v[16] = clamp01((float)cav.max_w / (float)max(1, W));
    out.v[17] = clamp01((float)cav.max_d / (float)max(1, D));
    out.v[18] = clamp01((float)cav.max_h / (float)max(1, H));
    out.v[19] = clamp01((float)pend_n / (float)OPEN_CAP);
    out.v[20] = clamp01((float)top4v / (float)bvol);
    out.v[21] = clamp01(top4e / typ);
    out.v[22] = clamp01((float)unpl / (float)bvol);
    out.v[23] = clamp01(compact_norm);
    out.v[24] = clamp01(last_lbb);
    out.v[25] = clamp01((float)st.last_waste / (float)bvol);
    out.v[26] = clamp01((float)height_mismatch_last(st) / (float)max(1, H));
    out.v[27] = clamp01((float)last_walls / 6.f);
    return out;
}

long long state_rank(const State& st) {
    int maxx, maxy, maxz, minx, miny;
    long long compact = compactness_raw(st, maxx, maxy, maxz, minx, miny);
    long long wall_last, item_last;
    contact_last(st, wall_last, item_last);
    (void)item_last;
    long long lid = 0;
    if (st.nplaced > 0) {
        const Place& a = st.placed[st.nplaced - 1];
        if (a.z + a.h == H) lid = 3LL * a.w * a.d * max(g_typical_edge, 180);
    }
    return st.g * W_G + st.nplaced * W_N - compact * W_COMPACT + (wall_last + item_last) / 3 +
           cavity_tiebreak(st) - st.last_waste - W_HM * height_mismatch_last(st) - lid -
           pending_unplaceable_vol(st) * W_UNPLACE;
}

// ---- JSONL dump ----

static ofstream g_dump_out;
static bool g_dump_on = false;
static int g_dump_bin = 0;
static int g_dump_layer = 0;

static const int DUMP_CAND_CAP = 4096;
static const int DUMP_PAIR_CAP = 512;

void rank_dump_open(const string& path) {
    rank_dump_close();
    g_dump_out.open(path, ios::out | ios::trunc);
    g_dump_on = g_dump_out.good();
    if (g_dump_on) cerr << "rank dump -> " << path << '\n';
}

void rank_dump_close() {
    if (g_dump_out.is_open()) g_dump_out.close();
    g_dump_on = false;
}

bool rank_dump_enabled() { return g_dump_on; }

void rank_dump_begin_bin(int bin_index) {
    g_dump_bin = bin_index;
    g_dump_layer = 0;
}

static void append_phi_json(ostringstream& o, const RankVec& phi) {
    o << "[";
    o << fixed << setprecision(6);
    for (int i = 0; i < RANK_DIM; ++i) {
        if (i) o << ',';
        o << phi.v[i];
    }
    o << "]";
}

void rank_dump_layer(const vector<State>& cand, const vector<long long>& rank,
                     const vector<int>& order, const vector<char>& kept, int m) {
    if (!g_dump_on || m <= 0) return;
    ++g_dump_layer;

    const int take = min(m, DUMP_CAND_CAP);
    vector<RankVec> phis((size_t)take);
    for (int t = 0; t < take; ++t) {
        int id = order[t];
        phis[t] = rank_features_vec(cand[id]);
    }

    for (int t = 0; t < take; ++t) {
        int id = order[t];
        ostringstream o;
        o << "{\"type\":\"cand\",\"bin\":" << g_dump_bin << ",\"layer\":" << g_dump_layer
          << ",\"g\":" << cand[id].g << ",\"nplaced\":" << cand[id].nplaced << ",\"rank\":" << rank[id]
          << ",\"kept\":" << (kept[id] ? 1 : 0) << ",\"phi\":";
        append_phi_json(o, phis[t]);
        o << "}\n";
        g_dump_out << o.str();
    }

    using Key = pair<long long, int>;
    map<Key, vector<int>> groups;
    for (int t = 0; t < take; ++t) {
        int id = order[t];
        groups[{cand[id].g, cand[id].nplaced}].push_back(id);
    }

    int npairs = 0;
    for (const auto& kv : groups) {
        if ((int)kv.second.size() < 2) continue;
        int best_kept = -1, best_reject = -1;
        long long r_kept = LLONG_MIN, r_reject = LLONG_MIN;
        for (int id : kv.second) {
            if (kept[id]) {
                if (rank[id] > r_kept) {
                    r_kept = rank[id];
                    best_kept = id;
                }
            } else if (rank[id] > r_reject) {
                r_reject = rank[id];
                best_reject = id;
            }
        }
        if (best_kept < 0 || best_reject < 0 || npairs >= DUMP_PAIR_CAP) continue;
        RankVec pa = rank_features_vec(cand[best_kept]);
        RankVec pb = rank_features_vec(cand[best_reject]);
        ostringstream o;
        o << "{\"type\":\"pair\",\"bin\":" << g_dump_bin << ",\"layer\":" << g_dump_layer
          << ",\"g\":" << cand[best_kept].g << ",\"nplaced\":" << cand[best_kept].nplaced
          << ",\"rank_a\":" << rank[best_kept] << ",\"rank_b\":" << rank[best_reject]
          << ",\"kept_a\":1,\"kept_b\":0,\"phi_a\":";
        append_phi_json(o, pa);
        o << ",\"phi_b\":";
        append_phi_json(o, pb);
        o << "}\n";
        g_dump_out << o.str();
        ++npairs;
    }
}
