#pragma once

#include <array>
#include <vector>

// Shared packing state. rank.cpp reads this; pack3d.cpp owns the geometry.

constexpr int MAX_PLACED = 256;
constexpr int MAXEP = 640;
constexpr int MAXEMS = 1280;
constexpr int BUFFER_CAP = 12;
constexpr int OPEN_CAP = BUFFER_CAP;

constexpr int ROT_PERM[6][3] = {
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

struct EmsBox {
    int x, y, z, w, d, h;
    bool operator<(const EmsBox& o) const {
        if (x != o.x) return x < o.x;
        if (y != o.y) return y < o.y;
        if (z != o.z) return z < o.z;
        if (w != o.w) return w < o.w;
        if (d != o.d) return d < o.d;
        return h < o.h;
    }
    bool operator==(const EmsBox& o) const {
        return x == o.x && y == o.y && z == o.z && w == o.w && d == o.d && h == o.h;
    }
};

struct Place {
    int id, x, y, z, w, d, h, rot;
};

struct Res {
    int x, y, z;
};

// last_waste / last_huge are the residual of the pose just applied, captured
// before the item occupied that origin. Rank uses them; search does not.
struct State {
    long long g = 0;
    int nplaced = 0;
    int nep = 1;
    int nems = 1;
    long long last_waste = 0;
    int last_huge = 0;
    Place placed[MAX_PLACED];
    Pt eps[MAXEP];
    EmsBox ems[MAXEMS];
    State() {
        eps[0] = {0, 0, 0};
        ems[0] = {0, 0, 0, 0, 0, 0};
    }
};

extern int W, D, H, N;
extern std::vector<std::array<int, 3>> orig;
extern std::vector<long long> vol;
extern std::vector<int> min_edge;
extern std::vector<int> g_open;
extern int g_seen;
extern int g_typical_edge;
extern bool g_use_ems;
extern bool g_use_ep;

void rotated_size(int id, int rot, int& w, int& d, int& h);
bool base_stable(int w, int d, int h);
bool is_packed(const State& st, int id);
Res residual_box(const State& st, int x, int y, int z);
