#pragma once

#include "pack_state.hpp"

#include <array>
#include <string>
#include <vector>

// Normalized feature vector for learned rank (training + inference).
// Order is fixed; train_rank.py must use the same RANK_DIM / names.
inline constexpr int RANK_DIM = 28;

struct RankVec {
    std::array<float, RANK_DIM> v{};
};

// Human-readable names, one per v[i]. Used in JSONL and training scripts.
const char* rank_feature_name(int i);

// Unitless features in [0, 1] (or small bounded range) for ML.
RankVec rank_features_vec(const State& st);

// Hand-tuned beam score (unchanged semantics). Uses raw geometry, not RankVec.
long long state_rank(const State& st);

// ---- JSONL dump (optional, for training) ----
// One line per candidate; pair lines for same (g, nplaced) kept vs rejected.
void rank_dump_open(const std::string& path);
void rank_dump_close();
bool rank_dump_enabled();
void rank_dump_begin_bin(int bin_index);
void rank_dump_layer(const std::vector<State>& cand, const std::vector<long long>& rank,
                     const std::vector<int>& order, const std::vector<char>& kept, int m);
