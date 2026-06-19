#pragma once
/*============================================================
 * search_util.hpp — shared move ordering + transposition table
 * helpers for AlphaBeta and PVS.
 *
 * Kept header-only so both algorithm files can include it without
 * needing a new .cpp added to the Makefile (wildcard picks up
 * alphabeta.cpp/pvs.cpp already; this header rides along for free).
 *============================================================*/

#include <cstdint>
#include <vector>
#include <algorithm>
#include "state.hpp"
#include "base_state.hpp"

/*------------------------------------------------------------
 * Transposition Table
 *
 * Fixed-size, no dynamic resizing, no replacement scheme beyond
 * "always overwrite" (simplest correct option for a time-boxed
 * improvement). Indexed by hash % TT_SIZE.
 *------------------------------------------------------------*/

enum class TTFlag : uint8_t { EMPTY = 0, EXACT, LOWER, UPPER };

struct TTEntry {
    uint64_t key   = 0;
    int      score = 0;
    int      depth = -1;       // -1 means "unused"
    TTFlag   flag  = TTFlag::EMPTY;
    Move     best_move{};
    bool     has_move = false;
};

/* 2^20 entries ~ 1M * sizeof(TTEntry). TTEntry is small (~40 bytes),
 * so this is ~40MB — comfortably under typical contest memory limits.
 * Power-of-two size lets us mask instead of mod. */
constexpr size_t TT_SIZE = 1u << 20;
constexpr uint64_t TT_MASK = TT_SIZE - 1;

/* A self-contained TT instance. Each algorithm (AlphaBeta, PVS, ...)
 * must own its own instance — sharing one global table between
 * algorithms is unsafe, since different algorithms can legitimately
 * compute different scores/flags for the "same" depth (e.g. PVS's
 * null-window sub-searches vs AlphaBeta's full-window search), and
 * one algorithm's stale entries can incorrectly short-circuit
 * another's search. */
struct TranspositionTable {
    std::vector<TTEntry> table;

    TranspositionTable() : table(TT_SIZE) {}

    void clear(){
        std::fill(table.begin(), table.end(), TTEntry{});
    }

    const TTEntry* probe(uint64_t key) const {
        const TTEntry& e = table[key & TT_MASK];
        if(e.flag != TTFlag::EMPTY && e.key == key){
            return &e;
        }
        return nullptr;
    }

    void store(
        uint64_t key, int depth, int score, TTFlag flag,
        const Move& best_move, bool has_move
    ){
        TTEntry& e = table[key & TT_MASK];
        if(e.flag == TTFlag::EMPTY || e.key != key || depth >= e.depth){
            e.key = key;
            e.depth = depth;
            e.score = score;
            e.flag = flag;
            e.best_move = best_move;
            e.has_move = has_move;
        }
    }
};


/*------------------------------------------------------------
 * Move ordering
 *
 * Score each legal move so the caller can std::sort descending:
 *   1. TT best move from a previous search of this exact position
 *      (highest priority — already proven good).
 *   2. Captures, ordered by MVV-LVA (Most Valuable Victim,
 *      Least Valuable Attacker).
 *   3. Killer moves (quiet moves that caused a cutoff at this ply
 *      in a sibling node).
 *   4. History heuristic (quiet moves that have caused cutoffs
 *      anywhere, weighted by depth^2).
 *   5. Everything else, unordered.
 *------------------------------------------------------------*/

/* Piece values for MVV-LVA. Index = piece type (0..6), matches
 * State::piece_at()'s encoding (0=empty,1=pawn,2=rook,3=knight,
 * 4=bishop,5=queen,6=king). Values only need relative ordering,
 * so plain material counts (not the 10x KP scale) are fine. */
inline const int& mvv_lva_value(int piece_type){
    static const int values[7] = {0, 1, 5, 3, 3, 9, 1000};
    if(piece_type < 0 || piece_type > 6) return values[0];
    return values[piece_type];
}

/* Killer moves: 2 slots per ply, shared across one search() call.
 * Indexed by ply; cleared at the start of every root search() so
 * stale killers from a previous position don't leak in (they would
 * still be "safe" since they're just an ordering hint, but clearing
 * keeps behavior predictable run-to-run). */
constexpr int MAX_PLY = 128;

struct KillerTable {
    Move slot[MAX_PLY][2];
    bool valid[MAX_PLY][2] = {};

    void clear(){
        for(int p = 0; p < MAX_PLY; p++){
            valid[p][0] = false;
            valid[p][1] = false;
        }
    }

    void add(int ply, const Move& m){
        if(ply < 0 || ply >= MAX_PLY) return;
        /* Avoid duplicate: if already slot 0, do nothing. */
        if(valid[ply][0] && slot[ply][0] == m) return;
        slot[ply][1] = slot[ply][0];
        valid[ply][1] = valid[ply][0];
        slot[ply][0] = m;
        valid[ply][0] = true;
    }

    bool is_killer(int ply, const Move& m) const {
        if(ply < 0 || ply >= MAX_PLY) return false;
        return (valid[ply][0] && slot[ply][0] == m)
            || (valid[ply][1] && slot[ply][1] == m);
    }
};

/* History heuristic: indexed by (player, from_square, to_square).
 * Squares packed as row*BOARD_W+col, so this is small and fast.
 * Shared across the whole search() call (not per-ply like killers). */
struct HistoryTable {
    int score[2][BOARD_H * BOARD_W][BOARD_H * BOARD_W] = {};

    void clear(){
        for(int p = 0; p < 2; p++)
            for(int f = 0; f < BOARD_H * BOARD_W; f++)
                for(int t = 0; t < BOARD_H * BOARD_W; t++)
                    score[p][f][t] = 0;
    }

    void add(int player, const Move& m, int depth){
        int f = (int)m.first.first  * BOARD_W + (int)m.first.second;
        int t = (int)m.second.first * BOARD_W + (int)m.second.second;
        if(f < 0 || f >= BOARD_H*BOARD_W || t < 0 || t >= BOARD_H*BOARD_W) return;
        score[player][f][t] += depth * depth;
    }

    int get(int player, const Move& m) const {
        int f = (int)m.first.first  * BOARD_W + (int)m.first.second;
        int t = (int)m.second.first * BOARD_W + (int)m.second.second;
        if(f < 0 || f >= BOARD_H*BOARD_W || t < 0 || t >= BOARD_H*BOARD_W) return 0;
        return score[player][f][t];
    }
};

/* Returns true if `m` is a capture for `state` (destination occupied
 * by an opponent piece). Cheap O(1) board lookup via piece_at(). */
inline bool is_capture_move(const State* state, const Move& m){
    int opp = 1 - state->player;
    int tr = (int)m.second.first, tc = (int)m.second.second;
    if(tr < 0 || tr >= BOARD_H || tc < 0 || tc >= BOARD_W) return false; // promo-encoded rows handled by caller if needed
    return state->piece_at(opp, tr, tc) != 0;
}

/* Order `actions` in place, descending by estimated usefulness.
 * `tt_move` may be a default-constructed Move{} if none is known —
 * pass has_tt_move=false in that case so we don't accidentally
 * prioritize move (0,0)->(0,0). */
inline void order_moves(
    std::vector<Move>& actions,
    const State* state,
    const Move& tt_move, bool has_tt_move,
    const KillerTable& killers, int ply,
    const HistoryTable& history
){
    std::stable_sort(actions.begin(), actions.end(),
        [&](const Move& a, const Move& b){
            auto move_score = [&](const Move& m) -> long long {
                if(has_tt_move && m == tt_move) return 1'000'000'000LL;

                if(is_capture_move(state, m)){
                    int opp = 1 - state->player;
                    int tr = (int)m.second.first, tc = (int)m.second.second;
                    int victim   = state->piece_at(opp, tr, tc);
                    int attacker = state->piece_at(state->player,
                                                    (int)m.first.first,
                                                    (int)m.first.second);
                    /* MVV-LVA: prioritize big victim, small attacker.
                     * Scaled so captures always outrank killers/history. */
                    long long s = 100'000'000LL;
                    s += (long long)mvv_lva_value(victim) * 1000;
                    s -= (long long)mvv_lva_value(attacker);
                    return s;
                }

                if(killers.is_killer(ply, m)) return 50'000'000LL;

                return (long long)history.get(state->player, m);
            };
            return move_score(a) > move_score(b);
        });
}
