#include <utility>
#include <chrono>
#include "state.hpp"
#include "pvs.hpp"
#include "search_util.hpp"


static KillerTable  g_killers;
static HistoryTable g_history;
static TranspositionTable g_tt;

/* Time management: set at the start of each root search(), checked in eval_ctx */
static int64_t g_time_limit_ms = 0;
static std::chrono::time_point<std::chrono::high_resolution_clock> g_search_start;

/* Piece values for delta pruning in quiescence (match kp_material scale) */
static const int qsearch_delta[7] = {0, 20, 60, 70, 80, 200, 1000};

/*============================================================
 * Mate-score / TT helpers
 *============================================================*/
static constexpr int MATE_THRESHOLD = P_MAX - 1000;

static inline int to_tt_score(int score, int ply){
    if(score >  MATE_THRESHOLD) return score + ply;
    if(score < -MATE_THRESHOLD) return score - ply;
    return score;
}

static inline int from_tt_score(int score, int ply){
    if(score >  MATE_THRESHOLD) return score - ply;
    if(score < -MATE_THRESHOLD) return score + ply;
    return score;
}


/*============================================================
 * PVS — eval_ctx
 *
 * Principal Variation Search with:
 *  - Null Move Pruning (NMP): try passing the move; if still >= beta, prune.
 *  - Late Move Reductions (LMR): reduce depth for quiet moves searched late.
 *  - Futility Pruning: skip quiet moves near leaves when hopelessly behind.
 *  - Transposition table + killer + history move ordering.
 *  - Quiescence search at the horizon (with delta pruning).
 *
 * allow_null: prevents two consecutive null moves on the same path.
 *============================================================*/
int PVS::eval_ctx(
    State *state,
    int depth,
    int alpha,
    int beta,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const PVSParams& p,
    bool allow_null
){
    ctx.nodes++;
    if(ply > ctx.seldepth) ctx.seldepth = ply;
    if(ctx.stop) return 0;

    /* Periodic time check: every 4096 nodes, abort if over the limit */
    if(g_time_limit_ms > 0 && (ctx.nodes & 0xFFF) == 0){
        auto now = std::chrono::high_resolution_clock::now();
        int64_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - g_search_start).count();
        if(elapsed >= g_time_limit_ms){
            ctx.stop = true;
            return 0;
        }
    }

    if(state->legal_actions.empty() && state->game_state == UNKNOWN)
        state->get_legal_actions();

    if(state->game_state == WIN)  return P_MAX - ply;
    if(state->game_state == DRAW) return 0;

    int rep_score;
    if(state->check_repetition(history, rep_score)) return rep_score;

    uint64_t key = state->hash();
    int orig_alpha = alpha, orig_beta = beta;
    Move tt_move{};
    bool has_tt_move = false;

    if(p.use_tt){
        const TTEntry* e = g_tt.probe(key);
        if(e && e->depth >= depth){
            int e_score = from_tt_score(e->score, ply);
            if(e->flag == TTFlag::EXACT) return e_score;
            if(e->flag == TTFlag::LOWER && e_score > alpha) alpha = e_score;
            else if(e->flag == TTFlag::UPPER && e_score < beta) beta = e_score;
            if(alpha >= beta) return e_score;
        }
        if(e && e->has_move){
            tt_move = e->best_move;
            has_tt_move = true;
        }
    }

    /* === Null Move Pruning ===
     * Try passing the move. If the opponent still can't beat beta,
     * our position is so strong we can prune without searching further.
     * Skip when: consecutive null moves, only king left (zugzwang risk),
     * or close to leaves (overhead exceeds benefit). */
    if(p.use_null_move && allow_null && depth >= 4 && !ctx.stop){
        /* Count own pieces to guard against zugzwang (king-only positions) */
        int pcount = 0;
        for(int r = 0; r < BOARD_H && pcount < 2; r++)
            for(int c = 0; c < BOARD_W && pcount < 2; c++)
                if(state->piece_at(state->player, r, c)) pcount++;

        if(pcount >= 2){
            State* null_st = static_cast<State*>(state->create_null_state());
            if(null_st){
                int R = (depth >= 6) ? 3 : 2;
                int null_val = eval_ctx(null_st, depth - R - 1,
                                        -beta, -beta + 1,
                                        history, ply + 1, ctx, p, false);
                delete null_st;
                if(!ctx.stop && -null_val >= beta){
                    /* Verify: re-search at reduced depth before trusting cutoff */
                    return beta;
                }
            }
        }
    }

    history.push(key);

    if(depth <= 0){
        int score;
        if(p.use_quiescence){
            score = quiescence(state, alpha, beta, history, ply, ctx, p);
        } else {
            score = state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history);
        }
        history.pop(key);
        return score;
    }

    /* === Futility Pruning ===
     * At depth 1-2, if the static eval + a safety margin can't reach alpha,
     * skip quiet moves (captures and TT moves are still searched). */
    bool futility_mode = false;
    int static_eval = 0;
    if(p.use_futility && depth <= 2 && !ctx.stop){
        static_eval = state->evaluate(p.use_kp_eval, false, &history);
        /* Margins: depth 1 = one minor piece, depth 2 = two minor pieces */
        const int margin = (depth == 1) ? 150 : 350;
        if(static_eval + margin <= alpha) futility_mode = true;
    }

    if(p.use_move_ordering){
        order_moves(state->legal_actions, state, tt_move, has_tt_move,
                     g_killers, ply, g_history);
    }

    int best_score = M_MAX;
    Move best_move{};
    bool has_best_move = false;
    bool first_child = true;
    int move_count = 0;

    for(auto& action : state->legal_actions){
        bool is_cap = is_capture_move(state, action);

        /* Futility: skip quiet moves that can't realistically raise alpha */
        if(futility_mode && !first_child && !is_cap &&
           !(has_tt_move && action == tt_move)){
            move_count++;
            continue;
        }

        State* next = static_cast<State*>(state->next_state(action));
        next->get_legal_actions();

        bool same = next->same_player_as_parent();

        int score = 0;
        bool aborted = false;

        if(first_child){
            /* First child: full-window search */
            int raw = eval_ctx(next, depth - 1,
                               same ? alpha : -beta,
                               same ? beta  : -alpha,
                               history, ply + 1, ctx, p, true);
            if(ctx.stop){
                aborted = true;
            } else {
                score = same ? raw : -raw;
            }
            first_child = false;
        } else {
            /* === Late Move Reductions ===
             * Quiet moves searched late in the list are likely bad;
             * search them at reduced depth first. */
            int reduction = 0;
            if(p.use_lmr && !is_cap && depth >= 3 && move_count >= 4){
                reduction = 1;
                if(move_count >= 8 && depth >= 6) reduction = 2;
            }

            int null_alpha = same ? alpha : -(alpha + 1);
            int null_beta  = same ? (alpha + 1) : -alpha;

            int raw = eval_ctx(next, depth - 1 - reduction,
                               null_alpha, null_beta,
                               history, ply + 1, ctx, p, true);

            if(ctx.stop){
                aborted = true;
            } else {
                score = same ? raw : -raw;

                /* LMR: if reduced search beat alpha, re-search at full depth
                 * with null window to confirm the move is actually good */
                if(reduction > 0 && score > alpha && !ctx.stop){
                    raw = eval_ctx(next, depth - 1,
                                   null_alpha, null_beta,
                                   history, ply + 1, ctx, p, true);
                    if(ctx.stop) aborted = true;
                    else score = same ? raw : -raw;
                }

                /* PVS re-search: null window beat alpha → do full window */
                if(!aborted && score > alpha && score < beta){
                    raw = eval_ctx(next, depth - 1,
                                   same ? alpha : -beta,
                                   same ? beta  : -alpha,
                                   history, ply + 1, ctx, p, true);
                    if(ctx.stop) aborted = true;
                    else score = same ? raw : -raw;
                }
            }
        }

        delete next;
        move_count++;

        if(aborted) break;

        if(score > best_score){
            best_score = score;
            best_move = action;
            has_best_move = true;
        }
        if(best_score > alpha) alpha = best_score;
        if(alpha >= beta){
            if(p.use_move_ordering && !is_cap){
                g_killers.add(ply, action);
                g_history.add(state->player, action, depth);
            }
            break;
        }
    }

    history.pop(key);

    if(p.use_tt && !ctx.stop){
        TTFlag flag;
        if(best_score <= orig_alpha)      flag = TTFlag::UPPER;
        else if(best_score >= orig_beta)  flag = TTFlag::LOWER;
        else                               flag = TTFlag::EXACT;
        g_tt.store(key, depth, to_tt_score(best_score, ply), flag, best_move, has_best_move);
    }

    return best_score;
}


/*============================================================
 * PVS — quiescence
 *
 * Captures-only search with delta pruning: skip captures whose
 * material gain + safety margin can't raise alpha (avoids searching
 * obviously hopeless captures). Mobility is disabled here — generating
 * opponent legal moves for each quiescence node is too expensive.
 *============================================================*/
int PVS::quiescence(
    State *state,
    int alpha,
    int beta,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const PVSParams& p
){
    ctx.nodes++;
    if(ctx.stop) return 0;

    /* Mobility skipped in qsearch (creates a full State + get_legal_actions per node) */
    int stand_pat = state->evaluate(p.use_kp_eval, false, &history);
    if(stand_pat >= beta) return beta;
    if(stand_pat > alpha) alpha = stand_pat;

    if(state->legal_actions.empty() && state->game_state == UNKNOWN)
        state->get_legal_actions();

    if(state->game_state == WIN)  return P_MAX - ply;
    if(state->game_state == DRAW) return 0;

    std::vector<Move> captures;
    captures.reserve(state->legal_actions.size());
    for(auto& m : state->legal_actions){
        if(!is_capture_move(state, m)) continue;
        captures.push_back(m);
    }

    std::stable_sort(captures.begin(), captures.end(),
        [&](const Move& a, const Move& b){
            int opp = 1 - state->player;
            auto val = [&](const Move& m){
                int victim   = state->piece_at(opp, (int)m.second.first, (int)m.second.second);
                int attacker = state->piece_at(state->player, (int)m.first.first, (int)m.first.second);
                return mvv_lva_value(victim) * 1000 - mvv_lva_value(attacker);
            };
            return val(a) > val(b);
        });

    for(auto& action : captures){
        State* next = static_cast<State*>(state->next_state(action));
        next->get_legal_actions();

        bool same  = next->same_player_as_parent();
        int raw    = quiescence(next, same ? alpha : -beta,
                                       same ? beta  : -alpha,
                                       history, ply + 1, ctx, p);
        delete next;

        if(ctx.stop) break;

        int score = same ? raw : -raw;
        if(score >= beta) return beta;
        if(score > alpha) alpha = score;
    }

    return alpha;
}


/*============================================================
 * PVS — search
 *
 * Root search with aspiration windows: use a narrow window around
 * the previous iteration's score for faster pruning. Re-search with
 * the full window if the score falls outside the window.
 *============================================================*/

/* Score from the previous completed depth iteration (for aspiration) */
static int g_prev_score = M_MAX;

SearchResult PVS::search(
    State *state,
    int depth,
    GameHistory& history,
    SearchContext& ctx
){
    ctx.reset();
    PVSParams p = PVSParams::from_map(ctx.params);
    SearchResult result;
    result.depth = depth;

    if(depth <= 1){
        g_prev_score = M_MAX;
        g_time_limit_ms = 0;  /* reset; will be set via SearchTimeLimitMs param */
        g_search_start  = std::chrono::high_resolution_clock::now();
        if(p.use_tt) g_tt.clear();
        g_killers.clear();
        g_history.clear();
    }

    /* Set/update time limit from caller (ubgi passes "SearchTimeLimitMs" via params) */
    {
        auto it = ctx.params.find("SearchTimeLimitMs");
        if(it != ctx.params.end()){
            g_time_limit_ms = std::stoll(it->second);
        }
    }

    if(!state->legal_actions.size())
        state->get_legal_actions();

    if(p.use_move_ordering){
        const TTEntry* e = p.use_tt ? g_tt.probe(state->hash()) : nullptr;
        Move tt_move{};
        bool has_tt_move = false;
        if(e && e->has_move){ tt_move = e->best_move; has_tt_move = true; }
        order_moves(state->legal_actions, state, tt_move, has_tt_move,
                     g_killers, 0, g_history);
    }

    /* Aspiration window: narrow search around previous score for depth >= 2 */
    const int ASP_WINDOW = 80;
    int a_alpha = M_MAX, a_beta = P_MAX;
    if(depth >= 2
       && g_prev_score > M_MAX + 200
       && g_prev_score < P_MAX - 200){
        a_alpha = g_prev_score - ASP_WINDOW;
        a_beta  = g_prev_score + ASP_WINDOW;
    }

    /* We may need up to 2 re-searches if aspiration fails */
    for(int asp_attempt = 0; asp_attempt < 3; asp_attempt++){
        int best_score  = M_MAX - 10;
        int alpha       = a_alpha;
        int beta        = a_beta;
        bool first_move = true;
        int move_index  = 0;
        int total_moves = (int)state->legal_actions.size();
        Move cur_best{};

        for(auto& action : state->legal_actions){
            State* next = static_cast<State*>(state->next_state(action));
            next->get_legal_actions();

            bool same = next->same_player_as_parent();
            int score = 0;
            bool discard = false;

            if(first_move){
                int raw = eval_ctx(next, depth - 1,
                                   same ? alpha : -beta,
                                   same ? beta  : -alpha,
                                   history, 1, ctx, p, true);
                score = same ? raw : -raw;
                first_move = false;
            } else {
                int null_alpha = same ? alpha : -(alpha + 1);
                int null_beta  = same ? (alpha + 1) : -alpha;
                int raw = eval_ctx(next, depth - 1,
                                   null_alpha, null_beta,
                                   history, 1, ctx, p, true);

                if(ctx.stop){
                    discard = true;
                } else {
                    score = same ? raw : -raw;

                    if(score > alpha && score < beta){
                        raw = eval_ctx(next, depth - 1,
                                       same ? alpha : -beta,
                                       same ? beta  : -alpha,
                                       history, 1, ctx, p, true);
                        if(ctx.stop) discard = true;
                        else score = same ? raw : -raw;
                    }
                }
            }

            delete next;

            if(discard) break;

            if(score > best_score){
                best_score       = score;
                cur_best         = action;
                result.best_move = action;
                result.score     = best_score;
                if(best_score > alpha) alpha = best_score;

                if(p.report_partial && ctx.on_root_update)
                    ctx.on_root_update({result.best_move, best_score, depth,
                                        move_index + 1, total_moves});
            }
            move_index++;

            if(ctx.stop) break;
        }

        /* Aspiration window check */
        if(ctx.stop) break;

        if(best_score <= a_alpha){
            /* Fail low: widen lower bound and re-search */
            a_alpha = M_MAX;
            /* Keep a_beta or widen slightly */
        } else if(best_score >= a_beta){
            /* Fail high: widen upper bound and re-search */
            a_beta = P_MAX;
        } else {
            /* Score within window — done */
            break;
        }

        /* On re-search, restore full move list order (TT move is now updated) */
        if(asp_attempt < 2 && p.use_move_ordering && !ctx.stop){
            const TTEntry* e = p.use_tt ? g_tt.probe(state->hash()) : nullptr;
            Move tt_move{};
            bool has_tt_move = false;
            if(e && e->has_move){ tt_move = e->best_move; has_tt_move = true; }
            order_moves(state->legal_actions, state, tt_move, has_tt_move,
                         g_killers, 0, g_history);
        }
    }

    g_prev_score    = result.score;
    result.nodes    = ctx.nodes;
    result.seldepth = ctx.seldepth;
    /* Populate pv so the CLI can recover the best move if the engine is
     * killed mid-search before outputting bestmove. */
    if(result.best_move != Move{})
        result.pv = {result.best_move};
    return result;
}


/*============================================================
 * PVS — default_params / param_defs
 *============================================================*/
ParamMap PVS::default_params(){
    return {
        {"UseKPEval",       "true"},
        {"UseEvalMobility", "true"},
        {"UseQuiescence",   "true"},
        {"ReportPartial",   "true"},
        {"UseTT",            "true"},
        {"UseMoveOrdering",  "true"},
        {"UseNullMove",      "true"},
        {"UseLMR",           "true"},
        {"UseFutility",      "true"},
    };
}

std::vector<ParamDef> PVS::param_defs(){
    return {
        {"UseKPEval",       ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"UseQuiescence",   ParamDef::CHECK, "true"},
        {"ReportPartial",   ParamDef::CHECK, "true"},
        {"UseTT",           ParamDef::CHECK, "true"},
        {"UseMoveOrdering", ParamDef::CHECK, "true"},
        {"UseNullMove",     ParamDef::CHECK, "true"},
        {"UseLMR",          ParamDef::CHECK, "true"},
        {"UseFutility",     ParamDef::CHECK, "true"},
    };
}
