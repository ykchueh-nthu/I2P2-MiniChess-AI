#include <utility>
#include "state.hpp"
#include "pvs.hpp"
#include "search_util.hpp"


static KillerTable  g_killers;
static HistoryTable g_history;
static TranspositionTable g_tt;


/*============================================================
 * PVS — eval_ctx
 *
 * Principal Variation Search (NegaScout) with transposition table
 * and move ordering. First child of every node gets a full-window
 * search; subsequent children get a null window [alpha, alpha+1]
 * and are only re-searched with the full window if they beat alpha.
 *============================================================*/
int PVS::eval_ctx(
    State *state,
    int depth,
    int alpha,
    int beta,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const PVSParams& p
){
    ctx.nodes++;
    if(ply > ctx.seldepth) ctx.seldepth = ply;
    if(ctx.stop) return 0;

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
            if(e->flag == TTFlag::EXACT) return e->score;
            if(e->flag == TTFlag::LOWER && e->score > alpha) alpha = e->score;
            else if(e->flag == TTFlag::UPPER && e->score < beta) beta = e->score;
            if(alpha >= beta) return e->score;
        }
        if(e && e->has_move){
            tt_move = e->best_move;
            has_tt_move = true;
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

    if(p.use_move_ordering){
        order_moves(state->legal_actions, state, tt_move, has_tt_move,
                     g_killers, ply, g_history);
    }

    int best_score = M_MAX;
    Move best_move{};
    bool has_best_move = false;
    bool first_child = true;

    for(auto& action : state->legal_actions){
        State* next = static_cast<State*>(state->next_state(action));
        next->get_legal_actions();

        bool same = next->same_player_as_parent();
        int score;

        if(first_child){
            int raw = eval_ctx(next, depth - 1,
                               same ? alpha : -beta,
                               same ? beta  : -alpha,
                               history, ply + 1, ctx, p);
            score = same ? raw : -raw;
            first_child = false;
        } else {
            int null_alpha = same ? alpha : -(alpha + 1);
            int null_beta  = same ? (alpha + 1) : -alpha;
            int raw = eval_ctx(next, depth - 1,
                               null_alpha, null_beta,
                               history, ply + 1, ctx, p);
            score = same ? raw : -raw;

            if(score > alpha && score < beta){
                raw = eval_ctx(next, depth - 1,
                               same ? alpha : -beta,
                               same ? beta  : -alpha,
                               history, ply + 1, ctx, p);
                score = same ? raw : -raw;
            }
        }

        delete next;

        if(score > best_score){
            best_score = score;
            best_move = action;
            has_best_move = true;
        }
        if(best_score > alpha) alpha = best_score;
        if(alpha >= beta){
            if(p.use_move_ordering && !is_capture_move(state, action)){
                g_killers.add(ply, action);
                g_history.add(state->player, action, depth);
            }
            break;
        }
    }

    history.pop(key);

    if(p.use_tt){
        TTFlag flag;
        if(best_score <= orig_alpha)      flag = TTFlag::UPPER;
        else if(best_score >= orig_beta)  flag = TTFlag::LOWER;
        else                               flag = TTFlag::EXACT;
        g_tt.store(key, depth, best_score, flag, best_move, has_best_move);
    }

    return best_score;
}


/*============================================================
 * PVS — quiescence
 *
 * Captures-only search at the horizon to avoid the horizon effect.
 * Not run through the main TT (separate, simpler concern — entries
 * here have no "depth" in the iterative-deepening sense).
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

    int stand_pat = state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history);
    if(stand_pat >= beta) return beta;
    if(stand_pat > alpha) alpha = stand_pat;

    if(state->legal_actions.empty() && state->game_state == UNKNOWN)
        state->get_legal_actions();

    if(state->game_state == WIN)  return P_MAX - ply;
    if(state->game_state == DRAW) return 0;

    /* Order captures by MVV-LVA so the strongest trades are explored
     * first — helps quiescence converge faster within its node budget. */
    std::vector<Move> captures;
    captures.reserve(state->legal_actions.size());
    for(auto& m : state->legal_actions){
        if(is_capture_move(state, m)) captures.push_back(m);
    }
    std::stable_sort(captures.begin(), captures.end(),
        [&](const Move& a, const Move& b){
            int opp = 1 - state->player;
            auto val = [&](const Move& m){
                int victim = state->piece_at(opp, (int)m.second.first, (int)m.second.second);
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
        int score  = same ? raw : -raw;
        delete next;

        if(score >= beta) return beta;
        if(score > alpha) alpha = score;
    }

    return alpha;
}


/*============================================================
 * PVS — search
 *============================================================*/
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

    if(p.use_tt && depth <= 1){
        /* Only clear on the first iterative-deepening call (depth==1)
         * for this go command, so shallower depths' TT entries continue
         * to seed move ordering for deeper ones within the same search. */
        g_tt.clear();
    }
    g_killers.clear();
    g_history.clear();

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

    int best_score  = M_MAX - 10;
    int alpha       = M_MAX;
    int beta        = P_MAX;
    bool first_move = true;
    int move_index  = 0;
    int total_moves = (int)state->legal_actions.size();

    for(auto& action : state->legal_actions){
        State* next = static_cast<State*>(state->next_state(action));
        next->get_legal_actions();

        bool same = next->same_player_as_parent();
        int score;

        if(first_move){
            int raw = eval_ctx(next, depth - 1,
                               same ? alpha : -beta,
                               same ? beta  : -alpha,
                               history, 1, ctx, p);
            score = same ? raw : -raw;
            first_move = false;
        } else {
            int null_alpha = same ? alpha : -(alpha + 1);
            int null_beta  = same ? (alpha + 1) : -alpha;
            int raw = eval_ctx(next, depth - 1,
                               null_alpha, null_beta,
                               history, 1, ctx, p);
            score = same ? raw : -raw;

            if(score > alpha && score < beta){
                raw = eval_ctx(next, depth - 1,
                               same ? alpha : -beta,
                               same ? beta  : -alpha,
                               history, 1, ctx, p);
                score = same ? raw : -raw;
            }
        }

        delete next;

        if(score > best_score){
            best_score       = score;
            result.best_move = action;
            result.score     = best_score;
            if(best_score > alpha) alpha = best_score;

            if(p.report_partial && ctx.on_root_update)
                ctx.on_root_update({result.best_move, best_score, depth,
                                    move_index + 1, total_moves});
        }
        move_index++;
    }

    result.score    = best_score;
    result.nodes    = ctx.nodes;
    result.seldepth = ctx.seldepth;
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
    };
}
