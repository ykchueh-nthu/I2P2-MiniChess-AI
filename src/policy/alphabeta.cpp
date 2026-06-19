#include <utility>
#include "state.hpp"
#include "alphabeta.hpp"
#include "search_util.hpp"


/* Per-search-call killer/history tables. Static so they persist across
 * eval_ctx recursive calls within one search(), reset at the start of
 * each root search() so stale data from a previous position doesn't
 * bias ordering (it would just be a missed opportunity, never wrong,
 * but resetting keeps behavior reproducible). */
static KillerTable  g_killers;
static HistoryTable g_history;
static TranspositionTable g_tt;


/*============================================================
 * AlphaBeta — eval_ctx
 *
 * Negamax with alpha-beta pruning, transposition table, and
 * move ordering (TT move > MVV-LVA captures > killers > history).
 *============================================================*/
int AlphaBeta::eval_ctx(
    State *state,
    int depth,
    int alpha,
    int beta,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const ABParams& p
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

    /* === Transposition table probe === */
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
        int score = state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history);
        history.pop(key);
        return score;
    }

    /* === Move ordering === */
    if(p.use_move_ordering){
        order_moves(state->legal_actions, state, tt_move, has_tt_move,
                     g_killers, ply, g_history);
    }

    int best_score = M_MAX;
    Move best_move{};
    bool has_best_move = false;

    for(auto& action : state->legal_actions){
        State* next = static_cast<State*>(state->next_state(action));
        next->get_legal_actions();

        bool same = next->same_player_as_parent();
        int raw   = eval_ctx(next, depth - 1,
                             same ? alpha : -beta,
                             same ? beta  : -alpha,
                             history, ply + 1, ctx, p);
        int score = same ? raw : -raw;
        delete next;

        if(score > best_score){
            best_score = score;
            best_move = action;
            has_best_move = true;
        }
        if(best_score > alpha) alpha = best_score;
        if(alpha >= beta){
            /* Beta cutoff — record killer/history for quiet moves only.
             * Captures already get strong ordering from MVV-LVA, so
             * polluting killer slots with them wastes the 2 slots. */
            if(p.use_move_ordering && !is_capture_move(state, action)){
                g_killers.add(ply, action);
                g_history.add(state->player, action, depth);
            }
            break;
        }
    }

    history.pop(key);

    /* === Transposition table store === */
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
 * AlphaBeta — search
 *============================================================*/
SearchResult AlphaBeta::search(
    State *state,
    int depth,
    GameHistory& history,
    SearchContext& ctx
){
    ctx.reset();
    ABParams p = ABParams::from_map(ctx.params);
    SearchResult result;
    result.depth = depth;

    if(p.use_tt && depth <= 1){
        /* Fresh game / shallow search heuristic: clear stale TT entries
         * from a prior unrelated position. Cheap relative to a full
         * search and avoids correctness issues from a stale exact-score
         * entry surviving across ucinewgame. Only done at low depth to
         * avoid repeatedly clearing during iterative deepening within
         * the same position. */
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

    int best_score = M_MAX - 10;
    int alpha      = M_MAX;
    int beta       = P_MAX;
    int move_index = 0;
    int total_moves = (int)state->legal_actions.size();

    for(auto& action : state->legal_actions){
        State* next = static_cast<State*>(state->next_state(action));
        next->get_legal_actions();

        bool same = next->same_player_as_parent();
        int raw   = eval_ctx(next, depth - 1,
                             same ? alpha : -beta,
                             same ? beta  : -alpha,
                             history, 1, ctx, p);
        int score = same ? raw : -raw;
        delete next;

        if(score > best_score){
            best_score      = score;
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
 * AlphaBeta — default_params / param_defs
 *============================================================*/
ParamMap AlphaBeta::default_params(){
    return {
        {"UseKPEval",       "true"},
        {"UseEvalMobility", "true"},
        {"ReportPartial",   "true"},
        {"UseTT",            "true"},
        {"UseMoveOrdering",  "true"},
    };
}

std::vector<ParamDef> AlphaBeta::param_defs(){
    return {
        {"UseKPEval",       ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"ReportPartial",   ParamDef::CHECK, "true"},
        {"UseTT",           ParamDef::CHECK, "true"},
        {"UseMoveOrdering", ParamDef::CHECK, "true"},
    };
}
