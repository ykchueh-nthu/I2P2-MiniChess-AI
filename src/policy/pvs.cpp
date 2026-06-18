#include <utility>
#include "state.hpp"
#include "pvs.hpp"


/*============================================================
 * PVS — eval_ctx
 *
 * Principal Variation Search (also called NegaScout).
 * Searches the first child with the full [alpha,beta] window.
 * All subsequent children are first searched with a null
 * window [alpha, alpha+1]. If a null-window search beats
 * alpha (a "surprise"), we re-search with the full window.
 * This is correct because the first move is expected to be
 * the best (good move ordering helps a lot here).
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
    history.push(state->hash());

    if(depth <= 0){
        int score;
        if(p.use_quiescence){
            score = quiescence(state, alpha, beta, history, ply, ctx, p);
        } else {
            score = state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history);
        }
        history.pop(state->hash());
        return score;
    }

    int best_score = M_MAX;
    bool first_child = true;

    for(auto& action : state->legal_actions){
        State* next = static_cast<State*>(state->next_state(action));
        next->get_legal_actions();

        bool same = next->same_player_as_parent();
        int score;

        if(first_child){
            // Full-window search on the first (expected best) child
            int raw = eval_ctx(next, depth - 1,
                               same ? alpha : -beta,
                               same ? beta  : -alpha,
                               history, ply + 1, ctx, p);
            score = same ? raw : -raw;
            first_child = false;
        } else {
            // Null-window search
            int null_alpha = same ? alpha : -(alpha + 1);
            int null_beta  = same ? (alpha + 1) : -alpha;
            int raw = eval_ctx(next, depth - 1,
                               null_alpha, null_beta,
                               history, ply + 1, ctx, p);
            score = same ? raw : -raw;

            // Re-search with full window if null window was beaten
            if(score > alpha && score < beta){
                raw = eval_ctx(next, depth - 1,
                               same ? alpha : -beta,
                               same ? beta  : -alpha,
                               history, ply + 1, ctx, p);
                score = same ? raw : -raw;
            }
        }

        delete next;

        if(score > best_score) best_score = score;
        if(best_score > alpha) alpha = best_score;
        if(alpha >= beta) break;   // beta cutoff
    }

    history.pop(state->hash());
    return best_score;
}


/*============================================================
 * PVS — quiescence
 *
 * Called at depth==0 (leaf) when use_quiescence is true.
 * Searches capture moves only until the position is quiet,
 * preventing the horizon effect where a bad trade is hidden
 * just beyond the search depth.
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

    // Stand-pat: assume we can choose not to capture
    int stand_pat = state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history);
    if(stand_pat >= beta) return beta;
    if(stand_pat > alpha) alpha = stand_pat;

    // Ensure legal actions are generated
    if(state->legal_actions.empty() && state->game_state == UNKNOWN)
        state->get_legal_actions();

    if(state->game_state == WIN)  return P_MAX - ply;
    if(state->game_state == DRAW) return 0;

    auto& oppn_board = state->board.board[1 - state->player];

    for(auto& action : state->legal_actions){
        int to_r = action.second.first;
        int to_c = action.second.second;

        // Only follow captures (destination has an opponent piece)
        if(!oppn_board[to_r][to_c]) continue;

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

    if(!state->legal_actions.size())
        state->get_legal_actions();

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
            // Null window
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
    };
}

std::vector<ParamDef> PVS::param_defs(){
    return {
        {"UseKPEval",       ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"UseQuiescence",   ParamDef::CHECK, "true"},
        {"ReportPartial",   ParamDef::CHECK, "true"},
    };
}
