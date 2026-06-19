#pragma once
#include "search_types.hpp"
#include "game_history.hpp"

struct ABParams {
    bool use_kp_eval       = true;
    bool use_eval_mobility = true;
    bool report_partial    = true;
    bool use_tt            = true;
    bool use_move_ordering = true;

    static ABParams from_map(const ParamMap& m){
        ABParams p;
        p.use_kp_eval       = param_bool(m, "UseKPEval",       true);
        p.use_eval_mobility = param_bool(m, "UseEvalMobility", true);
        p.report_partial    = param_bool(m, "ReportPartial",   true);
        p.use_tt            = param_bool(m, "UseTT",           true);
        p.use_move_ordering = param_bool(m, "UseMoveOrdering", true);
        return p;
    }
};

class AlphaBeta {
public:
    static int eval_ctx(
        State *state,
        int depth,
        int alpha,
        int beta,
        GameHistory& history,
        int ply,
        SearchContext& ctx,
        const ABParams& p
    );
    static SearchResult search(
        State *state,
        int depth,
        GameHistory& history,
        SearchContext& ctx
    );

    static ParamMap default_params();
    static std::vector<ParamDef> param_defs();
};
