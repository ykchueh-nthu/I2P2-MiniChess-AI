#pragma once
#include "search_types.hpp"
#include "game_history.hpp"

struct PVSParams {
    bool use_kp_eval       = true;
    bool use_eval_mobility = true;
    bool use_quiescence    = true;
    bool report_partial    = true;

    static PVSParams from_map(const ParamMap& m){
        PVSParams p;
        p.use_kp_eval       = param_bool(m, "UseKPEval",       true);
        p.use_eval_mobility = param_bool(m, "UseEvalMobility", true);
        p.use_quiescence    = param_bool(m, "UseQuiescence",   true);
        p.report_partial    = param_bool(m, "ReportPartial",   true);
        return p;
    }
};

class PVS {
public:
    static int eval_ctx(
        State *state,
        int depth,
        int alpha,
        int beta,
        GameHistory& history,
        int ply,
        SearchContext& ctx,
        const PVSParams& p
    );

    /* Quiescence search — captures only */
    static int quiescence(
        State *state,
        int alpha,
        int beta,
        GameHistory& history,
        int ply,
        SearchContext& ctx,
        const PVSParams& p
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
