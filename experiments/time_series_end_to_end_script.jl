# =============================================================================
# End-to-end reproduction of the consensus-x L2 final table
# =============================================================================
# Fits the final per-dataset configuration on all 8 (dataset, horizon) cells
# and prints the comparison table against the paper's Dynamic row and the CT
# arm (references hardcoded from results/etth_ct_table/table.txt).
#
# Final configuration (selected in paper_materials/
# etth2_direct_precision_hierarchy_findings.md):
#   shared    : consensus-x L2 hierarchy, scalar τ_y, Student-t x-messages
#               (Unscented tangent projection), α = 0.2 damped (no momentum),
#               anchor variance 1.0, 60 iterations, tol 1e-5, RFF seed 12345
#   ETTh2     : Matérn-3/2 RFF(400) basis, prior gain 3.0, β carrier (rate 1)
#   ETTh1     : linear 65-d basis, no β carrier, prior gain 5.0 (3.0 at h720 —
#               gain 5 exceeds the exp-site stability ceiling there)
#
# Each cell runs in its own Julia process (sequential — concurrent processes
# cause large slowdowns) via experiments/etth2_consensus_precision_hierarchy.jl
# in winner mode; each process fits both carrier variants (~2-6 min/cell
# including package load). Requires cache/dynamic_{etth1,etth2}_h*_cache.jld2
# (ETTh1 h96 may live at notebooks/vmp_vs_ngmp/).
#
# Usage:
#   julia --project=. experiments/reproduce_consensus_table.jl [results_dir]
#
# Default results_dir is results/consensus_table_repro (fresh — forces actual
# refits). Point it at results/etth2_consensus_hierarchy to reuse existing
# fits (completed arms are skipped, table assembled from disk).
# =============================================================================

using JLD2, Printf, Statistics

const ROOT = normpath(joinpath(@__DIR__, ".."))
const FIT_SCRIPT = joinpath(@__DIR__, "etth2_consensus_precision_hierarchy.jl")
const OUT_DIR = length(ARGS) >= 1 ? abspath(ARGS[1]) :
    joinpath(ROOT, "results", "consensus_table_repro")

# References: results/etth_ct_table/table.txt (NLL ± 1.96·SE, lower better).
const REFERENCES = Dict(
    ("ETTh2", 96) => (dyn = (0.93418, 0.03344), ct = (0.93973, 0.03861)),
    ("ETTh2", 192) => (dyn = (0.92366, 0.03571), ct = (0.86018, 0.03650)),
    ("ETTh2", 336) => (dyn = (0.96119, 0.03630), ct = (0.87019, 0.03483)),
    ("ETTh2", 720) => (dyn = (0.86993, 0.03214), ct = (0.97739, 0.03573)),
    ("ETTh1", 96) => (dyn = (0.41203, 0.01891), ct = (0.38879, 0.02098)),
    ("ETTh1", 192) => (dyn = (0.37006, 0.01862), ct = (0.33780, 0.01998)),
    ("ETTh1", 336) => (dyn = (0.31413, 0.01468), ct = (0.28773, 0.01611)),
    ("ETTh1", 720) => (dyn = (0.37633, 0.01502), ct = (0.35710, 0.01534)),
)

# Per-cell final configuration and the arm file the table reads.
struct CellSpec
    dataset::String
    horizon::Int
    setup::String
    gain::Float64
    carrier::String       # arm picked for the table
end

const CELLS = [
    CellSpec("ETTh2", 96, "rff_all", 3.0, "beta"),
    CellSpec("ETTh2", 192, "rff_all", 3.0, "beta"),
    CellSpec("ETTh2", 336, "rff_all", 3.0, "beta"),
    CellSpec("ETTh2", 720, "rff_all", 3.0, "beta"),
    CellSpec("ETTh1", 96, "linear_all", 5.0, "nobeta"),
    CellSpec("ETTh1", 192, "linear_all", 5.0, "nobeta"),
    CellSpec("ETTh1", 336, "linear_all", 5.0, "nobeta"),
    CellSpec("ETTh1", 720, "linear_all", 3.0, "nobeta"),
]

cell_tag(spec) = (spec.dataset == "ETTh2" ? "" :
    lowercase(spec.dataset) * "_") * "h$(spec.horizon)_sweep"

# Suffix layout must match hyper_suffix() + winner mode in the fit script:
# nobeta arms carry the default β-rate (1000.0) in their name, beta arms the
# tuned rate 1.0; both use α=0.2, momentum 0, damped, anchor_var 1.0, t-x.
function arm_file(spec)
    beta_rate = spec.carrier == "beta" ? "1.0" : "1000.0"
    return "$(spec.setup)_L2_$(spec.carrier)_scalar" *
        "_a0.2m0.0dg$(spec.gain)b$(beta_rate)av1.0_tx.jld2"
end

function run_cell(spec)
    target = joinpath(OUT_DIR, cell_tag(spec), arm_file(spec))
    if isfile(target)
        @printf("cell %s h%d: exists, skipping fit\n",
            spec.dataset, spec.horizon)
        return
    end
    environment = copy(ENV)
    environment["EDH_WINNER"] = "1"
    environment["EDH_XMSG"] = "student"
    environment["EDH_RESULTS_DIR"] = OUT_DIR
    environment["EDH_SWEEP_SETUP"] = spec.setup
    environment["EDH_W_GAIN"] = string(spec.gain)
    environment["EDH_W_BRATE"] = "1.0"
    command = setenv(
        `$(Base.julia_cmd()) --project=$(ROOT) $(FIT_SCRIPT) sweep $(spec.horizon) $(spec.dataset)`,
        environment,
    )
    @printf("cell %s h%d (%s, gain %.1f): fitting...\n",
        spec.dataset, spec.horizon, spec.setup, spec.gain)
    run(command)
    isfile(target) ||
        error("fit did not produce $(target) — check the cell log above")
    return
end

function load_cell(spec)
    result = load(joinpath(OUT_DIR, cell_tag(spec), arm_file(spec)))["result"]
    result["status"] == "ok" ||
        error("$(spec.dataset) h$(spec.horizon): $(result["error"])")
    metrics = result["metrics"]
    return (nll = metrics["nll"], ci = metrics["nll_ci"])
end

function verdict(ours, reference)
    difference = ours.nll - reference[1]
    resolved = abs(difference) > ours.ci + reference[2]
    difference < 0 && return resolved ? "win ✓" : "win (overlap)"
    difference > 0 && return resolved ? "loss ✓" : "tie"
    return "tie"
end

function main()
    mkpath(OUT_DIR)
    for spec in CELLS
        run_cell(spec)
    end

    rows = [(spec, load_cell(spec)) for spec in CELLS]
    table_path = joinpath(OUT_DIR, "final_table.md")
    open(table_path, "w") do io
        for stream in (stdout, io)
            println(stream, "\n# Consensus-x L2 vs paper Dynamic vs CT arm (NLL ± 95% CI)\n")
            println(stream, "| cell | ours L2 | paper Dyn | CT arm | vs Dyn | vs CT |")
            println(stream, "|---|---|---|---|---|---|")
            for (spec, ours) in rows
                reference = REFERENCES[(spec.dataset, spec.horizon)]
                @printf(stream,
                    "| %s %d | %.4f ± %.4f | %.4f ± %.4f | %.4f ± %.4f | %s | %s |\n",
                    spec.dataset, spec.horizon, ours.nll, ours.ci,
                    reference.dyn..., reference.ct...,
                    verdict(ours, reference.dyn), verdict(ours, reference.ct))
            end
            for dataset in ("ETTh2", "ETTh1")
                selected = [(s, o) for (s, o) in rows if s.dataset == dataset]
                @printf(stream, "| %s mean | %.4f | %.4f | %.4f | | |\n",
                    dataset,
                    mean(o.nll for (_, o) in selected),
                    mean(REFERENCES[(dataset, s.horizon)].dyn[1]
                        for (s, _) in selected),
                    mean(REFERENCES[(dataset, s.horizon)].ct[1]
                        for (s, _) in selected))
            end
            @printf(stream, "| all 8 mean | %.4f | %.4f | %.4f | | |\n",
                mean(o.nll for (_, o) in rows),
                mean(r.dyn[1] for r in values(REFERENCES)),
                mean(r.ct[1] for r in values(REFERENCES)))
            println(stream,
                "\nConfig: consensus-x L2, scalar τ_y, Student-t x-messages, " *
                "α=0.2 damped; ETTh2 = RFF gain 3 + β(rate 1); " *
                "ETTh1 = linear, no-β, gain 5 (3 at h720). Seed 12345.")
        end
    end
    @printf("\ntable written to %s\n", table_path)
end

main()
