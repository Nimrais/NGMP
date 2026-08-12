function usage(io::IO = stdout)
    println(io, """
    IVON posterior-gate ensemble benchmark

    Usage:
      julia --project=. run.jl smoke [--force]
      julia --project=. run.jl pilot [--force]
      julia --project=. run.jl full [--force]
      julia --project=. run.jl summarize

    Commands:
      smoke      one shortened ETTh1/H96 fit for each gate architecture
      pilot      ETTh1/H96 validation-only shared IVON grid selection
      full       16 final fits using the frozen pilot selection
      summarize  rebuild CSV and Markdown tables from final checkpoints
    """)
end

function main(args = ARGS)
    isempty(args) && (usage(stderr); return 2)
    command = first(args)
    flags = Set(args[2:end])
    unknown = setdiff(flags, Set(["--force"]))
    isempty(unknown) || error("Unknown flags: $(join(unknown, ", "))")
    force = "--force" in flags
    if command == "smoke"
        verify_dependency_pins()
        run_smoke(; force)
    elseif command == "pilot"
        verify_dependency_pins()
        run_pilot(; force)
    elseif command == "full"
        verify_dependency_pins()
        run_full(; force)
    elseif command == "summarize"
        isempty(flags) || error("summarize does not accept --force")
        summarize()
    elseif command in ("help", "--help", "-h")
        usage()
    else
        usage(stderr)
        error("Unknown command: $command")
    end
    return 0
end
