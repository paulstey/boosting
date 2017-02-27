
@everywhere begin
    include("smote_exs.jl")
    include("ub_smote.jl")
    include("Boosting.jl")
    include("simtools.jl")
end

@everywhere function sim(n)
    # running multiple simulations

    # n = 200
    xi = 0.6
    mu_err = 2
    pct_minority = 0.1

    p = 20

    simresults = run_simulations(10, n, p, mu_err, pct_minority, ξ = xi)
    # summarise_sims(simresults)
    writecsv("simres2/$(n)_$(mu_err)_$(xi)_$(p).csv", simresults)
end

# pmap(sim, [parse(x) for x in ARGS])
pmap(sim, [300, 500, 1000])
