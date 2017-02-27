
@everywhere begin
    using Distributions

    include("smote_exs.jl")
    include("ub_smote.jl")
    include("Boosting.jl")
    include("simtools.jl")
end

@everywhere function sim(xi)
    # running multiple simulations
    n = 200
    mu_err = 2
    simresults = run_simulations(20, n, 10, mu_err, 0.1, ξ = xi)
    # summarise_sims(simresults)
    writecsv("simres/$(n)_$(mu_err)_$(xi).csv", simresults)
end

# pmap(sim, [parse(x) for x in ARGS])
pmap(sim, [0.4, 0.5, 0.6])

#
# # running single simulation
# runsim(200, 10, 2, 0.1, seed = 111, ξ = 0.1)
