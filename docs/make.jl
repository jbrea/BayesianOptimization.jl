push!(LOAD_PATH, "../src/")

using Documenter
using BayesianOptimization

makedocs(sitename = "BayesianOptimization",
         format = Documenter.HTML(),
         modules = [BayesianOptimization])


deploydocs(
    repo = "github.com/jbrea/BayesianOptimization.jl.git",
)
