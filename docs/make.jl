push!(LOAD_PATH,"../src/")

using Documenter
using BayesianOptimization

makedocs(
    sitename = "BayesianOptimization",
    format = Documenter.HTML(),
    modules = [BayesianOptimization]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
