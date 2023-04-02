using BayesianOptimization, GaussianProcesses
const BO = BayesianOptimization
using Test, Random
include("acquisitionfunctions.jl")
include("acquisition.jl")
include("branin.jl")
include("warmstart.jl")
include("utils.jl")
include("BayesianOptimization.jl")
