using Lux, OrdinaryDiffEq, LinearAlgebra, ComponentArrays, NNlib

struct NeuralODE{M <: Lux.AbstractExplicitLayer, So, Se, T, K} <: Lux.AbstractExplicitContainerLayer{(:model,)}
    model::M
    solver::So
    sensealg::Se
    tspan::T
    kwargs::K
end

function NeuralODE(model::Lux.AbstractExplicitLayer; solver=Euler(), sensealg=ReverseDiffAdjoint(), tspan=(0.0f0, 1.0f0), kwargs...)
    return NeuralODE(model, solver, sensealg, tspan, kwargs)
end

function (n::NeuralODE)(x, ps, st)
    function make_new_func(func)
        function dudt(u, p, t)
            u_, st = n.model((u, func(t)), p, st)
            return u_
        end
        return dudt
    end
    function prob_func(prob, i, repeat)
        remake(prob, f = ODEFunction{false}(make_new_func(x.funcs[i])))
    end

    prob = ODEProblem{false}(ODEFunction{false}(make_new_func(x.funcs[1])), x.array, n.tspan, ps)
    ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
    return solve(ensemble_prob, n.solver, EnsembleThreads(), trajectories = length(x.funcs); sensealg=n.sensealg, n.kwargs...), st
end