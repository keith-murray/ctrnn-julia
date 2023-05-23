using Lux, OrdinaryDiffEq, LinearAlgebra, ComponentArrays, NNlib, SciMLSensitivity
include("load_SET_data.jl")

struct NeuralODE{M <: Lux.AbstractExplicitLayer, R, N <: Float32, So, Se, T, K} <: Lux.AbstractExplicitContainerLayer{(:model,)}
    model::M
    randGen::R
    noiseConstant::N
    solver::So
    sensealg::Se
    tspan::T
    kwargs::K
end

function NeuralODE(model::Lux.AbstractExplicitLayer, randGen; noiseConstant=1.0f0, solver=Euler(), sensealg=ReverseDiffAdjoint(), tspan=(0.00f0, 0.50f0), kwargs...)
    return NeuralODE(model, randGen, noiseConstant, solver, sensealg, tspan, kwargs)
end

function (n::NeuralODE)(x, ps, st)
    function make_new_func(func)
        function dudt(u, p, t)
            u_, st = n.model((u, func(t), Lux.randn32(n.randGen, 100)), p, st)
            return u_
        end
        return dudt
    end
    function prob_func(prob, i, repeat)
        remake(prob;
                    f = ODEFunction{false}(make_new_func(x.funcs[i])),
                    u0 = x.array + (n.noiseConstant .* Lux.randn32(n.randGen, 100)))
    end

    prob = ODEProblem{false}(ODEFunction{false}(make_new_func(x.funcs[1])), x.array, n.tspan, ps)
    ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)

    return solve(ensemble_prob, n.solver, EnsembleThreads(), trajectories = length(x.funcs); sensealg=n.sensealg, n.kwargs...), st
end

function Lux.apply(layer::NeuralODE, x::ArrayAndFuncs, ps, st::NamedTuple)
	y, st = layer(x, ps, st)
	return y, st
end

function ensemsol_to_array(x::EnsembleSolution)
	return Array(x)
end

function create_model(rng::AbstractRNG, gain_init::Float32, gain_recur::Float32, gain_out::Float32, tau::Float32, noise_IC::Float32, noise_recur::Float32)
	input_init(rng, dims...) = Lux.glorot_normal(rng, dims...; gain=gain_init)
	recurrent_init(rng, dims...) = Lux.glorot_normal(rng, dims...; gain=gain_recur)
	output_init(rng, dims...) = Lux.glorot_normal(rng, dims...; gain=gain_out)
	
	act_func = x -> NNlib.tanh_fast.(x)
	invtau_func = x -> (1/tau) .* x
    internal_noise = x -> noise_recur .* x
	
    model = Chain(Scale(100; use_bias=false, init_weight=Lux.glorot_normal),
                NeuralODE(Chain(Parallel(+,
                                SkipConnection(Chain(act_func, Dense(100,100; init_weight=recurrent_init, use_bias=false)), -),
                                Dense(100,100; init_weight=input_init),
                                WrappedFunction(internal_noise)),
                                invtau_func),
                    rng, noiseConstant=noise_IC, dt=0.01f0, save_start=false, adaptive=false), 
                ensemsol_to_array, 
                act_func,
                BranchLayer(Dense(100,1; init_weight=output_init),
                            NoOpLayer()))
	
    ps, st = Lux.setup(rng, model)
	ps = ComponentArray(ps)
	
    return model, ps, st
end