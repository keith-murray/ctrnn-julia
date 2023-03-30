module RecurrentNetworks

using Random

export loadData, FunctionArray, ArrayAndFunctionArray
include("load_SET_data.jl")

export NeuralODE, create_model
include("neuralode.jl")

export train
include("training_funcs.jl")

export main
function main(training::String, testing::String, seed::Int64, batch::Int64, epochs::Int64, gain_init::Float32, gain_recur::Float32, gain_out::Float32, tau::Float32, noise::Float32, L2_mag::Float32, AR_mag::Float32, lr::Float32)
    training_input_funcs, training_display_funcs, training_output = loadData(training)
    testing_input_funcs, testing_display_funcs, testing_output = loadData(testing)

    rng = Random.default_rng()
    Random.seed!(rng, seed)

    IC = ones(Float32, 100)
    training_data = (training_input_funcs, training_output)
    testing_data = (ArrayAndFunctionArray(IC, FunctionArray(testing_input_funcs)), testing_output)

    model, ps, st = create_model(rng, gain_init, gain_recur, gain_out, tau, noise)

    ps_out, accuracies = train(rng, batch, epochs, model, ps, st, training_data, testing_data, L2_mag, AR_mag, lr)

    return ps_out, accuracies
end

end # module RecurrentNetworks
