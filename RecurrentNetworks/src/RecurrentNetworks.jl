module RecurrentNetworks

using Random, DataFrames, JLD2, Serialization, ComponentArrays

export loadData, Interpolate, ArrayAndFuncs
include("load_SET_data.jl")

export create_model
include("neuralode.jl")

export train
include("training_funcs.jl")

export main
function main(training::String, testing::String, seed::Int64, batch::Int64, epochs::Int64, neurons::Int64, gain_init::Float32, gain_recur::Float32, gain_out::Float32, tau::Float32, noise_IC::Float32, noise_recur::Float32, L2_mag::Float32, AR_mag::Float32, lr::Float32)
    training_input_funcs, training_output = loadData(training)
    testing_input_funcs, testing_output = loadData(testing)

    rng = Random.default_rng()
    Random.seed!(rng, seed)

    IC = ones(Float32, neurons)
    training_data = (training_input_funcs, training_output)
    testing_data = (ArrayAndFuncs(IC, testing_input_funcs), testing_output)

    model, ps, st = create_model(rng, neurons, gain_init, gain_recur, gain_out, tau, noise_IC, noise_recur)

    ps_out, accuracies = train(rng, batch, epochs, model, ps, st, IC, training_data, testing_data, L2_mag, AR_mag, lr)

    return ps_out, accuracies
end

export schedule
function schedule(location::String, output::String, row_num::Int64)
    df_loaded = load(location, "df")
    row = eachrow(df_loaded)[row_num]

    ps_out, accuracies = main(row...)

    output_file = output * "model_$(row_num).jls"
    open(output_file, "w") do f
        serialize(f, ps_out)
        serialize(f, accuracies)
    end
end

end # module RecurrentNetworks
