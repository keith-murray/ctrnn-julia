module RecurrentNetworks

export loadData, FunctionArray, ArrayAndFunctionArray
include("load_SET_data.jl")

export NeuralODE, create_model
include("neuralode.jl")

end # module RecurrentNetworks
