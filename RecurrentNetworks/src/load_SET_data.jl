using Serialization, Lux, Random

function loadData(output_file::String)
	open(output_file, "r") do f
		input_funcs = deserialize(f)
		display_funcs = deserialize(f)
		output = deserialize(f)
	end
	return input_funcs, display_funcs, output
end

struct FunctionArray{F} <: AbstractArray{F, 1}
	data::Array{F, 1}
end

struct ArrayAndFunctionArray{A <: AbstractArray, B <: FunctionArray}
	array::A
	funcs::B
end

Base.size(A::FunctionArray) = size(A.data)
Base.getindex(A::FunctionArray, i::Int) = A.data[i]

function Lux.apply(layer::Lux.AbstractExplicitLayer, x::ArrayAndFunctionArray, ps, st::NamedTuple)
	y, st = layer(x.array, ps, st)
	return ArrayAndFunctionArray(y, x.funcs), st
end

function Lux.apply(layer::Lux.Chain, x::ArrayAndFunctionArray, ps, st::NamedTuple)
	y, st = layer(x, ps, st)
	return y, st
end

function Lux.apply(layer::NeuralODE, x::ArrayAndFunctionArray, ps, st::NamedTuple)
	y, st = layer(x, ps, st)
	return y, st
end