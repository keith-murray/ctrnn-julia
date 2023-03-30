using Serialization, Lux, Random

function constructFunctions(func_pieces)
	signal_values = func_pieces[1]
	signal_locs = func_pieces[2]
	vecs = func_pieces[3]

	function signal(t)
		val = signal_values[searchsortedfirst(signal_locs, t)]
		vec = zeros(Float32, 100)
		if val != 0
			vec = vecs[val,:]
		end
		return vec
	end
	return signal
end

function loadData(output_file::String)
	open(output_file, "r") do f
		input_funcs = deserialize(f)
		output = deserialize(f)
		return [constructFunctions(x) for x in input_funcs], output
	end
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