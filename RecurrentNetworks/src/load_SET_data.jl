using Serialization, Lux, Random

struct Interpolate
	SET::Array{Int64, 1}
	locations::Array{Float32, 1}
	vecs::Array{Float32, 2}
end

function (itp::Interpolate)(t::Float32)
	i = searchsortedfirst(itp.locations, t)
	@inbounds val = itp.SET[i]
	@inbounds itp.vecs[val, :]
end

struct ArrayAndFuncs
	array::Array{Float32, 1}
	funcs::Array{Interpolate, 1}
end

function loadData(output_file::String)
	open(output_file, "r") do f
		input_funcs = deserialize(f)
		output = deserialize(f)
		return [Interpolate(x[1], x[2], x[3]) for x in input_funcs], output
	end
end

function Lux.apply(layer::Lux.AbstractExplicitLayer, x::ArrayAndFuncs, ps, st::NamedTuple)
	y, st = layer(x.array, ps, st)
	return ArrayAndFuncs(y, x.funcs), st
end

function Lux.apply(layer::Lux.Chain, x::ArrayAndFuncs, ps, st::NamedTuple)
	y, st = layer(x, ps, st)
	return y, st
end