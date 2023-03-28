using CSV, DataFrames, Random, Lux

function loadSETdata(location::String)
	df = CSV.read(location, DataFrame, missingstring="?", delim=",", header=true)
	accepted = collect(skipmissing(df."3SYYA"))
	rejected = collect(skipmissing(df."3SYYR"))
	return rejected, accepted
end

function make_signal(rng::AbstractRNG, SET::Int64, lag::Tuple{Float32, Float32}, min_pulse_gap::Float32, pulse_width::Float32)
	
    SET_string = reverse(digits(SET))
	SET_length = length(SET_string)
	zero_string = zeros(Int64, SET_length)
	signal_values = [zero_string SET_string]'[:]
	append!(signal_values, zeros(Int64, 1))
	
	signal_begin = sort(rand(rng, Float32, SET_length).*(lag[2]-lag[1]).+lag[1])
	signal_gaps = [signal_begin[i]-signal_begin[i-1] for i in 2:length(signal_begin)]
	while minimum(signal_gaps) < min_pulse_gap
		signal_begin = sort(rand(rng, Float32, SET_length).*(lag[2]-lag[1]).+lag[1])
		signal_gaps = [signal_begin[i]-signal_begin[i-1] for i in 2:length(signal_begin)]
	end
	
	signal_end = signal_begin.+pulse_width
	signal_locs = [signal_begin signal_end]'[:]
	
	function signal(t)
		val = signal_values[searchsortedfirst(signal_locs, t)]
		vec = Lux.zeros32(rng, 3)
		if val != 0
			vec[val] = 1.0f0
		end
		return vec
	end
	return signal
end

function constructSetbatchFunction(batch::Int64, location::String, lag::Tuple{Float32, Float32}, min_pulse_gap::Float32, pulse_width::Float32, y_modifier)
    data = loadSETdata(location)

    function constructSetbatch(rng::AbstractRNG)
        funcs = []
        y_expected = Lux.zeros32(rng, batch)
        for i in 1:batch
            y = rand(rng, [1,2])
            SET_num = rand(rng, data[y])
            push!(funcs, make_signal(rng, SET_num, lag, min_pulse_gap, pulse_width))
            y_expected[i] = y_modifier(y)
        end
        return funcs, y_expected
    end

    return constructSetbatch
end