using Statistics, Lux, Optimisers, Zygote
include("load_SET_data.jl")

meansquarederror(y_pred, y) =  mean(abs2, y_pred .- y)

L2_reg(ps) = sum(abs2, ps.layer_2.layer_1.layer_2.weight) + sum(abs2, ps.layer_2.layer_1.layer_1.layer_2.weight) + sum(abs2, ps.layer_5.layer_1.weight)

AR_reg(y) = mean(abs2, y)

function construct_loss(L2_mag::Float32, AR_mag::Float32)
    function loss(x, y, model, ps, st)
        y_pred, st = model(x, ps, st)
        l = meansquarederror(y_pred[1], y) + L2_mag*L2_reg(ps) + AR_mag*AR_reg(y_pred[2])
        return l, st
    end
    return loss
end

function transformVector(v::Vector{Float32})
    return [x < -0.5f0 ? -1.0f0 : x > 0.5f0 ? 1.0f0 : 0.0f0 for x in v]
end

function test_accuracy(model, ps, st, x, y_expected)
    st = Lux.testmode(st)
    y_observed, st = model(x, ps, st)
    y_observed_end = transformVector(y_observed[1][1,end,:])
    num_matching = sum(y_observed_end .== y_expected[1,end,:])
    return num_matching / length(y_observed_end)
end

function constructIterator(rng::AbstractRNG, batch::Int64, training_data, IC::Vector{Float32})
    funs = training_data[1]
    y_expecteds = training_data[2]
    rand_indexes = randperm(rng, size(y_expecteds)[3])
    iterator = []
    training_iters = div(size(y_expecteds)[3], batch)

    iter_indexes = [rand_indexes[(i-1)*batch+1:i*batch] for i in 1:training_iters]
    iter_funcs = [FunctionArray(funs[iter_indexes[i]]) for i in 1:training_iters]
    iter_y_expecteds = [y_expecteds[:,:,iter_indexes[i]] for i in 1:training_iters]

    for i in 1:training_iters
        push!(iterator, (ArrayAndFunctionArray(IC, iter_funcs[i]), iter_y_expecteds[i]))
    end
    return iterator
end

function train(rng::AbstractRNG, batch::Int64, epochs::Int64, model, ps, st, training_data, testing_data, L2_mag::Float32, AR_mag::Float32, lr::Float32)	
    opt = Optimisers.ADAM(lr)
    st_opt = Optimisers.setup(opt, ps)
    loss = construct_loss(L2_mag, AR_mag)
    accuracies = Lux.zeros32(rng, epochs+1)
    IC = Lux.ones32(rng, 100)


    ### Warmup the Model
    iterator = constructIterator(rng, batch, training_data, IC)
    loss(iterator[1][1], iterator[1][2], model, ps, st)
    (l, _), back = pullback(p -> loss(iterator[1][1], iterator[1][2], model, p, st), ps)
    back((one(l), nothing))
    accuracies[1] = test_accuracy(model, ps, st, testing_data[1], testing_data[2])

    ### Lets train the model
    for epoch in 1:epochs
        iterator = constructIterator(rng, batch, training_data, IC)
        for (x, y) in iterator
            (l, st), back = pullback(p -> loss(x, y, model, p, st), ps)
            ### We need to add `nothing`s equal to the number of returned values - 1
            gs = back((one(l), nothing))[1]
            st_opt, ps = Optimisers.update(st_opt, ps, gs)
        end
        accuracies[epoch+1] = test_accuracy(model, ps, st, testing_data[1], testing_data[2])
    end
	return ps, accuracies
end