using Statistics, Lux, Optimisers, Zygote
include("load_SET_data.jl")

meansquarederror(y_pred, y) =  mean(abs2, y_pred .- y)

AR_reg(y) = 0.01f0 * mean(abs2, y)

function construct_loss(AR_mag::Float32)
    function loss(x, y, model, ps, st)
        y_pred, st = model(x, ps, st)
        l = meansquarederror(y_pred[1][:,46:end,:], y[:,46:end,:]) + AR_mag*AR_reg(y_pred[2])
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
    training_iters = div(size(y_expecteds)[3], batch)

    iter_indexes = [rand_indexes[(i-1)*batch+1:i*batch] for i in 1:training_iters]
    iterator = [(ArrayAndFuncs(IC, funs[iter_indexes[i]]), y_expecteds[:,:,iter_indexes[i]]) for i in 1:training_iters]
    return iterator
end

function constructResults(loss, model, ps, st, training_data, testing_data, IC)
    loss_training_current = loss(ArrayAndFuncs(IC, training_data[1]), training_data[2], model, ps, st)[1]
    loss_testing_current = loss(testing_data[1], testing_data[2], model, ps, st)[1]
    accuracy_training_current = test_accuracy(model, ps, st, ArrayAndFuncs(IC, training_data[1]), training_data[2])
    accuracy_test_current = test_accuracy(model, ps, st, testing_data[1], testing_data[2])
    return [loss_training_current, loss_testing_current, accuracy_training_current, accuracy_test_current]
end

function printUpdates(epoch, epochs, ttime, current_results)
    println("[$epoch/$epochs] \t Time $(round(ttime; digits=2))s \t Train Loss: " * "$(round(current_results[1]; digits=4)) \t " * "Test Loss: $(round(current_results[2]; digits=4))")
    println("[$epoch/$epochs] \t Time $(round(ttime; digits=2))s \t Train Accuracy: " * "$(round(current_results[3] * 100; digits=2))% \t " * "Test Accuracy: $(round(current_results[4] * 100; digits=2))%")
end

function train(rng::AbstractRNG, batch::Int64, epochs::Int64, model, ps, st, IC::Vector{Float32}, training_data, testing_data, L2_mag::Float32, AR_mag::Float32, lr::Float32)	
    opt = Optimisers.OptimiserChain(Optimisers.ClipGrad(5.0), Optimisers.AdamW(lr, (9f-1, 9.99f-1), L2_mag))
    st_opt = Optimisers.setup(opt, ps)
    loss = construct_loss(AR_mag)
    accuracies = Lux.zeros32(rng, 4, epochs+1)

    ### Warmup the Model
    stime = time()
    iterator = constructIterator(rng, batch, training_data, IC)
    loss(iterator[1][1], iterator[1][2], model, ps, st)
    (l, _), back = pullback(p -> loss(iterator[1][1], iterator[1][2], model, p, st), ps)
    back((one(l), nothing))
    current_results = constructResults(loss, model, ps, st, training_data, testing_data, IC)
    ttime = time() - stime
    accuracies[:, 1] = current_results
    printUpdates(0, epochs, ttime, current_results)

    ### Lets train the model
    for epoch in 1:epochs
        stime = time()
        iterator = constructIterator(rng, batch, training_data, IC)
        for (x, y) in iterator
            (l, st), back = pullback(p -> loss(x, y, model, p, st), ps)
            ### We need to add `nothing`s equal to the number of returned values - 1
            gs = back((one(l), nothing))[1]
            st_opt, ps = Optimisers.update(st_opt, ps, gs)
        end

        current_results = constructResults(loss, model, ps, st, training_data, testing_data, IC)
        ttime = time() - stime
        accuracies[:, epoch+1] = current_results
        printUpdates(epoch, epochs, ttime, current_results)
    end
	return ps, accuracies
end