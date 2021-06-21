# download 'atlas_data.csv' from "https://mega.nz/file/7sxx3ajJ#cgPEn1LGSutiJTCrNeeoVbuHGclxfYKZL2IhqhzpLHA"
# and put it in the same folder as this script

import MLJFlux
using DataFrames, DataFramesMeta, CSV, Alert, ProgressMeter, Plots, Flux, MLJ,  Random
using Flux: onehotbatch, onecold, @epochs, Data.DataLoader, Optimiser
using Chain: @chain
using MLDataUtils: splitobs, shuffleobs
using StatsBase: standardize, ZScoreTransform

function build_model(input, layers, output; activation = relu, use_softmax = true, use_last_activation = false)
	f = []
	in_layer = input

	for out_layer in layers
		append!(f, [Dense(in_layer, out_layer, activation)])
		in_layer = out_layer
	end

	if use_last_activation
		append!(f, [Dense(in_layer, output, activation)])
	else
		append!(f, [Dense(in_layer, output)])
	end

	if use_softmax	
		append!(f, [softmax])
	end

	Chain(f...)
end

function score_accuracy(X_output, y_output, classes = [1, 0])
	X = onecold(X_output, classes)
	y = onecold(y_output, classes)
	comparison = X .== y
	return sum(a -> a > 0, comparison) / length(comparison)
end

function flux_run_model(df, dataset_frac, hidden_layers, input_loss, opt, n_epochs, batchsize, λ = 0.0001, tol = 1e-4, iter_tol = 10)
	# converts the 'Label' column to a matrix of 0 and 1
	# compatible with the neural newtork
	X, y = select(df, Not(:Label)), @chain df begin
		select(_, :Label)
		Flux.onehotbatch(_.Label, ["s", "b"])
	end

	N_input  = length(names(X))
	N_output = size(y, 1)

	X = transpose(standardize(ZScoreTransform, Matrix(X)))
	X_train, X_test = splitobs(X, at = dataset_frac)
	y_train, y_test = splitobs(y, at = dataset_frac)

	model = build_model(N_input, hidden_layers, N_output)
	loss(a, b) = input_loss(model(a), b)

	ps = Flux.params(model)
	
	loader = DataLoader(
		(X_train, y_train),
		batchsize = batchsize,
		shuffle = false
	)

	tol_counter = 0
	loss_train_values = []
	loss_test_values = []
	acc_train_values = []
	acc_test_values = []
	
	prev = loss(X_train, y_train)
	p = Progress(n_epochs, dt = 1)
	generate_showvalues(i, n) = () -> [(:current_epoch, i), (:tot_epochs, n)]
	
	for i in 1:n_epochs 
		Flux.train!(loss, ps, loader, Optimiser(WeightDecay(λ), opt))

		l_train = loss(X_train, y_train)
		l_test = loss(X_test, y_test)
		append!(loss_train_values, l_train)
		append!(loss_test_values, l_test)
		append!(acc_train_values, score_accuracy(model(X_train), y_train))
		append!(acc_test_values, score_accuracy(model(X_test), y_test))

		if abs(l_train - prev) < tol
			tol_counter += 1
		else
			tol_counter = 0
		end

		if tol_counter == iter_tol
			break
		end

		prev = l_train

		ProgressMeter.next!(p; showvalues = generate_showvalues(i, n_epochs))
	end

	if tol_counter == iter_tol
		@warn "Terminated due to having reached tol = $tol for $iter_tol times in a row"
	end

	return loss_train_values, loss_test_values, acc_train_values, acc_test_values
end

mutable struct MyNetwork{F <: Function} <: MLJFlux.Builder
    layers :: Vector{Int64}
	activation :: F
	use_softmax :: Bool
	use_last_activation :: Bool
end

function MLJFlux.build(nn::MyNetwork, n_in, n_out)
	layers = nn.layers
	activation = nn.activation
	use_softmax = nn.use_softmax
	use_last_activation = nn.use_last_activation

    f = []
	in_layer = n_in

	for out_layer in layers
		append!(f, [Dense(in_layer, out_layer, activation)])
		in_layer = out_layer
	end

	if use_last_activation
		append!(f, [Dense(in_layer, n_out, activation)])
	else
		append!(f, [Dense(in_layer, n_out)])
	end

	if use_softmax	
		append!(f, [softmax])
	end

	Chain(f...)
end

function mljflux_run_model(df, holdout_frac, hidden_layers, input_loss, opt, n_epochs, batchsize, λ = 0.0001, α = 0.0)
	y, X = unpack(df, ==(:Label), colname -> true)

	X = coerce(X, Count => Continuous)
	y = coerce(y, Multiclass)

	NeuralNetworkClassifier = @load NeuralNetworkClassifier

	clf = NeuralNetworkClassifier(
		builder = MyNetwork(
			hidden_layers, 
			relu,
			false,
			false
		),
		finaliser = softmax,
		optimiser = opt,
		loss = input_loss,
		epochs = n_epochs,
		batch_size = batchsize,
		lambda = λ,
		alpha = α,
		optimiser_changes_trigger_retraining = false
	) 

	mach = machine(clf, X, y)

	evaluate!(
		mach,
		resampling = Holdout(
			fraction_train = holdout_frac
		),
		operation = predict_mode,
		measure = accuracy,
		verbosity = 2
	)
end

opt_dict = Dict([
	("momentum", Momentum(η, ρ)),
	("adam", ADAM(η, (β₁, β₂)))
])

# assuming 'atlas_data.csv' is in the same folder
filepath = joinpath(pwd(), "atlas_data.csv")

# after this, df contains 250k rows and 31 columns (labels included)
# this simply removes useless columns from the dataset
# and select data tagged with KaggleSet = "t"
df = @chain begin
	CSV.read(filepath, DataFrame)
	@where(_, :KaggleSet .== "t")
	select(_, Not([:Weight, :EventId, :KaggleSet, :KaggleWeight]))
end

# PARAMETERS ------------------------------------------------------------------------------------------------------------
hidden_layers = [20, 10, 5]

optimizer = "adam" # or "momentum"

η  = 1e-3 # learning rate
ρ  = 0.99 # for momentum opt
β₁ = 0.9  # for adam opt
β₂ = 0.999 # for adam opt
λ  = 0.0002 # L2 regularization
α  = 0.0 # used only in MLJ's NeuralNetworkClassifier

# for Flux model only
tol = 1e-4
iter_tol = 10

batchsize = 200
n_epochs = 10

# rows of dataset to use. Set it to `length(names(df))` to use the entire dataset
N_rows = 20000

# fraction of dataset for training (in Flux) or holdout fraction (in MLJFlux)
frac = 0.7

#
# FLUX-----------------------------------------------------------------------------------------------------------------
#

loss_train_values, loss_test_values, acc_train_values, acc_test_values = @alert "Flux finished" @time flux_run_model(
	df[1:N_rows, :],
	frac,
	hidden_layers,
	Flux.Losses.crossentropy,
	opt_dict[optimizer],
	n_epochs,
	batchsize,
	λ,
	tol,
	iter_tol
)

p_loss = plot(
	loss_train_values, 
	title = "Loss", 
	xlabel = "Epoch", 
	ylabel = "Loss", 
	label = "Training",
)

plot!(
	p_loss,
	loss_test_values, 
	label = "Testing"
)

p_acc = plot(
	acc_train_values, 
	title = "Accuracy", 
	xlabel = "Epoch", 
	ylabel = "Accuracy",
	label = "Training",
	legend = :bottomright
)

plot!(
	p_acc,
	acc_test_values, 
	label = "Testing"
)

#
# MLJFlux ---------------------------------------------------------------------------------------------------
#

ev = @alert "MLJFlux Finished" @time mljflux_run_model(
	df[1:N_rows, :],
	frac,
	hidden_layers,
	Flux.crossentropy,
	opt_dict[optimizer],
	n_epochs,
	batchsize,
	λ,
	α
)
