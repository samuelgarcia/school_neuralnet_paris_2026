# %% [markdown]
# # 05 · Decoding — **Julia / Judananas**
#
# In notebook 04 we looked for the main patterns of the population activity, without telling the
# methods anything about the task. **Decoding** goes the other way: we ask whether the activity of
# the population **contains information about the task**. We train a classifier to guess, from the
# spike counts of the units in one trial, a property of this trial, here its **block**:
#
# | Block | What happens at the start of the trial |
# |---|---|
# | 1 | a sound, before the conditioning (**pre**) |
# | 2 | the same sound, followed by an optogenetic stimulation (**sound + opto**) |
# | 3 | the same sound alone, after the conditioning (**post**) |
# | 4 | a blue noise followed by an air puff (**air puff**) |
#
# If the classifier guesses the block of **new trials**, not used to train it, better than chance,
# the population carries information about the block. This is the Julia version of
# `python/05-Decoding.ipynb`: same steps, same text, same figures. We use two classic **linear**
# classifiers:
#
# 1. **linear discriminant analysis (LDA)**;
# 2. a **support vector machine (SVM)** with a linear kernel;
#
# and then compare the four prefrontal areas. On the way we meet the main trap of decoding: what
# does the decoder really use?
#
# **Dataset.** Session `156131_20191112-probe0`, the same as in notebooks 01–04, trials of blocks
# 1–4 (notebook 01, section 1), units of ACA, PL, ILA and ORB.
#
# ## 0. Get the data
#
# The session is on the [DANDI Archive](https://dandiarchive.org), in
# [Dandiset 000473](https://dandiarchive.org/dandiset/000473): we use the file
# `sub-156131/sub-156131_ses-156131-20191112-probe0_ecephys+ogen.nwb` (5.6 GB), the same as in
# notebook 01. If you already downloaded it, just write its path in `NWB` below.
#
# **Download the file** to a folder of your choice on your computer
# (keep it out of the git repo: it is far too big for GitHub). Either:
#
# - in the browser: open the
#   [files of the Dandiset](https://dandiarchive.org/dandiset/000473/0.261102.1657/files?location=sub-156131),
#   and click the download icon next to the file;
# - or in a terminal, from the `python/` folder, with the DANDI command-line tool
#   (replace `/path/to/your/data/folder` by the folder where you want to save the file):
#
#   ```bash
#   uv run dandi download -o /path/to/your/data/folder "https://api.dandiarchive.org/api/assets/6b424b64-764a-4677-a828-49094420e930/download/"
#   ```
#
# Then write the full path to the file in `NWB` in the cell below.
# (Streaming the file from DANDI without downloading it is only available in the Python notebook
# for now.)

# %%
using Pkg
# julia/Project.toml: works whether Julia starts in julia/ or in the root of the repo
Pkg.activate(isfile(joinpath(@__DIR__, "Project.toml")) ? (@__DIR__) : "julia")
Pkg.instantiate()

# Judananas: time series and spike trains; Plots: figures; Statistics, LinearAlgebra: maths;
# LIBSVM: support vector machines; Random: random numbers
using Judananas
using Plots, Statistics, LinearAlgebra, Printf, Random
using LIBSVM
gr()
default(left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)  # room for the axis labels
TAB = palette(:tab10)  # the same 10 colors as matplotlib: blue, orange, green, red, purple, brown...

# Full path to the NWB file you downloaded. Change it to where you saved it.
NWB = "/path/to/your/data/folder/sub-156131_ses-156131-20191112-probe0_ecephys+ogen.nwb"

isfile(NWB) || error("No NWB file at $NWB: set NWB to the downloaded file")

# %% [markdown]
# ## 1. Load the units and the trials
#
# Same steps as in notebook 01 (sections 1 and 2): the trials of blocks 1–4, and the units of
# ACA, PL, ILA and ORB that pass the quality criteria.
# We do not remove the saturation periods here (notebook 01, section 1.1): they fall outside the
# trials and only touch a few windows between them, so their effect is negligible.

# %%
data = load_file(NWB)  # open the file and list its content
units = data["units"]  # all sorted units (TsGroup)
trials = data["trials"]  # task trials (IntervalSet)
trials = trials[trials[:Block] .<= 4]  # blocks 1-4

presence_ratio = vec(mean(values(count(units, tot_length(time_support(units)) / 100)) .> 0, dims = 1))
units[:isi_fp] = [q[2] for q in units[:quality]]  # ISI violation false-positive rate
units[:plausible] = (units[:FS] .== 1) .| (units[:RS] .== 1)  # FS or RS label of the file
units[:presence_ratio] = presence_ratio
units[:area] = [m === nothing ? "" : m.match for m in match.(r"^(ACA|PL|ILA|ORB)", units[:location])]
pfc = units[(units[:area] .!= "") .& (units[:isi_fp] .< 0.1) .& units[:plausible] .& (units[:presence_ratio] .> 0)]

block = Int.(trials[:Block])  # the label to decode: block of each trial
block_names = ["sound (pre)", "sound + opto", "sound (post)", "air puff"]  # names of blocks 1-4
block_colors = [TAB[1], TAB[5], TAB[10], TAB[2]]  # as in notebook 04
area_colors = ["ACA" => TAB[1], "PL" => TAB[2], "ILA" => TAB[3], "ORB" => TAB[5]]
println("trials: ", length(trials), " | PFC units: ", length(pfc))
for b in 1:4
    println(rpad(block_names[b], 14), sum(block .== b))
end

# %% [markdown]
# ## 2. One trial = one vector of spike counts
#
# For each trial, we count the spikes of every unit in the **first second after the onset**:
# `count` with an `IntervalSet` and no bin size gives one count per interval. The result is a
# table of **trials × units**: each trial is a point in a 192-dimensional space, and the block is
# its label.

# %%
after = IntervalSet(starts(trials), starts(trials) .+ 1)  # 0 to 1 s after each onset
X = Float64.(values(count(pfc; ep = after)))  # trials x units
y = block
println("X: ", size(X), " (trials x units) | y: ", size(y))

z = (X .- mean(X, dims = 1)) ./ (std(X, dims = 1, corrected = false) .+ 1e-9)  # z-score each unit, for the figure only
fig = heatmap(permutedims(z), color = cgrad(:RdBu, rev = true), clims = (-3, 3), yflip = true,
              xlabel = "Trial (in session order; black lines: start of blocks 2, 3, 4)", ylabel = "Unit",
              title = "Spike counts 0–1 s after the onset (z-scored)", size = (1200, 400))
vline!(fig, [findfirst(y .== b) - 0.5 for b in 2:4], color = :black, lw = 2, label = "")  # first trial of each block

# %% [markdown]
# Some units clearly change from one block to the next. The classifier will combine all of them.
#
# ## 3. Linear discriminant analysis (LDA)
#
# ### 3.1 LDA as a projection
#
# LDA models each class (block) as a cloud of points with its own **mean**, and the **same
# covariance** for all the classes. It then looks for the directions that separate the class means
# the most, relative to the spread inside the classes: the **discriminant axes** (at most
# number of classes − 1 = 3). A new trial is assigned to the class whose mean is the closest, along
# these axes.
#
# With 192 units and only 250 trials, the covariance is badly estimated. We **shrink** it toward a
# simpler, diagonal one (`shrinkage = 0.5`: half-way between the two): a classic regularization.
#
# LDA takes a few lines of linear algebra, so we write it ourselves (the same model as
# `LinearDiscriminantAnalysis(solver="eigen", shrinkage=0.5)` of scikit-learn in the Python
# notebook). Below, we fit LDA on **all** the trials and project them on the first two
# discriminant axes.

# %%
"""Covariance of the rows of X, shrunk toward a diagonal matrix (shrinkage between 0 and 1)."""
function shrunk_cov(X, shrinkage)
    S = cov(X, corrected = false)
    return (1 - shrinkage) .* S + shrinkage * tr(S) / size(S, 1) * I
end

"""Fit LDA on the rows of X with labels y. Returns the classes, the discriminant weights and the projection axes."""
function fit_lda(X, y; shrinkage = 0.5)
    classes = sort(unique(y))
    priors = [mean(y .== c) for c in classes]  # fraction of the trials in each class
    means = reduce(hcat, [vec(mean(X[y .== c, :], dims = 1)) for c in classes])  # units x classes
    Σ = sum(p .* shrunk_cov(X[y .== c, :], shrinkage) for (p, c) in zip(priors, classes))  # within-class
    W = Σ \ means  # one discriminant function per class: x' W[:, k] + b[k]
    b = [-0.5 * means[:, k]' * W[:, k] + log(priors[k]) for k in eachindex(classes)]
    between = shrunk_cov(X, shrinkage) - Σ  # between-class covariance
    e = eigen(Symmetric(between), Symmetric(Σ))  # directions that separate the classes the most
    axes_ = e.vectors[:, sortperm(e.values, rev = true)[1:length(classes)-1]]
    return (; classes, W, b, axes_)
end

"""Class of each row of X: the class with the largest discriminant function."""
predict_lda(model, X) = [model.classes[argmax(row)] for row in eachrow(X * model.W .+ model.b')]

lda = fit_lda(X, y)
projection = X * lda.axes_  # trials x 3 discriminant axes

fig = plot(xlabel = "Discriminant axis 1", ylabel = "Discriminant axis 2", title = "LDA projection (all trials)",
           size = (600, 500))
for b in 1:4
    scatter!(fig, projection[y .== b, 1], projection[y .== b, 2], ms = 3, msw = 0, color = block_colors[b],
             label = block_names[b])
end
fig

# %% [markdown]
# The four blocks are well separated. But this is **not** a proof that the activity carries
# information about the block: with 192 dimensions, a classifier can separate almost any
# labels on the trials it was trained on (try it with shuffled labels: `shuffle(rng, y)`).
# We must test it on trials it has **never seen**.
#
# ### 3.2 Cross-validation
#
# **k-fold cross-validation**: we cut the trials into 5 groups (the **folds**), train on 4 of them,
# test on the 5th, and repeat so that every trial is tested once. `stratified_folds` keeps the same
# proportion of each block in every fold, and `decode` returns the prediction for each trial, made
# by a classifier that did not see it. A classifier, here, is a function that learns from the
# training trials and predicts the test trials.
#
# **Chance level.** Block 2 has 100 trials, the others 50: always answering "block 2" would
# already be right 40 % of the time. A better reference: decode **shuffled labels** (the
# link between activity and block is destroyed), several times.

# %%
rng = Xoshiro(0)  # random generator with a fixed seed: same result for everyone

"""Fold (1 to k) of each trial: the trials of each class are shuffled and dealt to the k folds in turn."""
function stratified_folds(y, k, rng)
    folds = zeros(Int, length(y))
    for c in unique(y)
        trials_c = shuffle(rng, findall(y .== c))
        folds[trials_c] = mod1.(1:length(trials_c), k)
    end
    return folds
end

"""Cross-validated predictions and accuracy (fraction of trials decoded correctly)."""
function decode(classifier, X, y, folds)
    predicted = similar(y)
    for f in unique(folds)
        test = folds .== f
        predicted[test] = classifier(X[.!test, :], y[.!test], X[test, :])  # train without the test trials
    end
    return predicted, mean(predicted .== y)
end

lda_classifier(X_train, y_train, X_test) = predict_lda(fit_lda(X_train, y_train), X_test)

cv = stratified_folds(y, 5, rng)
predicted, accuracy = decode(lda_classifier, X, y, cv)
chance = [decode(lda_classifier, X, shuffle(rng, y), cv)[2] for _ in 1:20]  # 20 label shuffles
@printf("LDA accuracy: %.2f | shuffled labels: %.2f ± %.2f\n", accuracy, mean(chance), std(chance))

"""Confusion matrix in % of the trials of each true block (rows)."""
function plot_confusion(y, predicted, title)
    cm = [100 * mean(predicted[y .== t] .== d) for t in 1:4, d in 1:4]
    p = heatmap(cm, color = :Blues, clims = (0, 100), yflip = true, colorbar = false, aspect_ratio = 1,
                xticks = (1:4, block_names), yticks = (1:4, block_names), xrotation = 30,
                xlabel = "Decoded block", ylabel = "True block", title = title)
    annotate!(p, [(d, t, text(round(Int, cm[t, d]), 9, cm[t, d] > 50 ? :white : :black)) for t in 1:4 for d in 1:4])
    return p
end

fig = plot(plot_confusion(y, predicted, @sprintf("LDA, 0–1 s after the onset (%.0f %%)", 100 * accuracy)),
           size = (520, 500), bottom_margin = 10Plots.mm)

# %% [markdown]
# Almost all the trials are decoded correctly, far above the shuffled labels. The **confusion
# matrix** shows the rare mistakes: which block is taken for which.
#
# ## 4. Support vector machine (SVM)
#
# A linear SVM also separates the classes with flat boundaries (hyperplanes), but it chooses them
# differently: it places each boundary as **far as possible from the closest trials** (the
# "support vectors"), and ignores the trials far from the boundary. It makes no assumption about
# the distribution of the classes. With more than two classes, LIBSVM (the library behind
# scikit-learn's `SVC`) trains one SVM for each pair of blocks, and each trial gets the block that
# wins the most votes.
#
# Two practical points:
#
# - an SVM depends on the scale of each feature: we **z-score** the units first. The z-scoring is
#   learned on the training trials only, inside each fold;
# - `cost` (called `C` in scikit-learn) sets the trade-off between a wide margin and few training
#   errors (small `cost`: wider margin, more regularization).

# %%
function svm_classifier(X_train, y_train, X_test; cost = 0.01)
    m, s = mean(X_train, dims = 1), std(X_train, dims = 1) .+ 1e-9  # z-scoring learned on the training trials
    model = svmtrain(permutedims((X_train .- m) ./ s), y_train; kernel = Kernel.Linear, cost)  # one column per trial
    return svmpredict(model, permutedims((X_test .- m) ./ s))[1]
end

predicted_svm, accuracy_svm = decode(svm_classifier, X, y, cv)
@printf("SVM accuracy: %.2f\n", accuracy_svm)

fig = plot(plot_confusion(y, predicted, @sprintf("LDA (%.0f %%)", 100 * accuracy)),
           plot_confusion(y, predicted_svm, @sprintf("SVM (%.0f %%)", 100 * accuracy_svm)),
           layout = (1, 2), size = (1000, 500), bottom_margin = 10Plots.mm)

# %% [markdown]
# The two classifiers do equally well. With linear classifiers and this much information, the
# choice of the classifier matters much less than the question we ask, as we see now.
#
# ## 5. The trap: what does the decoder really use?
#
# The blocks follow each other in time: block 1 is the first 50 trials, block 4 the last 50. So
# anything that **changes slowly over the session** (the state of the mouse, its arousal, a slow
# drift of the probe) also differs between blocks. Is the decoder reading the response to the
# stimuli, or the time in the session?
#
# **Control: decode the block from the second *before* the onset**, when no stimulus has been
# given yet.

# %%
before = IntervalSet(starts(trials) .- 1, starts(trials))  # 1 s before each onset
X_before = Float64.(values(count(pfc; ep = before)))
predicted_before, accuracy_before = decode(lda_classifier, X_before, y, cv)
@printf("LDA accuracy before the onset: %.2f\n", accuracy_before)

# %% [markdown]
# The block is decoded almost as well **before** the stimulus as after it! Most of what the
# decoder uses is the slow change of the activity over the session, which we already saw in
# notebook 04 (PC2, and the trial factors of TCA). Two common fixes:
#
# 1. **Evoked activity**: subtract, in each trial, the count before the onset from the count
#    after it (`X - X_before`). What remains is the change caused by the stimulus.
# 2. **Cross-validation in chunks**: with random folds, the test trials have neighbors (trials
#    a few seconds away) in the training set. We cut each block into 5 chunks of **consecutive**
#    trials instead, and use the chunks as folds. The test trials are then further in time from the
#    training trials.

# %%
position = reduce(vcat, [0:sum(y .== b)-1 for b in 1:4])  # trial number in its block (from 0)
n_in_block = [sum(y .== b) for b in y]  # number of trials of the block of each trial
chunks = position .* 5 .÷ n_in_block .+ 1  # fold = chunk (1-5) of consecutive trials

windows = ["after (0–1 s)" => X, "before (-1–0 s)" => X_before, "evoked (after − before)" => X - X_before]
results = [decode(lda_classifier, Xw, y, folds)[2] for (_, Xw) in windows, folds in [cv, chunks]]  # windows x folds

fig = bar((1:3) .- 0.2, results[:, 1], bar_width = 0.4, color = :gray, label = "random folds",
          xticks = (1:3, first.(windows)), ylabel = "LDA accuracy", ylims = (0, 1),
          title = "What does the decoder use?", legend = :bottomright, size = (800, 400))
bar!(fig, (1:3) .+ 0.2, results[:, 2], bar_width = 0.4, color = :black, label = "chunks")
hline!(fig, [mean(chance)], color = :red, ls = :dash, label = "shuffled labels")
display(fig)
println(rpad("", 25), "random folds  chunks")
for (k, (name, _)) in enumerate(windows)
    @printf("%-25s %12.2f %7.2f\n", name, results[k, 1], results[k, 2])
end

# %% [markdown]
# - **before** the onset, the accuracy drops a little with the chunks, but stays far above chance:
#   the slow changes really differ from block to block;
# - the **evoked** activity is decoded less well than the raw activity after the onset, but still
#   far above chance: the **responses** themselves differ between blocks.
#
# Look at the confusion matrix of the evoked activity, with the chunks: which blocks are now
# confused?

# %%
predicted_evoked, accuracy_evoked = decode(lda_classifier, X - X_before, y, chunks)
fig = plot(plot_confusion(y, predicted_before, @sprintf("before the onset (%.0f %%)", 100 * accuracy_before)),
           plot_confusion(y, predicted_evoked, @sprintf("evoked, chunks (%.0f %%)", 100 * accuracy_evoked)),
           layout = (1, 2), size = (1000, 500), bottom_margin = 10Plots.mm)

# %% [markdown]
# Before the onset, the few mistakes are between blocks that are **neighbors in time** (block 4
# taken for block 3, ...): the signature of a slow change over the session.
#
# With the evoked activity, the mistakes are mostly between the three **sound** blocks, which
# start with the same stimulus. Blocks 1 and 3 (the same sound, before and after the conditioning)
# are still partly separated: the response to the sound may have changed with the conditioning,
# but this is also where the slow changes of the session are hardest to rule out.
#
# **Take-home message.** A decoder uses **anything** that differs between the classes. Always ask
# what else differs between them (time in the session, movements, licking, arousal...), and test
# it with controls.
#
# ## 6. Comparing the prefrontal areas
#
# Which area carries the most information about the block? The areas have different numbers of
# units, and the accuracy grows with the number of units. To compare them fairly, we draw the
# **same number of units** in each area (the number of units of the smallest area), 20 times, and
# decode the evoked activity with the chunks.

# %%
n_units = minimum(sum(pfc[:area] .== a) for (a, _) in area_colors)  # number of units of the smallest area
n_draws = 20
area_accuracy = Dict{String,Vector{Float64}}()
for (a, _) in area_colors
    in_area = findall(pfc[:area] .== a)  # columns of the units of this area
    area_accuracy[a] = [decode(lda_classifier, (X - X_before)[:, shuffle(rng, in_area)[1:n_units]], y,
                               chunks)[2] for _ in 1:n_draws]
end

fig = plot(ylabel = "LDA accuracy (evoked, chunks)", ylims = (0, 1), size = (600, 420),
           title = "$n_units units drawn per area, $n_draws times",
           xticks = (1:4, ["$a\n($(sum(pfc[:area] .== a)) units)" for (a, _) in area_colors]))
for (k, (a, color)) in enumerate(area_colors)
    acc = area_accuracy[a]
    scatter!(fig, k .+ 0.3 .* (rand(rng, n_draws) .- 0.5), acc, color = color, alpha = 0.5, msw = 0, label = "")
    scatter!(fig, [k], [mean(acc)], yerror = [std(acc)], color = :black, label = "")
end
hline!(fig, [mean(chance)], color = :red, ls = :dash, label = "shuffled labels")

# %% [markdown]
# With the same number of units, all the areas decode the block above chance, with similar
# accuracies. (ACA has only as many units as we draw: its 20 draws are all the same.) Differences
# between areas should be tested with more sessions and more animals: here, one session and one
# probe can only give a hint.
#
# **Questions.**
#
# - Decode only blocks 1 and 3 (same sound, before and after the conditioning:
#   `keep = in([1, 3]).(y)`). Is the difference still there with the evoked activity? And
#   before the onset?
# - Use the window 1–2 s after the onset instead of 0–1 s. Which blocks are best decoded now?
# - **Time-resolved decoding**: decode the block in successive 100 ms bins (with `count` and the
#   reshape of notebook 04). When does the accuracy rise above its level before the onset?
# - Replace the shrinkage of LDA by `shrinkage = 0` in `fit_lda`, and the `cost` of the SVM by
#   `cost = 1`. What changes?
