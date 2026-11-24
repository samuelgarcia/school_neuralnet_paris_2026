# %% [markdown]
# # 04 · Population activity and dimensionality reduction — **Julia / Judananas**
#
# So far we looked at one neuron (notebooks 01 and 02) or one pair of neurons (notebook 03) at a
# time. But we record ~200 prefrontal units at once. At each moment, the population activity is a
# point in a space with **one axis per neuron**: ~200 dimensions, impossible to look at. Yet the
# neurons are not independent: many of them go up and down together. **Dimensionality reduction**
# finds a few directions (combinations of neurons) that capture most of this shared activity.
#
# This is the Julia version of `python/04-Population_and_DimensionalityReduction.ipynb`: same
# steps, same text, same figures. It is a short tour of four methods, with what each of them is
# good for:
#
# 1. **PCA** (principal component analysis) on the trial-averaged activity: the main patterns of
#    the population response, and how it changes from block to block;
# 2. **TCA** (tensor component analysis) on the single-trial activity: patterns across neurons,
#    **time within a trial**, and **trials** at once;
# 3. **UMAP** and a **self-organizing map (SOM)**: non-linear maps of the units themselves,
#    described by the firing features of notebooks 01 and 02 (rate, burstiness, memory, waveform).
#
# **Dataset.** Session `156131_20191112-probe0`, the same as in notebooks 01–03, with the trials of
# blocks 1–4 (see the trial structure in notebook 01, section 1) and the units of the four
# prefrontal areas (ACA, PL, ILA, ORB).
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

# Judananas: time series and spike trains; HDF5: reads the NWB file; Plots: figures;
# Statistics, LinearAlgebra: maths;
# MultivariateStats: PCA; UMAP: UMAP; Clustering: k-means; Random: random numbers
using Judananas, HDF5
using Plots, Statistics, LinearAlgebra, Printf, Random
using MultivariateStats, UMAP, Clustering
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
# ACA, PL, ILA and ORB that pass the quality criteria. We also keep the trough-to-peak duration
# We do not remove the saturation periods here (notebook 01, section 1.1): they fall outside the
# trials and only touch a few windows between them, so their effect is negligible.
# of the waveform (`Peak_to_Trough`, read by Judananas) to classify the units into **narrow** and
# **wide** waveforms with the two thresholds of notebook 02 (section 3.1).

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

low_th, high_th = 0.380, 0.430  # ms, the thresholds of notebook 02
pfc[:wf_class] = [d < low_th ? "narrow" : d > high_th ? "wide" : "unclassified" for d in pfc[:Peak_to_Trough]]

block = Int.(trials[:Block])  # block of each trial: 1, 2, 3 or 4
block_names = Dict(1 => "sound (pre)", 2 => "sound + opto", 3 => "sound (post)", 4 => "air puff")
block_colors = Dict(1 => TAB[1], 2 => TAB[5], 3 => TAB[10], 4 => TAB[2])
area_colors = ["ACA" => TAB[1], "PL" => TAB[2], "ILA" => TAB[3], "ORB" => TAB[5]]
println("trials: ", length(trials), " | PFC units: ", length(pfc))
value_counts(pfc, :area)

# %% [markdown]
# ## 2. The population tensor
#
# For each unit and each trial, we count the spikes in 100 ms bins, from 1 s before to 2 s after
# the trial onset. `count` with the 250 windows as `ep` gives one row per bin (the 30 bins of the
# first trial, then the 30 bins of the second...) and one column per unit; we reshape it into a 3D
# array, a **tensor**, of shape (units × trials × time bins). We divide by the bin size to get rates
# (Hz), and smooth each trial in time with a Gaussian kernel (σ = 1 bin), to reduce the noise of
# single trials.

# %%
"""Gaussian smoothing of a vector, with standard deviation σ (in samples), as scipy's gaussian_filter1d."""
function gaussian_smooth(x, σ)
    r = ceil(Int, 4σ)  # the kernel is cut at 4 standard deviations
    kernel = exp.(-(-r:r) .^ 2 ./ (2σ^2))
    kernel ./= sum(kernel)
    n = length(x)
    reflect(i) = i < 1 ? 1 - i : i > n ? 2n + 1 - i : i  # mirror the signal at its edges
    return [sum(kernel[j + r + 1] * x[reflect(i + j)] for j in -r:r) for i in 1:n]
end

bin_size = 0.1  # s
n_bins = 30  # 3 s / 100 ms
window = IntervalSet(starts(trials) .- 1, starts(trials) .+ 2)  # -1 to +2 s around onset
counts = values(count(pfc, bin_size; ep = window))  # (trials x bins) x units
@assert size(counts, 1) == length(trials) * n_bins
rate_tensor = permutedims(reshape(counts, n_bins, length(trials), length(pfc)), (3, 2, 1)) ./ bin_size
rate_tensor = mapslices(x -> gaussian_smooth(x, 1), rate_tensor, dims = 3)  # smooth along time
t_bins = (0:n_bins-1) .* bin_size .- 1 .+ bin_size / 2  # center of each bin, in s
println("tensor (units x trials x bins): ", size(rate_tensor))

# %% [markdown]
# Averaging over the trials of each block gives one PSTH per unit and per block. We put the 4
# blocks side by side, and **z-score** each unit (subtract its mean, divide by its standard
# deviation): otherwise the fast units would dominate everything that follows. Below, one row per
# unit, sorted by its response to the pre-conditioning sound.

# %%
psth = reduce(hcat, [dropdims(mean(rate_tensor[:, block .== b, :], dims = 2), dims = 2) for b in 1:4])  # units x (4 x bins)
z = (psth .- mean(psth, dims = 2)) ./ std(psth, dims = 2, corrected = false)
order = sortperm(vec(mean(z[:, 1:n_bins][:, 0 .< t_bins .< 0.5], dims = 2)))  # response 0-0.5 s, block 1
onset = searchsortedfirst(t_bins, 0)  # first bin after the trial onset

fig = heatmap(z[order, :], color = cgrad(:RdBu, rev = true), clims = (-3, 3), yflip = true,
              xticks = ((0:3) .* n_bins .+ n_bins / 2, [block_names[b] for b in 1:4]),
              xlabel = "Time: -1 to 2 s around the trial onset (dotted line), in each block",
              ylabel = "Unit (sorted)", colorbar_title = "z-scored rate", size = (1300, 500))
vline!(fig, (0:3) .* n_bins .+ 0.5, color = :black, lw = 3, label = "")  # start of each block
vline!(fig, (0:3) .* n_bins .+ onset .- 0.5, color = :black, ls = :dot, label = "")  # trial onset
fig

# %% [markdown]
# Many units are excited just after the trial onset (red after the dotted line), others are
# inhibited (blue). The pattern is similar in the four blocks, but not identical: in blocks 2 and 4,
# a second wave of activity comes about 1 s after the onset.
#
# ## 3. PCA: the main patterns of the population
#
# ### 3.1 Principal components
#
# PCA looks for the direction in the 192-dimensional space (a weighted sum of the units) along
# which the data vary the most: the first **principal component** (PC1). Then the direction, at
# right angles to PC1, with the most remaining variance: PC2, and so on. Each PC has:
#
# - **loadings** (or weights): one value per unit, how much the unit contributes;
# - a **score** (or projection): one value per time point, the activity of the population
#   along this direction.
#
# Here the "samples" are the 4 × 30 time bins and the "features" are the units. (MultivariateStats
# wants one column per sample: `z` has one row per unit and one column per time bin, as needed.)

# %%
pca = fit(PCA, z; pratio = 1.0)  # keep all the components
scores = permutedims(predict(pca, z))  # (4 x bins) x components: one row per time bin
explained = principalvars(pca) ./ var(pca)  # fraction of the variance of each PC
n80 = findfirst(cumsum(explained) .>= 0.8)  # number of PCs for 80 % of the variance

fig = plot(1:20, 100 .* explained[1:20], marker = :o, color = :black, label = "each PC",
           xlabel = "Principal component", ylabel = "Variance explained (%)", title = "$n80 PCs explain 80 %",
           size = (600, 400))
plot!(fig, 1:20, 100 .* cumsum(explained)[1:20], marker = :o, color = :gray, label = "cumulative")
hline!(fig, [80], color = :gray, ls = :dot, label = "")

# %% [markdown]
# A few PCs explain most of the variance of 192 units: the trial-averaged activity is
# **low-dimensional**. (Careful: averaging over trials already removed most of the noise, which
# makes the data look more low-dimensional than single trials.)
#
# ### 3.2 What do the PCs look like?
#
# We plot the scores of the first 3 PCs against time, one curve per block, and the population
# **trajectory** in the plane of PC1 and PC2: each point is the state of the population at one
# moment, and the trajectory goes from 1 s before to 2 s after the onset.

# %%
scores_blocks = [scores[(b-1)*n_bins+1:b*n_bins, 1:3] for b in 1:4]  # one (bins x 3 PCs) matrix per block

panels = [plot(xlabel = "Time from trial onset (s)", ylabel = "Score",
               title = @sprintf("PC%d (%.0f %%)", pc, 100 * explained[pc])) for pc in 1:3]
trajectory = plot(xlabel = "PC1", ylabel = "PC2", title = "Trajectories (dot: -1 s, star: onset)")
for b in 1:4
    s = scores_blocks[b]
    for pc in 1:3
        plot!(panels[pc], t_bins, s[:, pc], color = block_colors[b], lw = 2, label = block_names[b])
    end
    plot!(trajectory, s[:, 1], s[:, 2], color = block_colors[b], lw = 2, label = "")
    scatter!(trajectory, [s[1, 1]], [s[1, 2]], color = block_colors[b], label = "")  # start: 1 s before onset
    scatter!(trajectory, [s[onset, 1]], [s[onset, 2]], color = :black, marker = :star5, ms = 8, label = "")
end
for pc in 1:3
    vline!(panels[pc], [0], color = :black, ls = :dot, label = "")
end
plot!(panels[2], legend = false)
plot!(panels[3], legend = false)
fig = plot(panels..., trajectory, layout = (1, 4), size = (1900, 420), legendfontsize = 7,
           bottom_margin = 10Plots.mm, left_margin = 8Plots.mm)

# %% [markdown]
# - **PC1** is the response common to all the blocks: the population jumps right after the trial
#   onset. In blocks 2 (sound + opto) and 4 (air puff) it jumps a second time, about 1 s after the
#   onset: the response to the aversive stimulus that follows the sound.
# - **PC2** is different in each block **already before the onset**, when nothing has happened
#   yet: it changes steadily from block to block. This is not a response, but a slow change of the
#   activity over the session (the blocks follow each other in time): a change of the state of
#   the mouse, or a **drift** of the recording. We meet it again in notebook 05.
# - In the PC1–PC2 plane, each block makes a loop that starts and ends near its own point (the
#   activity before the trial): the loops are shifted from block to block along PC2.
#
# ### 3.3 Which units make the PCs?
#
# The loadings tell which units contribute to each PC. Do the 4 areas contribute in the same way?

# %%
loadings = projection(pca)[:, 1:3]  # units x 3 PCs

panels = []
for pc in 1:3
    p = plot(xlabel = "Loading on PC$pc", ylabel = "Density")
    for (a, color) in area_colors
        stephist!(p, loadings[pfc[:area] .== a, pc], bins = range(-0.3, 0.3, length = 31),
                  normalize = :pdf, color = color, lw = 2, label = a)
    end
    push!(panels, p)
end
fig = plot(panels..., layout = (1, 3), size = (1500, 350), bottom_margin = 10Plots.mm, left_margin = 8Plots.mm)

# %% [markdown]
# The loadings of PC1 mostly have the same sign: most units go up together after the onset. (The
# sign of a PC is arbitrary: flipping both its loadings and its scores gives the same PC.) The
# loadings of the other PCs are spread around 0 in all the areas: every PC mixes units of the four
# areas, with positive and negative weights. A PC is a pattern of the **population**, not of an
# area.
#
# **Questions.**
#
# - Run the PCA without z-scoring (`psth .- mean(psth, dims = 2)` only). Which units dominate PC1
#   now?
# - Run the PCA on the units of one area only. Are the first PCs the same?
# - Fit the PCA on blocks 1 and 4 only, and project block 3 on these PCs (`predict`).
#
# ## 4. TCA: neurons, time and trials at once
#
# PCA needed the average over trials: it cannot see that the response changes **from trial to
# trial** (with learning, habituation, drift...). Tensor component analysis
# ([Williams et al., 2018](https://doi.org/10.1016/j.neuron.2018.05.015)) works on the whole
# tensor (units × time × trials). It writes the tensor as a sum of $R$ components, each the
# product of three vectors:
#
# $$ x_{n,t,k} \approx \sum_{r=1}^{R} a_r(n)\, b_r(t)\, c_r(k) $$
#
# - $a_r$: the **neuron factor**, which units take part in component $r$;
# - $b_r$: the **time factor**, its time course within a trial;
# - $c_r$: the **trial factor**, its strength in each trial.
#
# We use the **non-negative** version (all the factors ≥ 0, like the rates): the components then
# add up, and are easier to read. As Williams et al., we divide each unit by its maximum, so that
# all the units count. We fit the model ourselves, with **multiplicative updates** (the algorithm
# of `tensorly` in the Python notebook): starting from random factors, each factor is multiplied,
# again and again, by a ratio that brings the model closer to the data and keeps it ≥ 0. We fit it
# a few times (`n_fits`) with different random starts (`seed`), and keep the best fit.

# %%
tensor = permutedims(rate_tensor, (1, 3, 2))  # units x time x trials
tensor = tensor ./ maximum(tensor, dims = (2, 3))  # each unit divided by its maximum

"""Khatri-Rao product: column r is kron(U[:, r], V[:, r])."""
khatri_rao(U, V) = reduce(hcat, [kron(U[:, r], V[:, r]) for r in 1:size(U, 2)])

"""Non-negative TCA (CP decomposition) with multiplicative updates. Returns the 3 factors and the relative error."""
function fit_tca(tensor, rank; seed = 0, n_iter = 1000, tol = 1e-8)
    N, T, K = size(tensor)
    X1 = reshape(tensor, N, T * K)  # the tensor unfolded along each of its 3 dimensions
    X2 = reshape(permutedims(tensor, (2, 1, 3)), T, N * K)
    X3 = reshape(permutedims(tensor, (3, 1, 2)), K, N * T)
    rng = MersenneTwister(seed)
    A, B, C = rand(rng, N, rank), rand(rng, T, rank), rand(rng, K, rank)  # random start
    error, previous = Inf, Inf
    for it in 1:n_iter
        A .*= (X1 * khatri_rao(C, B)) ./ (A * ((B' * B) .* (C' * C)) .+ eps())
        B .*= (X2 * khatri_rao(C, A)) ./ (B * ((A' * A) .* (C' * C)) .+ eps())
        C .*= (X3 * khatri_rao(B, A)) ./ (C * ((A' * A) .* (B' * B)) .+ eps())
        if it % 10 == 0 || it == n_iter  # check the error every 10 iterations (it takes time)
            error = norm(X1 - A * khatri_rao(C, B)') / norm(X1)
            abs(previous - error) < 10tol && break  # stop when the error no longer decreases
            previous = error
        end
    end
    return (A, B, C), error
end

"""Fit TCA n_fits times from different random starts, and keep the fit with the lowest error."""
best_tca(tensor, rank; n_fits = 5) = argmin(fit -> fit[2], [fit_tca(tensor, rank; seed) for seed in 0:n_fits-1])

errors = [best_tca(tensor, rank)[2] for rank in 1:6]  # 30 fits: this cell takes 1-2 minutes

fig = plot(1:6, errors, marker = :o, color = :black, legend = false, xlabel = "Number of components R",
           ylabel = "Relative error", title = "TCA: error vs number of components", size = (500, 350))

# %% [markdown]
# The error decreases slowly, and stays high (~0.6): single-trial spike counts are **noisy**, and
# a large part of the variance is noise that no model can explain. There is no clear "elbow": we
# take $R = 4$ components, enough to see the main patterns. (With 4 components, most random starts
# give the same solution; with 3, they do not.)

# %%
rank = 4
(neuron_f, time_f, trial_f), error = best_tca(tensor, rank)
order_c = sortperm([argmax(time_f[:, r]) for r in 1:rank])  # components sorted by the time of their peak
neuron_f, time_f, trial_f = neuron_f[:, order_c], time_f[:, order_c], trial_f[:, order_c]

unit_order = sortperm(pfc[:area])  # units grouped by area
area_color = Dict(area_colors)
panels = []
for r in 1:rank
    push!(panels, bar(neuron_f[unit_order, r], bar_width = 1, lw = 0, legend = false,
                      color = [area_color[a] for a in pfc[:area][unit_order]], ylabel = "component $r"))
    push!(panels, plot(t_bins, time_f[:, r], color = :black, lw = 2, legend = false))
    vline!(panels[end], [0], color = :gray, ls = :dot)
    push!(panels, scatter(1:length(trials), trial_f[:, r], ms = 2.5, msw = 0, legend = false,
                          color = [block_colors[b] for b in block]))
end
plot!(panels[1], title = "Neuron factor (color: area)")
plot!(panels[2], title = "Time factor")
plot!(panels[3], title = "Trial factor (color: block)")
plot!(panels[end-2], xlabel = "Units, grouped by area: " * join(sort(first.(area_colors)), ", "))
plot!(panels[end-1], xlabel = "Time from trial onset (s)")
plot!(panels[end], xlabel = "Trial (in session order)")
fig = plot(panels..., layout = (rank, 3), size = (1500, 250 * rank), left_margin = 8Plots.mm)

# %% [markdown]
# Each component tells a small story: *these* units, with *this* time course, in *these* trials.
# The components are sorted by the time of the peak of their time factor:
#
# 1. active **before** the onset, and lower after it: the units **inhibited** by the stimuli.
#    Its trial factor slowly increases over the session;
# 2. an **onset response**, ~0.15 s after the onset, in all the blocks, and largest in the air
#    puff block;
# 3. another **onset response**, with other units, present in blocks 1–3 but **absent in
#    block 4**: block 4 starts with a different sound (a blue noise), so this is a response to the
#    **tone**. It decreases over the blocks (**habituation**?);
# 4. a **late response**, ~1 s after the onset, absent in block 1 and strong in blocks 2
#    (sound + opto) and 4 (air puff): the response to the **aversive stimulus** that follows the
#    sound. Look at its trial factor in block 3: after the conditioning, with the sound alone, a
#    small late response is still there in some trials.
#
# TCA found by itself, without knowing the blocks, that some trials have a late response and
# others do not. PCA on trial averages could not tell us whether this response is there in every
# trial of a block, or changes within a block.
#
# **Careful.** TCA has no unique solution: the components can come out in a different order, and
# a different random start (`seed`) can give different factors. Always fit the model several
# times and keep the components that come back every time.
#
# **Questions.**
#
# - Fit TCA with `fit_tca(tensor, 4; seed)` for `seed = 1, 2, 3`. Which components are stable?
# - Try $R = 3$ and $R = 5$. Which components split or merge?
# - Which units have the largest neuron factor in the response component? Look at their PSTH
#   (notebook 01, section 3.5).
#
# ## 5. A map of the units: UMAP and self-organizing maps
#
# PCA and TCA reduce the activity over **time**. We can also reduce the **description of the
# units**: in notebooks 01 and 02, we described each unit by a few numbers. Do the units form
# groups (cell types?) in this feature space? We use four features:
#
# - $\log_{10}$ of the firing rate (notebook 01, section 3.1: the rates span orders of magnitude);
# - the **burstiness** $B$ and the **memory** $M$ in the inter-trial intervals (notebook 01,
#   sections 4.3 and 5), without the corrections of the advanced sections;
# - the **waveform class**: 1 for narrow, 0 for wide (notebook 02, section 3.1).
#
# ### 5.1 The features
#
# We compute $B$ and $M$ in the 3 s windows before each trial, as in notebook 01: windows with at
# least 4 ISIs, units with at least 12 such windows. We keep the units with a narrow or wide
# waveform.

# %%
iti = IntervalSet(starts(trials) .- 3, starts(trials))  # 3 s before each trial
min_isi, min_windows = 4, 12

"""List of the ISI vectors of the windows that have at least min_isi ISIs."""
function isis_per_window(spikes, windows, min_isi = 4)
    isis = [diff(times(restrict(spikes, windows[k]))) for k in 1:length(windows)]
    return filter(isi -> length(isi) >= min_isi, isis)
end

"""Burstiness B = (std - mean) / (std + mean) of a vector of ISIs."""
burstiness(isi) = (std(isi, corrected = false) - mean(isi)) / (std(isi, corrected = false) + mean(isi))

"""Memory coefficient: correlation between consecutive ISIs, pooled over windows."""
function memory(isi_list)
    current = reduce(vcat, [isi[1:end-1] for isi in isi_list])
    following = reduce(vcat, [isi[2:end] for isi in isi_list])
    return cor(current, following)
end

B, M = Float64[], Float64[]
for i in ids(pfc)
    isi_list = isis_per_window(pfc[id = i], iti, min_isi)
    enough = length(isi_list) >= min_windows
    push!(B, enough ? mean(burstiness.(isi_list)) : NaN)
    push!(M, enough ? memory(isi_list) : NaN)
end
pfc[:burstiness], pfc[:memory], pfc[:log_rate] = B, M, log10.(rates(pfc))

units_f = pfc[.!isnan.(pfc[:burstiness]) .& (pfc[:wf_class] .!= "unclassified")]  # units with all features
feature_names = ["log10 rate", "burstiness", "memory", "narrow"]
features = hcat(units_f[:log_rate], units_f[:burstiness], units_f[:memory],
                Float64.(units_f[:wf_class] .== "narrow"))  # units x 4 features
features_z = (features .- mean(features, dims = 1)) ./ std(features, dims = 1)  # z-score: same weight for every feature
println("units with all the features: ", size(features, 1))
for (k, name) in enumerate(feature_names)
    q = quantile(features[:, k], [0, 0.25, 0.5, 0.75, 1])
    @printf("%-11s mean %5.2f  std %4.2f  min %5.2f  25%% %5.2f  50%% %5.2f  75%% %5.2f  max %5.2f\n",
            name, mean(features[:, k]), std(features[:, k]), q...)
end

# %% [markdown]
# As in PCA, we **z-score** each feature: the distances between units should not depend on the
# units of measurement (Hz, no unit...).
#
# ### 5.2 UMAP
#
# [UMAP](https://umap-learn.readthedocs.io) (Uniform Manifold Approximation and Projection) places
# the units on a 2D map so that units that are **neighbors** in the feature space stay neighbors
# on the map. Unlike PCA, it is non-linear: it can unfold curved groups. Its main parameter,
# `n_neighbors`, sets how local the map is. (UMAP.jl also wants one column per sample.)

# %%
Random.seed!(0)
embedding = UMAP.fit(permutedims(features_z), 2; n_neighbors = 15, min_dist = 0.1).embedding  # 2 x units

panels = [scatter(embedding[1, :], embedding[2, :], zcolor = features[:, k], color = :viridis, ms = 3,
                  msw = 0, legend = false, colorbar = true, title = "color: $(feature_names[k])")
          for k in 1:3]
p = plot(title = "color: waveform class")
for (c, color) in ["wide" => TAB[1], "narrow" => TAB[4]]
    sel = units_f[:wf_class] .== c
    scatter!(p, embedding[1, sel], embedding[2, sel], color = color, ms = 3, msw = 0, label = c)
end
push!(panels, p)
fig = plot(panels..., layout = (1, 4), size = (2000, 450), xlabel = "UMAP 1", ylabel = "UMAP 2",
           bottom_margin = 10Plots.mm, left_margin = 8Plots.mm)

# %% [markdown]
# The narrow units form their own island: the waveform class is a binary feature, so it splits
# the map in two. Inside each island, the units are ordered by rate, burstiness and memory, which
# vary **continuously**: no clear sub-groups.
#
# **Careful with UMAP.** The map keeps the neighbors, but not the distances: the size of the
# islands and the distance between them mean little. Change `n_neighbors` (5, 50) and the seed:
# the map changes, the neighbors much less.
#
# ### 5.3 Self-organizing map and clustering
#
# A **self-organizing map** (SOM, [Kohonen, 1982](https://doi.org/10.1007/BF00337288)) is a grid
# of "nodes", each with a prototype (a point in the feature space). During training, each unit
# pulls the prototype of its closest node, and a little the prototypes of the neighbor nodes on the
# grid. In the end, close nodes have similar prototypes: the grid is a smooth 2D map of the feature
# space. Each unit belongs to the node with the closest prototype.
#
# We write a small SOM ourselves (the same rules as `minisom` in the Python notebook) on a 6 × 6
# grid, then group the 36 prototypes into a few **clusters** with k-means: a common way to cluster
# with a SOM.

# %%
"""Node (row, column) whose prototype is the closest to the sample x."""
winner(prototypes, x) = Tuple(argmin([sum(abs2, prototypes[r, c, :] .- x) for r in axes(prototypes, 1),
                                                                         c in axes(prototypes, 2)]))

"""Train a grid x grid SOM on the rows of data, with n_iter samples drawn at random."""
function train_som(data, grid; sigma = 1.5, learning_rate = 0.5, n_iter = 5000, seed = 0)
    rng = MersenneTwister(seed)
    prototypes = rand(rng, grid, grid, size(data, 2)) .* 2 .- 1  # random start
    for t in 0:n_iter-1
        x = data[rand(rng, 1:size(data, 1)), :]  # one sample at random
        w = winner(prototypes, x)
        decay = 1 / (1 + t / (n_iter / 2))  # the learning rate and the neighborhood shrink with time
        s = sigma * decay
        for r in 1:grid, c in 1:grid  # pull the winner and its neighbors on the grid toward x
            h = exp(-((r - w[1])^2 + (c - w[2])^2) / (2s^2))
            prototypes[r, c, :] .+= learning_rate * decay * h .* (x .- prototypes[r, c, :])
        end
    end
    return prototypes
end

grid = 6
prototypes = train_som(features_z, grid)  # grid x grid x features (z-scored)
node = [winner(prototypes, features_z[u, :]) for u in 1:size(features_z, 1)]  # (row, column) of each unit

n_clusters = 3
flat = permutedims(reshape(prototypes, grid * grid, :))  # features x nodes, for kmeans
km = argmin(k -> k.totalcost, [kmeans(flat, n_clusters; rng = MersenneTwister(s)) for s in 0:9])  # best of 10
node_cluster = reshape(assignments(km), grid, grid)  # cluster of each node
cluster = [node_cluster[r, c] for (r, c) in node]  # cluster of each unit

hits = zeros(Int, grid, grid)
for (r, c) in node
    hits[r, c] += 1  # number of units in each node
end
panels = [heatmap(prototypes[:, :, k] .* std(features[:, k]) .+ mean(features[:, k]), color = :viridis,
                  yflip = true, aspect_ratio = 1, title = feature_names[k], ticks = false) for k in 1:4]
p = heatmap(node_cluster, color = palette(:Set2_8)[1:n_clusters], yflip = true, aspect_ratio = 1,
            colorbar = false, ticks = false, title = "clusters (number of units per node)")
annotate!(p, [(c, r, text(hits[r, c], 9)) for r in 1:grid for c in 1:grid])
push!(panels, p)
fig = plot(panels..., layout = (1, 5), size = (2200, 400))
display(fig)

units_f[:cluster] = cluster
println("cluster  n_units  rate_Hz  burstiness  memory  narrow")
for k in 1:n_clusters
    sel = cluster .== k
    @printf("%7d  %7d  %7.2f  %10.2f  %6.2f  %6.2f\n", k, sum(sel), median(10 .^ features[sel, 1]),
            median(features[sel, 2]), median(features[sel, 3]), mean(features[sel, 4]))
end

# %% [markdown]
# The first four maps (the **component planes**) show the value of each feature across the grid:
# the SOM has organized the units along gradients of rate, burstiness and memory, and put the
# narrow units in one corner. The k-means clusters cut the grid into regions; the table gives the
# median features of each cluster.
#
# Here the clusters mostly follow the waveform class and the rate. Are they **cell types**? Not
# necessarily: k-means always returns the number of clusters we ask for, even when the data form a
# continuum. Check how stable the clusters are (other seeds, other `n_clusters`), and whether they
# have a meaning outside the features that made them (area, depth, responses to the stimuli).
#
# **Questions.**
#
# - Change `n_clusters`. Is there a number of clusters that comes back for several seeds?
# - Remove the waveform class from the features. Do the narrow units still group together?
# - Are the clusters spread over the four areas (`[sum((cluster .== k) .& (units_f[:area] .== a))
#   for k in 1:n_clusters, a in first.(area_colors)]`)?
