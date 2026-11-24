# %% [markdown]
# # 02 · Spike waveforms and cell types — **Julia / Judananas**
#
# In notebook 01 we reduced each neuron to a list of spike times. But the spike sorter knows more
# about each unit: the **shape of its spikes**, as seen by the electrodes of the probe. This shape,
# the **spike waveform**, depends on the neuron (its ion channels, its morphology) and on where it
# sits relative to the probe. It is the main clue we have, in extracellular recordings, about the
# **type of neuron** behind a unit.
#
# The cortex has two big families of neurons:
#
# - **pyramidal neurons** (about 80 % of the neurons): excitatory (they release glutamate), they
#   project to other areas, and their spikes are relatively **wide**;
# - **interneurons** (about 20 %): inhibitory (GABA), they act locally. The largest group, the
#   **parvalbumin (PV)** interneurons, fire very short spikes at high rates: the **fast-spiking**
#   neurons. Their fast potassium channels (Kv3) repolarize the membrane very quickly.
#
# This is the Julia version of `python/02-Waveforms_and_CellTypes.ipynb`: same steps, same text,
# same figures. We:
#
# 1. look at the mean spike waveform of each unit, and measure its shape with a few **features**
#    (trough-to-peak duration, peak-to-baseline duration, half-width, peak-to-trough ratio);
# 2. split the units into **narrow** and **wide** waveforms with two thresholds, and (advanced) with
#    a **Gaussian mixture model**, and compare with the FS / RS labels stored in the file (used in
#    notebooks 01 and 03);
# 3. check that the two classes also **fire differently** (rate, autocorrelogram), with Judananas;
# 4. see where they are along the probe, and how they **respond to the stimuli** of the task;
# 5. meet a third category, the **positive (axonal) spikes**, and see the **limits** of a
#    classification based on the waveform alone.
#
# **Dataset.** Session `156131_20191112-probe0`, the same as in notebooks 01 and 03: a mouse doing
# an aversive conditioning task (sounds, optogenetic stimulations, air puffs) while a Neuropixels
# probe records in the medial prefrontal cortex (ACA, PL, ILA), the orbital cortex (ORBm), and in
# the areas below them (DP: dorsal peduncular area; ACB: nucleus accumbens; AON, OLF: olfactory
# areas).
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

# Judananas: time series and spike trains; HDF5: reads the NWB file (an HDF5 file);
# Plots: figures; Statistics: means, medians, quantiles...; Printf: formatted printing
using Judananas, HDF5
using Plots, Statistics, Printf
gr()
default(left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)  # room for the axis labels
TAB = palette(:tab10)  # the same 10 colors as matplotlib: blue, orange, green, red, purple, brown...

# Full path to the NWB file you downloaded. Change it to where you saved it.
NWB = "/path/to/your/data/folder/sub-156131_ses-156131-20191112-probe0_ecephys+ogen.nwb"

isfile(NWB) || error("No NWB file at $NWB: set NWB to the downloaded file")

# %% [markdown]
# ## 1. Load the units and their waveforms
#
# As in notebook 01, `load_file` gives the units as a `TsGroup` and the trials as an
# `IntervalSet`, and we keep the trials of blocks 1–4 (sound, sound + opto, air puff; see the
# trial structure in notebook 01, section 1).
# We do not remove the saturation periods here (notebook 01, section 1.1): they fall outside the
# trials and only touch a few windows between them, so their effect is negligible.
# This time, look at the **metadata columns** of the
# units: Judananas reads every
# column of the NWB units table, including several **waveform features** (`Peak_to_Trough`,
# `PeaktoBaseline`, `HW`, `PTR`, ...), one value per unit. We will compute them ourselves in
# section 2, and compare.

# %%
data = load_file(NWB)  # open the file and list its content
units = data["units"]  # all sorted units (TsGroup)
trials = data["trials"]  # task trials (IntervalSet)
trials = trials[trials[:Block] .<= 4]  # blocks 1-4, as in notebook 01
println("number of units: ", length(units))
println("metadata columns: ", sort(collect(keys(metadata(units)))))

# %% [markdown]
# **The mean waveforms.** The units table also has a column `waveform_means`: for each unit, the
# **average of all its spikes**, on the electrode where the spike is the largest (the "main"
# electrode). Averaging thousands of spikes removes the noise and the spikes of the other neurons,
# and leaves the typical shape of the spike of this unit: its **template**. Judananas does not read
# this column (it is not one value per unit, but a whole vector), so we read it with `HDF5`.
#
# Each waveform was cut from 82 samples of the recording (2.7 ms at 30 kHz, the sampling rate of
# Neuropixels), and stored **upsampled 1000 times** (82,000 points, i.e. one point every
# 0.033 µs!). We do not need that much: we keep one point in 100, i.e. 10 points per sample
# (one point every 3.3 µs). The values are in microvolts (µV). HDF5.jl gives the dimensions in
# the reverse order of h5py (Julia stores arrays by columns): we get **one column per unit**.
#
# We also prepare, as in notebook 01, the ISI violation rate (from `quality`) and the FS / RS
# labels. The brain area of each unit (`location`) and its depth (`DV`, dorso-ventral coordinate of
# its electrode, in mm) were already read by Judananas.

# %%
sampling_rate = 30_000  # Hz, sampling rate of the recording
upsampling = 10  # points per sample that we keep (the file has 1000)

waveforms = h5open(NWB) do f
    @assert read(f["units/id"]) == ids(units)  # same units, in the same order
    f["units/waveform_means"][1:1000÷upsampling:end, :]  # (points x units), in µV
end

dt = 1000 / (sampling_rate * upsampling)  # time between two points, in ms
t_wf = (0:size(waveforms, 1)-1) .* dt  # time axis of the waveforms, in ms
@printf("waveforms: %s (points x units), one point every %.1f µs\n", size(waveforms), dt * 1000)

units[:isi_fp] = [q[2] for q in units[:quality]]  # ISI violation false-positive rate
units[:file_label] = [f == 1 ? "FS" : r == 1 ? "RS" : "none" for (f, r) in zip(units[:FS], units[:RS])]

# %% [markdown]
# **Quality criteria.** We keep the units with an ISI violation rate below 0.1 and a presence
# ratio above 0, as in notebook 01 (section 2). Notebook 01 also required "a plausible waveform"
# (an FS or RS label): here we do **not** use the labels of the file, because we want to build
# our own. We keep the units of **all the brain areas** of the probe, to have more units in each
# class. `area` is the location without the layer (e.g. `PL6a` → `PL`).

# %%
units[:presence_ratio] = vec(mean(values(count(units, tot_length(time_support(units)) / 100)) .> 0,
                                  dims = 1))
units[:area] = [match(r"^([A-Za-z]+?)(?:\d|$)", l).captures[1] for l in units[:location]]  # remove the layer

is_good = (units[:isi_fp] .< 0.1) .& (units[:presence_ratio] .> 0)
good = units[is_good]
wf = waveforms[:, is_good]  # the waveforms of the good units, in the same order
println("good units: ", length(good), " out of ", length(units))
value_counts(good, :area)  # how many units per brain area

# %% [markdown]
# ## 2. The extracellular spike waveform
#
# ### 2.1 What does the probe see?
#
# During a spike, sodium ions rush **into** the neuron, near the soma. Seen from outside, the
# current leaves the extracellular space there: the electrode records a **negative** deflection,
# the **trough**. Then potassium ions flow out to repolarize the membrane: the potential goes back
# up, often above 0, the **peak**. So the extracellular spike is roughly upside down compared with
# the intracellular action potential (more precisely, it looks like its time derivative), and it is
# much smaller: tens to hundreds of µV, instead of ~100 mV inside the cell.
#
# Let's look at two waveforms: one unit with a narrow spike, recorded in the prelimbic area
# (PL, layer 5), and one with a wide spike, recorded in the infralimbic area (ILA, layer 6a): the
# bursty unit B of notebook 01. You will meet both again in notebook 03. `indexof(good, id)` gives the position of a
# unit in `good`, i.e. its column in `wf`.

# %%
examples = ["unit N" => 171, "unit W" => 58]  # name => unit id: a Narrow and a Wide spike
colors = Dict("unit N" => TAB[4], "unit W" => TAB[1])  # one color per unit in all figures
wave(i) = wf[:, indexof(good, i)]  # the waveform of unit i

p1 = plot(xlabel = "Time (ms)", ylabel = "Voltage (µV)", title = "Mean waveforms")
p2 = plot(xlabel = "Time (ms)", ylabel = "Voltage / trough depth", title = "Normalized waveforms")
for (name, i) in examples
    @printf("%s: id %d, %s, rate %.1f Hz\n", name, i, good[:location, id = i], good[:rate, id = i])
    plot!(p1, t_wf, wave(i), color = colors[name], lw = 2, label = "$name ($i)")
    # Normalized: divided by the depth of the trough, to compare the shapes
    plot!(p2, t_wf, wave(i) ./ -minimum(wave(i)), color = colors[name], lw = 2, label = "$name ($i)")
end
for p in (p1, p2)
    hline!(p, [0], color = :gray, lw = 0.5, label = "")
end
fig = plot(p1, p2, layout = (1, 2), size = (1200, 350), legend = :topleft)

# %% [markdown]
# Both units have the typical shape: a negative trough, then a positive peak. But unit N comes
# back up **much faster** than unit W, and its peak is higher relative to its trough. These are
# the two signatures of a fast-spiking interneuron.
#
# Now all the good units at once, normalized by their trough. The waveforms are aligned on the
# trough (around 1.35 ms) by the spike sorter.

# %%
normalized = wf ./ -minimum(wf, dims = 1)  # each waveform (column) divided by its trough depth

fig = plot(t_wf, normalized, color = :black, lw = 0.5, alpha = 0.1, label = "", size = (700, 400),
           xlabel = "Time (ms)", ylabel = "Voltage / trough depth",
           title = "Normalized waveforms of the $(length(good)) good units")

# %% [markdown]
# Most waveforms look alike, but the time of the peak after the trough varies a lot, and a few
# waveforms have a large positive peak **before** the trough. We come back to them in section 6.
#
# ### 2.2 Waveform features
#
# To compare hundreds of units, we summarize each waveform with a few numbers, the **features**:
#
# - **trough-to-peak duration**: the time from the trough to the following peak (the maximum
#   after the trough). Short for fast-spiking neurons;
# - **peak-to-baseline duration**: the time from the trough (the main, negative peak of the spike)
#   until the waveform comes back to the baseline (0 µV: the waveforms are centered on 0 before
#   the spike). It measures how fast the neuron **repolarizes**. (The file calls the trough the
#   "peak", hence the name);
# - **half-width**: the width of the trough at half of its depth. It measures how fast the
#   neuron depolarizes *and* repolarizes;
# - **peak-to-trough ratio**: the height of the peak divided by the trough (negative, since the
#   trough is negative): how big the repolarization is compared with the spike;
# - **amplitude**: from the trough to the peak, in µV. It depends mostly on the distance between
#   the neuron and the electrode, not on the type of the neuron.
#
# The durations are rounded to 0.1 µs, so that a unit exactly on a threshold (section 3) is
# classified in the same way on every computer.

# %%
"""Features of one waveform w (a vector, in µV), with dt the time between two points (ms)."""
function waveform_features(w, dt)
    trough = argmin(w)  # index of the trough
    peak = trough - 1 + argmax(w[trough:end])  # index of the highest point after the trough
    back = findfirst(>=(0), w[trough:end])  # first point after the trough at or above the baseline
    baseline = back === nothing ? length(w) : trough - 1 + back  # first return to the baseline
    half = w[trough] / 2  # half of the trough depth
    left, right = trough, trough  # walk left and right from the trough while below half depth
    while left > 1 && w[left-1] <= half
        left -= 1
    end
    while right < length(w) && w[right+1] <= half
        right += 1
    end
    return (trough_to_peak = round((peak - trough) * dt, digits = 4),  # ms
            peak_to_baseline = round((baseline - trough) * dt, digits = 4),  # ms
            half_width = round((right - left) * dt, digits = 4),  # ms
            peak_trough_ratio = w[peak] / w[trough],  # no unit, negative
            amplitude = w[peak] - w[trough],  # µV
            trough = trough, peak = peak, baseline = baseline,  # indices, for figures
            left = left, right = right)
end

"""A horizontal double arrow from x1 to x2 at height y, with a label on its right."""
function double_arrow!(p, x1, x2, y, label)
    plot!(p, [x1, x2], [y, y], arrow = arrow(:both), color = :black, label = "")
    annotate!(p, x2 + 0.05, y, text(label, 9, :left))
end

# The features of unit W, drawn on its waveform
i = 58
w, ft = wave(i), waveform_features(wave(i), dt)
above = w[ft.peak] + 0.1 * ft.amplitude  # height of the trough-to-peak arrow, above the peak
below = w[ft.trough] / 4  # height of the peak-to-baseline arrow, between the trough and 0

fig = plot(t_wf, w, color = colors["unit W"], lw = 2, label = "", size = (900, 450),
           ylims = (minimum(w) - 0.05 * ft.amplitude, above + 0.15 * ft.amplitude),
           xlabel = "Time (ms)", ylabel = "Voltage (µV)",
           title = @sprintf("unit W (%d): peak-to-trough ratio %.2f, amplitude %.0f µV",
                            i, ft.peak_trough_ratio, ft.amplitude))
scatter!(fig, t_wf[[ft.trough, ft.peak, ft.baseline]], w[[ft.trough, ft.peak, ft.baseline]], color = :black,
         label = "")
double_arrow!(fig, t_wf[ft.trough], t_wf[ft.peak], above, @sprintf("trough-to-peak: %.2f ms", ft.trough_to_peak))
double_arrow!(fig, t_wf[ft.trough], t_wf[ft.baseline], below,
              @sprintf("peak-to-baseline: %.2f ms", ft.peak_to_baseline))
double_arrow!(fig, t_wf[ft.left], t_wf[ft.right], w[ft.trough] / 2, @sprintf("half-width: %.2f ms", ft.half_width))
hline!(fig, [0], color = :gray, lw = 0.5, label = "")

# %% [markdown]
# We compute the features of all the good units, and store them as metadata of the `TsGroup`,
# next to the features of the file.

# %%
feature_names = (:trough_to_peak, :peak_to_baseline, :half_width, :peak_trough_ratio, :amplitude)
features = [waveform_features(w, dt) for w in eachcol(wf)]  # one NamedTuple per unit
for k in feature_names
    good[k] = [getfield(f, k) for f in features]  # one metadata column per feature
    q = quantile(good[k], [0, 0.25, 0.5, 0.75, 1])
    @printf("%-18s min %7.2f   25%% %7.2f   median %7.2f   75%% %7.2f   max %7.2f\n", k, q...)
end

# %% [markdown]
# **Do we agree with the file?** The NWB file has its own features (read by Judananas):
# `Peak_to_Trough` (ms), `PeaktoBaseline` (ms), `HW` (half-width, ms) and `PTR` (peak-to-trough
# ratio). We compare them with ours, unit by unit.

# %%
panels = []
for (ours, theirs) in [(:trough_to_peak, :Peak_to_Trough), (:peak_to_baseline, :PeaktoBaseline),
                       (:half_width, :HW), (:peak_trough_ratio, :PTR)]
    lims = extrema([good[theirs]; good[ours]])
    p = scatter(good[theirs], good[ours], color = :black, markersize = 2, markerstrokewidth = 0, label = "",
                xlabel = "$theirs (file)", ylabel = "$ours (ours)")
    plot!(p, collect(lims), collect(lims), color = TAB[3], lw = 1, label = "same value")
    push!(panels, p)
end
fig = plot(panels..., layout = (1, 4), size = (1800, 400), legend = :topleft)
display(fig)

disagree = ids(good)[abs.(good[:trough_to_peak] .- good[:Peak_to_Trough]) .> 0.05]  # more than 0.05 ms apart
println("units with a trough-to-peak differing by more than 0.05 ms: ", disagree)

# %% [markdown]
# For almost all the units, our features are the same as the file's (the points are on the
# diagonal; our peak-to-baseline is a little longer, by ~0.02 ms, because the file uses a slightly
# different definition of the baseline). A few units disagree completely: the two methods do not
# find the same "peak". These are unusual waveforms, and we look at them in section 6. Remember:
# **a feature is only as good as its definition**, and simple definitions fail on unusual shapes.
#
# ## 3. Narrow and wide waveforms
#
# ### 3.1 A dual threshold on the trough-to-peak duration
#
# If there are two families of neurons with different spike durations, the histogram of the
# trough-to-peak duration should have **two modes**.

# %%
low_th, high_th = 0.380, 0.430  # ms: narrow below low_th, wide above high_th

fig = histogram(good[:trough_to_peak], bins = 0:0.025:1.5, color = :gray, linecolor = :gray, label = "",
                xlabel = "Trough-to-peak duration (ms)", ylabel = "Number of units",
                title = "Trough-to-peak duration of the good units", size = (800, 400))
vspan!(fig, [low_th, high_th], color = TAB[2], alpha = 0.3, lw = 0, label = "$(low_th)–$(high_th) ms: unclassified")

# %% [markdown]
# Two groups indeed: a small one of **narrow** waveforms (about 0.2–0.35 ms) and a large one of
# **wide** waveforms (around 0.7 ms), with a gap around 0.4 ms. Instead of a single threshold in
# the gap (0.4 ms in [Siegle et al., 2021](https://doi.org/10.1038/s41586-020-03171-x)), we use
# **two thresholds**:
#
# - trough-to-peak < 0.380 ms: **narrow** waveform (putative fast-spiking interneuron);
# - trough-to-peak > 0.430 ms: **wide** waveform (putative pyramidal neuron, or another type of
#   interneuron, see section 6);
# - in between: **unclassified**. A unit in the gap could belong to either class, and a wrong
#   label is worse than no label.
#
# We keep one `TsGroup` per class in a dictionary.

# %%
good[:wf_class] = [d < low_th ? "narrow" : d > high_th ? "wide" : "unclassified" for d in good[:trough_to_peak]]
by_class = Dict(c => good[good[:wf_class] .== c] for c in ["narrow", "wide", "unclassified"])  # class => TsGroup
classes = ["narrow", "wide"]  # the two classes that we compare in sections 4 and 5
class_colors = Dict("narrow" => TAB[4], "wide" => TAB[1], "unclassified" => TAB[2])
println("unclassified units: ", ids(by_class["unclassified"]))
for c in ["narrow", "wide", "unclassified"]
    @printf("%s: %d units (%.0f %%)\n", c, length(by_class[c]), 100 * length(by_class[c]) / length(good))
end

# %% [markdown]
# About 15 % of narrow waveforms: close to the proportion of interneurons in the cortex
# (~20 %), of which only part are fast-spiking. **Compare with the labels of the file**, used in
# notebooks 01 and 03: `crosstab` (a small function, like `pd.crosstab` in Python) counts the units
# for each pair of labels.

# %%
"""Print the number of elements for each pair of values of a (rows) and b (columns)."""
function crosstab(a, b)
    ra, cb = sort(unique(a)), sort(unique(b))
    println(rpad("", 14), join(lpad.(string.(cb), 8)))
    for x in ra
        println(rpad(string(x), 14), join(lpad.([count((a .== x) .& (b .== y)) for y in cb], 8)))
    end
end

crosstab(good[:wf_class], good[:file_label])

# %% [markdown]
# Our classes match the FS / RS labels for almost all units:
#
# - the file also leaves a few units of the gap unclassified (**"none"**: their trough-to-peak
#   duration is between 0.4 and 0.44 ms). They are **not** units without a plausible waveform, as
#   we said in notebook 01;
# - the few disagreements are mostly the units of section 2.2 whose features disagree (and unit
#   56, just above our low threshold, at 0.383 ms).
#
# ### 3.2 Two features are better than one
#
# The trough-to-peak duration measures the whole repolarization, until the peak; the
# peak-to-baseline duration only its first part, until the waveform crosses 0. Plotted against each
# other, the two classes separate well. We also show the median waveform of each class (the median
# is less sensitive than the mean to a few unusual waveforms), with the 25th to 75th percentiles.

# %%
p1 = plot(xlabel = "Trough-to-peak duration (ms)", ylabel = "Peak-to-baseline duration (ms)",
          title = "Two waveform features")
p2 = plot(xlabel = "Time (ms)", ylabel = "Voltage / trough depth", title = "Median waveform (25–75 %) of each class")
for c in ["narrow", "unclassified", "wide"]
    g = by_class[c]
    scatter!(p1, g[:trough_to_peak], g[:peak_to_baseline], color = class_colors[c], markersize = 3,
             markerstrokewidth = 0, label = "$c (n=$(length(g)))")
end
for c in classes
    q = mapslices(x -> quantile(x, [0.25, 0.5, 0.75]), normalized[:, good[:wf_class] .== c], dims = 2)
    plot!(p2, t_wf, q[:, 2], ribbon = (q[:, 2] .- q[:, 1], q[:, 3] .- q[:, 2]), fillalpha = 0.2,
          color = class_colors[c], lw = 2, label = c)
end
vline!(p1, [low_th, high_th], color = :black, ls = :dash, lw = 0.8, label = "")
fig = plot(p1, p2, layout = (1, 2), size = (1300, 450))

# %% [markdown]
# Narrow waveforms have both a short trough-to-peak and a short peak-to-baseline duration, and a
# larger peak after the trough. In the literature, many more features are used: the half-width,
# the slopes of the repolarization and of the recovery, and features of the spike on the
# **neighbouring electrodes** (how far the spike spreads, how fast it travels along the probe).
# With a full spike sorting output, SpikeInterface (Python) computes them with the
# `template_metrics` extension of a `SortingAnalyzer` (see the spike sorting notebooks of this
# school).
#
# **Question.** Change the thresholds (e.g. a single threshold at 0.4 ms, or 0.35 and 0.45 ms).
# How many units change class?

# %% [markdown]
# ### 3.3 Advanced: a Gaussian mixture model
#
# The thresholds are chosen by eye on the histogram. A **Gaussian mixture model** (GMM) finds the
# two classes from the data: it describes the cloud of points as the sum of two Gaussian
# "blobs" (each with its own center, spread and orientation), fitted to the data by an iterative
# algorithm (expectation–maximization). Each unit then gets a **probability** to belong to each
# blob: units in the gap get probabilities in between, so the GMM also tells us **how sure** the
# classification is.
#
# We fit a GMM with **two clusters** to the two features of section 3.2. We first **standardize**
# each feature (subtract its mean, divide by its standard deviation), so that both features have
# the same weight.
#
# The Python notebook uses scikit-learn. Here we write the **expectation–maximization** (EM)
# algorithm ourselves, for two clusters and two features; it alternates two steps:
#
# 1. **M step**: given the probability of each unit to belong to each cluster (its
#    "responsibility"), compute the weight, the center and the covariance matrix of each cluster;
# 2. **E step**: given the clusters, compute the new probabilities of each unit (Bayes' rule).
#
# We start from the classes of the dual threshold, and repeat until the probabilities stop changing.

# %%
"""Density of a 2D Gaussian with center μ (2 values) and covariance Σ (2 x 2), at each row of Z."""
function gaussian2d(Z, μ, Σ)
    a, b, d = Σ[1, 1], Σ[1, 2], Σ[2, 2]
    det = a * d - b^2
    x, y = Z[:, 1] .- μ[1], Z[:, 2] .- μ[2]
    quad = (d .* x .^ 2 .- 2b .* x .* y .+ a .* y .^ 2) ./ det  # (z - μ)' Σ⁻¹ (z - μ)
    return exp.(-quad ./ 2) ./ (2π * sqrt(det))
end

"""GMM with 2 clusters on the rows of Z (units x 2 features), by expectation-maximization,
starting from the probabilities p_start of each unit to be in cluster 1."""
function fit_gmm2(Z, p_start; max_iter = 1000, tol = 1e-8)
    resp = [p_start 1 .- p_start]  # responsibilities: units x clusters
    weights, centers, covs = zeros(2), [zeros(2) for _ in 1:2], [zeros(2, 2) for _ in 1:2]
    for _ in 1:max_iter
        for k in 1:2  # M step
            r = resp[:, k]
            weights[k] = mean(r)
            centers[k] = vec(sum(r .* Z, dims = 1) ./ sum(r))
            D = Z .- centers[k]'
            covs[k] = (D .* r)' * D ./ sum(r) + 1e-6 * [1 0; 0 1]  # + a tiny regularization, as scikit-learn
        end
        dens = hcat([weights[k] .* gaussian2d(Z, centers[k], covs[k]) for k in 1:2]...)  # E step
        new_resp = dens ./ sum(dens, dims = 2)
        converged = maximum(abs.(new_resp .- resp)) < tol
        resp = new_resp
        converged && break
    end
    return (weights = weights, centers = centers, covs = covs, resp = resp)
end

"""Probability of each row of Z to be in cluster 1 of the fitted GMM."""
function gmm_proba(gmm, Z)
    dens = hcat([gmm.weights[k] .* gaussian2d(Z, gmm.centers[k], gmm.covs[k]) for k in 1:2]...)
    return dens[:, 1] ./ sum(dens, dims = 2)[:, 1]
end

X = [good[:trough_to_peak] good[:peak_to_baseline]]  # the 2 durations: units x 2
μX, σX = mean(X, dims = 1), std(X, dims = 1, corrected = false)
Z = (X .- μX) ./ σX  # standardized features

p_start = [c == "narrow" ? 1.0 : c == "wide" ? 0.0 : 0.5 for c in good[:wf_class]]  # cluster 1 = narrow
gmm = fit_gmm2(Z, p_start)
good[:p_narrow] = gmm.resp[:, 1]  # probability of each unit to be narrow
good[:gmm_class] = ifelse.(good[:p_narrow] .> 0.5, "narrow", "wide")

println("units with a probability between 0.1 and 0.9: ", ids(good)[0.1 .< good[:p_narrow] .< 0.9])
crosstab(good[:wf_class], good[:gmm_class])  # dual threshold (rows) vs GMM (columns)

# %% [markdown]
# We draw each unit with the color of its probability to be narrow, and the **boundary** of the GMM
# (probability 0.5), computed on a grid of points.

# %%
ttp_grid = range(0.15, 1.5, length = 300)
ptb_grid = range(0.05, 0.9, length = 300)
grid_points = [[t p] for p in ptb_grid, t in ttp_grid]  # one row [ttp, ptb] per grid point
p_grid = reshape(gmm_proba(gmm, (vcat(vec(grid_points)...) .- μX) ./ σX), size(grid_points))  # same transform as the units

fig = scatter(good[:trough_to_peak], good[:peak_to_baseline], zcolor = good[:p_narrow], color = :coolwarm,
              clims = (0, 1), markersize = 4, markerstrokewidth = 0, label = "", size = (850, 550),
              xlabel = "Trough-to-peak duration (ms)", ylabel = "Peak-to-baseline duration (ms)",
              colorbar_title = "Probability to be narrow", title = "GMM with two clusters (black line: probability 0.5)")
contour!(fig, ttp_grid, ptb_grid, p_grid, levels = [0.5], color = :black, lw = 1, colorbar_entry = false)
vline!(fig, [low_th, high_th], color = :black, ls = :dash, lw = 0.8, label = "")

# %% [markdown]
# The GMM agrees with the dual threshold for (almost) all the narrow and wide units, and splits the
# unclassified units of the gap between the two classes. Its boundary is not vertical: a unit with a
# trough-to-peak duration in the gap is called narrow if its peak-to-baseline duration is short.
# The units with a probability far from 0 and 1 are the ones to look at by hand.
#
# A GMM is not magic: it assumes that each class is a Gaussian blob, and it is sensitive to
# outliers, to the way we transform the features, and to its starting point (scikit-learn starts
# it 10 times from random points and keeps the best fit).
#
# **Questions.**
#
# - Fit the GMM on the **logarithm** of the durations (`log.(X)`): it is often recommended for
#   durations, which have long tails to the right. Do the classes change? Why?
# - Start the EM from another guess (e.g. `p_start = rand(length(good))`). Do you get the same
#   clusters?

# %% [markdown]
# ## 4. Do the two classes fire differently?
#
# The waveform is a property of the spike. If the narrow waveforms really come from fast-spiking
# interneurons, these units should also **fire** like interneurons: at higher rates, and with short
# refractory periods. These are measured on the spike trains only, independently of the waveform:
# a good test of our classification. From now on we compare the narrow and the wide units of the
# dual threshold (section 3.1), and leave the unclassified units out.
#
# ### 4.1 Firing rates
#
# We plot the session rate of each unit against its trough-to-peak duration (log scale for the
# rate, as in notebook 01).

# %%
rate_bins = 10 .^ range(-2, 2, length = 33)  # from 0.01 Hz to 100 Hz, evenly spaced in log
p1 = plot(yscale = :log10, xlabel = "Trough-to-peak duration (ms)", ylabel = "Session rate (Hz)")
p2 = plot(xscale = :log10, xlabel = "Session rate (Hz)", ylabel = "Fraction of the units of the class",
          title = "Rate of each class")
for c in classes
    g = by_class[c]
    scatter!(p1, g[:trough_to_peak], g[:rate], color = class_colors[c], markersize = 3,
             markerstrokewidth = 0, label = c)
    # weights: each unit counts 1 / (number of units of its class), to compare classes of different sizes
    stephist!(p2, g[:rate], bins = rate_bins, weights = fill(1 / length(g), length(g)), lw = 2,
              color = class_colors[c], label = @sprintf("%s: median %.1f Hz", c, median(g[:rate])))
end
vline!(p1, [low_th, high_th], color = :black, ls = :dash, lw = 0.8, label = "")
fig = plot(p1, p2, layout = grid(1, 2, widths = [0.56, 0.44]), size = (1300, 450))

# %% [markdown]
# Narrow units fire on average **about 3 times faster** than wide units, as expected for
# fast-spiking interneurons. But the two distributions overlap a lot: some wide units fire at
# 20 Hz, and some narrow units at about 1 Hz. **The rate alone would be a poor classifier**.
#
# ### 4.2 Autocorrelograms
#
# The autocorrelogram (ACG, see notebook 03) is the rate of a neuron at a time $\tau$ after one of
# its spikes. With `norm = true`, Judananas divides it by the mean rate of the unit, so that units
# with different rates can be compared: 1 means "as by chance". We compute the ACGs of the good
# units with 1 ms bins, up to ±50 ms (`acg.values` is a matrix lags × units), and show the two
# example units.

# %%
acg = compute_autocorrelogram(good; binsize = 0.001, windowsize = 0.05, norm = true)

panels = []
for (name, i) in examples
    p = bar(acg.lags .* 1000, acg.values[:, indexof(good, i)], bar_width = 1, color = colors[name],
            linecolor = colors[name], label = "", xlabel = "Lag (ms)",
            ylabel = name == "unit N" ? "Normalized ACG" : "",
            title = @sprintf("%s (%d, %s waveform), %.1f Hz", name, i, good[:wf_class, id = i], good[:rate, id = i]))
    hline!(p, [1], color = :black, ls = :dash, label = "as by chance", legend = :bottomright)
    push!(panels, p)
end
fig = plot(panels..., layout = (1, 2), size = (1200, 350))

# %% [markdown]
# - Both units almost never fire twice within ~3–4 ms: the **refractory period**.
# - **Unit N** then rises quickly to a flat plateau, from ~10 ms on: no preferred interval between
#   its spikes, a typical fast-spiking interneuron (~40 spikes/s).
# - **Unit W** has a sharp **peak about 6 ms after each spike**, 8 times higher than by chance: it
#   often fires a second spike 5–10 ms after the first one, i.e. it fires in **bursts**, like many
#   pyramidal neurons (we saw its bursts in notebook 01).
#
# Notice that both ACGs stay **above 1** even at 50 ms: as we saw in notebook 01, the rate
# of a unit changes during the session. After a spike, the unit is more likely to be in one of its
# active periods, so it fires more than its *session* mean rate. To compare the **shapes** of the
# ACGs, we divide each normalized ACG by its mean between 30 and 50 ms (its "plateau").
#
# Now the ACGs of the narrow and wide units, one row per unit, sorted by class and then by
# trough-to-peak duration, with the median ACG of each class. We keep only the positive lags (the
# ACG is symmetric).

# %%
plateau = mean(acg.values[0.0295 .<= acg.lags .<= 0.0505, :], dims = 1)  # mean of each ACG between 30 and 50 ms
positive_lags = acg.lags .> 0.0005  # positive lags only
shape = acg.values[positive_lags, :] ./ plateau  # ACG divided by its plateau
lags_ms = acg.lags[positive_lags] .* 1000
two_classes = findall(in(classes), good[:wf_class])  # columns of the narrow and wide units
order = two_classes[sortperm(collect(zip(good[:wf_class][two_classes], good[:trough_to_peak][two_classes])))]  # narrow first

"""Median of each row of M, ignoring the NaN and Inf values (units without spikes in the plateau)."""
row_median(M) = [median(filter(isfinite, r)) for r in eachrow(M)]

p1 = heatmap(lags_ms, 1:length(order), shape[:, order]', color = :viridis, clims = (0, 2), yflip = true,
             xlabel = "Lag (ms)", ylabel = "Unit (narrow on top, then wide)", title = "ACG / plateau")
hline!(p1, [length(by_class["narrow"]) + 0.5], color = :white, lw = 2, label = "")
p2 = plot(xlabel = "Lag (ms)", ylabel = "ACG / plateau", title = "Median ACG of each class")
for c in classes
    plot!(p2, lags_ms, row_median(shape[:, good[:wf_class] .== c]), color = class_colors[c], lw = 2, label = c)
end
hline!(p2, [1], color = :black, ls = :dash, label = "plateau")
fig = plot(p1, p2, layout = grid(1, 2, widths = [0.45, 0.55]), size = (1300, 500))

# %% [markdown]
# - Both classes have a refractory period of a few ms (dark at the left of the heatmap).
# - Most **wide** units have a **peak** between ~5 and 15 ms (yellow): they fire bursts.
# - Most **narrow** units rise more slowly to their plateau, with at most a small bump: they rarely
#   fire in bursts.
#
# These differences are clear on average, but individual ACGs vary a lot. Classifications such as
# the one of [CellExplorer](https://cellexplorer.org) (Petersen et al., 2021) use both the waveform
# and a fit of the ACG, to separate narrow interneurons, *wide* interneurons (wide waveforms, but no
# bursts) and pyramidal neurons.
#
# **Question.** Compute, for each unit, the fraction of ISIs shorter than 10 ms (a simple burst
# index, see notebook 01). Plot it against the trough-to-peak duration.

# %% [markdown]
# ## 5. Cell types along the probe and in the task
#
# ### 5.1 Where are the narrow units?
#
# Each unit has the depth of its main electrode (`DV`, read by Judananas). We plot the
# trough-to-peak duration of each unit against its depth, with the brain areas along the probe, and
# the fraction of narrow units (among the narrow and wide units) in each area, from top to bottom
# of the probe.

# %%
areas = unique(good[:area])
area_depth = Dict(a => mean(good[:DV][good[:area] .== a]) for a in areas)  # mean depth of the units of each area
sort!(areas, by = a -> area_depth[a])  # from the bottom to the top of the probe

p1 = plot(xlabel = "Trough-to-peak duration (ms)", ylabel = "Dorso-ventral coordinate (mm)",
          title = "Units along the probe", legend = :bottomleft)
for c in ["narrow", "unclassified", "wide"]
    scatter!(p1, by_class[c][:trough_to_peak], by_class[c][:DV], color = class_colors[c], markersize = 3,
             markerstrokewidth = 0, label = c)
end
annotate!(p1, [(1.45, area_depth[a], text(a, 8, :right)) for a in areas])  # name of each area, at its mean depth
vline!(p1, [low_th, high_th], color = :black, ls = :dash, lw = 0.8, label = "")

n_narrow = [count((good[:area] .== a) .& (good[:wf_class] .== "narrow")) for a in areas]
n_total = [count((good[:area] .== a) .& in.(good[:wf_class], Ref(classes))) for a in areas]  # narrow + wide
fraction = n_narrow ./ n_total  # fraction of narrow units in each area
p2 = bar(1:length(areas), fraction, orientation = :h, color = TAB[4], linecolor = TAB[4], label = "",
         yticks = (1:length(areas), areas), xlims = (0, 1.2 * maximum(fraction)),
         xlabel = "Fraction of narrow units", title = "Narrow units per area")
annotate!(p2, [(fraction[k] + 0.005, k, text("$(n_narrow[k]) / $(n_total[k])", 8, :left)) for k in eachindex(areas)])
fig = plot(p1, p2, layout = grid(1, 2, widths = [0.56, 0.44]), size = (1300, 600))

# %% [markdown]
# Narrow units are found all along the probe, from 5 % of the units (ACAd) to 30 % (ILA). But
# with a few units per area, these fractions are **not reliable**: the spike sorter also finds more easily the neurons that fire a lot (like
# fast-spiking interneurons) or that have large spikes, so these fractions are not the true
# proportions of the cell types in each area.
#
# ### 5.2 Responses to the stimuli
#
# Do the two classes respond differently to the task? We take the three types of trials (as in
# notebook 01): **sound**, **sound + opto** and **air puff** (a blue noise followed by an air puff). For each unit and each type of trial, `compute_eventcorrelogram`
# gives the PSTH in one line (here 50 ms bins, from 1 s before to 1 s after the trial onset).
#
# To compare units with different rates, we **z-score** each PSTH with its baseline (the second
# before the onset): $z = (\text{PSTH} - \text{mean of the baseline}) / \text{SD of the baseline}$.
# A unit responds if its mean $z$ in the first 500 ms is above 2 (an increase) or below −2 (a
# decrease).

# %%
trial_type = [a == 1 ? "air puff" : (o != 0 ? "sound + opto" : "sound")
              for (a, o) in zip(trials[:airpuff], trials[:optogenetics])]
stimuli = ["sound", "sound + opto", "air puff"]

onsets = Dict(s => starts(trials)[trial_type .== s] for s in stimuli)  # trial type => onsets (s)
psths = Dict(s => compute_eventcorrelogram(good, onsets[s]; binsize = 0.05, windowsize = 1.0, norm = false)
             for s in stimuli)
lags = psths["sound"].lags  # the bin centers (s), the same for all trial types

zscored = Dict()  # trial type => z-scored PSTHs (matrix: one row per bin, one column per unit)
for s in stimuli
    baseline = psths[s].values[lags .< 0, :]  # bins before the onset
    sd = std(baseline, dims = 1)
    sd[sd .== 0] .= NaN  # units without any spike in the baseline
    zscored[s] = (psths[s].values .- mean(baseline, dims = 1)) ./ sd
    println(s, ": ", length(onsets[s]), " trials")
end

"""Mean of each row of M, ignoring the NaN values."""
row_mean(M) = [mean(filter(isfinite, r)) for r in eachrow(M)]

tops, bottoms = [], []
for s in stimuli
    z = zscored[s]
    response = vec(mean(z[-1e-9 .<= lags .<= 0.5 + 1e-9, :], dims = 1))  # mean z in the first 500 ms, per unit
    # Heatmap: one row per unit, narrow units on top, each class sorted by response
    rows = Int[]
    for c in classes
        cols = findall(good[:wf_class] .== c)
        append!(rows, cols[sortperm(response[cols], rev = true, by = x -> isnan(x) ? -Inf : x)])
    end
    p = heatmap(lags, 1:length(rows), z[:, rows]', color = cgrad(:RdBu, rev = true), clims = (-5, 5),
                yflip = true, colorbar = false, title = s, ylabel = s == stimuli[1] ? "Unit (narrow on top)" : "")
    hline!(p, [length(by_class["narrow"]) + 0.5], color = :black, lw = 1.5, label = "")
    push!(tops, p)
    p = plot(xlabel = "Time from trial onset (s)", ylabel = s == stimuli[1] ? "Mean z-score" : "",
             legend = :topleft, legendfontsize = 7)
    for c in classes
        cols = good[:wf_class] .== c
        r = filter(isfinite, response[cols])
        plot!(p, lags, row_mean(z[:, cols]), color = class_colors[c], lw = 2,
              label = @sprintf("%s: %.0f %% up, %.0f %% down", c, 100 * mean(r .> 2), 100 * mean(r .< -2)))
    end
    vline!(p, [0], color = :gray, lw = 0.8, label = "")
    push!(bottoms, p)
end
fig = plot(tops..., bottoms..., layout = grid(2, 3, heights = [2 / 3, 1 / 3]), size = (1600, 800), link = :x,
           left_margin = 10Plots.mm)

# %% [markdown]
# In this session, the narrow units respond **more often** than the wide units (about 70 % against
# 30 %), and with an increase of their rate, for the three types of trials. A few wide units
# **decrease** their rate after the onset (blue rows at the bottom), which we do not see in the
# narrow units. In the sound + opto and air puff trials, a second response comes about 0.8 s after
# the onset: the optogenetic stimulation, or the air puff, which follow the sound.
#
# This is a common observation: interneurons receive many inputs and respond reliably, while each
# pyramidal neuron responds to fewer stimuli.
#
# But be careful before concluding: our test is **not fair between the two classes**. A unit with
# a high rate has many spikes in each bin, so its PSTH is less noisy, and the same relative change
# of rate gives a larger $z$. Narrow units fire faster (section 4.1): they are easier to detect as
# "responsive".
#
# **Question.** Compare the narrow units with wide units of **similar rates** (e.g. only the units
# above 5 Hz). Does the difference remain?

# %% [markdown]
# ## 6. A third category, and the limits of the method
#
# ### 6.1 Positive spikes: probably axons
#
# Everything so far assumed that a spike is a negative trough followed by a positive peak: the
# shape recorded **near the soma** of a neuron. But some units have a **positive** spike: their
# largest deflection is a positive peak, often **before** a small trough. We saw some of them in
# section 2.1, and they are behind most of the disagreements with the file (section 2.2).
#
# Such waveforms are typical of spikes recorded near an **axon** (or a dendrite), far from the
# soma: there, the membrane current during the spike has the opposite sign, and the electrode sees
# first a positive deflection. Axonal spikes are also very short, and can travel along the probe
# from one electrode to the next. Their trough-to-peak duration does not tell anything about the
# type of neuron, so they form a **third category**, outside the narrow / wide classification.
#
# We call a waveform **positive** if its largest deflection is positive (its maximum is larger
# than the depth of its trough).

# %%
is_positive = vec(maximum(wf, dims = 1) .> -minimum(wf, dims = 1))  # largest deflection is positive
good[:cell_class] = ifelse.(is_positive, "positive", good[:wf_class])  # 3 categories (+ unclassified)
positive = ids(good)[is_positive]
println("positive units: ", positive)
crosstab(good[:cell_class], good[:file_label])

# %%
n_col = min(5, length(positive))  # at most 5 panels per row
n_row = ceil(Int, length(positive) / n_col)
panels = []
for (k, i) in enumerate(positive)
    wi, fi = wave(i), waveform_features(wave(i), dt)
    c = good[:wf_class, id = i]
    p = plot(t_wf, wi, color = :black, label = "", titlefontsize = 9,
             xlabel = k > length(positive) - n_col ? "Time (ms)" : "", ylabel = k % n_col == 1 ? "Voltage (µV)" : "",
             title = @sprintf("%d (%s)\nours: %.2f ms, %s\nfile: %.2f ms, %s", i, good[:location, id = i],
                              fi.trough_to_peak, c, good[:Peak_to_Trough, id = i], good[:file_label, id = i]))
    scatter!(p, t_wf[[fi.trough, fi.peak]], wi[[fi.trough, fi.peak]], color = class_colors[c], label = "")  # trough and peak of section 3
    hline!(p, [0], color = :gray, lw = 0.5, label = "")
    push!(panels, p)
end
append!(panels, [plot(framestyle = :none) for _ in 1:(n_row*n_col-length(positive))])  # empty panels
fig = plot(panels..., layout = (n_row, n_col), size = (320 * n_col, 330 * n_row), top_margin = 8Plots.mm,
           left_margin = 10Plots.mm)

# %% [markdown]
# Only two units of this session have a positive spike, but they show the problem well: a large,
# short positive peak before the trough. Our method measures the time from the trough to the slow
# recovery after it: both end up in the wide class. The file measures, for unit 299, the time
# between the positive peak and the trough (short: FS), and for unit 205 the same duration as ours
# (RS). **Neither is right**: these units should not be classified as narrow or wide at all. In a
# real analysis, they are either excluded, or studied with more features (the shape on the
# neighbouring electrodes: an axonal spike travels along the probe). Once they are set apart, the
# wide class has no unit labeled FS by the file any more.
#
# Three other units of section 2.2 (27, 139 and 231) are not positive, but have a narrow spike
# whose peak decays very slowly back to 0: we take the sharp peak just after the trough (narrow),
# the file a point of this slow tail, more than 0.5 ms later (RS). Look at them!
#
# ### 6.2 What a waveform can and cannot tell
#
# - **Narrow ≠ PV interneuron, for sure.** Most narrow units in the mouse neocortex are PV
#   interneurons, but some pyramidal neurons have narrow spikes (e.g. in the motor cortex of
#   primates), and the other interneurons (somatostatin, VIP, ...) mostly have **wide** spikes:
#   they are in our wide class, mixed with the pyramidal neurons.
# - **The waveform depends on the position** of the electrode relative to the neuron: the same
#   neuron gives different shapes on different electrodes (larger and sharper near the soma, positive
#   near the axon). A feature measured on one electrode is a property of the pair neuron–electrode.
# - **Spike sorting matters.** The template is an average: if a unit mixes two neurons, its
#   template is a mix of two shapes.
# - **The thresholds are calibrated for the neocortex.** The rule "narrow = fast-spiking
#   interneuron, wide = pyramidal neuron" relies on the cell types of the cortex, and does not
#   transfer as is to other regions:
#   - in the **striatum**, most neurons (the medium spiny neurons) are inhibitory **and** have wide
#     waveforms; narrow waveforms come from fast-spiking interneurons, and the cholinergic
#     interneurons have wide waveforms but fire tonically;
#   - in the **thalamus**, almost all neurons are excitatory relay neurons (few local interneurons
#     in rodents), and the inhibitory neurons of the reticular nucleus can have narrow spikes;
#   - in the **midbrain**, dopamine neurons are recognized by their very wide waveforms (often more
#     than 1 ms), but this criterion alone misclassifies many neurons;
#   - in the **cerebellum**, Purkinje cells fire two kinds of spikes (simple and complex spikes)
#     with different shapes;
#   - even in this recording, the probe crosses the **nucleus accumbens** (ACB), which is part of the
#     striatum: there, the wide units are probably medium spiny neurons, which are inhibitory. And
#     the olfactory areas (AON, OLF) are a 3-layered cortex, with different cell types from the
#     6-layered prefrontal cortex: the same thresholds may not fit.
#
#   In each region, the classes and the thresholds must be checked again, ideally with a ground
#   truth.
# - **The ground truth** comes from other methods: **optotagging** (express a light-sensitive channel
#   in one cell type with a genetic tool, and see which units fire at short latency to the light),
#   juxtacellular labeling, or the **connectivity**: a unit whose spikes are followed by a short
#   inhibition of other units is inhibitory (see notebook 03, sections 3 and 6).
#
# **Questions.**
#
# - Classify the units with the half-width instead of the trough-to-peak duration (find the
#   thresholds on its histogram). Do you get the same classes?
# - Fit the GMM of section 3.3 again, without the positive units. Does it change the classes?
# - The amplitude depends on the distance to the neuron. Plot the amplitude against the depth: do
#   you see a relation with the classes?
