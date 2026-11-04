# %% [markdown]
# # 01 · Spike rates and spike intervals — **Julia / Judananas**
#
# After spike sorting, the activity of each neuron is reduced to a list of **spike times**.
# Mathematically, a spike train is a *point process*: a series of events in time,
# where only *when* the events happen matters, not their shape.
#
# In this notebook we describe spike trains with two simple tools:
#
# 1. the **firing rate**: how many spikes a neuron fires per second;
# 2. the **inter-spike intervals (ISIs)**: the time between two consecutive spikes,
#    which tells us *how regular* a neuron fires.
#
# **Dataset.** Session `216300_20200518-probe0.nwb`: a mouse doing an aversive conditioning task
# (sounds, sometimes followed by an air puff) while a Neuropixels probe records in the
# medial prefrontal cortex (mPFC). Here we will focus on three prefrontal areas:
#
# - **ACA**: anterior cingulate area,
# - **PL**: prelimbic area,
# - **ILA**: infralimbic area.
#
# The data are stored in an **NWB** file (Neurodata Without Borders), a standard format
# for neurophysiology data, that we read with [Judananas](https://github.com/PierreLeMerre/Judananas.jl),
# a small Julia package inspired by [pynapple](https://pynapple.org).
#
# This notebook is the Julia version of `python/01-SpikeRate_and_SpikeIntervals.ipynb`: same steps,
# same text, same figures.
#
# ## 0. Get the data
#
# The session is shared on the [DANDI Archive](https://dandiarchive.org), the public archive
# for neurophysiology data, in [Dandiset 000473](https://dandiarchive.org/dandiset/000473)
# (*Esr1+ hypothalamic-habenula neurons shape aversive states*). We use the file
# `sub-216300/sub-216300_ses-216300-20200518-probe0_ecephys+ogen.nwb` (5.3 GB).
#
# **Download the file** to a folder of your choice on your computer
# (keep it out of the git repo: it is far too big for GitHub). Either:
#
# - in the browser: open the
#   [files of the Dandiset](https://dandiarchive.org/dandiset/000473/0.261102.1657/files?location=sub-216300),
#   and click the download icon next to the file;
# - or in a terminal, from the `python/` folder, with the DANDI command-line tool
#   (replace `/path/to/your/data/folder` by the folder where you want to save the file):
#
#   ```bash
#   uv run dandi download -o /path/to/your/data/folder "https://api.dandiarchive.org/api/assets/1465c75e-34ee-4f88-954d-656c2aba0550/download/"
#   ```
#
# Then write the full path to the file in `NWB` in the cell below.
# (Streaming the file from DANDI without downloading it is only available in the Python notebook
# for now.)
#
# First we need to load the Julia packages and give the location of the data (NWB file).

# %%
using Pkg
# julia/Project.toml: works whether Julia starts in julia/ or in the root of the repo
Pkg.activate(isfile(joinpath(@__DIR__, "Project.toml")) ? (@__DIR__) : "julia")
Pkg.instantiate()

# Judananas: time series and spike trains; HDF5: reads the NWB file (an HDF5 file);
# Plots: figures; Statistics: means, medians...; StatsBase: histograms; Printf: formatted printing
using Judananas, HDF5
using Plots, Statistics, StatsBase, Printf, Random
gr()
default(left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)  # room for the axis labels
TAB = palette(:tab10)  # the same 10 colors as matplotlib: blue, orange, green, red, purple, brown...

# Full path to the NWB file you downloaded. Change it to where you saved it.
NWB = "/path/to/your/data/folder/sub-216300_ses-216300-20200518-probe0_ecephys+ogen.nwb"

isfile(NWB) || error("No NWB file at $NWB: set NWB to the downloaded file")

# %% [markdown]
# ## 1. Load the units and the trials
#
# `load_file` opens the NWB file and lists what it contains. Nothing heavy is read
# yet: each object is loaded only when we ask for it with `data["..."]`.
#
# - `units` is a **`TsGroup`**: a group of spike trains, one per unit (= putative neuron).
#   Each spike train is a `Ts` object, i.e. a vector of spike times in seconds.
# - `trials` is an **`IntervalSet`**: a table of time intervals (start, end) with one
#   row per trial, plus extra columns describing each trial.

# %%
data = load_file(NWB)  # open the file and list its content
display(data)

units = data["units"]  # all sorted units (TsGroup)
trials = data["trials"]  # task trials (IntervalSet)

# Displaying a TsGroup prints one row per unit with its metadata
units

# %% [markdown]
# Each unit can come with **metadata**: one value per unit, stored as columns of the `TsGroup`.
# Judananas computes `rate` itself (number of spikes / recording duration). It also reads every
# column of the NWB units table, and the columns of the electrode where each unit was recorded
# (e.g. its `location`): we use some of them in section 2.
#
# Let's look at the trials as well: each row is a trial with its start and end time
# in seconds, and the extra columns describe the trial (sound, air puff, optogenetics, ...).
# Each trial lasts 2 s.

# %%
println("number of units : ", length(units))
println("number of trials: ", length(trials))
trials

# %% [markdown]
# ## 2. Filter units: spike sorting, brain region, ...
#
# Not every unit found by the spike sorter is a well-isolated neuron: some are noise,
# others mix the spikes of several neurons (multi-unit activity, "MUA").
# We use inclusion criteria based on [Siegle et al. (2021)](https://doi.org/10.1038/s41586-020-03171-x),
# but you can also come up with your own set of filters:
#
# 1. **ISI violation ratio < 0.1**: few spikes closer than the refractory period
#    (we come back to ISI violations in section 4.6);
# 2. **a plausible spike waveform**: the waveform of the unit looks like an action potential;
# 3. **presence ratio > 0**: the unit fires during the session (see below);
# 4. **amplitude cutoff < 0.1**: few spikes are missed because they are too small.
#    This one needs the amplitude of every spike, which is not stored in this NWB file,
#    so we cannot apply it here.
#
# A last criterion, on the number of ISIs available before the trials, is applied in
# section 4.4, once we have introduced ISIs.
#
# **Extra information from the NWB file.** An NWB file is an HDF5 file: a tree of groups
# (like folders) and datasets (arrays), that we access with paths. Judananas reads the whole
# units table of the file, so these columns are already metadata of `units`:
#
# - `quality` (`units/quality` in the file) has two values per unit: the number of ISI
#   violations, and the **ISI violation false-positive rate**, an estimate of the fraction of
#   spikes that do not belong to the neuron. This is our ISI violation ratio.
# - `FS` and `RS` classify the waveform of each unit as **fast-spiking** (FS,
#   narrow spike, putative interneuron) or **regular-spiking** (RS, broad spike, putative
#   pyramidal neuron). A unit with neither label has no plausible waveform.
# - `location`: `units/electrodes` gives, for each unit, the row of the **electrode table** where
#   it was recorded, and the column `location` of that table holds the brain area, with its
#   cortical layer, using the Allen atlas names (e.g. `PL6a` = prelimbic area, layer 6a).
#   Judananas looks it up for us.

# %%
isi_fp = [q[2] for q in units[:quality]]  # 2nd value: ISI violation false-positive rate
fs = units[:FS] .== 1  # true for fast-spiking waveforms
rs = units[:RS] .== 1  # true for regular-spiking waveforms

# Waveform class of each unit: "FS", "RS", or "none" if the waveform is not plausible
cell_type = [f ? "FS" : r ? "RS" : "none" for (f, r) in zip(fs, rs)]

# Store them as new metadata columns of the TsGroup (one value per unit, same order)
units[:isi_fp] = isi_fp
units[:cell_type] = cell_type
value_counts(units, :location)  # how many units per brain location

# %% [markdown]
# **Presence ratio.** We cut the session into 100 blocks of equal duration and count the
# fraction of blocks in which the unit fires at least one spike. A unit that drifts away
# from the probe, or appears only late in the session, has a low presence ratio.
# `count` returns the number of spikes of every unit in every block (a `Tsd` with one column
# per unit).

# %%
block = tot_length(time_support(units)) / 100  # duration of one block, in seconds
counts_blocks = count(units, block)  # spike counts: one row per block, one column per unit
presence_ratio = vec(mean(values(counts_blocks) .> 0, dims = 1))  # fraction of blocks with spikes
units[:presence_ratio] = presence_ratio

println("presence ratio, min / median: ", round(minimum(presence_ratio), digits = 2), " / ",
        round(median(presence_ratio), digits = 2))

# %% [markdown]
# We extract the area name without the layer, and keep the units in ACA, PL or ILA that
# pass the criteria.

# %%
# Regular expression: take "ACA", "PL" or "ILA" at the start of the location.
# Units in other areas get an empty string "".
area = [m === nothing ? "" : m.match for m in match.(r"^(ACA|PL|ILA)", units[:location])]

# Store it as a new metadata column called "area"
units[:area] = area

in_pfc = units[:area] .!= ""  # in one of our 3 areas
good = (units[:isi_fp] .< 0.1) .& (units[:cell_type] .!= "none") .& (units[:presence_ratio] .> 0)

pfc = units[in_pfc .& good]  # selection with a vector of true/false, one per unit

println("PFC units: ", sum(in_pfc))
println("PFC units passing the criteria: ", length(pfc))
println(value_counts(pfc, :cell_type))
value_counts(pfc, :area)  # how many units per area

# %% [markdown]
# ## 3. Firing rates
#
# The **firing rate** is the number of spikes divided by the time during which we count them
# (the full experiment or a subselection):
#
# $$ \text{rate} = \frac{\text{number of spikes}}{\text{duration}} \quad \text{(spikes/s, or Hz)} $$
#
# It is the simplest description of a spike train, but *where* we count matters:
# over the whole session, or only during some periods of interest (here, before the trials).
#
# ### 3.1 Rate of all units over the whole session
#
# Every Judananas object has a **time support**: the `IntervalSet` on which it is defined.
# For our units, it is the whole recording session. The mean rate of a unit is its number of
# spikes divided by the total duration of the time support. Judananas already stores it in the
# `rate` column, but let's compute it by hand to see that it is that simple.

# %%
session = time_support(pfc)  # IntervalSet: start and end of the recording
duration = tot_length(session)  # total duration, in seconds
display(session)
println("session duration: ", round(duration / 60, digits = 1), " min")

# Number of spikes of each unit: length() of each spike train
n_spikes = [length(u) for u in pfc]
rate_by_hand = n_spikes ./ duration  # spikes / seconds, one value per unit

# Same numbers as the "rate" column computed by Judananas?
println("same as Judananas: ", rate_by_hand ≈ rates(pfc))
println("median rate: ", round(median(rate_by_hand), digits = 2), " Hz")
println("min / max rate: ", round(minimum(rate_by_hand), digits = 3), " / ",
        round(maximum(rate_by_hand), digits = 1), " Hz")

# %% [markdown]
# Firing rates span **several orders of magnitude**, from less than 0.1 Hz to tens of Hz.
# On a linear axis, the slow units would all be squeezed against zero, so we plot the
# histogram with **logarithmic bins** (each bin is a constant factor wider than the previous).

# %%
rate_bins = 10 .^ range(-2, 2, length = 33)  # 33 bin edges from 0.01 Hz to 100 Hz, evenly spaced in log
area_colors = ["ACA" => TAB[1], "PL" => TAB[2], "ILA" => TAB[3]]  # one color per area

fig = plot(size = (800, 400), xscale = :log10, xlabel = "Session rate (Hz, log scale)",
           ylabel = "Number of units", title = "Session firing rate of the PFC units")
for (a, color) in area_colors
    rates_area = rates(pfc)[pfc[:area] .== a]  # session rates of the units in this area
    stephist!(fig, rates_area, bins = rate_bins, lw = 2, color = color,
              label = "$a (n=$(length(rates_area)))")
end
fig

# %% [markdown]
# **Advanced: a smoothed density.** The histogram depends on where the bin edges fall, and
# the areas have different numbers of units, so their counts are hard to compare.
# Instead, we estimate the **density** of the rates in each area, with a
# **Savitzky–Golay filter**
# ([`savitzky_golay`](https://github.com/lnacquaroli/SavitzkyGolay.jl) from the SavitzkyGolay.jl package):
# in a sliding window, it fits a polynomial to the data and keeps the value of the polynomial
# (or of its derivative) at the center of the window.
#
# 1. We work with $\log_{10}(\text{rate})$, so that each decade (0.1–1 Hz, 1–10 Hz, ...)
#    has the same width.
# 2. We compute the **cumulative distribution** in fine bins (0.02 decade): the fraction of units
#    with a rate below each value. It goes from 0 to 1 by small steps, one per unit.
# 3. The density is the **derivative** of the cumulative distribution. The Savitzky–Golay filter
#    smooths the steps and differentiates at the same time (`deriv = 1`). Smoothing the cumulative
#    distribution works much better than smoothing the counts of a fine histogram: with a few
#    dozen units, most fine bins have 0 or 1 unit, and the result stays noisy.
#
# The area under each curve is 1, whatever the number of units, so the areas can be compared.
# The two parameters of the filter: `window` (number of bins in the sliding window: larger is
# smoother) and `polyorder` (degree of the polynomial, smaller than `window`). Try changing them!
# The ticks at the bottom show the rate of each unit. Careful with ACA: with only a few units,
# its curve is mostly the shape of the filter.

# %%
using SavitzkyGolay

bin_width = 0.02  # width of the fine bins, in decades of rate
window, polyorder = 75, 2  # Savitzky-Golay parameters: 75 bins = 1.5 decade, quadratic polynomial
# Bin edges in log10(rate), from 0.001 Hz to 1000 Hz: wider than the plot (0.01 to 100 Hz), because
# savitzky_golay mirrors the data at the edges, which would bend the curves near them
log_bins = -3:bin_width:3
centers = (log_bins[1:end-1] .+ log_bins[2:end]) ./ 2  # bin centers, in log10(rate)

fig = plot(size = (800, 400), xscale = :log10, xlims = (0.01, 100),
           xlabel = "Session rate (Hz, log scale)", ylabel = "Density (fraction of units per decade)",
           title = "Session firing rate of the PFC units, smoothed density")
for (a, color) in area_colors
    log_rates = log10.(rates(pfc)[pfc[:area] .== a])  # log10 of the session rates in this area
    n_in_bin = fit(Histogram, log_rates, log_bins).weights  # number of units in each fine bin
    cumulative = cumsum(n_in_bin) ./ length(log_rates)  # fraction of units below each bin, 0 to 1
    # Smoothed derivative of the cumulative distribution = density, in fraction of units per decade
    density = savitzky_golay(cumulative, window, polyorder; deriv = 1, rate = 1 / bin_width).y
    density = max.(density, 0)  # the filter can dip slightly below 0 at the edges

    plot!(fig, 10 .^ centers, density, color = color, lw = 2, fillrange = 0, fillalpha = 0.15,
          label = "$a (n=$(length(log_rates)))")
    # Rug: one small tick per unit, just below the curves
    scatter!(fig, 10 .^ log_rates, fill(-0.03, length(log_rates)), markershape = :vline,
             markersize = 6, color = color, markerstrokecolor = color, alpha = 0.6, label = "")
end
hline!(fig, [0], color = :black, lw = 0.5, label = "")
fig

# %% [markdown]
# ### 3.2 Two example units with the same rate
#
# We now look at two units recorded in the same area and layer (**ILA6a**, infralimbic
# area, layer 6a) with a similar **session rate**, about 8–10 Hz.
# If the rate were all there is to know about a spike train, these two units should look alike.
#
# In Judananas, `pfc[k]` is the k-th unit of the group (by position). To select a unit by its **id**,
# we write `pfc[id = 185]`, and `pfc[:location, id = 185]` gives one of its metadata values.

# %%
examples = ["unit A" => 185, "unit B" => 228]  # name => unit id
colors = Dict("unit A" => TAB[1], "unit B" => TAB[2])  # one color per unit in all figures

for (name, i) in examples
    @printf("%s: id %d, %s, %d spikes, session rate %.2f Hz\n", name, i, pfc[:location, id = i],
            length(pfc[id = i]), pfc[:rate, id = i])
end

# %% [markdown]
# Let's look at their activity during 10 seconds, drawn as a raster (one tick per spike).
# The trials are shaded in gray: outside them, the mouse waits for the next trial and we see
# the **spontaneous activity** of the neurons.

# %%
window = IntervalSet(3020, 3030)  # from 3020 s to 3030 s

panels = []
for (name, i) in examples
    spikes_in_window = restrict(pfc[id = i], window)  # keep only the spikes inside the window
    p = plot(spikes_in_window, color = colors[name], ylabel = name, xlims = (3020, 3030), xlabel = "")
    plot!(p, intersect(trials, window), color = :gray)  # shade the trials inside the window
    push!(panels, p)
    println("$name: $(length(spikes_in_window)) spikes in 10 s")
end
plot!(panels[1], title = "10 seconds of activity (gray: trials)")
plot!(panels[end], xlabel = "Time (s)")
fig = plot(panels..., layout = (2, 1), size = (1000, 300), link = :x, left_margin = 10Plots.mm,
           bottom_margin = 3Plots.mm)

# %% [markdown]
# The two units fire about the same number of spikes (~100 in 10 s), but **not in the same way**:
#
# - **unit A** fires spikes spread out rather evenly in time;
# - **unit B** fires **clusters of spikes** close together, separated by longer silences.
#
# The rate alone cannot capture this difference. We will quantify it in section 4
# with the inter-spike intervals.

# %% [markdown]
# ### 3.3 Rate on selected intervals: 3 s before trial onset
#
# Often we only care about some periods of the session. Here we look at the
# **3 seconds before the start of each trial**, when the mouse is waiting for the next
# trial (the inter-trial interval, ITI). In this task, all stimuli (sounds, air puffs,
# optogenetics) are delivered inside the trials, so these windows contain no stimulus.
#
# We take **all the trials**, and keep a window only if it does not overlap with the previous
# trial (in this session the gap between trials is always more than 3 s, so all are kept).
# `restrict` keeps only the spikes inside these intervals; the rate of the restricted spike
# train is computed on its new time support: number of windows × 3 s.

# %%
pre = 3.0  # length of each window, in seconds, before trial onset

# End of the previous trial, for each trial (-infinity for the first trial)
previous_end = [-Inf; stops(trials)[1:end-1]]
keep = starts(trials) .- pre .>= previous_end  # true if the window starts after the previous trial
onsets = starts(trials)[keep]  # trial start times (s) of the kept windows

# One interval [onset - 3 s, onset] per kept trial
pre_trial = IntervalSet(onsets .- pre, onsets)
display(pre_trial)
println("windows kept: ", length(pre_trial), " out of ", length(trials), " trials")
println("total duration of the windows: ", round(tot_length(pre_trial), digits = 1), " s")

for (name, i) in examples
    spikes_pre = restrict(pfc[id = i], pre_trial)  # spikes inside the windows only
    @printf("%s: %d spikes, rate before trials %.2f Hz, session rate %.2f Hz\n", name,
            length(spikes_pre), rate(spikes_pre), pfc[:rate, id = i])
end

# %% [markdown]
# ### 3.4 Raster plot aligned to trial onset
#
# A **raster plot** shows one row per trial and one tick per spike. To stack the trials,
# we need spike times **relative to the trial onset** rather than to the session start.
# `compute_perievent` does this alignment: for each event (here, each trial onset)
# it keeps the spikes inside the window and subtracts the event time.
# With the window `(-3, 0)`, we get spikes from 3 s before to the onset (time 0).
#
# The result is a `TsGroup` with one spike train per trial. We store one per example unit.

# %%
# Vector of pairs: unit name => TsGroup of aligned spike trains (one Ts per trial)
aligned = [name => compute_perievent(pfc[id = i], onsets, (-pre, 0)) for (name, i) in examples]

panels = []
for (name, al) in aligned
    # plot() of a TsGroup draws a raster: one row per trial, one tick per spike
    p = plot(al, color = colors[name], markersize = 2,
             xlims = (-pre, 0.05), xlabel = "Time from trial onset (s)", title = name)
    vline!(p, [0], color = :red, label = "")
    push!(panels, p)
end
fig = plot(panels..., layout = (1, 2), size = (1200, 700), link = :y)

# %% [markdown]
# Trials are sorted in time (trial 1 is the earliest, at the bottom). Notice that unit B barely
# fires before the first ~80 trials: the rate of a neuron is **not constant over a session**. It
# can change with the behavioural state of the animal, with learning, or with the recording itself
# (e.g. electrode drift). Keep this in mind when comparing rates!

# %% [markdown]
# **After the trial onset.** Obviously, what happens before the stimulus is a bit boring:
# the raster plots **after** the stimulus are much more interesting, to see whether a neuron
# responds to the stimulus or not. We now align the spikes from 1 s before to 3 s after the
# trial onset, and shade the trial (0 to 2 s).
#
# The trials are not all the same: there are sounds, sounds with optogenetic stimulation, air puffs
# (with a blue-noise sound), and trials with optogenetic stimulation only. We use the metadata
# columns of `trials` to give a type to each trial, and **group the rows of the raster by trial
# type** (inside a group, trials stay in time order). Next to units A and B, we add a third
# unit, **unit 288** (PL6a), that responds strongly.

# %%
post_units = [examples; "unit 288" => 288]  # name => unit id: units A, B and a responsive unit
colors["unit 288"] = TAB[5]

# Type of each trial, from the metadata columns of the trials
trial_type = [ap == 1 ? "air puff" : s == 1 ? (o == 1 ? "sound + opto" : "sound") : "opto only"
              for (ap, s, o) in zip(trials[:airpuff], trials[:sound], trials[:optogenetics])]
type_order = ["sound", "sound + opto", "air puff", "opto only"]  # order of the groups in the raster
order = reduce(vcat, [findall(==(t), trial_type) for t in type_order])  # trials sorted by type
sorted_onsets = starts(trials)[order]  # trial onsets, grouped by type (a vector: Ts would sort them)
group_ends = cumsum([count(==(t), trial_type) for t in type_order])  # last row of each group
group_starts = [1; group_ends[1:end-1] .+ 1]  # first row of each group
for (t, a, b) in zip(type_order, group_starts, group_ends)
    println("$t: $(b - a + 1) trials")
end

panels = []
for (name, i) in post_units
    al = compute_perievent(pfc[id = i], sorted_onsets, (-1, 3))  # 1 s before to 3 s after
    p = plot(al, color = colors[name], markersize = 1.5,
             xlims = (-1, 3), xlabel = "Time from trial onset (s)", ylabel = "", title = "$name ($i)",
             yflip = true)  # first trial at the top
    vspan!(p, [0, 2], color = :gray, alpha = 0.15, label = "")  # the trial: 0 to 2 s
    vline!(p, [0], color = :red, label = "")
    hline!(p, group_ends[1:end-1] .+ 0.5, color = :black, lw = 1, label = "")  # between trial types
    push!(panels, p)
end
plot!(panels[1], yticks = ((group_starts .+ group_ends) ./ 2, type_order))  # one label per trial type
for p in panels[2:end]
    plot!(p, yticks = false)
end
fig = plot(panels..., layout = (1, 3), size = (1500, 700), link = :y, left_margin = 10Plots.mm,
           bottom_margin = 5Plots.mm)

# %% [markdown]
# - **Unit A** fires about the same before and during the trials: it does not seem to respond.
# - **Unit B** is hard to judge by eye: it is almost silent in the first trials of the session
#   (most of the "sound" group), and its rate changes over the session.
# - **Unit 288** fires a sharp burst of spikes about 0.1 s after the onset of the trials, in
#   **every trial type**, even without a sound. In the air-puff trials, a second response comes
#   about 0.8 s after the onset.
#
# A raster shows the response trial by trial; the PSTH (next section) puts a number on it.

# %% [markdown]
# ### 3.5 Peri-stimulus time histogram (PSTH)
#
# The raster shows each trial, but it is hard to read the rate from it. The **PSTH**
# summarises the raster: we cut time into bins, count the spikes of all trials in each bin,
# and divide by (number of trials × bin size) to get a rate in Hz:
#
# $$ \text{PSTH}(t) = \frac{\text{spikes in the bin at } t \text{ (all trials)}}{N_\text{trials} \times \text{bin size}} $$
#
# The **bin size** is the main parameter of a PSTH. Change `bin_size` below and rerun the cell:
#
# - **small bins** (e.g. 10 ms): fine time resolution, but each bin has few spikes,
#   so the PSTH is very noisy;
# - **large bins** (e.g. 500 ms): smooth estimate, but fast changes of the rate are averaged out.

# %%
bin_size = 0.1  # bin size, in seconds. Try 0.01, 0.05, 0.5, 1 ...

"""PSTH of aligned spike trains (one Ts per trial): bin centers and rate in Hz."""
function psth(al, bin_size)
    # count() bins the spikes of each trial: a Tsd with one row per bin, one column per trial
    counts = count(al, bin_size)
    # Sum over trials, then divide by the number of trials and the bin size -> rate in Hz
    return times(counts), vec(sum(values(counts), dims = 2)) ./ (length(al) * bin_size)
end

rasters, histograms = [], []
for (name, al) in aligned
    t, rate_psth = psth(al, bin_size)
    mean_rate = mean(rate_psth)  # average rate over the 3 s window

    p1 = plot(al, color = colors[name], markersize = 2,
              xlims = (-pre, 0), xlabel = "",
              title = @sprintf("%s, bin size = %.0f ms", name, bin_size * 1000))
    p2 = bar(t, rate_psth, bar_width = bin_size, color = colors[name], linecolor = colors[name],
             xlims = (-pre, 0), xlabel = "Time from trial onset (s)", ylabel = "Rate (Hz)", label = "")
    hline!(p2, [mean_rate], color = :black, ls = :dash, label = "mean rate before trials",
           legend = :bottomleft)
    push!(rasters, p1); push!(histograms, p2)
end
fig = plot(rasters..., histograms..., layout = grid(2, 2, heights = [2 / 3, 1 / 3]),
           size = (1200, 700), left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)

# %% [markdown]
# To see the effect of the bin size directly, we compute the same PSTH with four bin sizes,
# all on the same y-axis. Note how the mean rate (dashed line) stays the same: the bin size
# only changes **how finely we look at time**, not how many spikes there are.
# With 10 ms bins, the bars jump up and down although nothing happens before the trial:
# these fluctuations are just counting noise.

# %%
bin_sizes = [0.01, 0.05, 0.25, 1.0]  # in seconds

panels = []
for b in bin_sizes, (name, al) in aligned  # row by row: one bin size per row, one unit per column
    t, rate_psth = psth(al, b)  # PSTH in Hz
    p = bar(t, rate_psth, bar_width = b, color = colors[name], linecolor = colors[name], label = "",
            xlims = (-pre, 0), title = @sprintf("%s, bin size = %.0f ms", name, b * 1000),
            ylabel = "Rate (Hz)", xlabel = b == bin_sizes[end] ? "Time from trial onset (s)" : "")
    hline!(p, [mean(rate_psth)], color = :black, ls = :dash, label = "")
    push!(panels, p)
end
# link = :all: same axes on all panels, so that we can compare the fluctuations
fig = plot(panels..., layout = (length(bin_sizes), 2), size = (1200, 900), link = :all,
           left_margin = 5Plots.mm)

# %% [markdown]
# ### 3.6 PSTH of the responses to the sound
#
# Now that we know how to build a PSTH, let's use it where it matters: **after the stimulus**.
# We keep the **sound only** trials (no optogenetics, no air puff, section 3.4), and compute the
# PSTH of units A, B and 288 from 1 s before to 3 s after the trial onset, with 50 ms bins.
# `compute_perievent` also accepts the trials themselves (an `IntervalSet`): it aligns the spikes on
# the start of each trial, and copies the trial metadata into the result.
# The dashed line is the mean rate in the second before the onset (the **baseline**): a response
# is a departure from it.

# %%
sound_trials = trials[trial_type .== "sound"]  # the sound only trials (IntervalSet, with metadata)
bin_size = 0.05  # 50 ms bins
println("sound only trials: ", length(sound_trials))

rasters, histograms = [], []
for (name, i) in post_units
    al = compute_perievent(pfc[id = i], sound_trials, (-1, 3))  # one Ts per trial, aligned on starts
    t, rate_psth = psth(al, bin_size)  # rate in Hz
    baseline = mean(rate_psth[t .< 0])  # mean rate in the second before the onset

    p1 = plot(al, color = colors[name], markersize = 2,
              xlims = (-1, 3), xlabel = "",
              title = "$name ($i), sound only trials")
    p2 = bar(t, rate_psth, bar_width = bin_size, color = colors[name], linecolor = colors[name],
             label = "", xlims = (-1, 3), xlabel = "Time from trial onset (s)",
             ylabel = i == post_units[1][2] ? "Rate (Hz)" : "")
    hline!(p2, [baseline], color = :black, ls = :dash, label = "baseline", legend = :topright)
    for p in (p1, p2)
        vspan!(p, [0, 2], color = :gray, alpha = 0.15, label = "")  # the trial: 0 to 2 s
        vline!(p, [0], color = :red, label = "")
    end
    push!(rasters, p1); push!(histograms, p2)
end
fig = plot(rasters..., histograms..., layout = grid(2, 3, heights = [2 / 3, 1 / 3]),
           size = (1500, 600), left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)

# %% [markdown]
# - **Unit 288** responds clearly to the sound: a sharp peak in the first 0.3 s.
# - **Unit B** seems to fire more between about 0.5 and 1.2 s after the onset. Careful: it is silent
#   in the first half of the sound trials (the start of the session), so this bump comes from the
#   last ~50 trials only.
# - **Unit A** does not change much.
#
# But is the bump of unit B a real response, or just counting noise? And would we see a shorter
# response with other bins? Answering by eye is not enough: we need a **statistical test**.
#
# ### 3.7 Advanced: beyond bins, the ZETA test
#
# **The problem with bins.** A PSTH depends on choices that we made ourselves:
#
# - the **bin size**: too small and the PSTH is noise, too large and a short response is averaged
#   out (section 3.5). There is no "right" value, and the result of a test can change with it;
# - the **bin edges**: moving all the bins by half a bin can split a response in two;
# - if we test every bin against the baseline, we make **many tests**, and some of them will be
#   "significant" by chance (multiple comparisons);
# - if instead we compare the **mean rate** during the trial with the baseline (one test, e.g. a
#   t-test), we miss responses that do not change the mean: a short burst followed by a pause,
#   or a change in the *timing* of the spikes rather than their number.
#
# **The ZETA test** ([Montijn et al., 2021, *eLife*](https://doi.org/10.7554/eLife.71969),
# Julia package [ZetaJu](https://github.com/PierreLeMerre/ZetaJu), a port of the Python package
# [zetapy](https://github.com/JorritMontijn/zetapy)) asks whether a neuron
# responds to an event **without any bin**:
#
# 1. it pools the spike times of all trials (relative to the event) between 0 and `dblUseMaxDur`,
#    and computes their **cumulative distribution**: the fraction of the spikes fired before each time;
# 2. if the neuron does not respond, spikes are spread evenly in time, and this curve is a straight
#    line. The **deviation** from the straight line shows when the neuron fires more (the curve rises
#    faster) or less (it rises slower) than average;
# 3. the largest deviation is the **ZETA** (Zenith of Event-based Time-locked Anomalies), and its time
#    is the **latency** of the response;
# 4. to know if this ZETA is large, the test **jitters** the event times at random many times,
#    and computes the deviation of these fake trials: this gives the deviations expected *by chance*,
#    hence a p-value. A **ZETA score** above 2 is significant.
#
# ZETA also gives an **instantaneous firing rate**, computed from the spike times at several time
# scales: a PSTH without bins.
#
# We run it on the sound only trials for units A, B and 288, and add **unit C** (302, ACAd5),
# that we will meet again in the optional section 5. With `[start stop]` for each trial,
# `zetatest` also gives the p-value of a t-test on the mean rate (trial vs baseline), to compare.

# %%
using ZetaJu

zeta_units = [post_units; "unit C" => 302]  # name => unit id: units A, B, 288 and C
colors["unit C"] = TAB[3]
sound_events = [starts(sound_trials) stops(sound_trials)]  # [start stop] of each sound only trial

zeta = []  # name => (ZETA results, instantaneous rate)
for (name, i) in zeta_units
    Random.seed!(0)  # zetatest jitters at random: fix the seed to get the same p-values for everyone
    p, zeta_data, rate_data = zetatest(times(pfc[id = i]), sound_events; dblUseMaxDur = 2.0,
                                       intResampNum = 100, boolReturnRate = true)
    push!(zeta, name => (zeta_data, rate_data))
    @printf("%s (%d): ZETA p = %.2g (score %.1f, latency %.0f ms) | mean-rate t-test p = %.2g\n",
            name, i, p, zeta_data["dblZETA"], zeta_data["dblLatencyZETA"] * 1000, zeta_data["dblMeanP"])
end

deviations, ifrs = [], []
for (name, (zd, rd)) in zeta
    i = Dict(zeta_units)[name]
    p1 = plot(title = @sprintf("%s (%d)\nZETA p = %.2g, t-test p = %.2g", name, i, zd["dblZetaP"],
                               zd["dblMeanP"]), titlefontsize = 10,
              ylabel = name == zeta[1][1] ? "Deviation from\nuniform spiking" : "")
    # Deviation of 20 jittered resamplings (gray): what we expect by chance
    for (t, dev) in first(collect(zip(zd["cellRandTime"], zd["cellRandDeviation"])), 20)
        plot!(p1, t, dev, color = :gray, lw = 0.5, alpha = 0.5, label = "")
    end
    plot!(p1, zd["vecSpikeT"], zd["vecRealDeviation"], color = colors[name], lw = 2, label = "")  # the data
    scatter!(p1, [zd["dblLatencyZETA"]], [zd["dblZETADeviation"]], color = :black, markersize = 6,
             label = "")  # the ZETA
    hline!(p1, [0], color = :black, lw = 0.5, label = "")

    # rate without bins
    p2 = plot(rd["vecT"], rd["vecRate"], color = colors[name], lw = 1.5, label = "", xlims = (0, 2),
              xlabel = "Time from trial onset (s)",
              ylabel = name == zeta[1][1] ? "Instantaneous rate (Hz)" : "")
    push!(deviations, p1); push!(ifrs, p2)
end
fig = plot(deviations..., ifrs..., layout = (2, 4), size = (1700, 650), link = :x,
           left_margin = 8Plots.mm, bottom_margin = 5Plots.mm)

# %% [markdown]
# Top: the deviation of the data (color) and of 20 jittered resamplings (gray); the black dot is the
# ZETA. Where the curve goes **up**, the neuron fires more than average; where it goes **down**,
# less. (The deviation is centered on 0, so the ZETA can also be the low point just *before* a
# steep rise, as for unit 288.) Bottom: the instantaneous firing rate.
#
# - **Unit 288**: a huge deviation, far outside the jittered ones: both tests agree, it responds.
# - **Unit A**: the deviation stays close to the jittered ones: no clear response (p > 0.05).
# - **Unit B**: fires *less* during the first 0.4 s, then more until ~1.5 s. Both tests find
#   a response (but remember that it comes from the last trials only).
# - **Unit C**: the most interesting case. ZETA finds a strong response, but the t-test on the mean
#   rate finds nothing (p ≈ 0.5): the neuron fires a brief burst ~80 ms after the onset, then a bit
#   less than usual, so the total number of spikes hardly changes. A test on the mean rate
#   cannot see it.
#
# **Questions.**
#
# - Plot the binned PSTH of unit C on the sound only trials (as in section 3.6). With which bin
#   size can you see its response?
# - Run `zetatest` on the "air puff" trials, or on the "opto only" trials. Do the same units respond?

# %% [markdown]
# ## 4. Inter-spike intervals (ISIs) and burstiness
#
# If a neuron fires spikes at times $t_1, t_2, t_3, \dots$, its **inter-spike intervals** are
# the differences between consecutive spike times:
#
# $$ \mathrm{ISI}_k = t_{k+1} - t_k $$
#
# The firing rate tells us *how many* spikes a neuron fires. The ISIs tell us *how* it fires:
# regularly like a clock, randomly, or in bursts of spikes close together.
#
# We go back to our two example units, which have the same rate but looked so different
# in section 3.2, and use all their spikes over the session.
#
# `time_diff` computes the difference between consecutive timestamps: these are the ISIs.
# The result is a `Tsd` (a time series with values), timed at the middle of each interval;
# `values` gives the ISIs themselves (a vector, in seconds).
#
# A useful sanity check: the **mean ISI** should be close to **1 / mean rate**.
# (They are not exactly equal because the recording does not start and end on a spike.)

# %%
isis = Dict(name => values(time_diff(pfc[id = i])) for (name, i) in examples)  # ISIs in seconds

for (name, i) in examples
    isi = isis[name]
    @printf("%s: mean ISI %.1f ms (1 / rate = %.1f ms), std of ISIs %.1f ms, shortest %.2f ms\n",
            name, mean(isi) * 1000, 1000 / pfc[:rate, id = i], std(isi, corrected = false) * 1000,
            minimum(isi) * 1000)
end

# %% [markdown]
# Similar mean ISI (similar rate), but a very different **spread** of the ISIs
# (standard deviation): more than 10 times larger for unit B.
#
# ### 4.1 ISI histograms
#
# The **ISI distribution** shows how often each interval occurs.
# `compute_isi_distribution` counts the ISIs in the bins we give it.
# We draw it three ways, because each view highlights something different:
#
# 1. **0 to 300 ms**: the overall shape. Unit A's ISIs form a broad bump: after a spike,
#    it rarely fires again within a few ms. Unit B has a sharp peak of very short ISIs
#    (below 10 ms): when it fires, it often fires several spikes in quick succession, a **burst**.
# 2. **zoom on 0 to 20 ms**: just after a spike, a neuron cannot fire again for ~1–2 ms:
#    this is the **refractory period**. A good single unit has (almost) no ISIs there.
#    Unit B's burst ISIs are around 7–8 ms.
# 3. **log scale**: ISIs range from milliseconds to minutes. With bins that are
#    evenly spaced on a logarithmic axis, we see all time scales at once. Unit A has a single
#    bump around 100 ms. Unit B has **two bumps**: the short ISIs inside bursts (~8 ms),
#    and the longer ISIs between bursts (~100 ms).

# %%
bins_ms = 0:0.005:0.3  # 5 ms bins from 0 to 300 ms (in seconds)
bins_zoom = 0:0.0005:0.02  # 0.5 ms bins from 0 to 20 ms
bins_log = 10 .^ range(-3, 2, length = 101)  # 101 edges from 1 ms (10^-3 s) to 100 s, evenly spaced in log

"""Histogram drawn as filled steps, from the bin edges and the counts."""
stairs!(p, edges, counts; kwargs...) =
    plot!(p, edges, [counts; counts[end]]; seriestype = :steppost, fillrange = 0, lw = 0, label = "",
          kwargs...)

panels = []
for (name, i) in examples
    # Each call returns the bin edges, the bin centers and the counts (number of ISIs in each bin)
    h_ms = compute_isi_distribution(pfc[id = i]; bins = bins_ms).counts
    h_zoom = compute_isi_distribution(pfc[id = i]; bins = bins_zoom).counts
    h_log = compute_isi_distribution(pfc[id = i]; bins = bins_log).counts

    p1 = stairs!(plot(ylabel = "$name\nCount"), bins_ms .* 1000, h_ms, color = colors[name])
    p2 = vspan!(plot(), [0, 1.5], color = :red, alpha = 0.2, label = "refractory period (1.5 ms)")
    stairs!(p2, bins_zoom .* 1000, h_zoom, color = colors[name])
    p3 = stairs!(plot(xscale = :log10, xticks = 10.0 .^ (-3:2)), bins_log, h_log, color = colors[name])  # log x-axis
    push!(panels, p1, p2, p3)
end
plot!(panels[1], title = "ISI histogram, 0–300 ms")
plot!(panels[2], title = "Zoom on 0–20 ms", legend = :topleft)
plot!(panels[3], title = "ISI histogram, log scale")
plot!(panels[4], xlabel = "ISI (ms)")
plot!(panels[5], xlabel = "ISI (ms)", legend = false)
plot!(panels[6], xlabel = "ISI (s, log scale)")
# Fraction of short and long ISIs, to put numbers on what we see
for (name, _) in examples
    isi = isis[name]
    @printf("%s: %.1f %% of ISIs < 20 ms, %.2f %% of ISIs > 1 s\n", name, 100 * mean(isi .< 0.02),
            100 * mean(isi .> 1))
end

fig = plot(panels..., layout = (2, 3), size = (1400, 600), left_margin = 8Plots.mm,
           bottom_margin = 5Plots.mm)

# %% [markdown]
# ### 4.2 Burstiness
#
# To summarise the shape of the ISI distribution in a single number, we compare its
# standard deviation $\sigma$ to its mean $\mu$. The **burstiness** of
# [Goh & Barabási (2008)](https://doi.org/10.1209/0295-5075/81/48002) is:
#
# $$ B = \frac{\sigma_\mathrm{ISI} - \mu_\mathrm{ISI}}{\sigma_\mathrm{ISI} + \mu_\mathrm{ISI}} $$
#
# $B$ is always between −1 and 1:
#
# - $B = -1$: perfectly **regular** firing, like a clock (all ISIs equal, $\sigma = 0$);
# - $B = 0$: **Poisson** firing, spikes occur at random times ($\sigma = \mu$,
#   exponential ISI distribution);
# - $B \to 1$: very **bursty** firing (a few very long silences and many very short ISIs,
#   $\sigma \gg \mu$).

# %%
"""Burstiness B = (std - mean) / (std + mean) of a vector of ISIs."""
function burstiness(isi)
    m, s = mean(isi), std(isi, corrected = false)
    return (s - m) / (s + m)
end

for (name, _) in examples
    @printf("%s: B = %+.3f\n", name, burstiness(isis[name]))
end

# %% [markdown]
# Unit A has $B < 0$ (slightly more regular than Poisson) and unit B has $B > 0$ (bursty).
#
# **Check with a Poisson process.** If spikes are thrown at random, uniformly in time,
# the ISIs follow an exponential distribution and $B$ should be close to 0.
# We simulate such a spike train with the same number of spikes as unit A.

# %%
rng = Xoshiro(0)  # random generator with a fixed seed: same result for everyone

n_poisson = length(pfc[id = 185])  # same number of spikes as unit A
t_start, t_end = starts(session)[1], stops(session)[1]  # same duration as the session

# n_poisson random spike times, uniformly distributed over the session (Ts sorts them in time)
poisson = Ts(t_start .+ (t_end - t_start) .* rand(rng, n_poisson))

println("Poisson rate: ", round(rate(poisson), digits = 2), " Hz")
@printf("Poisson B   : %+.3f\n", burstiness(values(time_diff(poisson))))

# %% [markdown]
# **Careful: the standard deviation is sensitive to a few very long ISIs.**
# Unit B is almost silent during the first minutes of the session, which gives it a few
# very long ISIs (up to two minutes). Let's see how much they weigh in $B$.

# %%
isi_B = sort(isis["unit B"])  # ISIs of unit B, sorted from shortest to longest
println("5 longest ISIs of unit B (s): ", round.(isi_B[end-4:end], digits = 1))
@printf("B with all ISIs           : %+.3f\n", burstiness(isi_B))
@printf("B without the 5 longest   : %+.3f\n", burstiness(isi_B[1:end-5]))  # still bursty, but less

# %% [markdown]
# Removing only 5 ISIs out of more than 40,000 changes $B$ a lot. Computed over the whole
# session, $B$ mixes the way the neuron fires with two other things that also make the ISIs
# more variable:
#
# - **slow changes of the rate** over the session (like the silent start of unit B);
# - **what happens during the trials**: sounds and air puffs can make a neuron fire many
#   spikes in a row, or stop firing, which artificially boosts $B$.
#
# ### 4.3 Burstiness in the inter-trial intervals
#
# To measure the firing pattern of a neuron *at rest*, we stay in the **inter-trial intervals
# (ITIs)**: we use the 3 s windows before the onset of every trial, built in section 3.3.
# In each window, we compute the ISIs and $B$, as long as there are **at least 4 ISIs**
# (with fewer, the standard deviation means nothing). Then we average $B$ over the windows.
#
# Because the windows are short, slow changes of the rate over the session no longer matter.

# %%
iti = pre_trial  # the 3 s windows before trial onset (section 3.3)
min_isi = 4  # minimum number of ISIs in a window to compute B
println("number of ITI windows: ", length(iti))

"""B in each window that has at least min_isi ISIs. Returns one value per valid window."""
function burstiness_per_window(spikes, windows, min_isi = 4)
    values = Float64[]
    for k in 1:length(windows)  # loop over the windows
        t = times(restrict(spikes, windows[k]))  # spike times inside window k
        isi = diff(t)  # ISIs inside window k only
        if length(isi) >= min_isi  # skip windows with too few ISIs
            push!(values, burstiness(isi))
        end
    end
    return values
end

B_windows = Dict(name => burstiness_per_window(pfc[id = i], iti, min_isi) for (name, i) in examples)

fig = plot(size = (700, 400), xlabel = "B in one 3 s ITI window", ylabel = "Number of windows",
           title = "Burstiness in each inter-trial interval")
for (name, _) in examples
    b = B_windows[name]
    @printf("%s: %d valid windows, mean B in ITIs = %+.3f\n", name, length(b), mean(b))
    stephist!(fig, b, bins = range(-1, 1, length = 41), lw = 2, color = colors[name],
              label = @sprintf("%s (mean %+.2f)", name, mean(b)))
    vline!(fig, [mean(b)], color = colors[name], ls = :dash, label = "")
end
fig

# %% [markdown]
# Unit B is still more bursty than unit A, but both values are lower than over the session.
#
# **Do the same with the Poisson spike train.** Over the whole session it gave $B \approx 0$.
# What about in the 3 s ITI windows?

# %%
B_poisson = burstiness_per_window(poisson, iti, min_isi)
@printf("Poisson (%.1f Hz): mean B in ITIs = %+.3f\n", rate(poisson), mean(B_poisson))

# %% [markdown]
# The Poisson train now has $B < 0$! This is a **bias of short windows**:
#
# - in a 3 s window, no ISI can be longer than 3 s: the long intervals are cut out;
# - with only a few ISIs per window, the standard deviation is underestimated.
#
# Both make $B$ smaller, and the effect is stronger for slow neurons (fewer ISIs per window).
# So in the ITIs, the reference for "random firing" is **not** $B = 0$: keep it in mind when
# reading the values below. The advanced section 4.5 shows how to correct for this bias.

# %% [markdown]
# ### 4.4 Burstiness of all PFC units
#
# We now compute the ITI burstiness for every unit. As a last inclusion criterion,
# a unit needs **at least 4 ISIs in at least 12 separate 3 s windows**: with fewer windows,
# the mean $B$ is not reliable. The other units get NaN (no value).
# We store the results as new metadata columns, like we did for the area.

# %%
min_windows = 12  # minimum number of valid windows per unit

B_iti, n_valid, rate_iti = Float64[], Int[], Float64[]  # one value per unit in each vector
for u in pfc
    b = burstiness_per_window(u, iti, min_isi)
    push!(n_valid, length(b))  # number of windows with at least min_isi ISIs
    push!(rate_iti, rate(restrict(u, iti)))  # spikes in the ITIs / total ITI duration
    push!(B_iti, length(b) >= min_windows ? mean(b) : NaN)  # NaN: not enough windows
end

pfc[:burstiness_iti] = B_iti
pfc[:n_valid_windows] = n_valid
pfc[:rate_iti] = rate_iti

valid = pfc[.!isnan.(pfc[:burstiness_iti])]  # units with at least 12 valid windows
println("units with >= 12 valid windows: ", length(valid), " out of ", length(pfc))
println("median B in ITIs: ", round(median(valid[:burstiness_iti]), digits = 3))
@printf("for comparison, Poisson train (%.0f Hz) in the ITIs: %+.3f\n", rate(poisson), mean(B_poisson))

# %% [markdown]
# To compare the areas, we draw the **smoothed density** of $B$ in each area, with the same
# Savitzky–Golay method as for the firing rates (section 3.1): the derivative of the smoothed
# cumulative distribution. We write it as a function, to use it again below.
#
# On the right, each dot is a unit. The two example units are drawn with their own colors and
# markers, different from the colors of the areas.

# %%
"""Smoothed density of values between lo and hi (Savitzky-Golay, as in section 3.1).

Returns the bin centers and the density (area under the curve = 1)."""
function smoothed_density(values, lo, hi; bin_width = 0.005, window = 41, polyorder = 2)
    edges = lo:bin_width:hi  # fine bins
    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
    cumulative = cumsum(fit(Histogram, values, edges).weights) ./ length(values)  # 0 to 1
    density = savitzky_golay(cumulative, window, polyorder; deriv = 1, rate = 1 / bin_width).y
    return centers, max.(density, 0)
end

"""Smoothed density of a metadata column, one curve per area, and one tick per unit below."""
function plot_density_by_area!(p, units, column, lo, hi)
    for (a, color) in area_colors
        x = units[column][units[:area] .== a]  # values of the units in this area
        centers, density = smoothed_density(x, lo, hi)
        plot!(p, centers, density, color = color, lw = 2, fillrange = 0, fillalpha = 0.15,
              label = "$a (n=$(length(x)))")
        scatter!(p, x, fill(-0.2, length(x)), markershape = :vline, markersize = 6, color = color,
                 markerstrokecolor = color, alpha = 0.6, label = "")  # rug
    end
    hline!(p, [0], color = :black, lw = 0.5, label = "")
    plot!(p, ylabel = "Density")
end

# Colors and markers of the example units in the population plots (not the colors of the areas)
highlight = Dict("unit A" => (:black, :circle), "unit B" => (TAB[4], :rect))

# Left: smoothed density of B in the ITIs, one curve per area
p1 = plot(xlabel = "Burstiness B in the ITIs", title = "Burstiness of PFC units in the ITIs")
plot_density_by_area!(p1, valid, :burstiness_iti, -0.5, 0.5)
for (name, i) in examples  # where the example units are
    color, marker = highlight[name]
    vline!(p1, [pfc[:burstiness_iti, id = i]], color = color, ls = :dash, lw = 1.5, label = name)
end

# Right: B against the ITI rate, one dot per unit
p2 = plot(xscale = :log10, xlabel = "Rate in the ITIs (Hz, log scale)",
          ylabel = "Burstiness B in the ITIs", title = "Burstiness vs rate")
for (a, color) in area_colors
    sel = valid[:area] .== a
    scatter!(p2, valid[:rate_iti][sel], valid[:burstiness_iti][sel], markersize = 3, color = color,
             markerstrokewidth = 0, label = a)
end
for (name, i) in examples  # the two example units
    color, marker = highlight[name]
    scatter!(p2, [pfc[:rate_iti, id = i]], [pfc[:burstiness_iti, id = i]], markershape = marker,
             markersize = 9, markercolor = RGBA(1, 1, 1, 0), markerstrokecolor = color,
             markerstrokewidth = 2, label = name)
end
hline!(p2, [0], color = :black, ls = :dot, label = "")
fig = plot(p1, p2, layout = (1, 2), size = (1300, 450), legendfontsize = 7, left_margin = 5Plots.mm,
           bottom_margin = 5Plots.mm)

# %% [markdown]
# Most units have $B$ slightly below 0. Remember the Poisson check of section 4.3: in 3 s windows,
# random firing already gives $B < 0$, and even more so for slow units, which fire fewer spikes per
# window. So $B < 0$ here does **not** mean that these units fire regularly.
#
# **Questions.**
#
# - Does burstiness depend on the area? On the rate? On the cell type (`cell_type`: FS or RS)?
# - Compare the ITI burstiness with the burstiness over the whole session
#   (`burstiness(values(time_diff(pfc[id = i])))`). Which units change the most?
# - Change `min_isi` and `min_windows`. How does it change the results?

# %% [markdown]
# ### 4.5 Advanced: correcting the bias of short windows
#
# How do we know whether a unit is more bursty or more regular than random firing, if the
# reference is not $B = 0$? We build a **reference for each unit: random spike timing**.
# For every valid window, we keep the number of spikes the unit fired, but we throw these
# spikes at random times inside the window, and compute $B$. We repeat this many times and
# average: this is the $B$ that the unit would have if its spikes were placed at random,
# *with the same number of spikes in each window*. It takes into account the rate of the unit
# and how it changes from window to window. The **corrected burstiness** is the difference:
#
# $$ B_\mathrm{corrected} = B_\mathrm{ITI} - B_\mathrm{random\ timing} $$
#
# $B_\mathrm{corrected} > 0$: more bursty than random; $B_\mathrm{corrected} < 0$: more regular.

# %%
"""Mean B over the valid windows, after placing each window's spikes at random times."""
function random_timing_burstiness(spikes, windows, min_isi = 4, n_rep = 20)
    values = Float64[]
    for k in 1:length(windows)
        n = length(restrict(spikes, windows[k]))  # number of spikes in window k
        n - 1 < min_isi && continue  # same rule as for the real data: at least min_isi ISIs
        start, stop = starts(windows)[k], stops(windows)[k]
        # n_rep rows of n random spike times inside the window, sorted along each row
        t = sort(start .+ (stop - start) .* rand(rng, n_rep, n), dims = 2)
        isi = diff(t, dims = 2)  # ISIs of each repetition
        m, s = mean(isi, dims = 2), std(isi, dims = 2, corrected = false)  # one per repetition
        push!(values, mean((s .- m) ./ (s .+ m)))  # B averaged over the repetitions
    end
    return isempty(values) ? NaN : mean(values)
end

for (name, i) in examples
    B_rand = random_timing_burstiness(pfc[id = i], iti, min_isi)
    b = mean(B_windows[name])
    @printf("%s: B in ITIs %+.3f, random timing %+.3f, corrected %+.3f\n", name, b, B_rand, b - B_rand)
end

# %% [markdown]
# We compute the corrected burstiness of every unit with enough windows (section 4.4), and draw
# the same two plots.

# %%
B_rand = [n >= min_windows ? random_timing_burstiness(u, iti, min_isi) : NaN
          for (u, n) in zip(pfc, pfc[:n_valid_windows])]
pfc[:burstiness_random] = B_rand
pfc[:burstiness] = pfc[:burstiness_iti] .- B_rand  # corrected burstiness

valid = pfc[.!isnan.(pfc[:burstiness])]  # same units as in section 4.4
println("median B in ITIs (not corrected): ", round(median(valid[:burstiness_iti]), digits = 3))
println("median corrected B              : ", round(median(valid[:burstiness]), digits = 3))
println("more bursty than random timing  : ", sum(valid[:burstiness] .> 0), " units")
println("more regular than random timing : ", sum(valid[:burstiness] .< 0), " units")

p1 = plot(xlabel = "Corrected burstiness in the ITIs", title = "Corrected burstiness of PFC units")
plot_density_by_area!(p1, valid, :burstiness, -0.5, 0.5)
for (name, i) in examples
    color, marker = highlight[name]
    vline!(p1, [pfc[:burstiness, id = i]], color = color, ls = :dash, lw = 1.5, label = name)
end
vline!(p1, [0], color = :black, ls = :dot, label = "random timing")

p2 = plot(xscale = :log10, xlabel = "Rate in the ITIs (Hz, log scale)",
          ylabel = "Corrected burstiness", title = "Corrected burstiness vs rate")
for (a, color) in area_colors
    sel = valid[:area] .== a
    scatter!(p2, valid[:rate_iti][sel], valid[:burstiness][sel], markersize = 3, color = color,
             markerstrokewidth = 0, label = a)
end
for (name, i) in examples
    color, marker = highlight[name]
    scatter!(p2, [pfc[:rate_iti, id = i]], [pfc[:burstiness, id = i]], markershape = marker,
             markersize = 9, markercolor = RGBA(1, 1, 1, 0), markerstrokecolor = color,
             markerstrokewidth = 2, label = name)
end
hline!(p2, [0], color = :black, ls = :dot, label = "")
fig = plot(p1, p2, layout = (1, 2), size = (1300, 450), legendfontsize = 7, left_margin = 5Plots.mm,
           bottom_margin = 5Plots.mm)

# %% [markdown]
# Units **above 0** fire more in bursts than random spikes with the same counts;
# units **below 0** fire more regularly. Compare with section 4.4: the correction shifts the
# values up, especially for the slow units.

# %% [markdown]
# ### 4.6 Refractory period violations
#
# A real neuron cannot fire two spikes less than ~1.5 ms apart. If a unit has many ISIs
# shorter than that, its spikes probably come from **more than one neuron**
# (bad isolation, contamination). This is one of the classic quality metrics used to
# decide whether a unit is "good".
#
# **Exercise.** Compute the percentage of violations for all the `pfc` units.
# Are some units still contaminated? Compare with the `isi_fp` column
# (ISI violation false-positive rate) that we used to filter the units in section 2.

# %%
refractory = 0.0015  # 1.5 ms, in seconds

for (name, _) in examples
    isi = isis[name]
    n_viol = sum(isi .< refractory)  # number of ISIs shorter than 1.5 ms
    @printf("%s: %d ISIs < 1.5 ms out of %d (%.3f %%)\n", name, n_viol, length(isi),
            100 * n_viol / length(isi))
end

# %% [markdown]
# ## 5. OPTIONAL: Memory: does an ISI predict the next one?
#
# *This whole section is optional: do it if you have time, or come back to it later.*
#
# Burstiness only looks at the **distribution** of the ISIs: it does not change if we
# shuffle the ISIs in a different order. But the **order** matters too. Two spike trains can
# have exactly the same ISIs, but in one of them short ISIs tend to follow short ones
# (spikes come in clusters), while in the other short and long ISIs alternate.
#
# [Goh & Barabási (2008)](https://doi.org/10.1209/0295-5075/81/48002) measure this with the
# **memory coefficient** $M$: the correlation coefficient between each ISI and the next one,
#
# $$ M = \frac{1}{n - 1} \sum_{k=1}^{n-1} \frac{(\tau_k - m_1)(\tau_{k+1} - m_2)}{\sigma_1 \sigma_2} $$
#
# where $\tau_k$ are the ISIs, $m_1, \sigma_1$ the mean and standard deviation of
# $\tau_1 \dots \tau_{n-1}$, and $m_2, \sigma_2$ those of $\tau_2 \dots \tau_n$.
# (Goh & Barabási call it $\mu$; we write $M$ to avoid confusion with the mean ISI.)
#
# - $M > 0$: a short ISI tends to be followed by a short one, and a long by a long one;
# - $M = 0$: no memory, each ISI is independent of the previous one (Poisson);
# - $M < 0$: short and long ISIs tend to alternate.
#
# As for burstiness, we stay in the **3 s ITI windows**, and use the windows with at
# least 4 ISIs. We only pair two ISIs from the **same window**: the time between two windows
# is not an ISI of the neuron at rest.

# %%
"""Vector of the ISI vectors of the windows that have at least min_isi ISIs."""
function isis_per_window(spikes, windows, min_isi = 4)
    out = Vector{Float64}[]
    for k in 1:length(windows)
        isi = diff(times(restrict(spikes, windows[k])))  # ISIs inside window k only
        length(isi) >= min_isi && push!(out, isi)
    end
    return out
end

"""Memory coefficient: correlation between consecutive ISIs, pooled over windows."""
function memory(isi_list)
    current = reduce(vcat, [isi[1:end-1] for isi in isi_list])  # tau_k, from every window
    following = reduce(vcat, [isi[2:end] for isi in isi_list])  # tau_(k+1), from the same window
    return cor(current, following)  # Pearson correlation coefficient
end

# %% [markdown]
# ### 5.1 Burstiness and memory are two different things
#
# Goh & Barabási illustrate the difference with synthetic spike trains (their Fig. 1).
# We rebuild it here. All trains last 50 s and have a mean ISI of 1 s (rate 1 Hz):
#
# - **a**: Poisson train, exponential ISIs ($B \approx 0$, $M \approx 0$);
# - **b**: bursty through the **ISI distribution**: a heavy-tailed distribution, many very short
#   and a few very long ISIs ($B > 0$);
# - **c**: regular through the **ISI distribution**: ISIs all close to 1 s ($B < 0$);
# - **d**: the **same ISIs as a**, reordered so that short ISIs follow short ones ($M > 0$):
#   it looks bursty, although its ISI distribution (and so $B$) is exactly the one of a;
# - **e**: the **same ISIs as a**, reordered so that short and long ISIs alternate ($M < 0$):
#   it looks more regular, with the same $B$ as a.
#
# To reorder the ISIs, we swap two ISIs at random many times, and keep a swap only if it
# brings $M$ closer to the value we want.

# %%
T = 50  # duration of the synthetic spike trains, in seconds

"""Keep the ISIs whose cumulative sum (the spike times) stays below T."""
keep_first_T(isi) = isi[cumsum(isi) .< T]

"""Reorder the ISIs by random swaps until their memory coefficient gets close to target."""
function reorder_isis(isi, target; n_steps = 5000)
    isi = copy(isi)
    for _ in 1:n_steps
        i, j = rand(rng, 1:length(isi), 2)  # two random positions
        new = copy(isi)
        new[i], new[j] = isi[j], isi[i]  # swap them
        if abs(memory([new]) - target) < abs(memory([isi]) - target)  # closer to target?
            isi = new
        end
    end
    return isi
end

n_draw = 200  # draw more ISIs than needed; keep_first_T keeps those that fit in 50 s
isi_a = keep_first_T(-log.(rand(rng, n_draw)))  # a: Poisson, exponential ISIs (mean 1 s)
isi_b = exp.(1.5 .* randn(rng, n_draw))  # b: heavy-tailed (log-normal) ISI distribution...
isi_b = keep_first_T(isi_b ./ mean(isi_b))  # ...rescaled to a mean of 1 s
isi_c = keep_first_T(1.0 .+ 0.1 .* randn(rng, n_draw))  # c: regular, Gaussian ISIs
isi_d = reorder_isis(isi_a, 0.6)  # d: same ISIs as a, positive memory
isi_e = reorder_isis(isi_a, -0.6)  # e: same ISIs as a, negative memory

trains = ["a" => isi_a, "b" => isi_b, "c" => isi_c, "d" => isi_d, "e" => isi_e]

panels = Dict()
for (label, isi) in trains
    p = plot(xlims = (0, T), ylims = (0, 1), yticks = false, titlefontsize = 10,
             title = @sprintf("%s:  B = %+.2f,  M = %+.2f", label, burstiness(isi), memory([isi])))
    for t in cumsum(isi)  # one vertical line per spike
        plot!(p, [t, t], [0, 1], color = :black, lw = 0.8, label = "")
    end
    panels[label] = p
end
plot!(panels["c"], xlabel = "t (s)")
plot!(panels["e"], xlabel = "t (s)")
# Layout of the original figure: a on top, b and c on the left, d and e on the right
fig = plot(panels["a"], panels["b"], panels["d"], panels["c"], panels["e"], size = (1200, 450),
           layout = @layout([_ a{0.5w} _; b d; c e]), bottom_margin = 3Plots.mm)

# %% [markdown]
# a, d and e have **exactly the same ISIs**, so the same $B$, but they look very different:
# the order of the ISIs, measured by $M$, matters. (With only ~50 ISIs, the values of a, b and c
# are not exactly 0 for $M$, or for $B$ in a: small samples are noisy. Rerun the cell to see
# them change.)
#
# **In our data.** We add a third example unit, **unit C** (`302`, ACAd5, met in section 3.7), which has almost
# the same burstiness as unit B but a **negative** memory. Below, we draw the 3 s ITI windows
# with the most spikes for units B and C, as in the figure above. Next to them, the same
# windows after **shuffling the order of their ISIs**: same ISIs, no memory.

# %%
memory_units = [examples; "unit C" => 302]  # name => unit id: units A, B and C
colors["unit C"] = TAB[3]  # color of unit C in the next figures

isi_windows = Dict(name => isis_per_window(pfc[id = i], iti, min_isi) for (name, i) in memory_units)
for (name, i) in memory_units
    isi_list = isi_windows[name]
    n_pairs = sum(length(isi) - 1 for isi in isi_list)  # number of (tau_k, tau_k+1) pairs
    @printf("%s (%d, %s): B in ITIs = %+.3f, %d pairs of consecutive ISIs, M = %+.3f\n", name, i,
            pfc[:location, id = i], pfc[:burstiness_iti, id = i], n_pairs, memory(isi_list))
end

n_show = 4  # number of windows to draw per unit
panels = []
for name in ["unit B", "unit C"]
    isi_list = isi_windows[name]
    busiest = sortperm(length.(isi_list))[end-n_show+1:end]  # windows with the most ISIs
    p_orig = plot(yticks = false, ylabel = name)
    p_shuf = plot(yticks = false)
    for (k, w) in enumerate(busiest)
        isi = isi_list[w]
        original = [0; cumsum(isi)]  # spike times from the window's first spike
        shuffled = [0; cumsum(shuffle(rng, isi))]  # same ISIs, random order
        for (p, spikes) in ((p_orig, original), (p_shuf, shuffled))
            for t in spikes
                plot!(p, [t, t], [k, k + 0.8], color = colors[name], lw = 0.8, label = "")
            end
        end
    end
    push!(panels, p_orig, p_shuf)
end
plot!(panels[1], title = "ISIs in their original order")
plot!(panels[2], title = "same ISIs, shuffled order")
plot!(panels[3], xlabel = "Time from the first spike of the window (s)")
plot!(panels[4], xlabel = "Time from the first spike of the window (s)")
fig = plot(panels..., layout = (2, 2), size = (1200, 400), link = :x, left_margin = 5Plots.mm,
           bottom_margin = 5Plots.mm)

# %% [markdown]
# Unit B fires **bursts of several spikes**: short ISIs come in a row. Unit C mostly fires
# **pairs of spikes** (doublets) followed by a pause: a short ISI is followed by a long one.
# After shuffling, the short ISIs of unit B no longer come in a row: its long bursts break
# into smaller pieces, although the ISIs are exactly the same.

# %% [markdown]
# ### 5.2 Return map
#
# To *see* the memory, we plot each ISI against the next one: this is a **return map**.
# Each dot is one pair $(\tau_k, \tau_{k+1})$. We use logarithmic axes because ISIs span
# several orders of magnitude. Without memory, the cloud has no particular orientation;
# with $M > 0$, it is stretched along the diagonal.

# %%
panels = []
for (name, _) in memory_units
    isi_list = isi_windows[name]
    current = reduce(vcat, [isi[1:end-1] for isi in isi_list])
    following = reduce(vcat, [isi[2:end] for isi in isi_list])
    p = scatter(current, following, markersize = 1.5, markerstrokewidth = 0, alpha = 0.3,
                color = colors[name], label = "", xscale = :log10, yscale = :log10,
                xlabel = "ISI k (s)", ylabel = name == memory_units[1][1] ? "ISI k+1 (s)" : "",
                title = @sprintf("%s, M = %+.2f", name, memory(isi_list)))
    plot!(p, [1e-3, 3], [1e-3, 3], color = :black, ls = :dot, lw = 1, label = "")  # diagonal
    push!(panels, p)
end
fig = plot(panels..., layout = (1, 3), size = (1500, 500), link = :all, left_margin = 5Plots.mm,
           bottom_margin = 5Plots.mm)

# %% [markdown]
# - **Unit B**: the dots gather in the bottom-left corner (a short ISI followed by a short one:
#   inside a burst) and along the diagonal.
# - **Unit C**: the map has the shape of a **cross**: a short ISI (~10 ms) is followed by a
#   long one (~100 ms), and a long one by a short one. Few pairs of short ISIs: no long bursts.
#
# ### 5.3 Memory of all PFC units, and the burstiness–memory plane
#
# We compute $M$ for the units that passed all the criteria (at least 4 ISIs in at least
# 12 windows, section 4.4). Then, following Goh & Barabási (their Fig. 3), we place every unit
# in the **(memory, burstiness) plane**: the two numbers describe two different ways in which a
# spike train can depart from a Poisson process. As in section 4.4, the left plot is the smoothed
# density of $M$ in each area, and the example units have their own colors and markers.

# %%
pfc[:memory_iti] = [n >= min_windows ? memory(isis_per_window(u, iti, min_isi)) : NaN
                    for (u, n) in zip(pfc, pfc[:n_valid_windows])]
valid = pfc[.!isnan.(pfc[:memory_iti])]  # same units as in section 4.4
println("units: ", length(valid))
println("median memory: ", round(median(valid[:memory_iti]), digits = 3))
println("units with positive memory: ", sum(valid[:memory_iti] .> 0), " out of ", length(valid))

highlight["unit C"] = (TAB[6], :diamond)  # color and marker of unit C in the population plots

"""Left: smoothed density of the memory per area. Right: the (memory, burstiness) plane."""
function plot_memory_burstiness(valid, m_column, b_column, m_label, b_label, title)
    p1 = plot(xlabel = m_label, title = title)
    plot_density_by_area!(p1, valid, m_column, -0.4, 0.6)
    for (name, i) in memory_units  # where the example units are
        color, marker = highlight[name]
        vline!(p1, [pfc[m_column, id = i]], color = color, ls = :dash, lw = 1.5, label = name)
    end
    vline!(p1, [0], color = :black, ls = :dot, label = "no memory")

    p2 = plot(xlabel = m_label, ylabel = b_label, title = "Burstiness–memory plane")
    for (a, color) in area_colors
        sel = valid[:area] .== a
        scatter!(p2, valid[m_column][sel], valid[b_column][sel], markersize = 4, color = color,
                 markerstrokewidth = 0, label = a)
    end
    for (name, i) in memory_units  # the three example units
        color, marker = highlight[name]
        scatter!(p2, [pfc[m_column, id = i]], [pfc[b_column, id = i]], markershape = marker,
                 markersize = 9, markercolor = RGBA(1, 1, 1, 0), markerstrokecolor = color,
                 markerstrokewidth = 2, label = name)
    end
    hline!(p2, [0], color = :black, ls = :dot, label = "")
    vline!(p2, [0], color = :black, ls = :dot, label = "")
    return plot(p1, p2, layout = (1, 2), size = (1300, 500), legendfontsize = 7,
                left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)
end

fig = plot_memory_burstiness(valid, :memory_iti, :burstiness_iti, "Memory M in the ITIs",
                             "Burstiness B in the ITIs", "Memory of PFC units in the ITIs")
println("correlation between B and M: ",
        round(cor(valid[:burstiness_iti], valid[:memory_iti]), digits = 3))
fig

# %% [markdown]
# Most units have a **positive memory**: short ISIs tend to follow short ones. But as for $B$,
# be careful: pooling many short windows biases $M$ (see section 5.4).
#
# ### 5.4 Advanced: a reference for memory, shuffling the ISIs
#
# Like $B$, $M$ computed over many short windows is not 0 for random firing. When we pool the
# pairs of all windows, windows where the neuron fires fast give pairs of short ISIs, and
# windows where it fires slowly give pairs of long ISIs: this creates a positive correlation
# even if, inside a window, the order of the ISIs is random.
#
# To remove this effect, we **shuffle the order of the ISIs inside each window** and compute
# $M$ again. Shuffling keeps exactly the same ISIs in every window (same rate, same
# burstiness) and only destroys their order. Averaged over many shuffles, this gives the
# memory expected *without* memory. The **corrected memory** is the difference:
#
# $$ M_\mathrm{corrected} = M - M_\mathrm{shuffled} $$
#
# This is how Goh & Barabási illustrate memory too (their Fig. 1d,e): same ISIs, different order.

# %%
"""Mean memory coefficient after shuffling the ISI order inside each window."""
function memory_shuffled(isi_list, n_rep = 20)
    values = Float64[]
    for _ in 1:n_rep
        shuffled = [shuffle(rng, isi) for isi in isi_list]  # new random order in each window
        push!(values, memory(shuffled))
    end
    return mean(values)
end

for (name, _) in memory_units
    isi_list = isi_windows[name]
    M, M_shuf = memory(isi_list), memory_shuffled(isi_list)
    @printf("%s: M = %+.3f, shuffled = %+.3f, corrected = %+.3f\n", name, M, M_shuf, M - M_shuf)
end

# Sanity check with the Poisson spike train: no memory at all
poisson_list = isis_per_window(poisson, iti, min_isi)
@printf("Poisson: M = %+.3f, shuffled = %+.3f\n", memory(poisson_list), memory_shuffled(poisson_list))

# %% [markdown]
# We compute the corrected memory of every unit, and draw the same plots with the **corrected**
# memory and the **corrected** burstiness (this needs the advanced section 4.5).

# %%
M_corr = fill(NaN, length(pfc))  # one value per unit, NaN for units without enough windows
for (k, u) in enumerate(pfc)
    if pfc[:n_valid_windows][k] >= min_windows
        isi_list = isis_per_window(u, iti, min_isi)
        M_corr[k] = memory(isi_list) - memory_shuffled(isi_list)
    end
end
pfc[:memory] = M_corr  # new metadata column: corrected memory

valid = pfc[.!isnan.(pfc[:memory])]  # same units as in section 4.4
println("median memory (not corrected): ", round(median(valid[:memory_iti]), digits = 3))
println("median corrected memory      : ", round(median(valid[:memory]), digits = 3))
println("units with positive corrected memory: ", sum(valid[:memory] .> 0), " out of ", length(valid))

fig = plot_memory_burstiness(valid, :memory, :burstiness, "Corrected memory M", "Corrected burstiness B",
                             "Corrected memory of PFC units")
println("correlation between corrected B and corrected M: ",
        round(cor(valid[:burstiness], valid[:memory]), digits = 3))
fig

# %% [markdown]
# Almost all units have a **positive memory**: in the ITIs, short ISIs tend to follow short
# ones, beyond what the window-to-window changes of rate explain. Memory and burstiness are
# barely correlated: they capture **two different properties** of the spike trains, so it is
# worth measuring both.
#
# **Questions.**
#
# - Is the memory different between areas? Between FS and RS units?
# - The correlation coefficient is dominated by the long ISIs. Compute $M$ on the logarithm
#   of the ISIs (`log.(isi)`) instead. Does the picture change?
# - Units in the top-right corner are both bursty and with memory. Look at their raster and
#   their return map: what do they have in common?
