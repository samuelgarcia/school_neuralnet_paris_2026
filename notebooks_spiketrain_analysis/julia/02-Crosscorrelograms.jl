# %% [markdown]
# # 02 · Cross-correlograms and functional connectivity — **Julia / Judananas**
#
# The brain is a network of neurons, and we can study it through its **connectivity**:
#
# - **structural connectivity** is the physical wiring: which neuron makes a synapse onto which
#   other one. It changes slowly (hours to days: plasticity, growth);
# - **functional connectivity** is the part of this wiring that is **active**: which neurons
#   influence each other, here and now. It can change within seconds, with the state of the animal
#   or the task.
#
# With extracellular recordings we cannot see the synapses, but we can see their effect on the spike
# trains: if neuron $i$ excites neuron $j$, then $j$ fires more often a few milliseconds after $i$.
# The **cross-correlogram** (CCG) measures exactly this, and it is the classic, "gold standard" tool
# to infer functional connections from spike trains.
#
# The difficulty is the **scale**: with Neuropixels probes we record hundreds of neurons at once, and
# the number of pairs grows as the square of the number of neurons (400 neurons: 160,000 ordered
# pairs). Testing each CCG for significance multiplies this cost again. So we need both the
# **theory** (what does a CCG count, how do we know a peak is real?) and **efficient tools**.
#
# This notebook is based on the internship work of Matthieu Trollé (ENS de Lyon, 2026), *Efficient
# cross-correlogram computations for large-scale functional connectivity analysis*. It is the Julia
# version of `python/02-Crosscorrelograms.ipynb`: same steps, same text, same figures. We:
#
# 1. build the CCG from its definition, and see three ways to compute it (counting, a sliding dot
#    product, the Fourier transform), and what they cost;
# 2. compute autocorrelograms and cross-correlograms, and learn to read them;
# 3. see how a shared response to a stimulus creates correlations;
# 4. correct the CCG with jittered spike trains, and test the significance of a connection;
# 5. scan **all pairs** of a recording, build a connectivity matrix, and (advanced) compare the
#    networks between the blocks of the task.
#
# **Dataset.** Session `156131_20191112-probe0` of the same study as notebook 01
# ([Calvigioni, Fuzik, Le Merre et al., 2023](https://doi.org/10.1038/s41593-023-01367-8)): a
# Vglut2-cre mouse, with a Neuropixels probe through the prefrontal cortex (prelimbic, infralimbic,
# orbital, anterior cingulate areas) and nearby regions. The session is made of **blocks** of trials:
#
# | Block | Trials | Stimulus |
# |---|---|---|
# | 1 | 50 | sound (pure tone) |
# | 2 | 100 | sound, followed by an optogenetic stimulation of the lateral hypothalamus → lateral habenula pathway (aversive) |
# | 3 | 50 | sound (as block 1) |
# | 4 | 50 | blue noise, followed by an air puff to the eye (aversive) |
# | 5–12 | 340 | optogenetic stimulation only |
#
# ## 0. Get the data
#
# The session is on the [DANDI Archive](https://dandiarchive.org), in
# [Dandiset 000473](https://dandiarchive.org/dandiset/000473): we use the file
# `sub-156131/sub-156131_ses-156131-20191112-probe0_ecephys+ogen.nwb` (5.6 GB).
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
# Plots: figures; Statistics: means, medians...; StatsBase: histograms; FFTW: Fourier transforms;
# Printf: formatted printing; Random: random numbers
using Judananas, HDF5
using Plots, Statistics, StatsBase, FFTW, Printf, Random
gr()
default(left_margin = 8Plots.mm, bottom_margin = 5Plots.mm)  # room for the axis labels
TAB = palette(:tab10)  # the same 10 colors as matplotlib: blue, orange, green, red, purple, brown...

# Full path to the NWB file you downloaded. Change it to where you saved it.
NWB = "/path/to/your/data/folder/sub-156131_ses-156131-20191112-probe0_ecephys+ogen.nwb"

isfile(NWB) || error("No NWB file at $NWB: set NWB to the downloaded file")

# %% [markdown]
# **Load the units and keep the good ones**, with the same quality criteria as in notebook 01
# (section 2): ISI violation ratio < 0.1, a plausible waveform (FS or RS), and presence ratio > 0.
# This time we keep the units of **all the brain regions** crossed by the probe, because we want to
# see connections within and between regions. Judananas reads all the columns we need from the
# units table: `quality`, `FS`, `RS`, the `electrode` of each unit, its `location` and its depth
# `DV` (dorso-ventral coordinate, in mm): two units on nearby electrodes are close in the brain.

# %%
data = load_file(NWB)  # open the file and list its content
units = data["units"]  # all sorted units (TsGroup)
trials = data["trials"]  # task trials (IntervalSet), with a "Block" column

units[:isi_fp] = [q[2] for q in units[:quality]]  # ISI violation false-positive rate
units[:cell_type] = [f == 1 ? "FS" : r == 1 ? "RS" : "none" for (f, r) in zip(units[:FS], units[:RS])]
units[:depth] = units[:DV]  # dorso-ventral coordinate of the electrode (mm)
units[:presence_ratio] = vec(mean(values(count(units, tot_length(time_support(units)) / 100)) .> 0,
                                  dims = 1))

good = units[(units[:isi_fp] .< 0.1) .& (units[:cell_type] .!= "none") .& (units[:presence_ratio] .> 0)]
println("good units: ", length(good), " out of ", length(units))
println(value_counts(good, :cell_type))
value_counts(good, :location)  # how many units per brain area

# %% [markdown]
# ## 1. Theoretical background
#
# ### 1.1 The definition: counting coincidences
#
# Take two spike trains: a **reference** neuron $i$ and a **target** neuron $j$. Each spike of $i$
# is taken in turn as the origin of time, and we place the spikes of $j$ relative to it: $t_j - t_i$.
# A negative value means that $j$ fired before $i$, a positive value that $j$ fired after $i$. The CCG
# is the **histogram of all these time differences** (the "lags"), in a window (e.g. ±50 ms):
#
# $$ \mathrm{CCG}_{ij}(\tau) = \text{number of pairs of spikes } (t_i, t_j) \text{ with } t_j - t_i \text{ in the bin around } \tau $$
#
# Each count is a **coincidence**: $j$ fired at lag $\tau$ after a spike of $i$. If $i$ excites $j$,
# then after many spikes of $i$ there is often a spike of $j$ just after: the CCG has a peak just
# after 0. Let's build it on two short simulated spike trains, where neuron $j$ tends to fire
# **2 ms after** neuron $i$, plus random spikes of its own.

# %%
rng = Xoshiro(0)  # random generator with a fixed seed: same result for everyone

T_toy = 20.0  # duration of the toy spike trains, in seconds
toy_i = sort(T_toy .* rand(rng, 100))  # neuron i: 100 random spikes (5 Hz)
followed = toy_i[rand(rng, length(toy_i)) .< 0.5]  # about half of the spikes of i...
follow = followed .+ 0.002 .+ 0.0003 .* randn(rng, length(followed))  # ...are followed by j, 2 ms later
toy_j = sort([follow; T_toy .* rand(rng, 100)])  # neuron j: followers + random spikes

bin_centers(edges) = (edges[1:end-1] .+ edges[2:end]) ./ 2  # bin centers, from the bin edges

# Bin edges from -20 to +20 ms, 1 ms bins, one bin centered on 0 (like pynapple)
edges = correlogram_edges(0.001, 0.02)

# The definition: all the differences t_j - t_i within the window, and their histogram
diffs = spike_time_differences(toy_i, toy_j, edges[end])
counts = fit(Histogram, diffs, edges).weights

# The same in one call: compute_crosscorrelogram(ref, target) counts the target relative to the ref.
# With counts = true it returns the number of coincidences per bin; h.lags are the bin centers (s).
h = compute_crosscorrelogram(Ts(toy_i), Ts(toy_j); edges = edges, counts = true)
println("histogram of the differences == compute_crosscorrelogram: ", counts == h.values)
centers = h.lags

p1 = scatter(toy_i, fill(1, length(toy_i)), markershape = :vline, markersize = 14, color = TAB[1],
             label = "", xlims = (0, 2), ylims = (-0.6, 1.6), yticks = ([1, 0], ["neuron i", "neuron j"]),
             xlabel = "Time (s)", title = "2 s of the two spike trains")
scatter!(p1, toy_j, fill(0, length(toy_j)), markershape = :vline, markersize = 14, color = TAB[2],
         label = "")
p2 = bar(centers .* 1000, counts, bar_width = 1, color = :black, label = "",
         xlabel = "Time of j relative to i (ms)", ylabel = "Number of coincidences",
         title = "Cross-correlogram, i → j")
vline!(p2, [0], color = :red, lw = 0.8, label = "")
fig = plot(p1, p2, layout = grid(1, 2, widths = [0.56, 0.44]), size = (1300, 350))

# %% [markdown]
# The raster alone does not tell much, but the CCG shows a clear **peak 2 ms after 0**. Outside the
# peak, the bins contain the coincidences **by chance**: with random spikes, their number is about
# the same in every bin.
#
# **Expected count by chance.** If the two neurons were independent, each spike of $i$ would see on
# average $\text{rate}_j \times \text{bin size}$ spikes of $j$ in each bin. So the flat level of the
# CCG is
#
# $$ \text{expected count} = N_i \times \text{rate}_j \times \text{bin size} $$
#
# Dividing the CCG by this expected count gives a **normalized CCG**, equal to 1 for independent
# neurons, whatever their rates: 2 means "twice as many coincidences as expected by chance".

# %%
expected = length(toy_i) * (length(toy_j) / T_toy) * 0.001  # N_i x rate_j x bin size
peak = counts[argmin(abs.(centers .- 0.002))]  # count in the 2 ms bin
@printf("expected count by chance: %.2f per bin\n", expected)
@printf("count in the 2 ms bin: %d (%.0f times more than by chance)\n", peak, peak / expected)

# %% [markdown]
# ### 1.2 The CCG as a sliding dot product
#
# Another way to see the CCG: cut time into small bins (here 1 ms) and write each spike train as a
# vector of spike counts, $A(t)$ for neuron $i$ and $B(t)$ for neuron $j$ (mostly 0, sometimes 1).
# Shift $B$ by a lag $\tau$, multiply it with $A$ bin by bin, and sum:
#
# $$ \mathrm{CCG}_{ij}(\tau) = \sum_t A(t) \, B(t + \tau) $$
#
# A product $A(t) B(t+\tau)$ is 1 exactly when $i$ fired in bin $t$ and $j$ in bin $t + \tau$: a
# coincidence at lag $\tau$. This is a **sliding dot product** (the cross-correlation of signal
# processing). It gives the same CCG as counting, if we first round the spike times to the bins.
#
# Let's check it on a real pair: **unit 339** (RS, putative pyramidal neuron) and **unit 98** (FS,
# putative interneuron), recorded on nearby electrodes in the orbital area (ORBm6a), during the first
# 10 minutes of the session. (`good[id = 339]` selects a unit by its id.)

# %%
ref_all, target_all = times(good[id = 339]), times(good[id = 98])  # spike times (s)
T_short = 600.0  # the first 10 minutes
ref, target = ref_all[ref_all .< T_short], target_all[target_all .< T_short]
binsize, n_lags = 0.001, 20  # 1 ms bins, lags from -20 to +20 ms
lags = -n_lags:n_lags
n_bins = round(Int, T_short / binsize)

# Binned spike trains: number of spikes in each 1 ms bin of the 10 minutes
A = zeros(Int, n_bins); B = zeros(Int, n_bins)
for t in ref; A[floor(Int, t / binsize) + 1] += 1; end
for t in target; B[floor(Int, t / binsize) + 1] += 1; end

# Sliding dot product: for each lag, multiply A(t) with B(t + lag) and sum over t
time_dot = @elapsed ccg_dot = [sum(A[max(1, 1 - k):n_bins - max(0, k)] .* B[max(1, 1 + k):n_bins - max(0, -k)])
                               for k in lags]

# Counting, on the spike times rounded to the bins: the same coincidences
time_count = @elapsed ccg_count = fit(Histogram,
    spike_time_differences(floor.(ref ./ binsize), floor.(target ./ binsize), n_lags),
    (-n_lags:n_lags+1) .- 0.5).weights

println("dot product == counting: ", ccg_dot == ccg_count)
@printf("time: dot product %.0f ms, counting %.0f ms\n", time_dot * 1000, time_count * 1000)

# %% [markdown]
# Same CCG. But the dot product visits **every bin of the recording for every lag**: its cost grows
# as $T \times M$ (duration in bins × number of lags), even where there is no spike. Counting only
# visits the spikes, and the spikes that fall within the window: spike trains are **sparse**
# (a neuron at 5 Hz has a spike in 0.5 % of the 1 ms bins), so counting is much faster here.
# (Run the cell twice: the first run also compiles the code.)
#
# ### 1.3 The Fourier transform
#
# The sliding dot product can also be computed with the **Fourier transform** $\mathcal{F}$: the
# cross-correlation of two signals is the inverse transform of the product of their transforms
# (one of them complex-conjugated, written $^*$):
#
# $$ \mathrm{CCG}_{ij} = \mathcal{F}^{-1}\left( \mathcal{F}(A)^* \cdot \mathcal{F}(B) \right) $$
#
# With the fast Fourier transform (FFT), this costs about $T \log T$ instead of $T \times M$, and it
# gives **all the lags at once**: it is the method of choice for long windows (seconds) or for
# continuous signals (e.g. the LFP). The signals must be padded with zeros, so that the end of one
# does not wrap around onto the start of the other.

# %%
time_fft = @elapsed begin
    n_fft = nextpow(2, 2 * n_bins)  # zero padding, to a power of 2 (fast)
    padded(x) = [Float64.(x); zeros(n_fft - length(x))]
    full = irfft(conj.(rfft(padded(A))) .* rfft(padded(B)), n_fft)
    ccg_fft = round.(Int, full[mod.(lags, n_fft) .+ 1])  # negative lags are at the end of the array
end

println("FFT == counting: ", ccg_fft == ccg_count)
@printf("time: FFT %.0f ms (all %d lags at once)\n", time_fft * 1000, n_fft)

fig = bar(lags, ccg_count, bar_width = 1, color = :lightgray, linecolor = :lightgray, label = "counting",
          xlabel = "Time of 98 relative to 339 (ms)", ylabel = "Number of coincidences",
          title = "Three ways to compute the same CCG (339 → 98, first 10 min)", size = (800, 350))
scatter!(fig, lags, ccg_dot, color = TAB[1], markersize = 5, markerstrokewidth = 0,
         label = "sliding dot product")
scatter!(fig, lags, ccg_fft, color = TAB[4], markershape = :xcross, markersize = 6,
         label = "Fourier transform")

# %% [markdown]
# The three methods give exactly the same numbers. Which one to use depends on the data:
#
# | Method | Cost | Best for |
# |---|---|---|
# | counting the spike time differences | number of spikes × spikes in the window | sparse spike trains, short windows (connectivity: ±50 ms) |
# | sliding dot product | duration (in bins) × number of lags | (didactic: it is the definition of a cross-correlation) |
# | Fourier transform | duration × log(duration), all lags at once | long windows, dense or continuous signals |
#
# In Matthieu's internship, two pipelines were compared on 31 recordings: a counting method built on
# SpikeInterface (**countCCG**) and an FFT method in Rust and Julia (**fftCCG**). For spike trains,
# counting was the fastest: more than 3 billion pairs in about 10 minutes on a 32-core computer.
#
# ### 1.4 What a CCG can show
#
# Before looking at real CCGs, the typical shapes and what they suggest:
#
# - a **sharp peak 1–5 ms after 0**: $j$ fires just after $i$: a putative **excitatory monosynaptic
#   connection** $i \to j$ (the delay is the time for the spike to travel and for the synapse to act);
# - a **trough 1–10 ms after 0**: $j$ is silenced just after $i$: a putative **inhibitory connection**;
# - a **narrow peak exactly at 0**: no synapse is that fast: the two units often share the *same*
#   spikes (one neuron detected twice by the spike sorter), or are driven by a common input with the
#   same delay;
# - a **broad peak** (tens to hundreds of ms): the two neurons are active in the same periods (common
#   slow inputs, brain state, a shared response to a stimulus): a **co-modulation**, not a synapse.
#
# A CCG shows a **correlation**, not a connection: a short-latency peak could also come from a third
# neuron that drives both units with different delays. This is why we speak of **putative**
# connections.
#
# ## 2. Autocorrelograms
#
# The **autocorrelogram** (ACG) is the CCG of a neuron with itself (the spike itself, at lag 0, is not
# counted). It is symmetric, and close to the ISIs of notebook 01: the ISIs only look at the **next**
# spike, the ACG at **all** the spikes in the window.
#
# Judananas computes it with `compute_autocorrelogram(x; binsize, windowsize, norm)`, with the same
# bins as pynapple (one bin centered on 0):
#
# - with `norm = false`, the result is a **rate** in spikes/s: the rate of the neuron at a time $\tau$
#   after one of its spikes;
# - with `norm = true` (the default), this rate is divided by the mean rate of the neuron: 1 means
#   "as by chance";
# - with `counts = true`, it gives the number of coincidences in each bin.
#
# The result is a named tuple: `h.lags` (the bin centers, in s) and `h.values` (a vector for one unit,
# a matrix lags × units for a `TsGroup`).

# %%
acg_units = ["unit 55 (RS, ILA6a)" => 55, "unit 58 (RS, ILA6a)" => 58, "unit 171 (FS, PL5)" => 171]
acg_colors = [TAB[1], TAB[2], TAB[4]]

panels = []
for ((name, i), color) in zip(acg_units, acg_colors)
    acg = compute_autocorrelogram(good[id = i]; binsize = 0.001, windowsize = 0.1, norm = false)  # spikes/s
    p = bar(acg.lags .* 1000, acg.values, bar_width = 1, color = color, linecolor = color, label = "",
            title = "$name, autocorrelogram", xlabel = "Lag (ms)", titlefontsize = 10,
            ylabel = i == acg_units[1][2] ? "Rate (spikes/s)" : "")
    hline!(p, [good[:rate, id = i]], color = :black, ls = :dash, label = "mean rate",
           legend = :bottomright)
    push!(panels, p)
end
fig = plot(panels..., layout = (1, 3), size = (1500, 350))

# %% [markdown]
# - All three units have a **gap around 0**: the **refractory period**. A unit without this gap is
#   probably a mix of several neurons: the ACG is a classic **quality check** of spike sorting.
# - **Unit 55** almost never fires within ~15 ms of one of its spikes, and fires most about 30 ms
#   later: a long gap followed by a peak, the signature of a **regular** (or rhythmic) neuron.
# - **Unit 58** fires much more than its mean rate in the first ~10 ms after a spike: it fires in
#   **bursts** of spikes close together.
# - **Unit 171** is a fast-spiking interneuron (about 40 spikes/s), with a short refractory period.
#
# ## 3. Cross-correlograms
#
# `compute_crosscorrelogram(ref, target; binsize, windowsize)` gives **the target relative to the
# reference**, as in our definition. With a `TsGroup`, it computes all the pairs `(i, j)` with `i`
# before `j` **in the order of the group**, `j` relative to `i`: `good[id = [339, 98]]` gives the pair
# `(339, 98)`. (Careful if you switch to pynapple: it sorts the units by id, and would give `(98, 339)`.)
#
# Let's check it against the definition (the histogram of the spike time differences), for the pair
# 339 → 98 over the whole session. With `norm = false` it returns a rate, in spikes/s: we multiply by
# $N_i \times \text{bin size}$ to get back the number of coincidences.

# %%
h = compute_crosscorrelogram(good[id = 339], good[id = 98]; binsize = 0.001, windowsize = 0.05,
                             norm = false)  # 98 relative to 339, in spikes/s
println("pairs of a group, in the order of the group: ",
        compute_crosscorrelogram(good[id = [339, 98]]; binsize = 0.001, windowsize = 0.05).pairs)

edges = correlogram_edges(0.001, 0.05)
centers = h.lags
counts = fit(Histogram, spike_time_differences(ref_all, target_all, edges[end]), edges).weights
counts_from_rate = h.values .* length(ref_all) .* 0.001  # rate -> number of coincidences
println("largest difference with the definition: ", round(maximum(abs.(counts_from_rate .- counts)), digits = 6),
        " (out of ", sum(counts), " coincidences)")

# %% [markdown]
# (Spike times are multiples of the sampling period, 1/30000 s, so many spike time differences fall
# **exactly on a bin edge**. Floating-point rounding then decides in which bin they go: two tools
# can disagree by a few coincidences, as we see in the Python notebook between pynapple and
# SpikeInterface.)
#
# ### 3.1 A gallery of cross-correlograms
#
# Here are five pairs of units of this session, each with a typical shape (section 1.4). We use the
# normalized CCG (`norm = true`): 1 = as many coincidences as expected by chance. We draw each pair at
# two time scales.

# %%
gallery = [  # (reference, target, what we see)
    (339, 98, "sharp peak just after 0"),
    (109, 127, "trough just after 0"),
    (98, 339, "peak before 0, trough after 0"),
    (35, 36, "narrow peak at 0"),
    (40, 114, "broad peak"),
]

"""Short description of a unit: id, cell type, area, electrode."""
unit_label(i) = "$i ($(good[:cell_type, id = i]), $(good[:location, id = i]), electrode $(good[:electrode, id = i]))"

panels = []
for (row, (i, j, what)) in enumerate(gallery)
    for (col, (b, w)) in enumerate([(0.001, 0.05), (0.0002, 0.01)])
        cc = compute_crosscorrelogram(good[id = i], good[id = j]; binsize = b, windowsize = w)
        p = bar(cc.lags .* 1000, cc.values, bar_width = b * 1000, color = :black, linecolor = :black, label = "",
                xlabel = row == length(gallery) ? "Time of $j relative to $i (ms)" : "",
                ylabel = col == 1 ? "$i → $j\nnormalized CCG" : "",
                title = col == 1 ? "$(unit_label(i)) → $(unit_label(j)): $what" :
                        (row == 1 ? "zoom: 0.2 ms bins, ±10 ms" : ""),
                titlefontsize = 9, titlelocation = :left)
        hline!(p, [1], color = TAB[3], ls = :dash, label = "")  # expected by chance
        vline!(p, [0], color = :red, lw = 0.8, label = "")
        push!(panels, p)
    end
end
fig = plot(panels..., layout = (length(gallery), 2), size = (1300, 280 * length(gallery)),
           left_margin = 8Plots.mm)

# %% [markdown]
# - **339 → 98**: a sharp peak about 1–2 ms after 0, on one side only: the interneuron 98 fires just
#   after the pyramidal neuron 339. A putative **excitatory** connection 339 → 98.
# - **109 → 127**: a trough a few ms after 0: the RS unit 127 is silenced just after the interneuron
#   109. A putative **inhibitory** connection. (Notice also the excess just *before* 0.)
# - **98 → 339**: the first pair, in the other direction. The excitatory peak is now *before* 0
#   (339 fires before 98), and **after 0 there is a trough**: after a spike of 98, unit 339 fires
#   less. The two neurons are connected **both ways**: 339 excites 98, and 98 inhibits 339, a classic
#   excitatory–inhibitory loop of the cortex.
# - **35 – 36**: two FS units on the **same electrode**, with a narrow peak around 0 (within ±0.5 ms):
#   probably one neuron split into two units by the spike sorter (or two neurons with almost the same
#   spikes).
# - **40 – 114**: two RS units in different areas (DP and ORBm6a), with a broad peak of ±50 ms and
#   nothing special at short lags: a **co-modulation** (shared slow inputs), not a synapse.
#
# **Questions.**
#
# - Draw the autocorrelograms of units 35 and 36, and their CCG with 0.1 ms bins. Is the peak at 0
#   exactly at 0? What would you do with these two units?
# - Units 337, 339 and 334 all excite unit 98 (try them!). What does it tell us about unit 98?

# %% [markdown]
# ## 4. Confounds: a shared response to the stimulus
#
# Two neurons that both respond to the stimulus fire more at the same times **because of the
# stimulus**, even if they are not connected. Their CCG then has a peak. Let's take **unit 164**
# (PL5) and **unit 337** (ORBm6a): they are far apart on the probe, but both respond to the trials.
#
# `compute_eventcorrelogram(x, events; binsize, windowsize)` is a CCG between a unit and a series of
# events: here, the trial onsets. It is the PSTH of notebook 01, in one line.

# %%
pair = [164, 337]
onsets = Ts(starts(trials))  # the trial onsets, as events

fig = plot(size = (800, 350), xlabel = "Time from trial onset (s)", ylabel = "Rate (spikes/s)",
           title = "Event correlogram: both units respond to the trials")
vspan!(fig, [0, 1], color = :gray, alpha = 0.15, label = "")
vline!(fig, [0], color = :red, lw = 0.8, label = "")
for (i, color) in zip(pair, [TAB[5], TAB[6]])
    peth = compute_eventcorrelogram(good[id = i], onsets; binsize = 0.02, windowsize = 1.0, norm = false)
    plot!(fig, peth.lags, peth.values, color = color, lw = 2, label = "unit $i ($(good[:location, id = i]))")
end
fig

# %% [markdown]
# We compute the CCG of 164 → 337 **during the trials only** (the first 2 s after each onset), with
# 10 ms bins and a ±500 ms window. To find the part of the CCG that is due to the stimulus, we use the
# **shift predictor**: we pair the spikes of unit 164 in trial $k$ with the spikes of unit 337 in
# trial $k+1$. Both units still respond to the stimulus in each trial, but any precise interaction
# between the two neurons is gone. What remains is the correlation **explained by the stimulus**.
# We also compute the CCG in the inter-trial intervals (the 3 s before each trial).
#
# `compute_crosscorrelogram` does all this with three keywords: `ep` (the epochs: here, the first
# 2 s of each trial), `same_epoch = true` (only pairs of spikes from the same epoch) and
# `epoch_shift = 1` (epoch $k$ of the reference with epoch $k+1$ of the target: the shift predictor).

# %%
during = IntervalSet(starts(trials), starts(trials) .+ 2)  # 0 to 2 s after each onset
iti = IntervalSet(starts(trials) .- 3, starts(trials))  # the 3 s before each onset
println("trial windows: ", length(during), ", inter-trial windows: ", length(iti))  # no overlap: one per trial

a, b = good[id = pair[1]], good[id = pair[2]]
kw = (binsize = 0.01, windowsize = 0.5, counts = true, same_epoch = true)  # 10 ms bins, ±500 ms
h = compute_crosscorrelogram(a, b; kw..., ep = during)
centers, ccg_trials = h.lags, h.values
ccg_shift = compute_crosscorrelogram(a, b; kw..., ep = during, epoch_shift = 1).values  # trial k vs k+1
ccg_iti = compute_crosscorrelogram(a, b; kw..., ep = iti).values

panels = []
for (k, (y, title)) in enumerate([(ccg_trials, "during the trials"),
                                  (ccg_shift, "shift predictor (trial k vs k+1)"),
                                  (ccg_trials .- ccg_shift, "trials − shift predictor"),
                                  (ccg_iti, "inter-trial intervals")])
    p = bar(centers .* 1000, y, bar_width = 10, color = :black, linecolor = :black, label = "",
            title = title, titlefontsize = 10, xlabel = "Time of $(pair[2]) relative to $(pair[1]) (ms)",
            ylabel = k == 1 ? "Number of coincidences" : "")
    vline!(p, [0], color = :red, lw = 0.8, label = "")
    push!(panels, p)
end
fig = plot(panels..., layout = (1, 4), size = (1700, 350))

# %% [markdown]
# During the trials, the CCG has a broad peak around 0, and the shift predictor has almost the same
# peak: this correlation is explained by the stimulus, which makes both neurons fire at the same
# times in every trial (the **signal correlation**). Once it is removed, only a small, noisy excess
# is left (the **noise correlation**, from trial to trial). In the inter-trial intervals, without any
# stimulus, the CCG still has a broad peak: the two units are also co-modulated at rest (brain state,
# shared slow inputs). Note also the shape of the trial CCGs: they decrease towards ±500 ms because, in
# a 2 s window, there are fewer pairs of spikes far apart in time. The shift predictor has the same
# bias, so the difference removes it.
#
# For **connectivity**, we care about the fast part of the CCG (a few ms), and the slow
# co-modulations (stimulus, brain state) are a nuisance. The next section removes them.

# %% [markdown]
# ## 5. Jitter correction and significance
#
# ### 5.1 The jitter-corrected CCG
#
# To separate the fast interactions from the slow co-modulations, we **jitter** the spikes: we move
# each spike of the target by a random amount, between −25 and +25 ms, and compute the CCG again.
# Jittering keeps everything slower than ~25 ms (the rate changes, the slow co-modulations, the
# responses to the stimulus) and destroys the precise timing at the scale of a few ms. We repeat it
# 40 times and average: this is the **jittered CCG**, the CCG expected *without* fast interactions.
#
# Following [Siegle et al. (2021)](https://doi.org/10.1038/s41586-020-03171-x) and Matthieu's report,
# the **jitter-corrected CCG** is the difference, divided by the geometric mean of the two firing
# rates (so that pairs of fast and slow neurons can be compared):
#
# $$ \mathrm{CCG}_{\text{corrected}, ij} = \frac{\mathrm{CCG}_{\text{real}, ij} - \mathrm{CCG}_{\text{jitter}, ij}}{\sqrt{\text{rate}_i \, \text{rate}_j}} $$

# %%
# Judananas: jitter(x, width) moves each spike by a random time in ±width, and
# jittered_crosscorrelogram(ref, target; jitter, n_rep) averages the CCG over n_rep jitters of the target.
# Bins of this section (own names, so that they do not depend on the cells run before)
jitter_edges = correlogram_edges(0.0005, 0.1)  # 0.5 ms bins, ±100 ms
jitter_lags = bin_centers(jitter_edges)  # bin centers (s)
duration = tot_length(time_support(good))

corrected = Dict()  # (i, j) => jitter-corrected CCG
tops, bottoms = [], []
for (i, j) in [(339, 98), (109, 127)]
    real = compute_crosscorrelogram(good[id = i], good[id = j]; edges = jitter_edges, counts = true).values
    jit = jittered_crosscorrelogram(good[id = i], good[id = j]; jitter = 0.025, n_rep = 40, edges = jitter_edges,
                                    counts = true, rng = rng).values
    corrected[(i, j)] = (real .- jit) ./ sqrt(good[:rate, id = i] * good[:rate, id = j])

    p_top = bar(jitter_lags .* 1000, real, bar_width = 0.5, color = :black, linecolor = :black,
                label = "real CCG", title = "$i → $j", ylabel = "Number of coincidences", xlims = (-30, 30))
    plot!(p_top, jitter_lags .* 1000, jit, color = TAB[3], lw = 2, label = "jittered CCG (±25 ms)",
          legendfontsize = 7)
    p_bottom = bar(jitter_lags .* 1000, corrected[(i, j)], bar_width = 0.5, color = :black, linecolor = :black,
                   label = "", xlims = (-30, 30), xlabel = "Time of $j relative to $i (ms)",
                   ylabel = "jitter-corrected CCG")
    push!(tops, p_top); push!(bottoms, p_bottom)
end
fig = plot(tops..., bottoms..., layout = (2, 2), size = (1400, 600))

# %% [markdown]
# The jittered CCG follows the slow shape of the real CCG, and the corrected CCG is flat at 0 except
# for the fast part: the excitatory peak of 339 → 98 and the inhibitory trough of 109 → 127 stand out.
#
# ### 5.2 Is the connection significant?
#
# We use the rule of Matthieu's report (adapted from Siegle et al., 2021):
#
# 1. the **baseline** is the corrected CCG far from 0, at lags between 50 and 100 ms on each side:
#    there, no monosynaptic effect is expected. We measure its mean and standard deviation (SD);
# 2. we look in the window **[0.8 ms, 10 ms]** after 0: fast enough for a monosynaptic connection
#    (one synapse, no intermediate neuron), and not at 0 (no synapse is instantaneous);
# 3. the connection is **excitatory** if at least **two consecutive bins** of this window are more than
#    **7 SD above** the baseline mean, and **inhibitory** if at least two consecutive bins are more than
#    **5 SD below** (troughs are smaller than peaks, so the threshold is lower).
#
# Requiring two consecutive bins and such high thresholds keeps the false positives low, even when
# we test thousands of pairs. We also require enough coincidences (at least 10 expected per bin in the
# window): with very few spikes, a few empty bins in a row can look like a trough.

# %%
in_window = (jitter_lags .> 0.0008) .& (jitter_lags .< 0.010)  # [0.8 ms, 10 ms]
flanks = abs.(jitter_lags) .> 0.050  # baseline: |lag| between 50 and 100 ms

"""True if at least two consecutive values are true."""
two_in_a_row(above) = any(above[2:end] .& above[1:end-1])

panels = []
for ((i, j), cc) in sort(collect(corrected), by = x -> x[1] != (339, 98))  # 339 → 98 first
    m, sd = mean(cc[flanks]), std(cc[flanks], corrected = false)
    high, low = m + 7 * sd, m - 5 * sd
    excitatory = two_in_a_row(cc[in_window] .> high)
    inhibitory = two_in_a_row(cc[in_window] .< low)

    p = plot(xlabel = "Time of $j relative to $i (ms)", ylabel = "jitter-corrected CCG",
             title = "$i → $j: excitatory $excitatory, inhibitory $inhibitory", titlefontsize = 10)
    vspan!(p, [-100, -50], color = TAB[1], alpha = 0.07, label = "")  # baseline
    vspan!(p, [50, 100], color = TAB[1], alpha = 0.07, label = "")
    vspan!(p, [0.8, 10], color = :orange, alpha = 0.2, label = "[0.8, 10] ms")
    bar!(p, jitter_lags .* 1000, cc, bar_width = 0.5, color = :black, linecolor = :black, label = "")
    hline!(p, [high], color = TAB[4], ls = :dash, label = "mean + 7 SD")
    hline!(p, [low], color = TAB[1], ls = :dash, label = "mean − 5 SD", legendfontsize = 7)
    push!(panels, p)
end
fig = plot(panels..., layout = (1, 2), size = (1400, 380))

# %% [markdown]
# (The light blue areas, at ±50–100 ms, are the baseline.)
#
# ### 5.3 Transmission probability
#
# How strong is a connection? The **transmission probability** is the number of *extra* coincidences
# in the window (real − jittered), divided by the number of spikes of the reference: the fraction of
# the spikes of $i$ that make $j$ fire ([English et al., 2017](https://doi.org/10.1016/j.neuron.2017.09.033)).

# %%
i, j = 339, 98
real = compute_crosscorrelogram(good[id = i], good[id = j]; edges = jitter_edges, counts = true).values
jit = jittered_crosscorrelogram(good[id = i], good[id = j]; jitter = 0.025, n_rep = 40, edges = jitter_edges,
                                counts = true, rng = rng).values
extra = sum((real .- jit)[in_window])  # extra coincidences in [0.8, 10] ms
@printf("%d → %d: %.0f extra coincidences for %d spikes of %d: transmission probability %.3f\n",
        i, j, extra, length(good[id = i]), i, extra / length(good[id = i]))

# %% [markdown]
# About one spike of 339 in ten is followed by an extra spike of 98: for a connection between two
# neurons recorded extracellularly, this is a strong one.
#
# ### 5.4 Pitfalls: why a human still checks the CCGs
#
# An automatic rule is never perfect. In Matthieu's report, every significant pair was checked by eye,
# and several kinds of false positives came up (his Figure 3):
#
# - **too few spikes**: the CCG is noisy, and a few empty or full bins can pass the threshold;
# - a **broad increase** around 0: the CCG reflects a population of neurons firing together, not one
#   connection;
# - a **peak that starts before 0** and reaches its maximum after 0: the target may need several
#   inputs, or the two neurons share an input;
# - a **peak at 0 in an autocorrelogram-like shape**: one neuron recorded twice, on two parts of the
#   probe, and not merged by the spike sorter (like 35 – 36);
# - a **trough and a peak on each side of 0**: two neurons connected both ways (like 98 – 339), which is
#   real, but must be counted as two connections with opposite signs.
#
# **Question.** Change the thresholds (7 and 5 SD), the window ([0.8, 10] ms) or the jitter (±25 ms),
# and see how the decisions change for the two pairs.

# %% [markdown]
# ## 6. All pairs at once
#
# To find all the putative connections of a recording, we need the CCGs of **all the pairs**. With
# 239 units, that is 239 × 238 = 56,882 ordered pairs. In Python, SpikeInterface computes all the
# correlograms of a sorting in one fast call, compiled with numba (this is the basis of the countCCG
# pipeline). In Julia, a plain loop is compiled too, so we can write the same algorithm directly.
#
# ### 6.1 All the correlograms in one pass
#
# We merge the spikes of all the units into one sorted list (remembering the unit of each spike), and
# walk through it once: for each spike, we look at the following spikes within the window, and add one
# coincidence to the CCG of the corresponding pair of units, in both directions. The result is an
# array of counts of size (units × units × bins), where `ccgs[a, b, :]` is the CCG of unit `b`
# relative to unit `a` (our convention: **reference `a`, target `b`**; careful, SpikeInterface uses
# the opposite order).

# %%
"""CCGs of all the pairs of spike trains (and ACGs on the diagonal), in the bins given by edges.
ccgs[a, b, :] counts the spikes of train b relative to the spikes of train a."""
function all_correlograms(trains::Vector{Vector{Float64}}, edges)
    n, n_bins = length(trains), length(edges) - 1
    t = reduce(vcat, trains)  # all the spikes...
    u = reduce(vcat, [fill(k, length(x)) for (k, x) in enumerate(trains)])  # ...and their unit
    order = sortperm(t)
    t, u = t[order], u[order]
    lo, hi, width = edges[1], edges[end], edges[2] - edges[1]
    ccgs = zeros(Int32, n, n, n_bins)
    for k in eachindex(t)
        for m in k+1:length(t)  # the following spikes, until the end of the window
            d = t[m] - t[k]
            d > hi && break
            bin = floor(Int, (d - lo) / width) + 1  # spike m relative to spike k
            1 <= bin <= n_bins && (ccgs[u[k], u[m], bin] += 1)
            bin = floor(Int, (-d - lo) / width) + 1  # spike k relative to spike m
            1 <= bin <= n_bins && (ccgs[u[m], u[k], bin] += 1)
        end
    end
    return ccgs
end

unit_ids = ids(good)
trains = [times(good[id = i]) for i in unit_ids]
all_correlograms(trains[1:2], correlogram_edges(0.001, 0.05))  # first call: compiles the function

time_all = @elapsed ccgs = all_correlograms(trains, correlogram_edges(0.001, 0.05))  # 1 ms bins, ±50 ms
@printf("%.1f s for all pairs, array of size %s\n", time_all, size(ccgs))

a, b = indexof(good, 339), indexof(good, 98)  # positions of the units in the array
println("CCG 339 → 98, largest difference with section 3: ",
        maximum(abs.(ccgs[a, b, :] .- compute_crosscorrelogram(good[id = 339], good[id = 98];
                                                               binsize = 0.001, windowsize = 0.05,
                                                               counts = true).values)),
        " coincidences (spike time differences exactly on a bin edge)")

# %% [markdown]
# A few seconds for all the pairs, and the same CCG as before for 339 → 98 (up to a few coincidences
# exactly on a bin edge, section 3).
#
# ### 6.2 Jittering without jittering
#
# Jittering 40 times means computing all the CCGs 41 times: for all the pairs of a recording, this
# takes minutes. There is a shortcut. If each target spike moves by a random amount, uniform between
# −25 and +25 ms, each coincidence of the CCG moves by the same random amount. So **on average**, the
# jittered CCG is the real CCG **smoothed by a ±25 ms box** (a moving average over 101 bins of
# 0.5 ms). This is the idea of the convolution method of
# [Stark & Abeles (2009)](https://doi.org/10.1016/j.jneumeth.2008.12.029), discussed in Matthieu's
# report: no random numbers, a single CCG per pair. Let's check it on the pair 339 → 98.

# %%
"""Moving average over 2 half + 1 bins along the last dimension (values beyond the edges are
repeated, like scipy's mode = "nearest")."""
function box_smooth(x::AbstractArray, half::Int)
    d = ndims(x)
    first_ = selectdim(x, d, 1:1); last_ = selectdim(x, d, size(x, d):size(x, d))
    padded = cat(repeat(first_, outer = (ntuple(_ -> 1, d - 1)..., half)), x,
                 repeat(last_, outer = (ntuple(_ -> 1, d - 1)..., half)); dims = d)
    c = cumsum(padded; dims = d)
    c = cat(zeros(eltype(c), size(c)[1:d-1]..., 1), c; dims = d)  # c[k] = sum of the k-1 first values
    w = 2 * half + 1
    return (selectdim(c, d, w+1:size(c, d)) .- selectdim(c, d, 1:size(c, d)-w)) ./ w
end

wide_edges = correlogram_edges(0.0005, 0.125)  # compute the CCG up to ±125 ms, to smooth it up to ±100 ms
real_wide = compute_crosscorrelogram(good[id = 339], good[id = 98]; edges = wide_edges, counts = true).values
smoothed = box_smooth(Float64.(real_wide), 50)[51:end-50]  # ±25 ms box, back to ±100 ms

fig = plot(jitter_lags .* 1000, jit, color = TAB[3], lw = 3, label = "mean of 40 jittered CCGs",
           xlabel = "Time of 98 relative to 339 (ms)", ylabel = "Number of coincidences",
           title = "Jittered CCG = smoothed CCG", size = (900, 350), legend = :bottom)
plot!(fig, jitter_lags .* 1000, smoothed, color = :black, lw = 1, label = "real CCG smoothed by a ±25 ms box")
@printf("largest difference: %.1f coincidences per bin\n", maximum(abs.(smoothed .- jit)))
fig

# %% [markdown]
# The two curves are the same, up to the randomness of the 40 jitters. Notice how the jitter spreads
# the sharp excitatory peak into a plateau of ±25 ms: subtracting it removes the slow part of the CCG
# and keeps the fast peak.
#
# ### 6.3 Scanning all the pairs
#
# We now apply section 5 to all the pairs at once: the CCGs of all pairs (0.5 ms bins, ±125 ms), the
# jittered CCGs by smoothing, the corrected CCGs, and the significance rule. The arrays are
# (units × units × bins), and each step is applied to all the pairs together. With our convention,
# a significant peak after 0 in `ccgs[a, b, :]` is a connection **from `a` to `b`**.

# %%
"""Putative excitatory and inhibitory connections between all the spike trains.

Returns two boolean matrices (units x units), and the z-score of the corrected CCGs in the window:
exc[a, b] is true for a putative excitatory connection from unit a to unit b."""
function find_connections(trains, duration; min_count = 10)
    real = Float32.(all_correlograms(trains, correlogram_edges(0.0005, 0.125)))  # ±125 ms, 0.5 ms bins
    jit = box_smooth(real, 50)  # ±25 ms jitter
    lags = bin_centers(correlogram_edges(0.0005, 0.125)) .* 1000  # bin centers, in ms
    keep = abs.(lags) .<= 100  # keep ±100 ms
    real, jit, lags = real[:, :, keep], jit[:, :, keep], lags[keep]

    rates = length.(trains) ./ duration
    corrected = (real .- jit) ./ sqrt.(rates .* rates')

    flanks = abs.(lags) .> 50  # baseline
    window = (lags .> 0.8) .& (lags .< 10)  # [0.8, 10] ms
    m = mean(corrected[:, :, flanks], dims = 3)
    sd = std(corrected[:, :, flanks], dims = 3, corrected = false)
    sd[sd .== 0] .= Inf  # pairs without any coincidence
    z = (corrected[:, :, window] .- m) ./ sd

    enough = dropdims(mean(jit[:, :, window], dims = 3), dims = 3) .>= min_count
    above, below = z .> 7, z .< -5
    two(x) = dropdims(any(x[:, :, 2:end] .& x[:, :, 1:end-1], dims = 3), dims = 3)  # two bins in a row
    exc, inh = two(above) .& enough, two(below) .& enough
    for k in eachindex(trains)  # no connection of a unit with itself
        exc[k, k] = false; inh[k, k] = false
    end
    return exc, inh, z
end

time_scan = @elapsed exc, inh, z_window = find_connections(trains, duration)
@printf("%.0f s for %d ordered pairs\n", time_scan, length(unit_ids) * (length(unit_ids) - 1))
println("putative excitatory connections: ", sum(exc))
println("putative inhibitory connections: ", sum(inh))

# %% [markdown]
# **Who connects to whom?** With the cell types, we can check that the connections make sense: the
# excitatory connections should come mostly from RS units (pyramidal neurons), the inhibitory ones
# from FS units (interneurons).

# %%
types = good[:cell_type]  # cell type of each unit, in the order of the group
for (name, conn) in [("excitatory", exc), ("inhibitory", inh)]
    links = ["$(types[I[1]]) → $(types[I[2]])" for I in findall(conn)]  # conn[a, b]: from a to b
    println(name, ": ", join(["$k: $v" for (k, v) in sort(collect(countmap(links)))], ", "))
end

# %% [markdown]
# Most excitatory connections go from RS to FS units, and most inhibitory ones from FS to RS units,
# as expected from pyramidal neurons and interneurons. (Some RS → RS "inhibitions" may be indirect,
# through an interneuron that was not recorded.)
#
# The Python notebook finds a few connections less (76 excitatory and 51 inhibitory): SpikeInterface
# puts a bin **edge** at 0 while our bins are **centered** on 0, so the window [0.8, 10] ms does not
# contain exactly the same bins. Pairs close to the thresholds can then be decided differently: a
# reminder that these numbers depend on choices, not only on the data.
#
# ### 6.4 The connectivity matrix
#
# We draw all the pairs as a **matrix**: one row per target unit, one column per source unit, with the
# units **sorted by brain area and by depth**. The color is the largest z-score of the corrected CCG
# in the window [0.8, 10] ms (red: excess, blue: deficit). On the right, only the significant
# connections, as dots: excitatory in red, inhibitory in blue.

# %%
unit_area, unit_depth = good[:location], good[:depth]
area_depth = Dict(a => mean(unit_depth[unit_area .== a]) for a in unique(unit_area))  # mean depth
order = sortperm(collect(zip([area_depth[a] for a in unit_area], unit_depth)))  # areas along the probe, then depth
areas = unit_area[order]
boundaries = findall(areas[2:end] .!= areas[1:end-1]) .+ 0.5  # where the area changes
area_names = unique(areas)
area_centers = [mean(findall(areas .== a)) for a in area_names]

zmax = dropdims(maximum(z_window, dims = 3), dims = 3)
zmin = dropdims(minimum(z_window, dims = 3), dims = 3)
strongest = ifelse.(abs.(zmax) .> abs.(zmin), zmax, zmin)  # largest excess or deficit
for k in eachindex(unit_ids); strongest[k, k] = 0; end
# rows = targets, columns = sources: transpose our matrices (conn[source, target])
M = permutedims(strongest[order, order])

p1 = heatmap(M, color = cgrad(:RdBu, rev = true), clims = (-15, 15), yflip = true, colorbar_title = "z-score in [0.8, 10] ms",
             title = "All pairs", aspect_ratio = :equal)
p2 = plot(title = "Significant connections", yflip = true, aspect_ratio = :equal, legend = :bottomleft,
          xlims = (0.5, length(order) + 0.5), ylims = (0.5, length(order) + 0.5))
exc_o, inh_o = exc[order, order], inh[order, order]
scatter!(p2, [I[1] for I in findall(exc_o)], [I[2] for I in findall(exc_o)], color = TAB[4],
         markerstrokewidth = 0, markersize = 4, label = "excitatory")
scatter!(p2, [I[1] for I in findall(inh_o)], [I[2] for I in findall(inh_o)], color = TAB[1],
         markerstrokewidth = 0, markersize = 4, label = "inhibitory")
for p in (p1, p2)
    vline!(p, boundaries, color = :gray, lw = 0.5, label = "")
    hline!(p, boundaries, color = :gray, lw = 0.5, label = "")
    plot!(p, xticks = (area_centers, area_names), yticks = (area_centers, area_names), xrotation = 90,
          xlabel = "source unit (sorted by area and depth)", ylabel = "target unit")
end
fig = plot(p1, p2, layout = (1, 2), size = (1500, 650))

# %% [markdown]
# Most connections are **close to the diagonal**, inside each area: neurons recorded close to each
# other are more often connected (and the CCG method only sees connections between neurons that are
# both recorded, so near the probe). A few connections link different areas.
#
# ### 6.5 A matrix of correlograms
#
# To check a few units by eye, a matrix of correlograms is handy: the ACGs on the diagonal, and the
# CCGs between the units of each row (reference) and column (target). In Python, SpikeInterface draws
# it with `plot_crosscorrelograms`; here we draw it ourselves, from the `ccgs` array of section 6.1.

# %%
check = [339, 98, 109, 127]
k = indexof(good, check)  # positions of the units in the array
lag_ms = bin_centers(correlogram_edges(0.001, 0.05)) .* 1000
panels = []
for (r, a) in enumerate(k), (c, b) in enumerate(k)
    color = r == c ? TAB[3] : :black  # ACGs in green
    p = bar(lag_ms, ccgs[a, b, :], bar_width = 1, color = color, linecolor = color, label = "",
            xlims = (-20, 20), title = r == 1 ? "target $(check[c])" : "", titlefontsize = 9,
            ylabel = c == 1 ? "ref $(check[r])" : "", xlabel = r == length(k) ? "lag (ms)" : "")
    push!(panels, p)
end
fig = plot(panels..., layout = (length(k), length(k)), size = (900, 750), left_margin = 3Plots.mm,
           bottom_margin = 2Plots.mm)

# %% [markdown]
# Find the excitatory peak from 339 to 98, and the trough from 109 to 127: on which side of 0 are
# they, in the panel (ref 339, target 98)? And in the panel (ref 98, target 339)?
#
# ### 6.6 Advanced: does the network change with the task?
#
# Functional connectivity can change with the state of the animal. In Matthieu's report, the
# connections were found separately in each block of the task (section 0), and compared between
# blocks with the **Jaccard index** of the two sets of connections $A$ and $B$:
#
# $$ J = \frac{|A \cap B|}{|A \cup B|} $$
#
# the number of connections found in both blocks, divided by the number found in at least one of
# them: 1 for the same network, 0 for completely different ones. We restrict the spikes to each of the
# blocks 1–4 (from 3 s before its first trial to 3 s after its last one) and run the scan again.

# %%
block_connections = Dict()
for b in 1:4
    in_block = trials[:Block] .== b
    period = IntervalSet(minimum(starts(trials)[in_block]) - 3, maximum(stops(trials)[in_block]) + 3)
    block_trains = [times(restrict(good[id = i], period)) for i in unit_ids]
    e, n, _ = find_connections(block_trains, tot_length(period))
    block_connections[b] = e .| n  # excitatory or inhibitory
    @printf("block %d: %.1f min, %d excitatory, %d inhibitory\n", b, tot_length(period) / 60, sum(e), sum(n))
end

jaccard = [sum(block_connections[a] .& block_connections[b]) /
           max(sum(block_connections[a] .| block_connections[b]), 1) for a in 1:4, b in 1:4]

fig = heatmap(1:4, 1:4, [a == b ? NaN : jaccard[a, b] for a in 1:4, b in 1:4], color = :Reds,
              clims = (0, 0.5), yflip = true, aspect_ratio = :equal, xlabel = "Block", ylabel = "Block",
              title = "Jaccard index between blocks", size = (450, 400), xticks = 1:4, yticks = 1:4)
annotate!(fig, [(b, a, text(@sprintf("%.2f", jaccard[a, b]), 9)) for a in 1:4, b in 1:4 if a != b])
fig

# %% [markdown]
# Each block is short (7–14 minutes), so far fewer connections pass the significance rule than in the
# whole session: only the strongest ones (only 2 in block 1). The blocks 2, 3 and 4 are much more
# similar to each other (0.35 to 0.65) than to block 1 (0.11 to 0.17), the same trend as in Matthieu's
# report for this recording: after the first aversive stimulation (block 2), the network may stay in a
# different state. But with so few connections, these numbers are fragile (the Python notebook, with
# slightly different bins, finds the same trend with lower values): they would need more recordings,
# and a statistical test, to be conclusive.
#
# ### 6.7 Correlograms in spike sorting curation
#
# In a spike sorting workflow, correlograms are also used to curate the units, for example:
#
# - the **refractory period violations** of each unit, from its ACG (a quality metric);
# - the **automatic merging** of units: two units whose CCG looks like their ACGs (with a refractory
#   gap at 0) and whose waveforms are similar are probably the same neuron, split in two;
# - units that share many spikes at lag 0 (like 35 and 36) are flagged as **duplicates**.
#
# These tools are in SpikeInterface (Python): see the spike sorting notebooks of this school
# (`notebooks_sorting/`) to run them on a recording.
#
# **Questions.**
#
# - Which unit receives the most excitatory connections? Draw its CCGs with its sources.
# - Find the connections between different areas. Are they excitatory or inhibitory?
# - Change `min_count` in `find_connections`, or the thresholds. How many connections are left in the
#   blocks?
