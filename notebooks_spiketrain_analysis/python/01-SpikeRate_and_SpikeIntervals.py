# %% [markdown]
# # 01 · Spike rates and spike intervals — **Python / pynapple**
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
# for neurophysiology data, that we read with [pynapple](https://pynapple.org).
#
# ## 0. Get the data
#
# The session is shared on the [DANDI Archive](https://dandiarchive.org), the public archive
# for neurophysiology data, in [Dandiset 000473](https://dandiarchive.org/dandiset/000473)
# (*Esr1+ hypothalamic-habenula neurons shape aversive states*). We use the file
# `sub-216300/sub-216300_ses-216300-20200518-probe0_ecephys+ogen.nwb` (5.3 GB).
#
# **Option 1 (recommended): download the file** to a folder of your choice on your computer
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
#
# **Option 2 (advanced): stream the file from DANDI**, without downloading it. Set
# `STREAM = True` below (`NWB` is then not used). h5py then reads over the internet only the
# parts of the file that we use (the spike times, the trials, a few unit columns), with the
# [DANDI Python client](https://dandi.readthedocs.io) and [remfile](https://github.com/flatironinstitute/remfile).
# It needs a good internet connection, and each read is slower than from a local file.
#
# First we need to import the python packages and give the location of the data (NWB file).

# %%
# numpy: arrays and maths; matplotlib: figures; pynapple: time series and spike trains;
# h5py: low-level access to the NWB file (an HDF5 file), to read extra unit information
import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
import h5py
from pathlib import Path

# Option 1: full path to the NWB file you downloaded. Change it to where you saved it.
NWB = "/path/to/your/data/folder/sub-216300_ses-216300-20200518-probe0_ecephys+ogen.nwb"

# Option 2: set to True to stream the file from the DANDI Archive instead
STREAM = False

if not STREAM and not Path(NWB).is_file():
    raise FileNotFoundError(f"No NWB file at {NWB}: set NWB to the downloaded file, or STREAM = True")

# %% [markdown]
# **Streaming (Option 2 only).** The DANDI client finds the file in the Dandiset and gives its
# URL on the archive's storage. `remfile.File` makes that URL look like a local file to h5py.
# If `STREAM = False`, this cell only defines `open_nwb`.

# %%
if STREAM:
    import remfile
    from dandi.dandiapi import DandiAPIClient

    DANDISET, VERSION = "000473", "0.261102.1657"  # Dandiset id and published version
    ASSET = "sub-216300/sub-216300_ses-216300-20200518-probe0_ecephys+ogen.nwb"
    with DandiAPIClient() as client:
        asset = client.get_dandiset(DANDISET, VERSION).get_asset_by_path(ASSET)
        url = asset.get_content_url(follow_redirects=1, strip_query=True)
    print("streaming from", url)


def open_nwb():
    """Open the NWB file with h5py: the local file, or the remote one if STREAM is True."""
    return h5py.File(remfile.File(url) if STREAM else NWB, "r")

# %% [markdown]
# ## 1. Load the units and the trials
#
# `nap.load_file` opens the NWB file and lists what it contains. Nothing heavy is read
# yet: each object is loaded only when we ask for it with `data["..."]`.
# (When streaming, we open the file with `pynwb`, the reference NWB library, and give it to
# `nap.NWBFile`: this is what `nap.load_file` does for a local file.)
#
# - `units` is a **`TsGroup`**: a group of spike trains, one per unit (= putative neuron).
#   Each spike train is a `Ts` object, i.e. a vector of spike times in seconds.
# - `trials` is an **`IntervalSet`**: a table of time intervals (start, end) with one
#   row per trial, plus extra columns describing each trial.

# %%
if STREAM:
    import pynwb
    data = nap.NWBFile(pynwb.NWBHDF5IO(file=open_nwb()).read())  # remote file
else:
    data = nap.load_file(NWB)  # open the file and list its content
print(data)

units = data["units"]  # all sorted units (TsGroup)
trials = data["trials"]  # task trials (IntervalSet)

# Displaying a TsGroup prints one row per unit with its metadata
units

# %% [markdown]
# Each unit can come with **metadata**: one value per unit, stored as columns of the `TsGroup`.
# pynapple computes `rate` itself (number of spikes / recording duration). It also reads some
# columns of the NWB units table (e.g. the electrode `location` and waveform features), but
# not all of them: we add the cell type and a quality metric ourselves in section 2.
#
# Let's look at the trials as well: each row is a trial with its start and end time
# in seconds, and the extra columns describe the trial (sound, air puff, optogenetics, ...).
# Each trial lasts 2 s.

# %%
print("number of units :", len(units))
print("number of trials:", len(trials))
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
# **Extra information from the NWB file.** This file stores more columns in its units table
# than pynapple reads, so we open it directly with `h5py`. An NWB file is an HDF5 file:
# a tree of groups (like folders) and datasets (arrays), that we access with paths.
#
# - `units/quality` has two columns: the number of ISI violations, and the
#   **ISI violation false-positive rate**, an estimate of the fraction of spikes that do not
#   belong to the neuron. This is our ISI violation ratio.
# - `units/FS` and `units/RS` classify the waveform of each unit as **fast-spiking** (FS,
#   narrow spike, putative interneuron) or **regular-spiking** (RS, broad spike, putative
#   pyramidal neuron). A unit with neither label has no plausible waveform.
# - `units/electrodes` gives, for each unit, the row of the **electrode table** where it was
#   recorded. The column `location` of that table holds the brain area, with its cortical
#   layer, using the Allen atlas names (e.g. `PL6a` = prelimbic area, layer 6a).

# %%
with open_nwb() as f:  # open the file read-only (section 0); it is closed at the end of the block
    isi_fp = f["units/quality"][:, 1]  # 2nd column: ISI violation false-positive rate
    fs = f["units/FS"][:] == 1  # True for fast-spiking waveforms
    rs = f["units/RS"][:] == 1  # True for regular-spiking waveforms
    electrode = f["units/electrodes"][:]  # electrode-table row of each unit (0-based)
    all_locations = f["general/extracellular_ephys/electrodes/location"][:]  # one per electrode
    location = all_locations[electrode].astype(str)  # location of each unit, as text

# Waveform class of each unit: "FS", "RS", or "none" if the waveform is not plausible
cell_type = np.where(fs, "FS", np.where(rs, "RS", "none"))

# Store them as new metadata columns of the TsGroup (one value per unit, same order)
units.set_info(location=location, isi_fp=isi_fp, cell_type=cell_type)
units.location.value_counts()  # how many units per brain location

# %% [markdown]
# **Presence ratio.** We cut the session into 100 blocks of equal duration and count the
# fraction of blocks in which the unit fires at least one spike. A unit that drifts away
# from the probe, or appears only late in the session, has a low presence ratio.
# `count` returns the number of spikes of every unit in every block (a `TsdFrame`).

# %%
block = units.time_support.tot_length() / 100  # duration of one block, in seconds
counts_blocks = units.count(block)  # spike counts: one row per block, one column per unit
presence_ratio = (counts_blocks.values > 0).mean(axis=0)  # fraction of blocks with spikes
units.set_info(presence_ratio=presence_ratio)

print("presence ratio, min / median:", presence_ratio.min().round(2), "/",
      np.median(presence_ratio).round(2))

# %% [markdown]
# We extract the area name without the layer, and keep the units in ACA, PL or ILA that
# pass the criteria.

# %%
# Regular expression: take "ACA", "PL" or "ILA" at the start of the location.
# Units in other areas get NaN (missing value).
area = units.location.str.extract(r"^(ACA|PL|ILA)", expand=False)

# Store it as a new metadata column called "area"
units.set_info(area=area)

in_pfc = units.area.notna()  # in one of our 3 areas
good = (units.isi_fp < 0.1) & (units.cell_type != "none") & (units.presence_ratio > 0)

pfc = units[in_pfc & good]  # boolean selection, like with a pandas table

print("PFC units:", in_pfc.sum())
print("PFC units passing the criteria:", len(pfc))
print(pfc.cell_type.value_counts().to_string())
pfc.area.value_counts()  # how many units per area

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
# Every pynapple object has a **time support**: the `IntervalSet` on which it is defined.
# For our units, it is the whole recording session. The mean rate of a unit is its number of
# spikes divided by the total duration of the time support. pynapple already stores it in the
# `rate` column, but let's compute it by hand to see that it is that simple.

# %%
session = pfc.time_support  # IntervalSet: start and end of the recording
duration = session.tot_length()  # total duration, in seconds
print(session)
print("session duration:", round(duration / 60, 1), "min")

# Number of spikes of each unit: len() of each spike train
n_spikes = np.array([len(pfc[i]) for i in pfc.keys()])
rate_by_hand = n_spikes / duration  # spikes / seconds, one value per unit

# Same numbers as the "rate" column computed by pynapple?
print("same as pynapple:", np.allclose(rate_by_hand, pfc.rate.values))
print("median rate:", round(np.median(rate_by_hand), 2), "Hz")
print("min / max rate:", round(rate_by_hand.min(), 3), "/", round(rate_by_hand.max(), 1), "Hz")

# %% [markdown]
# Firing rates span **several orders of magnitude**, from less than 0.1 Hz to tens of Hz.
# On a linear axis, the slow units would all be squeezed against zero, so we plot the
# histogram with **logarithmic bins** (each bin is a constant factor wider than the previous).

# %%
rate_bins = np.logspace(-2, 2, 33)  # 33 bin edges from 0.01 Hz to 100 Hz, evenly spaced in log

fig, ax = plt.subplots(figsize=(8, 4))
for a in ["ACA", "PL", "ILA"]:
    rates_area = pfc.rate[pfc.area == a]  # session rates of the units in this area
    ax.hist(rates_area, bins=rate_bins, histtype="step", lw=2, label=f"{a} (n={len(rates_area)})")
ax.set_xscale("log")
ax.set(xlabel="Session rate (Hz, log scale)", ylabel="Number of units",
       title="Session firing rate of the PFC units")
ax.legend()
plt.show()

# %% [markdown]
# **Advanced: a smoothed density.** The histogram depends on where the bin edges fall, and
# the areas have different numbers of units, so their counts are hard to compare.
# Instead, we estimate the **density** of the rates in each area, with a
# **Savitzky–Golay filter**
# ([`scipy.signal.savgol_filter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html)):
# in a sliding window, it fits a polynomial to the data and keeps the value of the polynomial
# (or of its derivative) at the center of the window.
#
# 1. We work with $\log_{10}(\text{rate})$, so that each decade (0.1–1 Hz, 1–10 Hz, ...)
#    has the same width.
# 2. We compute the **cumulative distribution** in fine bins (0.02 decade): the fraction of units
#    with a rate below each value. It goes from 0 to 1 by small steps, one per unit.
# 3. The density is the **derivative** of the cumulative distribution. The Savitzky–Golay filter
#    smooths the steps and differentiates at the same time (`deriv=1`). Smoothing the cumulative
#    distribution works much better than smoothing the counts of a fine histogram: with a few
#    dozen units, most fine bins have 0 or 1 unit, and the result stays noisy.
#
# The area under each curve is 1, whatever the number of units, so the areas can be compared.
# The two parameters of the filter: `window` (number of bins in the sliding window: larger is
# smoother) and `polyorder` (degree of the polynomial, smaller than `window`). Try changing them!
# The ticks at the bottom show the rate of each unit. Careful with ACA: with only a few units,
# its curve is mostly the shape of the filter.

# %%
from scipy.signal import savgol_filter

bin_width = 0.02  # width of the fine bins, in decades of rate
window, polyorder = 75, 2  # Savitzky-Golay parameters: 75 bins = 1.5 decade, quadratic polynomial
log_bins = np.arange(-2, 2 + bin_width / 2, bin_width)  # bin edges in log10(rate): 0.01 Hz to 100 Hz
centers = (log_bins[:-1] + log_bins[1:]) / 2  # bin centers, in log10(rate)
area_colors = {"ACA": "C0", "PL": "C1", "ILA": "C2"}  # same colors as in the histogram above

fig, ax = plt.subplots(figsize=(8, 4))
for a, color in area_colors.items():
    log_rates = np.log10(pfc.rate[pfc.area == a])  # log10 of the session rates in this area
    counts, _ = np.histogram(log_rates, bins=log_bins)
    cumulative = np.cumsum(counts) / len(log_rates)  # fraction of units below each bin, 0 to 1
    # Smoothed derivative of the cumulative distribution = density, in fraction of units per decade
    density = savgol_filter(cumulative, window, polyorder, deriv=1, delta=bin_width, mode="nearest")
    density = np.clip(density, 0, None)  # the filter can dip slightly below 0 at the edges

    ax.plot(10 ** centers, density, color=color, lw=2, label=f"{a} (n={len(log_rates)})")
    ax.fill_between(10 ** centers, density, color=color, alpha=0.15, lw=0)
    # Rug: one small tick per unit, just below the curves
    ax.plot(10 ** log_rates, np.full(len(log_rates), -0.03), "|", color=color, ms=8, alpha=0.6)
ax.set_xscale("log")
ax.axhline(0, color="k", lw=0.5)
ax.set(xlabel="Session rate (Hz, log scale)", ylabel="Density (fraction of units per decade)",
       title="Session firing rate of the PFC units, smoothed density")
ax.legend()
plt.tight_layout()  # avoid cutting the axis labels
plt.show()

# %% [markdown]
# ### 3.2 Two example units with the same rate
#
# We now look at two units recorded in the same area and layer (**ILA6a**, infralimbic
# area, layer 6a) with a similar **session rate**, about 8–10 Hz.
# If the rate were all there is to know about a spike train, these two units should look alike.

# %%
examples = {"unit A": 185, "unit B": 228}  # name -> unit id
colors = {"unit A": "tab:blue", "unit B": "tab:orange"}  # one color per unit in all figures

for name, i in examples.items():
    print(f"{name}: id {i}, {pfc.location[i]}, {len(pfc[i])} spikes, "
          f"session rate {pfc.rate[i]:.2f} Hz")

# %% [markdown]
# Let's look at their activity during 10 seconds, drawn as a raster (one tick per spike).
# The trials are shaded in gray: outside them, the mouse waits for the next trial and we see
# the **spontaneous activity** of the neurons.

# %%
window = nap.IntervalSet(start=3020, end=3030)  # from 3020 s to 3030 s

fig, axes = plt.subplots(2, 1, figsize=(10, 3), sharex=True)
for ax, (name, i) in zip(axes, examples.items()):
    spikes_in_window = pfc[i].restrict(window)  # keep only the spikes inside the window
    ax.eventplot(spikes_in_window.t, colors=colors[name])
    # Shade the trials that fall inside the window
    for start, end in trials.intersect(window).values:
        ax.axvspan(start, end, color="gray", alpha=0.2, lw=0)
    ax.set(yticks=[], ylabel=name)
    print(f"{name}: {len(spikes_in_window)} spikes in 10 s")
axes[0].set_title("10 seconds of activity (gray: trials)")
axes[-1].set(xlim=(3020, 3030), xlabel="Time (s)")
plt.tight_layout()  # avoid cutting the axis labels
plt.show()

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
previous_end = np.r_[-np.inf, trials.end[:-1]]
keep = trials.start - pre >= previous_end  # True if the window starts after the previous trial
onsets = trials.start[keep]  # trial start times (s) of the kept windows

# One interval [onset - 3 s, onset] per kept trial
pre_trial = nap.IntervalSet(start=onsets - pre, end=onsets)
print(pre_trial)
print("windows kept:", len(pre_trial), "out of", len(trials), "trials")
print("total duration of the windows:", round(pre_trial.tot_length(), 1), "s")

for name, i in examples.items():
    spikes_pre = pfc[i].restrict(pre_trial)  # spikes inside the windows only
    print(f"{name}: {len(spikes_pre)} spikes, rate before trials {spikes_pre.rate:.2f} Hz, "
          f"session rate {pfc.rate[i]:.2f} Hz")

# %% [markdown]
# ### 3.4 Raster plot aligned to trial onset
#
# A **raster plot** shows one row per trial and one tick per spike. To stack the trials,
# we need spike times **relative to the trial onset** rather than to the session start.
# `nap.compute_perievent` does this alignment: for each event (here, each trial onset)
# it keeps the spikes inside `window` and subtracts the event time.
# With `window=(-3, 0)`, we get spikes from 3 s before to the onset (time 0).
#
# The result is a `TsGroup` with one spike train per trial. We store one per example unit.

# %%
onset_events = nap.Ts(t=onsets)  # the trial onsets as a series of events

# Dictionary: unit name -> TsGroup of aligned spike trains (one Ts per trial)
aligned = {name: nap.compute_perievent(pfc[i], onset_events, window=(-pre, 0))
           for name, i in examples.items()}

fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
for ax, (name, al) in zip(axes, aligned.items()):
    # One row per trial: al[k].t are the spike times of trial k, relative to its onset
    ax.eventplot([al[k].t for k in al.keys()], colors=colors[name], linewidths=0.5)
    ax.axvline(0, color="red")
    ax.set(xlim=(-pre, 0.05), xlabel="Time from trial onset (s)", title=name)
axes[0].set_ylabel("Trial")
plt.show()

# %% [markdown]
# Trials are sorted in time (trial 0 is the earliest). Notice that unit B barely fires before
# the first ~80 trials: the rate of a neuron is **not constant over a session**. It can change
# with the behavioural state of the animal, with learning, or with the recording itself
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
post_units = {**examples, "unit 288": 288}  # name -> unit id: units A, B and a responsive unit
colors["unit 288"] = "tab:purple"

# Type of each trial, from the metadata columns of the trials
trial_type = np.where(trials.airpuff == 1, "air puff",
                      np.where(trials.sound == 1,
                               np.where(trials.optogenetics, "sound + opto", "sound"),
                               "opto only"))
type_order = ["sound", "sound + opto", "air puff", "opto only"]  # order of the groups in the raster
order = np.argsort([type_order.index(t) for t in trial_type], kind="stable")  # trials sorted by type
sorted_onsets = nap.Ts(t=trials.start[order])  # trial onsets, grouped by type
group_ends = np.cumsum([np.sum(trial_type == t) for t in type_order])  # last row + 1 of each group
group_starts = np.r_[0, group_ends[:-1]]  # first row of each group
for t, n in zip(type_order, group_ends - group_starts):
    print(f"{t}: {n} trials")

fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
for ax, (name, i) in zip(axes, post_units.items()):
    al = nap.compute_perievent(pfc[i], sorted_onsets, window=(-1, 3))  # 1 s before to 3 s after
    ax.eventplot([al[k].t for k in al.keys()], colors=colors[name], linewidths=0.5)
    ax.axvspan(0, 2, color="gray", alpha=0.15, lw=0)  # the trial: 0 to 2 s
    ax.axvline(0, color="red")
    for end in group_ends[:-1]:  # a line between two trial types
        ax.axhline(end - 0.5, color="k", lw=1)
    ax.set(xlim=(-1, 3), xlabel="Time from trial onset (s)", title=f"{name} ({i})")
axes[0].set_yticks((group_starts + group_ends) / 2 - 0.5, type_order)  # one label per trial type
axes[0].set_ylim(len(order) - 0.5, -0.5)  # first trial at the top
plt.tight_layout()
plt.show()

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

fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, height_ratios=[2, 1])
for col, (name, al) in enumerate(aligned.items()):
    # count() bins the spikes of each trial: a TsdFrame with one row per bin, one column per trial
    counts = al.count(bin_size)
    # Sum over trials, then divide by the number of trials and the bin size -> rate in Hz
    psth = counts.values.sum(axis=1) / (len(al) * bin_size)
    mean_rate = psth.mean()  # average rate over the 3 s window

    axes[0, col].eventplot([al[k].t for k in al.keys()], colors=colors[name], linewidths=0.5)
    axes[0, col].set(title=f"{name}, bin size = {bin_size * 1000:.0f} ms")
    axes[1, col].bar(counts.t, psth, width=bin_size, color=colors[name])  # one bar per bin
    axes[1, col].axhline(mean_rate, color="k", ls="--", label="mean rate before trials")
    axes[1, col].set(xlim=(-pre, 0), xlabel="Time from trial onset (s)")
    axes[1, col].legend(loc="lower left")
axes[0, 0].set_ylabel("Trial")
axes[1, 0].set_ylabel("Rate (Hz)")
plt.show()

# %% [markdown]
# To see the effect of the bin size directly, we compute the same PSTH with four bin sizes,
# all on the same y-axis. Note how the mean rate (dashed line) stays the same: the bin size
# only changes **how finely we look at time**, not how many spikes there are.
# With 10 ms bins, the bars jump up and down although nothing happens before the trial:
# these fluctuations are just counting noise.

# %%
bin_sizes = [0.01, 0.05, 0.25, 1.0]  # in seconds

# sharey=True: same y-axis on all panels, so that we can compare the fluctuations
fig, axes = plt.subplots(len(bin_sizes), 2, figsize=(12, 9), sharex=True, sharey=True)
for col, (name, al) in enumerate(aligned.items()):
    for row, b in enumerate(bin_sizes):
        c = al.count(b)  # spike counts per bin and per trial
        rate = c.values.sum(axis=1) / (len(al) * b)  # PSTH in Hz
        ax = axes[row, col]
        ax.bar(c.t, rate, width=b, color=colors[name])
        ax.axhline(rate.mean(), color="k", ls="--")
        ax.set(title=f"{name}, bin size = {b * 1000:.0f} ms")
    axes[-1, col].set(xlim=(-pre, 0), xlabel="Time from trial onset (s)")
for row in range(len(bin_sizes)):
    axes[row, 0].set_ylabel("Rate (Hz)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.6 PSTH of the responses to the sound
#
# Now that we know how to build a PSTH, let's use it where it matters: **after the stimulus**.
# We keep the **sound only** trials (no optogenetics, no air puff, section 3.4), and compute the
# PSTH of units A, B and 288 from 1 s before to 3 s after the trial onset, with 50 ms bins.
# The dashed line is the mean rate in the second before the onset (the **baseline**): a response
# is a departure from it.

# %%
sound_onsets = nap.Ts(t=trials.start[trial_type == "sound"])  # onsets of the sound only trials
bin_size = 0.05  # 50 ms bins
print("sound only trials:", len(sound_onsets))

fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharex=True, height_ratios=[2, 1])
for col, (name, i) in enumerate(post_units.items()):
    al = nap.compute_perievent(pfc[i], sound_onsets, window=(-1, 3))  # one Ts per trial
    counts = al.count(bin_size)  # one row per bin, one column per trial
    psth = counts.values.sum(axis=1) / (len(al) * bin_size)  # rate in Hz
    baseline = psth[counts.t < 0].mean()  # mean rate in the second before the onset

    axes[0, col].eventplot([al[k].t for k in al.keys()], colors=colors[name], linewidths=0.5)
    axes[0, col].set(title=f"{name} ({i}), sound only trials")
    axes[1, col].bar(counts.t, psth, width=bin_size, color=colors[name])
    axes[1, col].axhline(baseline, color="k", ls="--", label="baseline")
    axes[1, col].set(xlim=(-1, 3), xlabel="Time from trial onset (s)")
    axes[1, col].legend(loc="upper right")
    for ax in axes[:, col]:
        ax.axvspan(0, 2, color="gray", alpha=0.15, lw=0)  # the trial: 0 to 2 s
        ax.axvline(0, color="red")
axes[0, 0].set_ylabel("Trial")
axes[1, 0].set_ylabel("Rate (Hz)")
plt.tight_layout()
plt.show()

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
# Python package [zetapy](https://github.com/JorritMontijn/zetapy)) asks whether a neuron
# responds to an event **without any bin**:
#
# 1. it pools the spike times of all trials (relative to the event) between 0 and `max_duration`,
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
# that we will meet again in the optional section 5. With `[start, end]` for each trial, `zetatest` also gives
# the p-value of a t-test on the mean rate (trial vs baseline), to compare.

# %%
from zetapy import zetatest

zeta_units = {**post_units, "unit C": 302}  # name -> unit id: units A, B, 288 and C
colors["unit C"] = "tab:green"
sound_trials = np.c_[trials.start, trials.end][trial_type == "sound"]  # [start, end] of each trial

zeta = {}  # name -> (ZETA results, instantaneous rate)
for name, i in zeta_units.items():
    np.random.seed(0)  # zetatest jitters at random: fix the seed to get the same p-values for everyone
    p, zeta_data, rate_data = zetatest(pfc[i].t, sound_trials, max_duration=2.0,
                                       resampling_number=100, return_rate=True)
    zeta[name] = (zeta_data, rate_data)
    print(f"{name} ({i}): ZETA p = {p:.2g} (score {zeta_data['zeta_score']:.1f}, "
          f"latency {zeta_data['latency_zeta'] * 1000:.0f} ms) | mean-rate t-test p = "
          f"{zeta_data['ttest_p_value']:.2g}")

fig, axes = plt.subplots(2, 4, figsize=(17, 6.5), sharex=True)
for col, (name, (zd, rd)) in enumerate(zeta.items()):
    ax = axes[0, col]
    # Deviation of 20 jittered resamplings (gray): what we expect by chance
    for t, dev in list(zip(zd["random_times"], zd["random_deviations"]))[:20]:
        ax.plot(t, dev, color="gray", lw=0.5, alpha=0.5)
    ax.plot(zd["spike_time_vector"], zd["real_deviation"], color=colors[name], lw=2)  # the data
    ax.plot(zd["latency_zeta"], zd["zeta_deviation"], "o", color="k", ms=8)  # the ZETA
    ax.axhline(0, color="k", lw=0.5)
    ax.set(title=f"{name} ({zeta_units[name]})\nZETA p = {zd['zeta_p_value']:.2g}, "
                 f"t-test p = {zd['ttest_p_value']:.2g}")

    ax = axes[1, col]
    ax.plot(rd["timestamps"], rd["rate_vector"], color=colors[name], lw=1.5)  # rate without bins
    ax.set(xlim=(0, 2), xlabel="Time from trial onset (s)")
axes[0, 0].set_ylabel("Deviation from\nuniform spiking")
axes[1, 0].set_ylabel("Instantaneous rate (Hz)")
plt.tight_layout()
plt.show()

# %% [markdown]
# Top: the deviation of the data (color) and of 20 jittered resamplings (gray); the black dot is the
# ZETA. Where the curve goes **up**, the neuron fires more than average; where it goes **down**,
# less. (The deviation is centered on 0, so the ZETA can also be the low point just *before* a
# steep rise, as for unit 288.) Bottom: the instantaneous firing rate.
#
# - **Unit 288**: a huge deviation, far outside the jittered ones: both tests agree, it responds.
# - **Unit A**: the deviation stays close to the jittered ones: no clear response (p ≈ 0.06).
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
# The result is a `Tsd` (a time series with values), timed at the middle of each interval.
#
# A useful sanity check: the **mean ISI** should be close to **1 / mean rate**.
# (They are not exactly equal because the recording does not start and end on a spike.)

# %%
isis = {name: pfc[i].time_diff() for name, i in examples.items()}  # ISIs in seconds

for name, isi in isis.items():
    i = examples[name]
    print(f"{name}: mean ISI {isi.mean() * 1000:.1f} ms (1 / rate = {1000 / pfc.rate[i]:.1f} ms), "
          f"std of ISIs {isi.std() * 1000:.1f} ms, shortest {isi.min() * 1000:.2f} ms")

# %% [markdown]
# Similar mean ISI (similar rate), but a very different **spread** of the ISIs
# (standard deviation): more than 10 times larger for unit B.
#
# ### 4.1 ISI histograms
#
# The **ISI distribution** shows how often each interval occurs.
# `nap.compute_isi_distribution` counts the ISIs in the bins we give it.
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
bins_ms = np.arange(0, 0.3005, 0.005)  # 5 ms bins from 0 to 300 ms (in seconds)
bins_zoom = np.arange(0, 0.0205, 0.0005)  # 0.5 ms bins from 0 to 20 ms
bins_log = np.logspace(-3, 2, 101)  # 101 edges from 1 ms (10^-3 s) to 100 s, evenly spaced in log

fig, axes = plt.subplots(2, 3, figsize=(14, 6))
for row, (name, i) in enumerate(examples.items()):
    # Each call returns a pandas DataFrame: index = bin centers (s), one column per unit
    h_ms = nap.compute_isi_distribution(pfc[i], bins=bins_ms)
    h_zoom = nap.compute_isi_distribution(pfc[i], bins=bins_zoom)
    h_log = nap.compute_isi_distribution(pfc[i], bins=bins_log)

    # stairs(counts, edges) draws a histogram from the counts and the bin edges
    axes[row, 0].stairs(h_ms.values[:, 0], bins_ms * 1000, fill=True, color=colors[name])
    axes[row, 0].set(ylabel=f"{name}\nCount")

    axes[row, 1].stairs(h_zoom.values[:, 0], bins_zoom * 1000, fill=True, color=colors[name])
    axes[row, 1].axvspan(0, 1.5, color="red", alpha=0.2, label="refractory period (1.5 ms)")

    axes[row, 2].stairs(h_log.values[:, 0], bins_log, fill=True, color=colors[name])
    axes[row, 2].set_xscale("log")  # logarithmic x-axis

axes[0, 0].set_title("ISI histogram, 0–300 ms")
axes[0, 1].set_title("Zoom on 0–20 ms")
axes[0, 1].legend(loc="upper left")
axes[0, 2].set_title("ISI histogram, log scale")
axes[1, 0].set_xlabel("ISI (ms)")
axes[1, 1].set_xlabel("ISI (ms)")
axes[1, 2].set_xlabel("ISI (s, log scale)")
plt.tight_layout()
plt.show()

# Fraction of short and long ISIs, to put numbers on what we see
for name, isi in isis.items():
    print(f"{name}: {100 * np.mean(isi.values < 0.02):.1f} % of ISIs < 20 ms, "
          f"{100 * np.mean(isi.values > 1):.2f} % of ISIs > 1 s")

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
def burstiness(isi):
    """Burstiness B = (std - mean) / (std + mean) of an array of ISIs."""
    m, s = np.mean(isi), np.std(isi)
    return (s - m) / (s + m)


for name, isi in isis.items():
    print(f"{name}: B = {burstiness(isi.values):+.3f}")

# %% [markdown]
# Unit A has $B < 0$ (slightly more regular than Poisson) and unit B has $B > 0$ (bursty).
#
# **Check with a Poisson process.** If spikes are thrown at random, uniformly in time,
# the ISIs follow an exponential distribution and $B$ should be close to 0.
# We simulate such a spike train with the same number of spikes as unit A.

# %%
rng = np.random.default_rng(seed=0)  # random generator with a fixed seed: same result for everyone

n_poisson = len(pfc[examples["unit A"]])  # same number of spikes as unit A
t_start, t_end = session.start[0], session.end[0]  # same duration as the session

# n_poisson random spike times, uniformly distributed over the session, sorted in time
poisson = nap.Ts(t=np.sort(rng.uniform(t_start, t_end, size=n_poisson)))

print("Poisson rate:", round(poisson.rate, 2), "Hz")
print(f"Poisson B   : {burstiness(poisson.time_diff().values):+.3f}")

# %% [markdown]
# **Careful: the standard deviation is sensitive to a few very long ISIs.**
# Unit B is almost silent during the first minutes of the session, which gives it a few
# very long ISIs (up to two minutes). Let's see how much they weigh in $B$.

# %%
isi_B = np.sort(isis["unit B"].values)  # ISIs of unit B, sorted from shortest to longest
print("5 longest ISIs of unit B (s):", np.round(isi_B[-5:], 1))
print(f"B with all ISIs           : {burstiness(isi_B):+.3f}")
print(f"B without the 5 longest   : {burstiness(isi_B[:-5]):+.3f}")  # still bursty, but less

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
print("number of ITI windows:", len(iti))


def burstiness_per_window(spikes, windows, min_isi=4):
    """B in each window that has at least min_isi ISIs. Returns one value per valid window."""
    values = []
    for k in range(len(windows)):  # loop over the windows
        t = spikes.restrict(windows[k]).t  # spike times inside window k
        isi = np.diff(t)  # ISIs inside window k only
        if len(isi) >= min_isi:  # skip windows with too few ISIs
            values.append(burstiness(isi))
    return np.array(values)


B_windows = {name: burstiness_per_window(pfc[i], iti, min_isi) for name, i in examples.items()}

fig, ax = plt.subplots(figsize=(7, 4))
for name, b in B_windows.items():
    print(f"{name}: {len(b)} valid windows, mean B in ITIs = {b.mean():+.3f}")
    ax.hist(b, bins=np.linspace(-1, 1, 41), histtype="step", lw=2, color=colors[name],
            label=f"{name} (mean {b.mean():+.2f})")
    ax.axvline(b.mean(), color=colors[name], ls="--")
ax.set(xlabel="B in one 3 s ITI window", ylabel="Number of windows",
       title="Burstiness in each inter-trial interval")
ax.legend()
plt.show()

# %% [markdown]
# Unit B is still more bursty than unit A, but both values are lower than over the session.
#
# **Do the same with the Poisson spike train.** Over the whole session it gave $B \approx 0$.
# What about in the 3 s ITI windows?

# %%
B_poisson = burstiness_per_window(poisson, iti, min_isi)
print(f"Poisson ({poisson.rate:.1f} Hz): mean B in ITIs = {B_poisson.mean():+.3f}")

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

B_iti, n_valid, rate_iti = [], [], []  # one value per unit in each list
for i in pfc.keys():
    b = burstiness_per_window(pfc[i], iti, min_isi)
    n_valid.append(len(b))  # number of windows with at least min_isi ISIs
    rate_iti.append(pfc[i].restrict(iti).rate)  # spikes in the ITIs / total ITI duration
    B_iti.append(b.mean() if len(b) >= min_windows else np.nan)  # NaN: not enough windows

pfc.set_info(burstiness_iti=np.array(B_iti), n_valid_windows=np.array(n_valid),
             rate_iti=np.array(rate_iti))

valid = pfc[pfc.burstiness_iti.notna()]  # units with at least 12 valid windows
print("units with >= 12 valid windows:", len(valid), "out of", len(pfc))
print("median B in ITIs:", round(valid.burstiness_iti.median(), 3))
print(f"for comparison, Poisson train ({poisson.rate:.0f} Hz) in the ITIs: {B_poisson.mean():+.3f}")

# %% [markdown]
# To compare the areas, we draw the **smoothed density** of $B$ in each area, with the same
# Savitzky–Golay method as for the firing rates (section 3.1): the derivative of the smoothed
# cumulative distribution. We write it as a function, to use it again below.
#
# On the right, each dot is a unit. The two example units are drawn with their own colors and
# markers, different from the colors of the areas.

# %%
def smoothed_density(values, lo, hi, bin_width=0.005, window=41, polyorder=2):
    """Smoothed density of values between lo and hi (Savitzky-Golay, as in section 3.1).

    Returns the bin centers and the density (area under the curve = 1)."""
    edges = np.arange(lo, hi + bin_width / 2, bin_width)  # fine bins
    centers = (edges[:-1] + edges[1:]) / 2
    cumulative = np.cumsum(np.histogram(values, bins=edges)[0]) / len(values)  # 0 to 1
    density = savgol_filter(cumulative, window, polyorder, deriv=1, delta=bin_width, mode="nearest")
    return centers, np.clip(density, 0, None)


def plot_density_by_area(ax, units, column, lo, hi):
    """Smoothed density of a metadata column, one curve per area, and one tick per unit below."""
    for a, color in area_colors.items():
        x = units[column][units.area == a].values  # values of the units in this area
        centers, density = smoothed_density(x, lo, hi)
        ax.plot(centers, density, color=color, lw=2, label=f"{a} (n={len(x)})")
        ax.fill_between(centers, density, color=color, alpha=0.15, lw=0)
        ax.plot(x, np.full(len(x), -0.2), "|", color=color, ms=8, alpha=0.6)  # rug
    ax.axhline(0, color="k", lw=0.5)
    ax.set(ylabel="Density")


# Colors and markers of the example units in the population plots (not the colors of the areas)
highlight = {"unit A": ("black", "o"), "unit B": ("tab:red", "s")}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

# Left: smoothed density of B in the ITIs, one curve per area
plot_density_by_area(ax1, valid, "burstiness_iti", -0.5, 0.5)
for name, (color, marker) in highlight.items():  # where the example units are
    ax1.axvline(pfc.burstiness_iti[examples[name]], color=color, ls="--", lw=1.5, label=name)
ax1.set(xlabel="Burstiness B in the ITIs", title="Burstiness of PFC units in the ITIs")
ax1.legend(fontsize=8)

# Right: B against the ITI rate, one dot per unit
for a, color in area_colors.items():
    sel = valid.area == a
    ax2.scatter(valid.rate_iti[sel], valid.burstiness_iti[sel], s=15, color=color, label=a)
for name, (color, marker) in highlight.items():  # the two example units
    i = examples[name]
    ax2.scatter(pfc.rate_iti[i], pfc.burstiness_iti[i], s=150, marker=marker, facecolors="none",
                edgecolors=color, lw=2, label=name)
ax2.axhline(0, color="k", ls=":")
ax2.set_xscale("log")
ax2.set(xlabel="Rate in the ITIs (Hz, log scale)", ylabel="Burstiness B in the ITIs",
        title="Burstiness vs rate")
ax2.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# Most units have $B$ slightly below 0. Remember the Poisson check of section 4.3: in 3 s windows,
# random firing already gives $B < 0$, and even more so for slow units, which fire fewer spikes per
# window. So $B < 0$ here does **not** mean that these units fire regularly.
#
# **Questions.**
#
# - Does burstiness depend on the area? On the rate? On the cell type (`cell_type`: FS or RS)?
# - Compare the ITI burstiness with the burstiness over the whole session
#   (`burstiness(pfc[i].time_diff().values)`). Which units change the most?
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
def random_timing_burstiness(spikes, windows, min_isi=4, n_rep=20):
    """Mean B over the valid windows, after placing each window's spikes at random times."""
    values = []
    for k in range(len(windows)):
        n = len(spikes.restrict(windows[k]))  # number of spikes in window k
        if n - 1 < min_isi:  # same rule as for the real data: at least min_isi ISIs
            continue
        start, end = windows.start[k], windows.end[k]
        # n_rep rows of n random spike times inside the window, sorted along each row
        t = np.sort(rng.uniform(start, end, size=(n_rep, n)), axis=1)
        isi = np.diff(t, axis=1)  # ISIs of each repetition
        m, s = isi.mean(axis=1), isi.std(axis=1)  # mean and std of each repetition
        values.append(np.mean((s - m) / (s + m)))  # B averaged over the repetitions
    return np.mean(values) if values else np.nan


for name, i in examples.items():
    B_rand = random_timing_burstiness(pfc[i], iti, min_isi)
    b = B_windows[name].mean()
    print(f"{name}: B in ITIs {b:+.3f}, random timing {B_rand:+.3f}, corrected {b - B_rand:+.3f}")

# %% [markdown]
# We compute the corrected burstiness of every unit with enough windows (section 4.4), and draw
# the same two plots.

# %%
B_rand = np.array([random_timing_burstiness(pfc[i], iti, min_isi)
                   if pfc.n_valid_windows[i] >= min_windows else np.nan for i in pfc.keys()])
pfc.set_info(burstiness_random=B_rand,
             burstiness=pfc.burstiness_iti.values - B_rand)  # corrected burstiness

valid = pfc[pfc.burstiness.notna()]  # same units as in section 4.4
print("median B in ITIs (not corrected):", round(valid.burstiness_iti.median(), 3))
print("median corrected B              :", round(valid.burstiness.median(), 3))
print("more bursty than random timing  :", np.sum(valid.burstiness > 0), "units")
print("more regular than random timing :", np.sum(valid.burstiness < 0), "units")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

plot_density_by_area(ax1, valid, "burstiness", -0.5, 0.5)
for name, (color, marker) in highlight.items():
    ax1.axvline(pfc.burstiness[examples[name]], color=color, ls="--", lw=1.5, label=name)
ax1.axvline(0, color="k", ls=":", label="random timing")
ax1.set(xlabel="Corrected burstiness in the ITIs", title="Corrected burstiness of PFC units")
ax1.legend(fontsize=8)

for a, color in area_colors.items():
    sel = valid.area == a
    ax2.scatter(valid.rate_iti[sel], valid.burstiness[sel], s=15, color=color, label=a)
for name, (color, marker) in highlight.items():
    i = examples[name]
    ax2.scatter(pfc.rate_iti[i], pfc.burstiness[i], s=150, marker=marker, facecolors="none",
                edgecolors=color, lw=2, label=name)
ax2.axhline(0, color="k", ls=":")
ax2.set_xscale("log")
ax2.set(xlabel="Rate in the ITIs (Hz, log scale)", ylabel="Corrected burstiness",
        title="Corrected burstiness vs rate")
ax2.legend(fontsize=8)
plt.tight_layout()
plt.show()

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

for name, isi in isis.items():
    n_viol = np.sum(isi.values < refractory)  # number of ISIs shorter than 1.5 ms
    print(f"{name}: {n_viol} ISIs < 1.5 ms out of {len(isi)} "
          f"({100 * n_viol / len(isi):.3f} %)")

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
def isis_per_window(spikes, windows, min_isi=4):
    """List of the ISI arrays of the windows that have at least min_isi ISIs."""
    out = []
    for k in range(len(windows)):
        isi = np.diff(spikes.restrict(windows[k]).t)  # ISIs inside window k only
        if len(isi) >= min_isi:
            out.append(isi)
    return out


def memory(isi_list):
    """Memory coefficient: correlation between consecutive ISIs, pooled over windows."""
    current = np.concatenate([isi[:-1] for isi in isi_list])  # tau_k, from every window
    following = np.concatenate([isi[1:] for isi in isi_list])  # tau_(k+1), from the same window
    return np.corrcoef(current, following)[0, 1]  # Pearson correlation coefficient


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


def keep_first_T(isi):
    """Keep the ISIs whose cumulative sum (the spike times) stays below T."""
    return isi[np.cumsum(isi) < T]


def reorder_isis(isi, target, n_steps=5000):
    """Reorder the ISIs by random swaps until their memory coefficient gets close to target."""
    isi = isi.copy()
    for _ in range(n_steps):
        i, j = rng.integers(len(isi), size=2)  # two random positions
        new = isi.copy()
        new[i], new[j] = isi[j], isi[i]  # swap them
        if abs(memory([new]) - target) < abs(memory([isi]) - target):  # closer to target?
            isi = new
    return isi


n_draw = 200  # draw more ISIs than needed; keep_first_T keeps those that fit in 50 s
isi_a = keep_first_T(rng.exponential(1.0, n_draw))  # a: Poisson, exponential ISIs
isi_b = rng.lognormal(0, 1.5, n_draw)  # b: heavy-tailed ISI distribution...
isi_b = keep_first_T(isi_b / isi_b.mean())  # ...rescaled to a mean of 1 s
isi_c = keep_first_T(rng.normal(1.0, 0.1, n_draw))  # c: regular, Gaussian ISIs
isi_d = reorder_isis(isi_a, target=0.6)  # d: same ISIs as a, positive memory
isi_e = reorder_isis(isi_a, target=-0.6)  # e: same ISIs as a, negative memory

trains = {"a": isi_a, "b": isi_b, "c": isi_c, "d": isi_d, "e": isi_e}

# Layout of the original figure: a on top, b and c on the left, d and e on the right
fig = plt.figure(figsize=(12, 4.5))
grid = fig.add_gridspec(3, 4, hspace=0.9)
places = {"a": grid[0, 1:3], "b": grid[1, 0:2], "c": grid[2, 0:2], "d": grid[1, 2:4], "e": grid[2, 2:4]}
for label, isi in trains.items():
    ax = fig.add_subplot(places[label])
    ax.vlines(np.cumsum(isi), 0, 1, color="k", lw=0.8)  # one vertical line per spike
    ax.set(xlim=(0, T), ylim=(0, 1), yticks=[])
    ax.set_title(f"{label}:  B = {burstiness(isi):+.2f},  M = {memory([isi]):+.2f}", fontsize=10)
ax.set_xlabel("t (s)")
plt.show()

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
memory_units = {**examples, "unit C": 302}  # name -> unit id: units A, B and C
colors["unit C"] = "tab:green"  # color of unit C in the next figures

isi_windows = {name: isis_per_window(pfc[i], iti, min_isi) for name, i in memory_units.items()}
for name, isi_list in isi_windows.items():
    n_pairs = sum(len(isi) - 1 for isi in isi_list)  # number of (tau_k, tau_k+1) pairs
    i = memory_units[name]
    print(f"{name} ({i}, {pfc.location[i]}): B in ITIs = {pfc.burstiness_iti[i]:+.3f}, "
          f"{n_pairs} pairs of consecutive ISIs, M = {memory(isi_list):+.3f}")

n_show = 4  # number of windows to draw per unit
fig, axes = plt.subplots(2, 2, figsize=(12, 4), sharex=True)
for row, name in enumerate(["unit B", "unit C"]):
    isi_list = isi_windows[name]
    busiest = np.argsort([len(isi) for isi in isi_list])[-n_show:]  # windows with the most ISIs
    for k, w in enumerate(busiest):
        isi = isi_list[w]
        original = np.r_[0, np.cumsum(isi)]  # spike times from the window's first spike
        shuffled = np.r_[0, np.cumsum(rng.permutation(isi))]  # same ISIs, random order
        axes[row, 0].vlines(original, k, k + 0.8, color=colors[name], lw=0.8)
        axes[row, 1].vlines(shuffled, k, k + 0.8, color=colors[name], lw=0.8)
    axes[row, 0].set(yticks=[], ylabel=name)
    axes[row, 1].set(yticks=[])
axes[0, 0].set_title("ISIs in their original order")
axes[0, 1].set_title("same ISIs, shuffled order")
axes[1, 0].set_xlabel("Time from the first spike of the window (s)")
axes[1, 1].set_xlabel("Time from the first spike of the window (s)")
plt.tight_layout()
plt.show()

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
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
for ax, (name, isi_list) in zip(axes, isi_windows.items()):
    current = np.concatenate([isi[:-1] for isi in isi_list])
    following = np.concatenate([isi[1:] for isi in isi_list])
    ax.scatter(current, following, s=2, alpha=0.3, color=colors[name])
    ax.plot([1e-3, 3], [1e-3, 3], "k:", lw=1)  # diagonal: tau_(k+1) = tau_k
    ax.set(xscale="log", yscale="log", xlabel="ISI k (s)", title=f"{name}, M = {memory(isi_list):+.2f}")
axes[0].set_ylabel("ISI k+1 (s)")
plt.tight_layout()
plt.show()

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
pfc.set_info(memory_iti=np.array([memory(isis_per_window(pfc[i], iti, min_isi))
                                  if pfc.n_valid_windows[i] >= min_windows else np.nan
                                  for i in pfc.keys()]))
valid = pfc[pfc.memory_iti.notna()]  # same units as in section 4.4
print("units:", len(valid))
print("median memory:", round(valid.memory_iti.median(), 3))
print("units with positive memory:", np.sum(valid.memory_iti > 0), "out of", len(valid))

highlight["unit C"] = ("tab:brown", "D")  # color and marker of unit C in the population plots


def plot_memory_burstiness(valid, m_column, b_column, m_label, b_label, title):
    """Left: smoothed density of the memory per area. Right: the (memory, burstiness) plane."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    plot_density_by_area(ax1, valid, m_column, -0.4, 0.6)
    for name, (color, marker) in highlight.items():  # where the example units are
        ax1.axvline(pfc[m_column][memory_units[name]], color=color, ls="--", lw=1.5, label=name)
    ax1.axvline(0, color="k", ls=":", label="no memory")
    ax1.set(xlabel=m_label, title=title)
    ax1.legend(fontsize=8)

    for a, color in area_colors.items():
        sel = valid.area == a
        ax2.scatter(valid[m_column][sel], valid[b_column][sel], s=20, color=color, label=a)
    for name, (color, marker) in highlight.items():  # the three example units
        i = memory_units[name]
        ax2.scatter(pfc[m_column][i], pfc[b_column][i], s=150, marker=marker, facecolors="none",
                    edgecolors=color, lw=2, label=name)
    ax2.axhline(0, color="k", ls=":")
    ax2.axvline(0, color="k", ls=":")
    ax2.set(xlabel=m_label, ylabel=b_label, title="Burstiness–memory plane")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


plot_memory_burstiness(valid, "memory_iti", "burstiness_iti", "Memory M in the ITIs",
                       "Burstiness B in the ITIs", "Memory of PFC units in the ITIs")
print("correlation between B and M:",
      round(np.corrcoef(valid.burstiness_iti, valid.memory_iti)[0, 1], 3))

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
def memory_shuffled(isi_list, n_rep=20):
    """Mean memory coefficient after shuffling the ISI order inside each window."""
    values = []
    for _ in range(n_rep):
        shuffled = [rng.permutation(isi) for isi in isi_list]  # new random order in each window
        values.append(memory(shuffled))
    return np.mean(values)


for name, isi_list in isi_windows.items():
    M, M_shuf = memory(isi_list), memory_shuffled(isi_list)
    print(f"{name}: M = {M:+.3f}, shuffled = {M_shuf:+.3f}, corrected = {M - M_shuf:+.3f}")

# Sanity check with the Poisson spike train: no memory at all
poisson_list = isis_per_window(poisson, iti, min_isi)
print(f"Poisson: M = {memory(poisson_list):+.3f}, shuffled = {memory_shuffled(poisson_list):+.3f}")

# %% [markdown]
# We compute the corrected memory of every unit, and draw the same plots with the **corrected**
# memory and the **corrected** burstiness (this needs the advanced section 4.5).

# %%
M_corr = np.full(len(pfc), np.nan)  # one value per unit, NaN for units without enough windows
for k, i in enumerate(pfc.keys()):
    if pfc.n_valid_windows[i] >= min_windows:
        isi_list = isis_per_window(pfc[i], iti, min_isi)
        M_corr[k] = memory(isi_list) - memory_shuffled(isi_list)
pfc.set_info(memory=M_corr)  # new metadata column: corrected memory

valid = pfc[pfc.memory.notna()]  # same units as in section 4.4
print("median memory (not corrected):", round(valid.memory_iti.median(), 3))
print("median corrected memory      :", round(valid.memory.median(), 3))
print("units with positive corrected memory:", np.sum(valid.memory > 0), "out of", len(valid))

plot_memory_burstiness(valid, "memory", "burstiness", "Corrected memory M", "Corrected burstiness B",
                       "Corrected memory of PFC units")
print("correlation between corrected B and corrected M:",
      round(np.corrcoef(valid.burstiness, valid.memory)[0, 1], 3))

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
#   of the ISIs (`np.log(isi)`) instead. Does the picture change?
# - Units in the top-right corner are both bursty and with memory. Look at their raster and
#   their return map: what do they have in common?
