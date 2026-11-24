# %% [markdown]
# # 02 · Spike waveforms and cell types — **Python / pynapple**
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
# In this notebook we:
#
# 1. look at the mean spike waveform of each unit, and measure its shape with a few **features**
#    (trough-to-peak duration, peak-to-baseline duration, half-width, peak-to-trough ratio);
# 2. split the units into **narrow** and **wide** waveforms with two thresholds, and (advanced) with
#    a **Gaussian mixture model**, and compare with the FS / RS labels stored in the file (used in
#    notebooks 01 and 03);
# 3. check that the two classes also **fire differently** (rate, autocorrelogram), with
#    [pynapple](https://pynapple.org);
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
# **Option 1 (recommended): download the file** to a folder of your choice on your computer
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
#
# **Option 2 (advanced): stream the file from DANDI**, without downloading it: set `STREAM = True`
# below (see section 0 of notebook 01). This notebook reads the mean waveforms of all the units
# (about 270 MB), so the first cells take a few minutes when streaming.

# %%
# numpy: arrays and maths; matplotlib: figures; pandas: tables; pynapple: spike trains;
# h5py: low-level access to the NWB file (an HDF5 file), to read the waveforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pynapple as nap
import h5py
from pathlib import Path

# Option 1: full path to the NWB file you downloaded. Change it to where you saved it.
NWB = "/path/to/your/data/folder/sub-156131_ses-156131-20191112-probe0_ecephys+ogen.nwb"

# Option 2: set to True to stream the file from the DANDI Archive instead
STREAM = False

if not STREAM and not Path(NWB).is_file():
    raise FileNotFoundError(f"No NWB file at {NWB}: set NWB to the downloaded file, or STREAM = True")

if STREAM:
    import remfile
    from dandi.dandiapi import DandiAPIClient

    DANDISET, VERSION = "000473", "0.261102.1657"  # Dandiset id and published version
    ASSET = "sub-156131/sub-156131_ses-156131-20191112-probe0_ecephys+ogen.nwb"
    with DandiAPIClient() as client:
        asset = client.get_dandiset(DANDISET, VERSION).get_asset_by_path(ASSET)
        url = asset.get_content_url(follow_redirects=1, strip_query=True)
    print("streaming from", url)


def open_nwb():
    """Open the NWB file with h5py: the local file, or the remote one if STREAM is True."""
    return h5py.File(remfile.File(url) if STREAM else NWB, "r")

# %% [markdown]
# ## 1. Load the units and their waveforms
#
# As in notebook 01, `nap.load_file` gives the units as a `TsGroup` and the trials as an
# `IntervalSet`, and we keep the trials of blocks 1–4 (sound, sound + opto, air puff; see the
# trial structure in notebook 01, section 1).
# We do not remove the saturation periods here (notebook 01, section 1.1): they fall outside the
# trials and only touch a few windows between them, so their effect is negligible.
# This time, look at the **metadata columns** of the
# units: pynapple already read
# several **waveform features** stored in the NWB units table (`Peak_to_Trough`, `PeaktoBaseline`,
# `HW`, `PTR`, ...), one value per unit. We will compute them ourselves in section 2, and compare.

# %%
if STREAM:
    import pynwb
    data = nap.NWBFile(pynwb.NWBHDF5IO(file=open_nwb()).read())  # remote file
else:
    data = nap.load_file(NWB)  # open the file and list its content

units = data["units"]  # all sorted units (TsGroup)
trials = data["trials"]  # task trials (IntervalSet)
trials = trials[trials.Block <= 4]  # blocks 1-4, as in notebook 01
print("number of units:", len(units))
print("metadata columns:", units.metadata_columns)

# %% [markdown]
# **The mean waveforms.** The units table also has a column `waveform_means`: for each unit, the
# **average of all its spikes**, on the electrode where the spike is the largest (the "main"
# electrode). Averaging thousands of spikes removes the noise and the spikes of the other neurons,
# and leaves the typical shape of the spike of this unit: its **template**. pynapple does not read
# this column (it is not one value per unit, but a whole vector), so we read it with `h5py`.
#
# Each waveform was cut from 82 samples of the recording (2.7 ms at 30 kHz, the sampling rate of
# Neuropixels), and stored **upsampled 1000 times** (82,000 points, i.e. one point every
# 0.033 µs!). We do not need that much: we keep one point in 100, i.e. 10 points per sample
# (one point every 3.3 µs). The values are in microvolts (µV).
#
# We also read, as in notebook 01, the ISI violation rate (`quality`) and the FS / RS labels. The
# brain area of each unit (`location`) and its depth (`DV`, dorso-ventral coordinate of its
# electrode, in mm) were already read by pynapple.

# %%
sampling_rate = 30_000  # Hz, sampling rate of the recording
upsampling = 10  # points per sample that we keep (the file has 1000)

with open_nwb() as f:
    assert np.array_equal(f["units/id"][:], units.index)  # same units, in the same order
    waveforms = f["units/waveform_means"][:, ::1000 // upsampling]  # (units x points), in µV
    isi_fp = f["units/quality"][:, 1]  # ISI violation false-positive rate
    fs = f["units/FS"][:] == 1  # fast-spiking label of the file
    rs = f["units/RS"][:] == 1  # regular-spiking label of the file

dt = 1000 / (sampling_rate * upsampling)  # time between two points, in ms
t_wf = np.arange(waveforms.shape[1]) * dt  # time axis of the waveforms, in ms
print("waveforms:", waveforms.shape, f"(units x points), one point every {dt * 1000:.1f} µs")

file_label = np.where(fs, "FS", np.where(rs, "RS", "none"))  # label stored in the file
units.set_info(isi_fp=isi_fp, file_label=file_label)

# %% [markdown]
# **Quality criteria.** We keep the units with an ISI violation rate below 0.1 and a presence
# ratio above 0, as in notebook 01 (section 2). Notebook 01 also required "a plausible waveform"
# (an FS or RS label): here we do **not** use the labels of the file, because we want to build
# our own. We keep the units of **all the brain areas** of the probe, to have more units in each
# class. `area` is the location without the layer (e.g. `PL6a` → `PL`).

# %%
presence_ratio = (units.count(units.time_support.tot_length() / 100).values > 0).mean(axis=0)
area = units.location.str.extract(r"^([A-Za-z]+?)(?:\d|$)", expand=False)  # remove the layer
units.set_info(presence_ratio=presence_ratio, area=area)

is_good = (units.isi_fp < 0.1) & (units.presence_ratio > 0)
good = units[is_good]
wf = waveforms[is_good.values]  # the waveforms of the good units, in the same order
print("good units:", len(good), "out of", len(units))
good.area.value_counts()  # how many units per brain area

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
# bursty unit B of notebook 01. You will meet both again in notebook 03.

# %%
examples = {"unit N": 171, "unit W": 58}  # name -> unit id: a Narrow and a Wide spike
colors = {"unit N": "tab:red", "unit W": "tab:blue"}  # one color per unit in all figures
row = {i: k for k, i in enumerate(good.keys())}  # unit id -> row of the waveform array `wf`

fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
for name, i in examples.items():
    print(f"{name}: id {i}, {good.location[i]}, rate {good.rate[i]:.1f} Hz")
    axes[0].plot(t_wf, wf[row[i]], color=colors[name], lw=2, label=f"{name} ({i})")
    # Normalized: divided by the depth of the trough, to compare the shapes
    axes[1].plot(t_wf, wf[row[i]] / -wf[row[i]].min(), color=colors[name], lw=2, label=f"{name} ({i})")
axes[0].set(xlabel="Time (ms)", ylabel="Voltage (µV)", title="Mean waveforms")
axes[1].set(xlabel="Time (ms)", ylabel="Voltage / trough depth", title="Normalized waveforms")
for ax in axes:
    ax.axhline(0, color="gray", lw=0.5)
    ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# Both units have the typical shape: a negative trough, then a positive peak. But unit N comes
# back up **much faster** than unit W, and its peak is higher relative to its trough. These are
# the two signatures of a fast-spiking interneuron.
#
# Now all the good units at once, normalized by their trough. The waveforms are aligned on the
# trough (around 1.35 ms) by the spike sorter.

# %%
normalized = wf / -wf.min(axis=1, keepdims=True)  # each waveform divided by its trough depth

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t_wf, normalized.T, color="k", lw=0.5, alpha=0.1)
ax.set(xlabel="Time (ms)", ylabel="Voltage / trough depth",
       title=f"Normalized waveforms of the {len(good)} good units")
plt.show()

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
def waveform_features(w, dt):
    """Features of one waveform w (a vector, in µV), with dt the time between two points (ms)."""
    trough = np.argmin(w)  # index of the trough
    peak = trough + np.argmax(w[trough:])  # index of the highest point after the trough
    back = np.flatnonzero(w[trough:] >= 0)  # points after the trough at or above the baseline
    baseline = trough + back[0] if len(back) else len(w) - 1  # first return to the baseline
    half = w[trough] / 2  # half of the trough depth
    left, right = trough, trough  # walk left and right from the trough while below half depth
    while left > 0 and w[left - 1] <= half:
        left -= 1
    while right < len(w) - 1 and w[right + 1] <= half:
        right += 1
    return {"trough_to_peak": round((peak - trough) * dt, 4),  # ms
            "peak_to_baseline": round((baseline - trough) * dt, 4),  # ms
            "half_width": round((right - left) * dt, 4),  # ms
            "peak_trough_ratio": w[peak] / w[trough],  # no unit, negative
            "amplitude": w[peak] - w[trough],  # µV
            "trough": trough, "peak": peak, "baseline": baseline,  # indices, for figures
            "left": left, "right": right}


# The features of unit W, drawn on its waveform
i = examples["unit W"]
w, ft = wf[row[i]], waveform_features(wf[row[i]], dt)
above = w[ft["peak"]] + 0.1 * ft["amplitude"]  # height of the trough-to-peak arrow, above the peak
below = w[ft["trough"]] / 4  # height of the peak-to-baseline arrow, between the trough and 0


def double_arrow(ax, x1, x2, y, label):
    """A horizontal double arrow from x1 to x2 at height y, with a label on its right."""
    ax.annotate("", xy=(x2, y), xytext=(x1, y), arrowprops=dict(arrowstyle="<->"))
    ax.text(x2 + 0.05, y, label, va="center")


fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(t_wf, w, color=colors["unit W"], lw=2)
ax.plot(t_wf[[ft["trough"], ft["peak"], ft["baseline"]]], w[[ft["trough"], ft["peak"], ft["baseline"]]], "ko")
double_arrow(ax, t_wf[ft["trough"]], t_wf[ft["peak"]], above, f"trough-to-peak: {ft['trough_to_peak']:.2f} ms")
double_arrow(ax, t_wf[ft["trough"]], t_wf[ft["baseline"]], below,
             f"peak-to-baseline: {ft['peak_to_baseline']:.2f} ms")
double_arrow(ax, t_wf[ft["left"]], t_wf[ft["right"]], w[ft["trough"]] / 2,
             f"half-width: {ft['half_width']:.2f} ms")
ax.axhline(0, color="gray", lw=0.5)
ax.set_ylim(top=above + 0.15 * ft["amplitude"])  # room for the text
ax.set(xlabel="Time (ms)", ylabel="Voltage (µV)",
       title=f"unit W ({i}): peak-to-trough ratio {ft['peak_trough_ratio']:.2f}, "
             f"amplitude {ft['amplitude']:.0f} µV")
plt.show()

# %% [markdown]
# We compute the features of all the good units, and store them as metadata of the `TsGroup`,
# next to the features of the file.

# %%
feature_names = ["trough_to_peak", "peak_to_baseline", "half_width", "peak_trough_ratio", "amplitude"]
features = pd.DataFrame([waveform_features(w, dt) for w in wf], index=good.index)[feature_names]
good.set_info(features)  # one metadata column per feature
features.describe().round(2)

# %% [markdown]
# **Do we agree with the file?** The NWB file has its own features (read by pynapple):
# `Peak_to_Trough` (ms), `PeaktoBaseline` (ms), `HW` (half-width, ms) and `PTR` (peak-to-trough
# ratio). We compare them with ours, unit by unit.

# %%
pairs = [("trough_to_peak", "Peak_to_Trough"), ("peak_to_baseline", "PeaktoBaseline"), ("half_width", "HW"),
         ("peak_trough_ratio", "PTR")]
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, (ours, theirs) in zip(axes, pairs):
    ax.plot(good[theirs], good[ours], "k.", ms=4)
    lims = [min(good[theirs].min(), good[ours].min()), max(good[theirs].max(), good[ours].max())]
    ax.plot(lims, lims, color="tab:green", lw=1, label="same value")
    ax.set(xlabel=f"{theirs} (file)", ylabel=f"{ours} (ours)")
    ax.legend()
plt.tight_layout()
plt.show()

disagree = good[np.abs(good.trough_to_peak - good.Peak_to_Trough) > 0.05]  # more than 0.05 ms apart
print("units with a trough-to-peak differing by more than 0.05 ms:", list(disagree.keys()))

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

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(good.trough_to_peak, bins=np.arange(0, 1.5, 0.025), color="gray")
ax.axvspan(low_th, high_th, color="orange", alpha=0.3, lw=0, label=f"{low_th}–{high_th} ms: unclassified")
ax.set(xlabel="Trough-to-peak duration (ms)", ylabel="Number of units",
       title="Trough-to-peak duration of the good units")
ax.legend()
plt.show()

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
# `groupby` (a pynapple method) gives the ids of the units of each class, and `getby_category`
# gives one `TsGroup` per class.

# %%
wf_class = np.select([good.trough_to_peak < low_th, good.trough_to_peak > high_th], ["narrow", "wide"],
                     default="unclassified")
good.set_info(wf_class=wf_class)
print("unclassified units:", good.groupby("wf_class")["unclassified"])  # groupby: class -> unit ids
by_class = good.getby_category("wf_class")  # class -> TsGroup
classes = ["narrow", "wide"]  # the two classes that we compare in sections 4 and 5
class_colors = {"narrow": "tab:red", "wide": "tab:blue", "unclassified": "tab:orange"}
for c in ["narrow", "wide", "unclassified"]:
    print(f"{c}: {len(by_class[c])} units ({100 * len(by_class[c]) / len(good):.0f} %)")

# %% [markdown]
# About 15 % of narrow waveforms: close to the proportion of interneurons in the cortex
# (~20 %), of which only part are fast-spiking. **Compare with the labels of the file**, used in
# notebooks 01 and 03: `pd.crosstab` counts the units for each pair of labels.

# %%
pd.crosstab(good.wf_class, good.file_label)

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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
for c, g in by_class.items():
    ax1.scatter(g.trough_to_peak, g.peak_to_baseline, s=12, color=class_colors[c], label=f"{c} (n={len(g)})")
for c in classes:
    rows = [row[i] for i in by_class[c].keys()]
    q25, median, q75 = np.percentile(normalized[rows], [25, 50, 75], axis=0)
    ax2.plot(t_wf, median, color=class_colors[c], lw=2, label=c)
    ax2.fill_between(t_wf, q25, q75, color=class_colors[c], alpha=0.2, lw=0)
for th in (low_th, high_th):
    ax1.axvline(th, color="k", ls="--", lw=0.8)
ax1.set(xlabel="Trough-to-peak duration (ms)", ylabel="Peak-to-baseline duration (ms)",
        title="Two waveform features")
ax2.set(xlabel="Time (ms)", ylabel="Voltage / trough depth", title="Median waveform (25–75 %) of each class")
ax1.legend()
ax2.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# Narrow waveforms have both a short trough-to-peak and a short peak-to-baseline duration, and a
# larger peak after the trough. In the literature, many more features are used: the half-width,
# the slopes of the repolarization and of the recovery, and features of the spike on the
# **neighbouring electrodes** (how far the spike spreads, how fast it travels along the probe).
# With a full spike sorting output, SpikeInterface computes them with the `template_metrics`
# extension of a `SortingAnalyzer` (see the spike sorting notebooks of this school).
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
# We fit a GMM with **two clusters** to the two features of section 3.2, with
# [scikit-learn](https://scikit-learn.org). We first **standardize** each feature (subtract its
# mean, divide by its standard deviation), so that both features have the same weight.

# %%
from sklearn.mixture import GaussianMixture

X = good.metadata[["trough_to_peak", "peak_to_baseline"]].values  # the 2 durations: units x 2
mean, sd = X.mean(axis=0), X.std(axis=0)
Z = (X - mean) / sd  # standardized features: one row per unit

# n_init=10: 10 fits from different starting points, keep the best; random_state: same result for all
gmm = GaussianMixture(n_components=2, covariance_type="full", n_init=10, random_state=0).fit(Z)
narrow_cluster = np.argmin(gmm.means_[:, 0])  # the cluster with the shortest trough-to-peak
p_narrow = gmm.predict_proba(Z)[:, narrow_cluster]  # probability of each unit to be narrow
good.set_info(p_narrow=p_narrow, gmm_class=np.where(p_narrow > 0.5, "narrow", "wide"))

uncertain = good[(good.p_narrow > 0.1) & (good.p_narrow < 0.9)]
print("units with a probability between 0.1 and 0.9:", list(uncertain.keys()))
pd.crosstab(good.wf_class, good.gmm_class)  # dual threshold (rows) vs GMM (columns)

# %% [markdown]
# We draw each unit with the color of its probability to be narrow, and the **boundary** of the GMM
# (probability 0.5), computed on a grid of points.

# %%
ttp_grid, ptb_grid = np.meshgrid(np.linspace(0.15, 1.5, 300), np.linspace(0.05, 0.9, 300))
grid = (np.c_[ttp_grid.ravel(), ptb_grid.ravel()] - mean) / sd  # same transform as the units
p_grid = gmm.predict_proba(grid)[:, narrow_cluster].reshape(ttp_grid.shape)

fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(good.trough_to_peak, good.peak_to_baseline, c=good.p_narrow, cmap="coolwarm", s=14,
                vmin=0, vmax=1)
ax.contour(ttp_grid, ptb_grid, p_grid, levels=[0.5], colors="k", linewidths=1)
for th in (low_th, high_th):
    ax.axvline(th, color="k", ls="--", lw=0.8)
ax.set(xlabel="Trough-to-peak duration (ms)", ylabel="Peak-to-baseline duration (ms)",
       title="GMM with two clusters (black line: probability 0.5)")
fig.colorbar(sc, ax=ax, label="Probability to be narrow")
plt.show()

# %% [markdown]
# The GMM agrees with the dual threshold for (almost) all the narrow and wide units, and splits the
# unclassified units of the gap between the two classes. Its boundary is not vertical: a unit with a
# trough-to-peak duration in the gap is called narrow if its peak-to-baseline duration is short.
# The units with a probability far from 0 and 1 are the ones to look at by hand.
#
# A GMM is not magic: it assumes that each class is a Gaussian blob, and it is sensitive to
# outliers and to the way we transform the features (see the questions).
#
# **Questions.**
#
# - Fit the GMM on the **logarithm** of the durations (`np.log(X)`): it is often recommended for
#   durations, which have long tails to the right. Do the classes change? Why?
# - Add the half-width as a third feature. Do the classes change?

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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), width_ratios=[1.3, 1])
rate_bins = np.logspace(-2, 2, 33)  # from 0.01 Hz to 100 Hz, evenly spaced in log
for c in classes:
    g = by_class[c]
    ax1.scatter(g.trough_to_peak, g.rate, s=12, color=class_colors[c], label=c)
    # weights: each unit counts 1 / (number of units of its class), to compare classes of different sizes
    ax2.hist(g.rate, bins=rate_bins, histtype="step", lw=2, weights=np.full(len(g), 1 / len(g)),
             color=class_colors[c], label=f"{c}: median {np.median(g.rate):.1f} Hz")
for th in (low_th, high_th):
    ax1.axvline(th, color="k", ls="--", lw=0.8)
ax1.set(yscale="log", xlabel="Trough-to-peak duration (ms)", ylabel="Session rate (Hz)")
ax2.set(xscale="log", xlabel="Session rate (Hz)", ylabel="Fraction of the units of the class",
        title="Rate of each class")
ax1.legend()
ax2.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# Narrow units fire on average **about 3 times faster** than wide units, as expected for
# fast-spiking interneurons. But the two distributions overlap a lot: some wide units fire at
# 20 Hz, and some narrow units at about 1 Hz. **The rate alone would be a poor classifier**.
#
# ### 4.2 Autocorrelograms
#
# The autocorrelogram (ACG, see notebook 03) is the rate of a neuron at a time $\tau$ after one of
# its spikes. With `norm=True`, pynapple divides it by the mean rate of the unit, so that units
# with different rates can be compared: 1 means "as by chance". We compute the ACGs of the good
# units with 1 ms bins, up to ±50 ms, and show the two example units.

# %%
acg = nap.compute_autocorrelogram(good, binsize=0.001, windowsize=0.05, norm=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), sharex=True)
for ax, (name, i) in zip(axes, examples.items()):
    ax.bar(acg.index * 1000, acg[i], width=1, color=colors[name])
    ax.axhline(1, color="k", ls="--", label="as by chance")
    ax.set(title=f"{name} ({i}, {good.wf_class[i]} waveform), {good.rate[i]:.1f} Hz", xlabel="Lag (ms)")
    ax.legend(loc="lower right")
axes[0].set_ylabel("Normalized ACG")
plt.tight_layout()
plt.show()

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
plateau = acg.loc[0.03:0.05].mean()  # mean of each ACG between 30 and 50 ms
shape = (acg / plateau).loc[0.0005:]  # ACG divided by its plateau, positive lags only
two_classes = good.metadata[good.metadata.wf_class.isin(classes)]  # narrow and wide units
order = two_classes.sort_values(["wf_class", "trough_to_peak"]).index  # narrow first ("n" < "w")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), width_ratios=[1, 1.2])
im = ax1.imshow(shape[order].T, aspect="auto", cmap="viridis", vmin=0, vmax=2, interpolation="nearest",
                extent=[0, shape.index[-1] * 1000 + 0.5, len(order), 0])
ax1.axhline(len(by_class["narrow"]), color="w", lw=2)
ax1.set(xlabel="Lag (ms)", ylabel="Unit (narrow on top, then wide)", title="ACG / plateau")
fig.colorbar(im, ax=ax1)
for c in classes:
    ax2.plot(shape.index * 1000, shape[by_class[c].index].median(axis=1), color=class_colors[c], lw=2, label=c)
ax2.axhline(1, color="k", ls="--", label="plateau")
ax2.set(xlabel="Lag (ms)", ylabel="ACG / plateau", title="Median ACG of each class")
ax2.legend()
plt.tight_layout()
plt.show()

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
# Each unit has the depth of its main electrode (`DV`, read by pynapple). We plot the
# trough-to-peak duration of each unit against its depth, with the brain areas along the probe, and
# the fraction of narrow units (among the narrow and wide units) in each area, from top to bottom
# of the probe.

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), width_ratios=[1.3, 1])
for c, g in by_class.items():
    ax1.scatter(g.trough_to_peak, g.DV, s=12, color=class_colors[c], label=c)
area_depth = good.metadata.groupby("area").DV.mean().sort_values()  # mean depth of the units of each area
for a, d in area_depth.items():  # name of each area, at the mean depth of its units
    ax1.text(1.45, d, a, va="center", ha="right", fontsize=9)
for th in (low_th, high_th):
    ax1.axvline(th, color="k", ls="--", lw=0.8)
ax1.set(xlabel="Trough-to-peak duration (ms)", ylabel="Dorso-ventral coordinate (mm)",
        title="Units along the probe")
ax1.legend(loc="lower left")

counts = pd.crosstab(good.area, good.wf_class).loc[area_depth.index]  # units per area and class, by depth
fraction = counts.narrow / (counts.narrow + counts.wide)  # fraction of narrow units in each area
ax2.barh(counts.index, fraction, color="tab:red")
for k, a in enumerate(counts.index):
    ax2.text(fraction[a] + 0.005, k, f"{counts.narrow[a]} / {counts.narrow[a] + counts.wide[a]}", va="center")
ax2.set(xlabel="Fraction of narrow units", title="Narrow units per area")
plt.tight_layout()
plt.show()

# %% [markdown]
# Narrow units are found all along the probe, from 5 % of the units (ACAd) to 30 % (ILA). But
# with a few units per area, these fractions are **not reliable**: the spike sorter also finds more easily the neurons that fire a lot (like
# fast-spiking interneurons) or that have large spikes, so these fractions are not the true
# proportions of the cell types in each area.
#
# ### 5.2 Responses to the stimuli
#
# Do the two classes respond differently to the task? We take the three types of trials (as in
# notebook 01): **sound**, **sound + opto** and **air puff** (a blue noise followed by an air puff). For each unit and each type of trial, `nap.compute_eventcorrelogram`
# gives the PSTH in one line (here 50 ms bins, from 1 s before to 1 s after the trial onset).
#
# To compare units with different rates, we **z-score** each PSTH with its baseline (the second
# before the onset): $z = (\text{PSTH} - \text{mean of the baseline}) / \text{SD of the baseline}$.
# A unit responds if its mean $z$ in the first 500 ms is above 2 (an increase) or below −2 (a
# decrease).

# %%
trial_type = np.where(trials.airpuff == 1, "air puff",
                      np.where(trials.optogenetics, "sound + opto", "sound"))
stimuli = ["sound", "sound + opto", "air puff"]

zscored = {}  # trial type -> z-scored PSTHs (DataFrame: one row per bin, one column per unit)
for s in stimuli:
    onsets = nap.Ts(t=trials.start[trial_type == s])
    psth = nap.compute_eventcorrelogram(good, onsets, binsize=0.05, windowsize=1.0, norm=False)
    baseline = psth.loc[:-0.001]  # bins before the onset
    zscored[s] = (psth - baseline.mean()) / baseline.std().replace(0, np.nan)
    print(f"{s}: {len(onsets)} trials")

fig, axes = plt.subplots(2, 3, figsize=(16, 8), height_ratios=[2, 1], sharex=True)
for col, s in enumerate(stimuli):
    z = zscored[s]
    response = z.loc[0:0.5].mean()  # mean z in the first 500 ms, one value per unit
    # Heatmap: one row per unit, narrow units on top, each class sorted by response
    rows = []
    for c in classes:
        rows += list(response[by_class[c].index].sort_values(ascending=False).index)
    axes[0, col].imshow(z[rows].T, aspect="auto", cmap="RdBu_r", vmin=-5, vmax=5, interpolation="nearest",
                        extent=[z.index[0] - 0.025, z.index[-1] + 0.025, len(rows), 0])
    axes[0, col].axhline(len(by_class["narrow"]), color="k", lw=1.5)
    axes[0, col].set(title=s, ylabel="Unit (narrow on top)" if col == 0 else "")
    for c in classes:
        r = response[by_class[c].index].dropna()
        up, down = (r > 2).mean(), (r < -2).mean()
        axes[1, col].plot(z.index, z[by_class[c].index].mean(axis=1), color=class_colors[c], lw=2,
                          label=f"{c}: {100 * up:.0f} % up, {100 * down:.0f} % down")
    axes[1, col].axvline(0, color="gray", lw=0.8)
    axes[1, col].set(xlabel="Time from trial onset (s)", ylabel="Mean z-score" if col == 0 else "")
    axes[1, col].legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.show()

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
is_positive = wf.max(axis=1) > -wf.min(axis=1)  # largest deflection is positive
good.set_info(cell_class=np.where(is_positive, "positive", good.wf_class))  # 3 categories (+ unclassified)
positive = good[good.cell_class == "positive"]
print("positive units:", list(positive.keys()))
pd.crosstab(good.cell_class, good.file_label)

# %%
n_col = min(5, len(positive))  # at most 5 panels per row
n_row = int(np.ceil(len(positive) / n_col))
fig, axes = plt.subplots(n_row, n_col, figsize=(3.2 * n_col, 3 * n_row), sharex=True, squeeze=False)
for ax, i in zip(axes.flat, positive.keys()):
    w, ft = wf[row[i]], waveform_features(wf[row[i]], dt)
    ax.plot(t_wf, w, color="k")
    ax.plot(t_wf[[ft["trough"], ft["peak"]]], w[[ft["trough"], ft["peak"]]], "o",
            color=class_colors[good.wf_class[i]])  # the trough and peak that we used in section 3
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title(f"{i} ({good.location[i]})\nours: {good.trough_to_peak[i]:.2f} ms, {good.wf_class[i]}\n"
                 f"file: {good.Peak_to_Trough[i]:.2f} ms, {good.file_label[i]}", fontsize=9)
for ax in axes.flat[len(positive):]:
    ax.axis("off")  # empty panels
for ax in axes[-1]:
    ax.set_xlabel("Time (ms)")
for ax in axes[:, 0]:
    ax.set_ylabel("Voltage (µV)")
plt.tight_layout()
plt.show()

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
