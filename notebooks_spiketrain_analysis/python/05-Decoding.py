# %% [markdown]
# # 05 · Decoding — **Python / pynapple**
#
# In notebook 04 we looked for the main patterns of the population activity, without telling the
# methods anything about the task. **Decoding** goes the other way: we ask whether the activity of
# the population **contains information about the task**. We train a classifier to guess, from the
# spike counts of the units in one trial, a property of this trial, here its **block** (but it could be many other features):
#
# | Block | What happens at the start of the trial |
# |---|---|
# | 1 | a sound, before the conditioning (**pre**) |
# | 2 | the same sound, followed by an optogenetic stimulation (**sound + opto**) |
# | 3 | the same sound alone, after the conditioning (**post**) |
# | 4 | a blue noise followed by an air puff (**air puff**) |
#
# The classifier needs to guess the block identity at the trial level. Importantly, the classifier guesses the block of **new trials**, not used to train it, better than chance,
# the population carries information about the block. We use two classic **linear** classifiers:
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
# below (see section 0 of notebook 01).

# %%
# numpy: arrays and maths; matplotlib: figures; pandas: tables; pynapple: spike trains;
# h5py: low-level access to the NWB file; scikit-learn: the classifiers and cross-validation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pynapple as nap
import h5py
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, PredefinedSplit, cross_val_predict
from sklearn.metrics import confusion_matrix

# Option 1: full path to the NWB file you downloaded. Change it to where you saved it.
NWB = "/path/to/your/data/folder/sub-156131_ses-156131-20191112-probe0_ecephys+ogen.nwb"

# Option 2: set to True to stream the file from the DANDI Archive instead
STREAM = True

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
# ## 1. Load the units and the trials
#
# Same steps as in notebook 01 (sections 1 and 2): the trials of blocks 1–4, and the units of
# ACA, PL, ILA and ORB that pass the quality criteria.
# We do not remove the saturation periods here (notebook 01, section 1.1): they fall outside the
# trials and only touch a few windows between them, so their effect is negligible.

# %%
if STREAM:
    import pynwb
    data = nap.NWBFile(pynwb.NWBHDF5IO(file=open_nwb()).read())  # remote file
else:
    data = nap.load_file(NWB)  # open the file and list its content

units = data["units"]  # all sorted units (TsGroup)
trials = data["trials"]  # task trials (IntervalSet)
trials = trials[trials.Block <= 4]  # blocks 1-4

with open_nwb() as f:
    isi_fp = f["units/quality"][:, 1]  # ISI violation false-positive rate
    fs = f["units/FS"][:] == 1  # fast-spiking label of the file
    rs = f["units/RS"][:] == 1  # regular-spiking label of the file

presence_ratio = (units.count(units.time_support.tot_length() / 100).values > 0).mean(axis=0)
area = units.location.str.extract(r"^(ACA|PL|ILA|ORB)", expand=False)  # NaN outside the 4 areas
units.set_info(isi_fp=isi_fp, plausible=fs | rs, presence_ratio=presence_ratio, area=area)
pfc = units[units.area.notna() & (units.isi_fp < 0.1) & units.plausible & (units.presence_ratio > 0)]

block = trials.Block.astype(int)  # the label to decode: block of each trial
block_names = {1: "sound (pre)", 2: "sound + opto", 3: "sound (post)", 4: "air puff"}
block_colors = {1: "tab:blue", 2: "tab:purple", 3: "tab:cyan", 4: "tab:orange"}  # as in notebook 04
area_colors = {"ACA": "C0", "PL": "C1", "ILA": "C2", "ORB": "C4"}
print("trials:", len(trials), "| PFC units:", len(pfc))
print(pd.Series(block).map(block_names).value_counts().to_string())

# %% [markdown]
# ## 2. One trial = one vector of spike counts
#
# For each trial, we count the spikes of every unit in the **first 2 seconds after the onset**:
# `count` with an `IntervalSet` and no bin size gives one count per interval. So very important here: we will feed a single count value per unit and per trial (we loose so much temporal information 😭!!). The result is a
# table of **trials × units**: each trial is a point in a 192-dimensional space, and the block is
# its label.

# %%
after = nap.IntervalSet(start=trials.start, end=trials.start + 2)  # 0 to 2 s after each onset
X = pfc.count(ep=after).values  # trials x units
y = block
print("X:", X.shape, "(trials x units) | y:", y.shape)

fig, ax = plt.subplots(figsize=(12, 4))
z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)  # z-score each unit, for the figure only
im = ax.imshow(z.T, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, interpolation="none")
for b in [2, 3, 4]:
    ax.axvline(np.flatnonzero(y == b)[0] - 0.5, color="k", lw=2)  # first trial of each block
ax.set(xlabel="Trial (in session order; black lines: start of blocks 2, 3, 4)", ylabel="Unit",
       title="Spike counts 0–2 s after the onset (z-scored)")
plt.colorbar(im)
plt.show()

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
# simpler, diagonal one (`shrinkage=0.5`: half-way between the two): a classic regularization.
#
# Below, we fit LDA on **all** the trials and project them on the first two discriminant axes.

# %%
lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage=0.5)
projection = lda.fit(X, y).transform(X)  # trials x 3 discriminant axes

fig, ax = plt.subplots(figsize=(6, 5))
for b, name in block_names.items():
    ax.scatter(*projection[y == b, :2].T, s=12, color=block_colors[b], label=name)
ax.set(xlabel="Discriminant axis 1", ylabel="Discriminant axis 2", title="LDA projection (all trials)")
ax.legend()
plt.show()

# %% [markdown]
# The four blocks are well separated. But this is **not** a proof that the activity carries
# information about the block: with 192 dimensions, a classifier can separate almost any
# labels on the trials it was trained on (try it with shuffled labels: `rng.permutation(y)`).
# We must test it on trials it has **never seen**.
#
# ### 3.2 Cross-validation
#
# **k-fold cross-validation**: we cut the trials into 5 groups (the **folds**), train on 4 of them,
# test on the 5th, and repeat so that every trial is tested once. `StratifiedKFold` keeps the same
# proportion of each block in every fold, and `cross_val_predict` returns the prediction for each
# trial, made by a classifier that did not see it.
#
# **Chance level.** Block 2 has 100 trials, the others 50: always answering "block 2" would
# already be right 40 % of the time. A better reference: decode **shuffled labels** (the
# link between activity and block is destroyed), several times.

# %%
rng = np.random.default_rng(0)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


def decode(clf, X, y, cv):
    """Cross-validated predictions and accuracy (fraction of trials decoded correctly)."""
    predicted = cross_val_predict(clf, X, y, cv=cv)
    return predicted, np.mean(predicted == y)


predicted, accuracy = decode(lda, X, y, cv)
chance = [decode(lda, X, rng.permutation(y), cv)[1] for _ in range(20)]  # 20 label shuffles
print(f"LDA accuracy: {accuracy:.2f} | shuffled labels: {np.mean(chance):.2f} ± {np.std(chance):.2f}")


def plot_confusion(ax, y, predicted, title):
    """Confusion matrix in % of the trials of each true block (rows)."""
    cm = confusion_matrix(y, predicted, normalize="true") * 100
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
    for r in range(4):
        for c in range(4):
            ax.text(c, r, f"{cm[r, c]:.0f}", ha="center", va="center", color="w" if cm[r, c] > 50 else "k")
    ax.set(xticks=range(4), yticks=range(4), xticklabels=block_names.values(), yticklabels=block_names.values(),
           xlabel="Decoded block", ylabel="True block", title=title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


fig, ax = plt.subplots(figsize=(5, 4.5))
plot_confusion(ax, y, predicted, f"LDA, 0–2 s after the onset ({100 * accuracy:.0f} %)")
plt.show()

# %% [markdown]
# Almost all the trials are decoded correctly, far above the shuffled labels. The **confusion
# matrix** shows the rare mistakes: which block is taken for which.
#
# ## 4. Support vector machine (SVM)
#
# A linear SVM also separates the classes with flat boundaries (hyperplanes), but it chooses them
# differently: it places each boundary as **far as possible from the closest trials** (the
# "support vectors"), and ignores the trials far from the boundary. It makes no assumption about
# the distribution of the classes. With more than two classes, `SVC` trains one SVM for each pair
# of blocks, and each trial gets the block that wins the most votes.
#
# Two practical points:
#
# - an SVM depends on the scale of each feature: we **z-score** the units first. With
#   `make_pipeline`, the z-scoring is learned on the training trials only, inside each fold;
# - `C` sets the trade-off between a wide margin and few training errors (small `C`: wider margin,
#   more regularization).

# %%
svm = make_pipeline(StandardScaler(), SVC(kernel="linear", C=0.01))
predicted_svm, accuracy_svm = decode(svm, X, y, cv)
print(f"SVM accuracy: {accuracy_svm:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
plot_confusion(axes[0], y, predicted, f"LDA ({100 * accuracy:.0f} %)")
plot_confusion(axes[1], y, predicted_svm, f"SVM ({100 * accuracy_svm:.0f} %)")
plt.tight_layout()
plt.show()

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
# **Control: decode the block from the 2 seconds *before* the onset**, when no stimulus has been
# given yet. We use a window of the same length as after the onset (2 s), so that the counts are
# comparable. Two trials are always more than 3 s apart: this window never overlaps with the
# previous trial.

# %%
before = nap.IntervalSet(start=trials.start - 2, end=trials.start)  # 2 s before each onset
X_before = pfc.count(ep=before).values
predicted_before, accuracy_before = decode(lda, X_before, y, cv)
print(f"LDA accuracy before the onset: {accuracy_before:.2f}")

# %% [markdown]
# The block is decoded almost as well **before** the stimulus as after it! Most of what the
# decoder uses is the slow change of the activity over the session, which we already saw in
# notebook 04 (the state coding direction, and the trial factors of TCA). Two common fixes:
#
# 1. **Evoked activity**: subtract, in each trial, the count before the onset from the count
#    after it (`X - X_before`). What remains is the change caused by the stimulus.
# 2. **Cross-validation in chunks**: with random folds, the test trials have neighbors (trials
#    a few seconds away) in the training set. We cut each block into 5 chunks of **consecutive**
#    trials instead, and use the chunks as folds (`PredefinedSplit`). The test trials are then
#    further in time from the training trials.

# %%
position = np.concatenate([np.arange(np.sum(y == b)) for b in block_names])  # trial number in its block
n_in_block = np.array([np.sum(y == b) for b in y])  # number of trials of the block of each trial
chunks = PredefinedSplit(position * 5 // n_in_block)  # fold = chunk (0-4) of consecutive trials

windows = {"after (0–2 s)": X, "before (-2–0 s)": X_before, "evoked (after − before)": X - X_before}
results = pd.DataFrame({cv_name: {w: decode(lda, Xw, y, folds)[1] for w, Xw in windows.items()}
                        for cv_name, folds in {"random folds": cv, "chunks": chunks}.items()})

fig, ax = plt.subplots(figsize=(8, 4))
results.plot.bar(ax=ax, rot=0, color=["gray", "k"])
ax.axhline(np.mean(chance), color="r", ls="--", label="shuffled labels")
ax.set(ylabel="LDA accuracy", ylim=(0, 1), title="What does the decoder use?")
ax.legend(loc="lower right")
plt.show()
results.round(2)

# %% [markdown]
# - **before** the onset, the accuracy drops a little with the chunks, but stays far above chance:
#   the slow changes really differ from block to block;
# - the **evoked** activity is decoded less well than the raw activity after the onset, but still
#   far above chance: the **responses** themselves differ between blocks.
#
# Look at the confusion matrix of the evoked activity, with the chunks: which blocks are now
# confused?

# %%
predicted_evoked, accuracy_evoked = decode(lda, X - X_before, y, chunks)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
plot_confusion(axes[0], y, predicted_before, f"before the onset ({100 * accuracy_before:.0f} %)")
plot_confusion(axes[1], y, predicted_evoked, f"evoked, chunks ({100 * accuracy_evoked:.0f} %)")
plt.tight_layout()
plt.show()

# %% [markdown]
# Before the onset, the few mistakes are between blocks that are **neighbors in time** (block 4
# taken for block 3, ...): the signature of a slow change over the session.
#
# With the evoked activity, the mistakes are mostly between blocks **1 and 3**: the same sound
# alone, before and after the conditioning. Blocks 2 and 4 are recognized by their response to the
# aversive stimulus (opto or air puff), which comes within the 2 s window. Blocks 1 and 3 are still
# partly separated: the response to the sound may have changed with the conditioning, but this is
# also where the slow changes of the session are hardest to rule out.
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
n_units = pfc.area.value_counts().min()  # number of units of the smallest area
n_draws = 20
area_accuracy = {}
for a in area_colors:
    in_area = np.flatnonzero(pfc.area.values == a)  # columns of the units of this area
    area_accuracy[a] = [decode(lda, (X - X_before)[:, rng.choice(in_area, n_units, replace=False)], y,
                               chunks)[1] for _ in range(n_draws)]

fig, ax = plt.subplots(figsize=(6, 4))
for k, (a, acc) in enumerate(area_accuracy.items()):
    ax.plot(k + rng.uniform(-0.15, 0.15, n_draws), acc, ".", color=area_colors[a], alpha=0.5)
    ax.errorbar(k, np.mean(acc), yerr=np.std(acc), fmt="o", color="k", capsize=4)
ax.axhline(np.mean(chance), color="r", ls="--", label="shuffled labels")
ax.set(xticks=range(4), xticklabels=[f"{a}\n({np.sum(pfc.area == a)} units)" for a in area_colors],
       ylabel="LDA accuracy (evoked, chunks)", ylim=(0, 1),
       title=f"{n_units} units drawn per area, {n_draws} times")
ax.legend()
plt.show()

# %% [markdown]
# With the same number of units, all the areas decode the block above chance, with similar
# accuracies. (ACA has only as many units as we draw: its 20 draws are all the same.) Differences
# between areas should be tested with more sessions and more animals: here, one session and one
# probe can only give a hint.
#
# **Questions.**
#
# - Decode only blocks 1 and 3 (same sound, before and after the conditioning:
#   `keep = np.isin(y, [1, 3])`). Is the difference still there with the evoked activity? And
#   before the onset?
# - Use the windows 0–1 s and 1–2 s after the onset instead of 0–2 s (and a 1 s window before the
#   onset). Which blocks are best decoded in each?
# - **Time-resolved decoding**: decode the block in successive 100 ms bins (with
#   `nap.build_tensor`, as in notebook 04). When does the accuracy rise above its level before
#   the onset?
# - Replace the shrinkage of LDA by `shrinkage=None`, and `C` of the SVM by `C=1`. What changes?
