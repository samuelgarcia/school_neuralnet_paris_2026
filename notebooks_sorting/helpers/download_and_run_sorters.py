import os
from pathlib import Path

import numpy as np
import spikeinterface.full as si

import requests

from tqdm.auto import tqdm


# change this path to point to the folder your data is in
# base_folder = Path("/home/samuel/Documents/DataSpikeSorting/school_neuralnet_2026/")
base_folder = Path("/mnt/data/sam/DataSpikeSorting/school_neuralnet_2026/")
# base_folder = Path("/home/pierre/school_neuralnet_2026/")

data_folder = base_folder / "data"
name = "sub-A3702_ses-191126-3600s"
result_folder = base_folder / "results" / name
result_folder.mkdir(exist_ok=True)


# the file is here https://dandiarchive.org/dandiset/000939/0.260512.1701/files?location=sub-A3702
download_link = "https://api.dandiarchive.org/api/assets/6fe925dd-ea41-4a84-9cee-ca6bb039edcb/download?content_disposition=inline"
local_nwb_file = data_folder / "sub-A3702_ses-191126_behavior+ecephys.nwb"

preprocessed_file = result_folder / f"{name}-preprocessed"

job_kwargs = dict(n_jobs=16, chunk_duration='1s')
si.set_global_job_kwargs(**job_kwargs)

def download_dandiset():
    with requests.get(download_link, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        bar = tqdm(total=total, unit="B", unit_scale=True)
        chunk_size= 1024 * 1024
        with open(local_nwb_file, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=chunk_size)):
                f.write(chunk)
                bar.update(chunk_size)


def _fix_probecosmetics(rec):
    # the nwb file do not have the real probe but only the contact position
    # here a make a better looking probe
    # also force the probe to be 2D
    probe = rec.get_probe()
    probe = probe.to_2d(axes="xy")
    n = probe.get_contact_count()
    probe._contact_shapes = np.array(['rect']*n)
    probe._contact_shape_params = np.array([{"width": 11.0,"height": 15.0}]*n)
    probe.create_auto_shape()
    rec._probegroup.probes[0] = probe

def do_preprocess():
    raw_rec = si.read_nwb_recording(local_nwb_file, electrical_series_path="acquisition/ElectricalSeries")
    print(raw_rec)
    _fix_probecosmetics(raw_rec)

    motion_folder = result_folder / "sub-A3702_ses-191126-motion"

    rec = raw_rec
    rec = rec.time_slice(0., 3600.)
    print(rec)
    rec = si.bandpass_filter(rec, freq_min=300, freq_max=6000)
    print(rec)
    rec = si.detect_and_interpolate_bad_channels(rec)
    print(rec)
    rec = si.common_reference(rec)
    print(rec)
    rec, motion_info = si.correct_motion(
        rec,
        preset="dredge_fast",
        output_motion=True,
        estimate_motion_kwargs={"rigid": True},
        folder=motion_folder,
    )
    print(rec)
    rec = rec.save(folder=preprocessed_file, format="binary")
    print(rec)




by_sorter_params = {
    "lupin": {
        "apply_preprocessing": True,
        "apply_motion_correction": False,
    },
    "kilosort4": {
        "use_binary_file": False,
        "do_correction": False,
    },
    "dartsort": {
        "do_motion_estimation": False,
    },
}


def run_some_sorters():
    sorter_names = by_sorter_params.keys()


    for sorter_name in sorter_names:
        rec = si.load(preprocessed_file)
        print(rec)

        if sorter_name == 'dartsort':
            rec = si.zscore(rec)

        sorter_folder = result_folder / f"sorter_{sorter_name}"
        si.run_sorter(
            sorter_name=sorter_name,
            recording=rec,
            folder=sorter_folder,
            verbose=True,
            **by_sorter_params[sorter_name]
        )



def run_some_analyzers():
    sorter_names = by_sorter_params.keys()

    rec = si.load(preprocessed_file)
    print(rec)


    for sorter_name in sorter_names:

        sorter_folder = result_folder / f"sorter_{sorter_name}"
        sorting = si.read_sorter_folder(sorter_folder)
        print(sorting)

        analyzer_folder = result_folder / f"analyzer_{sorter_name}"
        analyzer = si.create_sorting_analyzer(
            recording=rec,
            sorting=sorting,
            format='binary_folder',
            folder=analyzer_folder,
        )
        print(analyzer)

        analyzer.compute({
            "random_spikes": {"max_spikes_per_unit": 300},
            "templates": {},
            "waveforms": {},
            "correlograms": {},
            "noise_levels": {},
            "spike_amplitudes": {},
            "spike_locations": {},
            "template_metrics": {},
            "unit_locations": {},
            "template_similarity": {'method': 'l1'},
            "quality_metrics": {},
        })



if __name__ == '__main__':
    # download_dandiset()
    # do_preprocess()
    # run_some_sorters()
    run_some_analyzers()


