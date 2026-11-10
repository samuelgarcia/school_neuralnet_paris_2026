import os
from pathlib import Path

import numpy as np
import spikeinterface.full as si


from spikeinterface_gui import run_mainwindow_comparison


# change this path to point to the folder your data is in
# change this path to point to the folder your data is in
# base_folder = Path("/home/samuel/Documents/DataSpikeSorting/school_neuralnet_2026/")
base_folder = Path("/mnt/data/sam/DataSpikeSorting/school_neuralnet_2026/")
# base_folder = Path("/home/pierre/school_neuralnet_2026/")

data_folder = base_folder / "data"
name = "sub-A3702_ses-191126-3600s"
result_folder = base_folder / "results" / name
result_folder.mkdir(exist_ok=True)

sorter_name1 = "lupin"
# sorter_name1 = "kilosort4"
# sorter_name1 = "dartsort"
analyzer_folder1 = result_folder / f"analyzer_{sorter_name1}"
analyzer1 = si.load(analyzer_folder1)
print(analyzer1)

# sorter_name2 = "lupin"
# sorter_name2 = "kilosort4"
sorter_name2 = "dartsort"
analyzer_folder2 = result_folder / f"analyzer_{sorter_name2}"
analyzer2 = si.load(analyzer_folder2)
print(analyzer2)



def open_gui_compare_two_sorters():

    run_mainwindow_comparison(
        analyzer1,
        analyzer2,
        analyzer1_name=sorter_name1,
        analyzer2_name=sorter_name2,
        match_score=0.5,
        mode="desktop",
    )

if __name__ == '__main__':
    open_gui_compare_two_sorters()

    