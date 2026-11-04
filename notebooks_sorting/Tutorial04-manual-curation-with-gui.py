import os
from pathlib import Path

import numpy as np
import spikeinterface.full as si


from spikeinterface_gui import run_mainwindow


# change this path to point to the folder your data is in

base_data_folder = Path("/mnt/data/sam/DataSpikeSorting/school_neuralnet_2026")
# base_data_folder = Path("/home/samuel/Documents/DataSpikeSorting/data_school_neuralnet_paris_2026/")


analyzer_folder = base_data_folder / 'full_analyzer_lupin_000939.zarr'
print(analyzer_folder)


analyzer = si.load(analyzer_folder)
print(analyzer)


def example1():
    # standard mode
    run_mainwindow(analyzer)

def example2():
    # options : run in web mode
    run_mainwindow(
        analyzer,
        # mode='desktop', # 
        mode='web', # 
        with_traces=True,
    )


def example3():
    # curation mode
    run_mainwindow(
        analyzer,
        curation=True
    )

def example4():
    # TODO with Pierre
    curation_dict = {}

    # curation mode with pre existing curation format
    run_mainwindow(
        analyzer,
        curation=True,
        curation_dict=curation_dict,
    )



if __name__ == '__main__':
    # example1()
    # example2()
    # example3()
    example4()

    