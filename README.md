# school_neuralnet_paris_2026


Material for SpikeInterface Tutorial @ Neuralnet 2026

## Data

You can go through the notebooks in this course yourself, or code along with us on the day. To do so, please download `dandiset-000939-90s.zip` and `full_analyzer_lupin_000939.zarr.zip` from this Zenodo page: https://zenodo.org/records/23373455

This data was originally collated and organised for the [Ephys track](https://neuroinformatics.dev/open-software-summer-school/2026/extracellular-electrophysiology.html) of the [Open Software Summer School](https://neuroinformatics.dev/open-software-summer-school/index.html). You can read more about the data [here](https://neuroinformatics.dev/course_ephys_osss/course-data.html).


## Installation

We recommend that you [install uv](https://docs.astral.sh/uv/getting-started/installation/). This is an unbelievably good new package
management tool for Python.

Open your Terminal app and cd (change directory) to where you like to keep github repos. Then clone this repo, and navigate to the `notebooks` folder of it by entering the following commands into your Terminal:

``` shell
git clone https://github.com/SpikeInterface/SpikeInterface-Training-OpenLabOnline-Oct26.git
cd SpikeInterface-Training-OpenLabOnline-Oct26/notebooks/
```

Now open jupyter lab (you don't need to install this - uv will take care of it) using uv:

``` shell
uv run jupyter lab
```

Or if you prefer to use vscode:

``` shell
uv run code .
```

If you like traditional virtual environments, please visit [this page](https://github.com/SpikeInterface/spikeinterface/tree/main/installation_tips) to install on your laptop python+spikeinterface.


## Schedule

**November 23rd - 14:00-18:00 CET**


### Introduction

Display installation instructions and check that everyone has the necessary software ready
while people are entering the room

3. (mention various projects and papers)



**November 24th - 9:00-12:00 CET**

 * Introduction to Spike Sorting (PierreY 60min)
 * Introduction to `SpikeInterface` (Sam 60min)
 * Pause
 * Install checks (10min)
 * Tutorial 1 Reading + Manipulation (45 min - Sam)
    - Read and visualise data
    - The importance of probes
    - Splitting by channels 
    - (optional) Make your own probe
 * Tutorial 2 Preprocessing (45 min - PierreY)
    - Apply and visualize preprocessing 
      - filter
      - CMR
    - Detect bad channels
    - Saturation
    - Motion


**November 24th - 9:00-12:00 CET**

 * Tutorial 3 Spike sort and Analysis(60 min - PierreY)
    - Run a spike sorter (LUPIN)
    - Create a `SortingAnalyzer` to explore your sorting
    - Compute some extensions
    - Play with the extension outputs

 * Tutorial 4 Curation manual : GUI (45 min Sam)
    - GUI - manual curation
 * Pause
 * Tutorial 5 Curation (60 min - PierreY)
    - Automatic:
      - Simple, threshold-based curation
      - Curation using ML models / Bombcell
      - Merging units
    - Using GUI  + Automated curation


**November 24th - 14:00-18:00 CET**

 * Spike sorting on your own dataset.
 * Or spike sorting on theses availables datasets


# Overview of available datasets

If you have to pick only one file to download, because of disk usage, it has to be the first one, **M25_D23_2024-11-11_13-11-10_OF1**, since it will be the most used during the Tutorials. The download link is https://filesender.renater.fr/?s=download&token=f94a08b3-e779-4911-90bf-a312c3c34a01
**Please do think about unzipping the data prior to the school!**

## In vivo

### M25_D23_2024-11-11_13-11-10_OF1

Neuropixels 2.0 dataset (384 channels) acquired with the Open Ephys GUI in binary format.
You can use the `spikeinterface.extractor.read_openephys()` function to read the data.

### 1544_2023-04-21_09-55-34_of

Recording with 16 channels organized in 4 tetrodes acquired with the Open Ephys GUI in "openephys" format.
You can use the `spikeinterface.extractor.read_openephys()` function to read the data.



## In vitro

### mouse_CTX_14D.brw

Primary mouse cortex for E18 embryos. Plated 80K on CorePlate 1W38/60 3Brain, recorded after 14 days in culture 
with 3Brain BioCAM system. 
You can use the `spikeinterface.extractor.read_biocam()` function to read the data. 


### human_ngn2_100K_c_55D.brw

Human glutamatergic neurons induced via ngn2 method. Plated 100k on CorePlate 1W38/60 3Brain, recorded after 55 days in 
culture with 3Brain BioCAM system. 
You can use the `spikeinterface.extractor.read_biocam()` function to read the data. 

> [!IMPORTANT]  
> The 3Brain BioCAM system saves the data in unsigned integers. You should convert it to signed integers for a smooth
> processing. See documentation [here](https://spikeinterface.readthedocs.io/en/latest/how_to/unsigned_to_signed.html). Note that the `bit_depth` for this system is 12!



