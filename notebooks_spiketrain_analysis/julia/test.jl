# %%
using Pkg
# julia/Project.toml: works whether the kernel starts in julia/ or in the repo root
Pkg.activate(isfile("Project.toml") ? "." : "julia")

using Judananas
using HDF5

NWB = "/Users/pierre/Documents/Minischool_GDR2026_SpikeTrains/data/986235_20251209_adaptivesensorimotortask_rec.nwb"

# %%
data   = load_file(NWB)
println(data)

# %%
units  = data["units"]                 # TsGroup
units


# %%
trials = data["trials"]                # IntervalSet
# %%
