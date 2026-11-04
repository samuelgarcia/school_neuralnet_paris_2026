# %%
import pynapple as nap

NWB = "/Users/pierre/Documents/Minischool_GDR2026_SpikeTrains/data/986235_20251209_adaptivesensorimotortask_rec.nwb"
data = nap.load_file(NWB)
print(data)

# %%
units = data["units"]
units
#units.count(bin_size = 1, time_units="s")

# %%
sounds = data["passive_bluenoise"]
sounds
sounds.mean(axis=0)
