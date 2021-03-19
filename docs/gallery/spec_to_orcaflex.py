import matplotlib.pyplot as plt
from wavespectra.construct import ochihubble



ds = ochihubble(hs=[1,1], tp=[3,15] ,L=[10,10],  dp = [30,30], dspr = [20,20])
print(ds.spec.dm())
ds.spec.plot.contour()

plt.title(ds.spec.hs().values)

plt.show()

plt.plot(ds.dir, ds.spec.sum('freq').efth)
plt.show()


import OrcFxAPI
model = OrcFxAPI.Model()

ds.spec.to_orcaflex(model, minEnergy=1e-5)  # write to the model, but only those direction that have > 1e-5 of energy

model.SaveData(r'c:\data\test.dat')

# -- check results --
print('Running 3hr simulation')
model.general.StageDuration[1] = 10800
model.RunSimulation()

# Calculate significant waveheight from time-trace standard deviation
el = model.environment.TimeHistory('Elevation', None, OrcFxAPI.oeEnvironment(0,0,0))
import numpy as np
np.std(el)
print(4* np.std(el))

# And compare with that of the spectrum
print(ds.spec.hs())


