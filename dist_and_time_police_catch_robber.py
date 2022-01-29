import numpy as npy
import matplotlib.pyplot as mplt
time = npy.linspace(0, 100, 1000)
dist_rb = 2.5*time
dist_pl = 3*(time-5)
fig, axis = mplt.subplots()
mplt.title('Time and Distance to catch a run away Bank Robber')
mplt.xlabel('Time (in Minutes)')
mplt.ylabel('Distance (in Km)')
axis.set_xlim([0, 50])
axis.set_ylim([0, 100])
axis.plot(time, dist_rb, c='red', label='dist and time of rubber')
axis.plot(time, dist_pl, c='green', label='dist and time of police')
mplt.legend(title='Legend', loc='lower left', bbox_to_anchor=(1.0, 0), ncol=1, fancybox=True, shadow=True)
mplt.axvline(x=30, color='purple', linestyle='--')
mplt.axhline(y=75, color='purple', linestyle='--')
