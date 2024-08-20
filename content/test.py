#%%

from metocean_api import ts
from metocean_stats.stats import general_stats, dir_stats, extreme_stats, profile_stats
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose, STL
import matplotlib.pyplot as plt
#%%

# Define TimeSeries-object
df_ts = ts.TimeSeries(lon=3.73, lat=64.60,start_time='2020-01-01', end_time='2020-12-31' , product='NORA3_wind_wave')
#df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2021-01-01', end_time='2021-01-15' , product='NORA3_wind_sub')
#df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2021-01-01', end_time='2021-03-31' , product='NORA3_wave_sub')
#df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2000-01-01', end_time='2001-03-31' , product='NORA3_stormsurge')
#df_ts = ts.TimeSeries(lon=1.320, lat=53.324,start_time='2021-01-01', end_time='2021-03-31' , product='NORA3_atm_sub')
#df_ts = ts.TimeSeries(lon=3.7, lat=61.8, start_time='2023-01-01', end_time='2023-02-01', product='NORA3_atm3hr_sub')

#%%
# Import data from thredds.met.no and save it as csv
df_ts.import_data(save_csv=True)

#print(df_ts.data)
# Load data from a local csv-file
df_ts.load_data(local_file=df_ts.datafile)
# %%
df_ts.data.header
# %%
general_stats.scatter_diagram(data=df_ts.data, var1='hs', step_var1=1, var2='tp', step_var2=1, output_file='scatter_hs_tp_diagram.png')


# %%
dir_stats.var_rose(df_ts.data, 'thq','hs','windrose.png',method='overall')
# %%
np.corrcoef(df_ts.data.wind_speed_10m,df_ts.data.hs_swell)
# %%
pd.Series.autocorr(df_ts.data.wind_speed_10m)
# %%
result_mul = seasonal_decompose(df_ts.data.wind_speed_10m, model='multiplicative', extrapolate_trend='freq')
# %%
# Multiplicative Decomposition 
result_mul = seasonal_decompose(df_ts.data.wind_speed_10m, model='multiplicative')#, extrapolate_trend='freq')

# Additive Decomposition
result_add = seasonal_decompose(df_ts.data.wind_speed_10m, model='additive')#, extrapolate_trend='freq')

# Plot
plt.rcParams.update({'figure.figsize': (10,10)})
result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
result_add.plot().suptitle('Additive Decompose', fontsize=22)
plt.show()
# %%
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import RecurrencePlot
#from pyts.datasets import load_gunpoint

# Load the GunPoint dataset
X, _, _, _ = load_gunpoint(return_X_y=True)

# Get the recurrence plots for all the time series
rp = RecurrencePlot(threshold='point', percentage=20)
X_rp = rp.fit_transform(df_ts.data.wind_speed_10m.values.reshape(-1, 1))

# Plot the 50 recurrence plots
fig = plt.figure(figsize=(10, 5))

grid = ImageGrid(fig, 111, nrows_ncols=(5, 10), axes_pad=0.1, share_all=True)
for i, ax in enumerate(grid):
    ax.imshow(X_rp[i], cmap='binary', origin='lower')
grid[0].get_yaxis().set_ticks([])
grid[0].get_xaxis().set_ticks([])

fig.suptitle(
    "Recurrence plots for the 50 time series in the 'GunPoint' dataset",
    y=0.92
)

plt.show()
# %%
stl = STL(df_ts.data.wind_speed_10m)#, seasonal=13)
result = stl.fit()
# Plot the original time series and the decomposed components
plt.figure(figsize=(8,7))

plt.subplot(411)
plt.plot(result.observed)
plt.title('Original Series', fontsize=16)

plt.subplot(412)
plt.plot(result.trend)
plt.title('Trend Component', fontsize=16)

plt.subplot(413)
plt.plot(result.seasonal)
plt.title('Seasonal Component', fontsize=16)

plt.subplot(414)
plt.plot(result.resid)
plt.title('Residual Component', fontsize=16)

plt.tight_layout()
plt.show()
# %%
