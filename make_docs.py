import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
import numpy as np
import seaborn as sns
from scipy import stats
import math
from sympy import sec
from jacobi import propagate

from iminuit import Minuit
from iminuit.cost import LeastSquares

df = pd.read_csv('all_pulsar_fits.csv')

# grab model specific data frames
spl_df  = df[df["Model"] == "simple_power_law"]
bpl_df  = df[df["Model"] == "broken_power_law"]
lps_df  = df[df["Model"] == "log_parabolic_spectrum"]
hfto_df = df[df["Model"] == "high_frequency_cut_off_power_law"]
lfto_df = df[df["Model"] == "low_frequency_turn_over_power_law"]
dtos_df = df[df["Model"] == "double_turn_over_spectrum"]
nomod_df = df[df["Model"] == ""]
smart_df = df[df["SMART"]]
all_to_df = df[df["vpeak"].notnull()]
sync_df = all_to_df[all_to_df["beta"] < 2.05]
print(len(sync_df))
thermal_df = all_to_df[all_to_df["beta"] >= 2.05]
print(len(thermal_df))

msp_cutoff = 0.03 # seconds

# MSPs
msp_spl_df  = spl_df[spl_df["ATNF Period (s)"] < msp_cutoff]
msp_bpl_df  = bpl_df[bpl_df["ATNF Period (s)"] < msp_cutoff]
msp_lps_df  = lps_df[lps_df["ATNF Period (s)"] < msp_cutoff]
msp_hfto_df = hfto_df[hfto_df["ATNF Period (s)"] < msp_cutoff]
msp_lfto_df = lfto_df[lfto_df["ATNF Period (s)"] < msp_cutoff]
msp_dtos_df = dtos_df[dtos_df["ATNF Period (s)"] < msp_cutoff]
msp_df = df[df["ATNF Period (s)"] < msp_cutoff]

# normal pulsars
np_spl_df  = spl_df[spl_df["ATNF Period (s)"]   >= msp_cutoff]
np_bpl_df  = bpl_df[bpl_df["ATNF Period (s)"]   >= msp_cutoff]
np_lps_df  = lps_df[lps_df["ATNF Period (s)"]   >= msp_cutoff]
np_hfto_df = hfto_df[hfto_df["ATNF Period (s)"] >= msp_cutoff]
np_lfto_df = lfto_df[lfto_df["ATNF Period (s)"] >= msp_cutoff]
np_dtos_df = dtos_df[dtos_df["ATNF Period (s)"] >= msp_cutoff]
np_df = df[df["ATNF Period (s)"] >= msp_cutoff]


if len(lps_df) == 0:
    del df["lps_a"]
    del df["lps_u_a"]
    del df["lps_b"]
    del df["lps_u_b"]
    del df["lps_c"]
    del df["lps_u_c"]


# Set up the docs

# Record summary results on homepage
with open(f'{os.path.dirname(os.path.realpath(__file__))}/docs/index.rst', 'w') as file:
    file.write(f'''
Pulsar Spectra all pulsars fit results
======================================

.. toctree::
   :maxdepth: 1
   :caption: Spectral Model Gallerys:

   spl_gallery
   bpl_gallery
   lfto_gallery
   hfco_gallery
   dtos_gallery


.. toctree::
   :maxdepth: 1
   :caption: Other Gallerys:

   smart_gallery
   msp_gallery


Fit Summary Pulsar Count
------------------------
.. csv-table::
   :header: "Model", "Total", "MSP", "%", "Normal", "%"

   "simple_power_law",                  "{len(spl_df)}",  "{len(msp_spl_df)}",  "{len(msp_spl_df) /len(msp_df)*100:.1f} %",  "{len(np_spl_df)}", "{len(np_spl_df) /len(np_df)*100:.1f} %"
   "broken_power_law",                  "{len(bpl_df)}",  "{len(msp_bpl_df)}",  "{len(msp_bpl_df) /len(msp_df)*100:.1f} %",  "{len(np_bpl_df)}", "{len(np_bpl_df) /len(np_df)*100:.1f} %"
   "high_frequency_cut_off_power_law",  "{len(hfto_df)}", "{len(msp_hfto_df)}", "{len(msp_hfto_df)/len(msp_df)*100:.1f} %", "{len(np_hfto_df)}", "{len(np_hfto_df)/len(np_df)*100:.1f} %"
   "low_frequency_turn_over_power_law", "{len(lfto_df)}", "{len(msp_lfto_df)}", "{len(msp_lfto_df)/len(msp_df)*100:.1f} %", "{len(np_lfto_df)}", "{len(np_lfto_df)/len(np_df)*100:.1f} %"
   "double_turn_over_spectrum",         "{len(dtos_df)}", "{len(msp_dtos_df)}", "{len(msp_dtos_df)/len(msp_df)*100:.1f} %", "{len(np_dtos_df)}", "{len(np_dtos_df)/len(np_df)*100:.1f} %"
   "Total",                             "{len(df)}",      "{len(msp_df)}",      "{len(msp_df)     /len(msp_df)*100:.1f} %",      "{len(np_df)}", "{len(np_df)     /len(np_df)*100:.1f} %"

Analysis Summary
----------------
.. csv-table::
   :header: "Parameter", "All Mean", "MSP Mean", "Normal Mean"

   "spectral index",          "{df["a"].mean():.2f}±{df["a"].std():.2f}",                 "{msp_df["a"].mean():.2f}±{msp_df["a"].std():.2f}",                 "{np_df["a"].mean():.2f}±{np_df["a"].std():.2f}"
   "Peak Frequency (GHz)",    "{df["vpeak"].mean()/1e9:.2f}±{df["vpeak"].std()/1e9:.2f}", "{msp_df["vpeak"].mean()/1e9:.2f}±{msp_df["vpeak"].std()/1e9:.2f}", "{np_df["vpeak"].mean()/1e9:.2f}±{np_df["vpeak"].std()/1e9:.2f}"
   "Cut off frequency (GHz)", "{df["vc"].mean()/1e9:.2f}±{df["vc"].std()/1e9:.2f}",       "{msp_df["vc"].mean()/1e9:.2f}±{msp_df["vc"].std()/1e9:.2f}",       "{np_df["vc"].mean()/1e9:.2f}±{np_df["vc"].std()/1e9:.2f}"
   "Beta",                    "{df["beta"].mean():.2f}±{df["beta"].std():.2f}",           "{msp_df["beta"].mean():.2f}±{msp_df["beta"].std():.2f}",           "{np_df["beta"].mean():.2f}±{np_df["beta"].std():.2f}"

Spectral Index Mean Summary
----------------------
.. csv-table::
   :header: "Model", "All Mean", "MSP Mean", "Normal Mean"

   "simple_power_law",                  "{spl_df["a"].mean():.2f}±{spl_df["a"].std():.2f}",   "{msp_spl_df["a"].mean():.2f}±{ msp_spl_df["a"].std():.2f}",  "{np_spl_df["a"].mean():.2f}±{ np_spl_df["a"].std():.2f}"
   "high_frequency_cut_off_power_law",  "{hfto_df["a"].mean():.2f}±{hfto_df["a"].std():.2f}", "{msp_hfto_df["a"].mean():.2f}±{msp_hfto_df["a"].std():.2f}", "{np_hfto_df["a"].mean():.2f}±{np_hfto_df["a"].std():.2f}"
   "low_frequency_turn_over_power_law", "{lfto_df["a"].mean():.2f}±{lfto_df["a"].std():.2f}", "{msp_lfto_df["a"].mean():.2f}±{msp_lfto_df["a"].std():.2f}", "{np_lfto_df["a"].mean():.2f}±{np_lfto_df["a"].std():.2f}"
   "double_turn_over_spectrum",         "{dtos_df["a"].mean():.2f}±{dtos_df["a"].std():.2f}", "{msp_dtos_df["a"].mean():.2f}±{msp_dtos_df["a"].std():.2f}", "{np_dtos_df["a"].mean():.2f}±{np_dtos_df["a"].std():.2f}"
   "Total",                             "{df["a"].mean():.2f}±{df["a"].std():.2f}",           "{msp_df["a"].mean():.2f}±{     msp_df["a"].std():.2f}",      "{np_df["a"].mean():.2f}±{     np_df["a"].std():.2f}"

Spectral Index Median Summary
----------------------
.. csv-table::
   :header: "Model", "All Median", "MSP Median", "Normal Median"

   "simple_power_law",                  "{spl_df["a"].median():.2f}±{spl_df["a"].std():.2f}",   "{msp_spl_df["a"].median():.2f}±{ msp_spl_df["a"].std():.2f}",  "{np_spl_df["a"].median():.2f}±{ np_spl_df["a"].std():.2f}"
   "high_frequency_cut_off_power_law",  "{hfto_df["a"].median():.2f}±{hfto_df["a"].std():.2f}", "{msp_hfto_df["a"].median():.2f}±{msp_hfto_df["a"].std():.2f}", "{np_hfto_df["a"].median():.2f}±{np_hfto_df["a"].std():.2f}"
   "low_frequency_turn_over_power_law", "{lfto_df["a"].median():.2f}±{lfto_df["a"].std():.2f}", "{msp_lfto_df["a"].median():.2f}±{msp_lfto_df["a"].std():.2f}", "{np_lfto_df["a"].median():.2f}±{np_lfto_df["a"].std():.2f}"
   "double_turn_over_spectrum",         "{dtos_df["a"].median():.2f}±{dtos_df["a"].std():.2f}", "{msp_dtos_df["a"].median():.2f}±{msp_dtos_df["a"].std():.2f}", "{np_dtos_df["a"].median():.2f}±{np_dtos_df["a"].std():.2f}"
   "Total",                             "{df["a"].median():.2f}±{df["a"].std():.2f}",           "{msp_df["a"].median():.2f}±{     msp_df["a"].std():.2f}",      "{np_df["a"].median():.2f}±{     np_df["a"].std():.2f}"

Single Power Law Results
------------------------
.. csv-table::
   :header: "Pulsar", "a"

''')
    for index, row in spl_df.iterrows():
        data_str = f'   ":ref:`{row["Pulsar"]}`", '
        for val, error in [("a", "u_a")]:
            if "v" in val:
                data_str += f'"{int(row[val]/1e6):d}±{int(row[error]/1e6):d}", '
            else:
                data_str += f'"{row[val]:.2f}±{row[error]:.2f}", '
        file.write(f'{data_str[:-2]}\n')

    file.write(f'''


Broken Power Law Results
------------------------
.. csv-table::
   :header: "Pulsar", "vb (MHz)", "a1", "a2"

''')
    for index, row in bpl_df.iterrows():
        data_str = f'   ":ref:`{row["Pulsar"]}`", '
        for val, error in [("vb", "u_vb"), ("a1", "u_a1"), ("a2","u_a2")]:
            if "v" in val:
                data_str += f'"{int(row[val]/1e6):d}±{int(row[error]/1e6):d}", '
            else:
                data_str += f'"{row[val]:.2f}±{row[error]:.2f}", '
        file.write(f'{data_str[:-2]}\n')

    file.write(f'''


Low Frequency Turn Over Results
-------------------------------
.. csv-table::
   :header: "Pulsar", "vpeak (MHz)", "a", "beta"

''')
    for index, row in lfto_df.iterrows():
        data_str = f'   ":ref:`{row["Pulsar"]}`", '
        for val, error in [("vpeak", "u_vpeak"), ("a", "u_a"), ("beta", "u_beta")]:
            if "v" in val:
                data_str += f'"{int(row[val]/1e6):d}±{int(row[error]/1e6):d}", '
            else:
                data_str += f'"{row[val]:.2f}±{row[error]:.2f}", '
        file.write(f'{data_str[:-2]}\n')

    file.write(f'''


High Frequency Cut Off Results
------------------------------
.. csv-table::
   :header: "Pulsar", "vc (MHz)", "a"

''')
    for index, row in hfto_df.iterrows():
        data_str = f'   ":ref:`{row["Pulsar"]}`", '
        for val, error in [("vc", "u_vc"), ("a", "u_a")]:
            if "v" in val:
                data_str += f'"{int(row[val]/1e6):d}±{int(row[error]/1e6):d}", '
            else:
                data_str += f'"{row[val]:.2f}±{row[error]:.2f}", '
        file.write(f'{data_str[:-2]}\n')

    file.write(f'''


Double Turn Over Spectrum Results
---------------------------------
.. csv-table::
   :header: "Pulsar", "vc (MHz)", "vpeak (MHz)", "a", "beta"

''')
    for index, row in dtos_df.iterrows():
        data_str = f'   ":ref:`{row["Pulsar"]}`", '
        for val, error in [("vc", "u_vc"), ("vpeak", "u_vpeak"), ("a", "u_a"), ("beta", "u_beta")]:
            if "v" in val:
                data_str += f'"{int(row[val]/1e6):d}±{int(row[error]/1e6):d}", '
            else:
                data_str += f'"{row[val]:.2f}±{row[error]:.2f}", '
        file.write(f'{data_str[:-2]}\n')

    file.write(f'''


No Model Results
----------------
.. csv-table::
   :header: "Pulsar", "N data"

''')
    for index, row in nomod_df.iterrows():
        file.write(f'   ":ref:`{row["Pulsar"]}`", "{row["N data flux"]}"')


# Set up the gallerys
with open(f'{os.path.dirname(os.path.realpath(__file__))}/docs/spl_gallery.rst', 'w') as file:
    file.write(f'''
Simple Power Law Gallery
========================

''')
    for index, row in spl_df.iterrows():
        file.write(f'''

.. _{row["Pulsar"]}:

{row["Pulsar"]}
{"-"*len(row["Pulsar"])}
.. image:: best_fits/{row["Pulsar"]}_fit.png
  :width: 800
''')

with open(f'{os.path.dirname(os.path.realpath(__file__))}/docs/bpl_gallery.rst', 'w') as file:
    file.write(f'''
Broken Power Law Gallery
========================

''')
    for index, row in bpl_df.iterrows():
        file.write(f'''

.. _{row["Pulsar"]}:

{row["Pulsar"]}
{"-"*len(row["Pulsar"])}
.. image:: best_fits/{row["Pulsar"]}_fit.png
  :width: 800
''')

with open(f'{os.path.dirname(os.path.realpath(__file__))}/docs/lfto_gallery.rst', 'w') as file:
    file.write(f'''
Low Frequency Turn Over Gallery
===============================

''')
    for index, row in lfto_df.iterrows():
        file.write(f'''

.. _{row["Pulsar"]}:

{row["Pulsar"]}
{"-"*len(row["Pulsar"])}
.. image:: best_fits/{row["Pulsar"]}_fit.png
  :width: 800
''')

with open(f'{os.path.dirname(os.path.realpath(__file__))}/docs/hfco_gallery.rst', 'w') as file:
    file.write(f'''
High Frequency Cut Off Gallery
==============================

''')
    for index, row in hfto_df.iterrows():
        file.write(f'''

.. _{row["Pulsar"]}:

{row["Pulsar"]}
{"-"*len(row["Pulsar"])}
.. image:: best_fits/{row["Pulsar"]}_fit.png
  :width: 800
''')

with open(f'{os.path.dirname(os.path.realpath(__file__))}/docs/dtos_gallery.rst', 'w') as file:
    file.write(f'''
Double Turn Over Spectrum Gallery
=================================

''')
    for index, row in dtos_df.iterrows():
        file.write(f'''

.. _{row["Pulsar"]}:

{row["Pulsar"]}
{"-"*len(row["Pulsar"])}
.. image:: best_fits/{row["Pulsar"]}_fit.png
  :width: 800
''')

with open(f'{os.path.dirname(os.path.realpath(__file__))}/docs/smart_gallery.rst', 'w') as file:
    file.write(f'''
SMART Gallery
=============

All pulsar detections from the SMART pulsar survey (these will be in other galleries).

''')
    for index, row in smart_df.iterrows():
        file.write(f'''

.. _{row["Pulsar"]}:

{row["Pulsar"]}
{"-"*len(row["Pulsar"])}
.. image:: best_fits/{row["Pulsar"]}_fit.png
  :width: 800
''')


with open(f'{os.path.dirname(os.path.realpath(__file__))}/docs/msp_gallery.rst', 'w') as file:
    file.write(f'''
MSP Gallery
===========

All millisecond pulsar detections (these will be in other galleries).

''')
    for index, row in msp_df.iterrows():
        file.write(f'''

.. _{row["Pulsar"]}:

{row["Pulsar"]}
{"-"*len(row["Pulsar"])}
.. image:: best_fits/{row["Pulsar"]}_fit.png
  :width: 800
''')

# Make spectral index histogram plots
n_bins = 20
hist_range = (min(df["a"]), max(df["a"]))

all_indexs = np.array([
    spl_df ["a"],
    hfto_df["a"],
    lfto_df["a"],
    dtos_df["a"],
])
colours = [
    'blue',
    'green',
    'orange',
    'purple',
]

fig, axes = plt.subplots(nrows=5, figsize=(5, 3*5))

axes[0].hist(all_indexs, n_bins, density=True, histtype='bar', stacked=True, label=["spl", "hfto", "lfto", "dtos"], color=colours)
axes[0].set_title("All spectral indexs")
axes[0].legend(prop={'size': 10})

axes[1].hist(spl_df ["a"], n_bins, histtype='bar', color=colours[0], range=hist_range)
axes[1].set_title('Simple power law')

axes[2].hist(hfto_df["a"], n_bins, histtype='bar', color=colours[1], range=hist_range)
axes[2].set_title('High-frequency cut off')

axes[3].hist(lfto_df["a"], n_bins, histtype='bar', color=colours[2], range=hist_range)
axes[3].set_title('Low-frequency turn over')

axes[4].hist(dtos_df["a"], n_bins, histtype='bar', color=colours[3], range=hist_range)
axes[4].set_title('Double turn over spectrum')

fig.tight_layout()
fig.savefig("spectral_index_histogram.png")




# Remove unessiary columns
del df["Pulsar"]
del df["Model"]
del df["Probability Best"]
del df["N data flux"]

# Remove broken power law stuff
del df["vb"]
del df["a1"]
del df["a2"]
# for col_name in df.keys():
#     if "u_" in col_name or col_name.endswith("_c"):
#         del df[col_name]

f, ax = plt.subplots(figsize=(10, 5))
# Plot correlations
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True,
            cmap='seismic', mask=mask,
            vmin=-1, vmax=1,)
corr.to_csv(f"pulsar_coor.csv")
plt.tight_layout(pad=0.5)
plt.savefig("pulsar_coor.png")
plt.clf()

log_df = df#[df["beta"] < 2.05]
# convert to log data
for col_name in log_df.keys():
    if "ATNF" in col_name or col_name in ("L400 (mJy kpc^2)", "L1400 (mJy kpc^2)", "Age (Yr)"):
        log_df[col_name + " log"] = np.log10(log_df[col_name])
        #del log_df[col_name]
    elif col_name.startswith("v"):
        log_df[col_name + " log"] = np.log10(log_df[col_name])
        log_df["u_" + col_name + " log"] = log_df["u_" + col_name] / log_df[col_name]
        #del log_df[col_name]

print(log_df.keys())
log_df_sync = log_df[log_df["beta"] < 2.05]

# Plot correlation
corr = log_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True,
            cmap='seismic', mask=mask,
            vmin=-1, vmax=1,)
corr.to_csv(f"pulsar_coor_log.csv")
plt.tight_layout(pad=0.5)
plt.savefig("pulsa_coor_log.png")

def weighted_correlation(x, y, yerr):
    """Weighted correlation based on:
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Weighted_correlation_coefficient
    """
    # weights based on y uncertainty
    w = 1 / yerr**2
    # weighted mean
    mx = np.sum( w * x ) / np.sum( w )
    my = np.sum( w * y ) / np.sum( w )
    # weighted covariance
    cov_xy = np.sum( w * (x - mx) * (y - my) ) / np.sum( w )
    cov_xx = np.sum( w * (x - mx) * (x - mx) ) / np.sum( w )
    cov_yy = np.sum( w * (y - my) * (y - my) ) / np.sum( w )
    # weighted correlation
    corr = cov_xy / np.sqrt( cov_xx * cov_yy )
    return corr

def yline(x, g, xmean, ymean):
    return ymean + g * (x - xmean)

def xline(y, g, ymean, xmean):
    return xmean + g * (y - ymean)

def line_fit(
        data_x, data_y, data_yerr, raw_data_x, raw_data_y, raw_data_yerr,
        msp_x, msp_y, msp_yerr, msp_raw_x, msp_raw_y, msp_raw_yerr,
        np_x, np_y, np_yerr, np_raw_x, np_raw_y, np_raw_yerr,
        vpeak_x, vpeak_y, vpeak_yerr,
        xcol, ycol, lax, weighted_corr):
    xmean=np.mean(data_x)
    ymean=np.mean(data_y)

    # fit y line
    least_squares = LeastSquares(data_x, data_y, data_yerr, yline)
    m = Minuit(least_squares, g=0, xmean=xmean, ymean=ymean)  # starting values for α and β
    m.fixed["xmean"] = True
    m.fixed["ymean"] = True
    m.migrad()  # finds minimum of least_squares function
    m.hesse()   # accurately computes uncertaintiesax):

    # # fit x line
    # least_squares = LeastSquares(data_y, data_x, data_x*0.001, xline)
    # mx = Minuit(least_squares, g=0, xmean=xmean, ymean=ymean)  # starting values for α and β
    # mx.fixed["xmean"] = True
    # mx.fixed["ymean"] = True
    # mx.migrad()  # finds minimum of least_squares function
    # mx.hesse()   # accurately computes uncertainties

    # m1 = my.values["g"]
    # m2 = mx.values["g"]

    # # Delta from the difference between regression lines
    # delta = math.atan( (m1 - m2) / (1 + m1 * m2) )

    # Delta from fot product
    delta = math.acos( np.dot(data_x, data_y) / ( np.linalg.norm(data_x) * np.linalg.norm(data_y) ) )
    # print(delta)
    #r = float(sec(delta) - math.tan(delta))
    r = (1 - np.sqrt(1 - np.cos(delta))) / np.cos(delta)
    #print(r)

    capsize = 1.5
    errorbar_linewidth = 0.7
    marker_border_thickness = 0.5
    markersize=2
    f, ax = plt.subplots()
    # Add normal pulsar data
    colour = "b"
    (_, caps, _) = ax.errorbar(np_raw_x, np_raw_y, np_raw_yerr,
        fmt="o",
        markeredgewidth=marker_border_thickness,
        elinewidth=errorbar_linewidth,
        capsize=capsize,
        markersize=markersize,
        label="Normal pulsars",
        ecolor=colour,
        c=colour,
    )
    for cap in caps:
        cap.set_markeredgewidth(errorbar_linewidth)
    (_, caps, _) = lax.errorbar(np_raw_x, np_raw_y, np_raw_yerr,
        fmt="o",
        markeredgewidth=marker_border_thickness,
        elinewidth=errorbar_linewidth,
        capsize=capsize,
        markersize=markersize,
        label="Normal pulsars",
        ecolor=colour,
        c=colour,
    )
    colour = "g"
    for cap in caps:
        cap.set_markeredgewidth(errorbar_linewidth)
    # Add MSP data
    (_, caps, _) = ax.errorbar(msp_raw_x, msp_raw_y, msp_raw_yerr,
        fmt="o",
        markeredgewidth=marker_border_thickness,
        elinewidth=errorbar_linewidth,
        capsize=capsize,
        markersize=markersize,
        label="MSPs",
        ecolor=colour,
        c=colour,
    )
    for cap in caps:
        cap.set_markeredgewidth(errorbar_linewidth)
    (_, caps, _) = lax.errorbar(msp_raw_x, msp_raw_y, msp_raw_yerr,
        fmt="o",
        markeredgewidth=marker_border_thickness,
        elinewidth=errorbar_linewidth,
        capsize=capsize,
        markersize=markersize,
        label="MSPs",
        ecolor=colour,
        c=colour,
    )
    for cap in caps:
        cap.set_markeredgewidth(errorbar_linewidth)

    # plot fit line and error bar
    fitted_line, fitted_line_cov = propagate(lambda p: yline(data_x, *p), m.values, m.covariance)
    fitted_line_err = abs(np.diag(fitted_line_cov) ** 0.5)
    if "log" in ycol:
        fitted_line = 10**fitted_line
        fitted_line_err = 10**fitted_line_err
    #print(f"error percent: {np.mean(fitted_line_err/fitted_line)}")
    ax.plot( raw_data_x, fitted_line, label="fit", c='orange')
    lax.plot(raw_data_x, fitted_line, label="fit", c='orange')
    ax.fill_between( raw_data_x, fitted_line - fitted_line_err, fitted_line + fitted_line_err, facecolor="C1", alpha=0.5)
    lax.fill_between(raw_data_x, fitted_line - fitted_line_err, fitted_line + fitted_line_err, facecolor="C1", alpha=0.5)
    # print(fitted_line)
    # print(fitted_line_err)

    if ycol == "vpeak log":
        #print(vpeak_x, vpeak_y, vpeak_yerr)
        # Add vpeak limits
        colour = 'orange'
        (_, caps, _) = ax.errorbar(
            vpeak_x, vpeak_y, vpeak_yerr,
            uplims=True,
            fmt="o",
            markeredgewidth=marker_border_thickness,
            elinewidth=errorbar_linewidth,
            capsize=capsize,
            markersize=markersize,
            label="MSPs",
            ecolor=colour,
            c=colour,
        )
        for cap in caps:
            cap.set_markeredgewidth(errorbar_linewidth)
        (_, caps, _) = lax.errorbar(
            vpeak_x, vpeak_y, vpeak_yerr,
            uplims=True,
            fmt="o",
            markeredgewidth=marker_border_thickness,
            elinewidth=errorbar_linewidth,
            capsize=capsize,
            markersize=markersize,
            label="MSPs",
            ecolor=colour,
            c=colour,
        )
        for cap in caps:
            cap.set_markeredgewidth(errorbar_linewidth)

    # Labels
    ax.set_xlabel(xcol.replace(" log", ""))
    lax.set_xlabel(xcol.replace(" log", ""))
    ax.set_ylabel(ycol.replace(" log", ""))
    lax.set_ylabel(ycol.replace(" log", ""))
    ax.legend( title=f"Corr={weighted_corr:.3f}")
    lax.legend(title=f"Corr={weighted_corr:.3f}")
    if 'log' in xcol:
        ax.set_xscale('log')
        lax.set_xscale('log')
        # ax.get_xaxis().set_major_formatter(FormatStrFormatter('%g'))
        # lax.get_xaxis().set_major_formatter(FormatStrFormatter('%g'))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, p:
                '$\mathdefault{10^{%i}}$' % x
                if x > 1e5 else
                '%g' % x
            )
        )
        lax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, p:
                '$\mathdefault{10^{%i}}$' % np.log10(x)
                if x > 1e5 else
                '%g' % x
            )
        )
    if 'log' in ycol:
        ax.set_yscale('log')
        lax.set_yscale('log')
        # ax.get_yaxis().set_major_formatter(FormatStrFormatter('%g'))
        # lax.get_yaxis().set_major_formatter(FormatStrFormatter('%g'))
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, p:
                '$\mathdefault{10^{%i}}$' % x
                if x > 1e5 else
                '%g' % x
            )
        )
        lax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, p:
                '$\mathdefault{10^{%i}}$' % np.log10(x)
                if x > 1e5 else
                '%g' % x
            )
        )
    ax.tick_params(which='both', direction='in', top=1, right=1)
    lax.tick_params(which='both', direction='in', top=1, right=1)

    # # display legend with some fit info
    # fit_info = [
    #     f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {len(data_x) - m.nfit}",
    # ]
    # for p, v, e in zip(m.parameters, m.values, m.errors):
    #     fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")

    # f.legend(title="\n".join(fit_info))
    f.savefig(f"corr_line_{xcol}_{ycol}.png".replace(" ", "_").replace("/", "_"))
    return r

# Make individual correlation plots
# df, x, y
corr_plots = [
    (log_df, "ATNF E_dot (ergs/s) log", "vpeak"),
]
xcols = [
    "ATNF Period (s) log",
    "ATNF DM log",
    "ATNF B_surf (G) log",
    "ATNF E_dot (ergs/s) log",
    "L400 (mJy kpc^2) log",
    "L1400 (mJy kpc^2) log",
    "Age (Yr) log",
]
ycols = [
        "a"         ,
        # "c"         ,
        # "vb"        ,
        # "a1"        ,
        # "a2"        ,
        "vc log"        ,
        "vpeak log"     ,
        "beta"      ,
]
nx = len(xcols)
ny = len(ycols)
sfig, saxes = plt.subplots(ny, nx, figsize=(5*nx, 5*ny))
lfig, laxes = plt.subplots(ny, nx, figsize=(5*nx, 5*ny))
corr_matrix = log_df.corr()
for xa, xcol in enumerate(xcols):
    for ya, ycol in enumerate(ycols):
        if ycol == "beta":
            this_df = log_df_sync
        else:
            this_df = log_df
        if "log" in xcol:
            raw_xcol = xcol[:-4]
        else:
            raw_xcol = xcol
        if "log" in ycol:
            raw_ycol = ycol[:-4]
        else:
            raw_ycol = ycol
        x = []
        y = []
        yerr = []
        raw_x = []
        raw_y = []
        raw_yerr = []
        msp_x = []
        msp_y = []
        msp_yerr = []
        msp_raw_x = []
        msp_raw_y = []
        msp_raw_yerr = []
        np_x = []
        np_y = []
        np_yerr = []
        np_raw_x = []
        np_raw_y = []
        np_raw_yerr = []
        vpeak_x = []
        vpeak_y = []
        vpeak_yerr = []
        for xi, yi, yerri, raw_xi, raw_yi, raw_yerri, period, min_freq in zip(
                list(this_df[xcol]), list(this_df[ycol]), list(this_df["u_"+ycol]),
                list(this_df[raw_xcol]), list(this_df[raw_ycol]), list(this_df["u_"+raw_ycol]),
                list(this_df["ATNF Period (s)"]), list(this_df["Min freq (MHz)"])):
            # if ycol == "vc log":
            #     print(xi, yi, yerri)
            if (not np.isnan(xi)) and ( not np.isnan(yi) ) and ( not np.isnan(yerri) ) and (not np.isinf(xi)):
                x.append(xi)
                y.append(yi)
                yerr.append(yerri)
                raw_x.append(raw_xi)
                raw_y.append(raw_yi)
                raw_yerr.append(raw_yerri)
                # msp check
                if period < msp_cutoff:
                    #msp
                    msp_x.append(xi)
                    msp_y.append(yi)
                    msp_yerr.append(yerri)
                    msp_raw_x.append(raw_xi)
                    msp_raw_y.append(raw_yi)
                    msp_raw_yerr.append(raw_yerri)
                else:
                    #normal
                    np_x.append(xi)
                    np_y.append(yi)
                    np_yerr.append(yerri)
                    np_raw_x.append(raw_xi)
                    np_raw_y.append(raw_yi)
                    np_raw_yerr.append(raw_yerri)
            elif ycol == "vpeak log":
                vpeak_x.append(raw_xi)
                vpeak_y.append(min_freq*1e6)
                vpeak_yerr.append(min_freq*1e6*0.1)


        x = np.array(x)
        y = np.array(y)
        yerr = np.array(yerr)
        raw_x = np.array(raw_x)
        raw_y = np.array(raw_y)
        raw_yerr = np.array(raw_yerr)
        print(f"\nCorrelations of x: {xcol} and y: {ycol}. N: {len(x)}")
        print(f"-------------------------------------------------")
        print(f"pandas:        {corr_matrix[xcol][ycol]:6.3f}")
        print(f"numpy:         {np.corrcoef(x,y)[0][1]:6.3f}")
        weighted_corr = weighted_correlation(x, y ,yerr)
        print(f"weighted corr: {weighted_correlation(x, y ,yerr):6.3f}")
        rho, pval = stats.spearmanr(x, y, nan_policy="omit")
        print(f"spearmanr:     {rho:6.3f}")

        weighted_line_corr = line_fit(
            x, y, yerr, raw_x, raw_y, raw_yerr,
            msp_x, msp_y, msp_yerr, msp_raw_x, msp_raw_y, msp_raw_yerr,
            np_x, np_y, np_yerr, np_raw_x, np_raw_y, np_raw_yerr,
            vpeak_x, vpeak_y, vpeak_yerr,
            xcol, ycol, laxes[ya][xa], weighted_corr)
        print(f"line:          {weighted_line_corr:6.3f}")

        sns.regplot(data=this_df, x=xcol, y=ycol, ax=saxes[ya][xa])

sfig.savefig(f'coor_all.png', dpi=300)
lfig.savefig(f'corr_all_weighted.png', dpi=300)
