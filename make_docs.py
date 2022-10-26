import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
import numpy as np
import seaborn as sns
from scipy import stats
import math
from jacobi import propagate
import yaml

from iminuit import Minuit
from iminuit.cost import LeastSquares


from pulsar_spectra.catalogue import CAT_DIR
from pulsar_spectra.spectral_fit import huber_loss_function


df = pd.read_csv('all_pulsar_fits.csv')

# filter to use only jankowski pulsars
# with open(f"{CAT_DIR}/Jankowski_2018.yaml", "r") as stream:
#     cat_dict = yaml.safe_load(stream)
# jankowski_pulsars = cat_dict.keys()
# df = df[df["Pulsar"].isin(jankowski_pulsars)]


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
print(list(df[df["u_vc"]/df["vc"] > 3]["Pulsar"]))

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

print(f'''
Model & Total & \% & MSP & \% & Normal & \% \\\\

SPL &                  {len(spl_df)} &  {len(spl_df) /len(df)*100:.1f} \% &  {len(msp_spl_df)} &  {len(msp_spl_df) /len(msp_df)*100:.1f} \% &  {len(np_spl_df)} & {len(np_spl_df) /len(np_df)*100:.1f} \% \\\\
BPL &                  {len(bpl_df)} &  {len(bpl_df) /len(df)*100:.1f} \% &  {len(msp_bpl_df)} &  {len(msp_bpl_df) /len(msp_df)*100:.1f} \% &  {len(np_bpl_df)} & {len(np_bpl_df) /len(np_df)*100:.1f} \% \\\\
HFCO &  {len(hfto_df)} & {len(hfto_df)/len(df)*100:.1f} \% & {len(msp_hfto_df)} & {len(msp_hfto_df)/len(msp_df)*100:.1f} \% & {len(np_hfto_df)} & {len(np_hfto_df)/len(np_df)*100:.1f} \% \\\\
LFTO & {len(lfto_df)} & {len(lfto_df)/len(df)*100:.1f} \% & {len(msp_lfto_df)} & {len(msp_lfto_df)/len(msp_df)*100:.1f} \% & {len(np_lfto_df)} & {len(np_lfto_df)/len(np_df)*100:.1f} \% \\\\
DTOS &         {len(dtos_df)} & {len(dtos_df)/len(df)*100:.1f} \% & {len(msp_dtos_df)} & {len(msp_dtos_df)/len(msp_df)*100:.1f} \% & {len(np_dtos_df)} & {len(np_dtos_df)/len(np_df)*100:.1f} \% \\\\
Total &                             {len(df)} &      {len(df)     /len(df)*100:.1f} \% & {len(msp_df)} &      {len(msp_df)     /len(msp_df)*100:.1f} \% &      {len(np_df)} & {len(np_df)     /len(np_df)*100:.1f} \%) \\\\
''')

print(np.std(spl_df["a"], ddof=1) / np.sqrt(np.size(spl_df["a"])))

print(f'''
\hline
Model & All Mean & MSP Mean & Normal Mean \\\\
\hline
SPL &   ${spl_df["a"].mean():.2f}\pm {spl_df["a"].std():.2f}  $ & ${msp_spl_df["a"].mean():.2f}\pm { msp_spl_df["a"].std():.2f} $ & $ {np_spl_df["a"].mean():.2f}\pm { np_spl_df["a"].std():.2f}$ \\\\
HFCO &  ${hfto_df["a"].mean():.2f}\pm {hfto_df["a"].std():.2f}$ & ${msp_hfto_df["a"].mean():.2f}\pm {msp_hfto_df["a"].std():.2f}$ & ${np_hfto_df["a"].mean():.2f}\pm {np_hfto_df["a"].std():.2f}$ \\\\
LFTO &  ${lfto_df["a"].mean():.2f}\pm {lfto_df["a"].std():.2f}$ & ${msp_lfto_df["a"].mean():.2f}\pm {msp_lfto_df["a"].std():.2f}$ & ${np_lfto_df["a"].mean():.2f}\pm {np_lfto_df["a"].std():.2f}$ \\\\
DTOS &  ${dtos_df["a"].mean():.2f}\pm {dtos_df["a"].std():.2f}$ & ${msp_dtos_df["a"].mean():.2f}\pm {msp_dtos_df["a"].std():.2f}$ & ${np_dtos_df["a"].mean():.2f}\pm {np_dtos_df["a"].std():.2f}$ \\\\
Total & ${df["a"].mean():.2f}\pm {df["a"].std():.2f}          $ & ${msp_df["a"].mean():.2f}\pm {     msp_df["a"].std():.2f}     $ & $     {np_df["a"].mean():.2f}\pm {     np_df["a"].std():.2f}$ \\\\
\hline
\\\\
\hline
Model & All Median & MSP Median & Normal Median \\\\
\hline
SPL &   ${spl_df["a"].median():.2f}\pm {spl_df["a"].std():.2f}  $ & ${msp_spl_df["a"].median():.2f}\pm { msp_spl_df["a"].std():.2f} $ & ${np_spl_df["a"].median():.2f}\pm { np_spl_df["a"].std():.2f}  $ \\\\
HFCO &  ${hfto_df["a"].median():.2f}\pm {hfto_df["a"].std():.2f}$ & ${msp_hfto_df["a"].median():.2f}\pm {msp_hfto_df["a"].std():.2f}$ & ${np_hfto_df["a"].median():.2f}\pm {np_hfto_df["a"].std():.2f} $ \\\\
LFTO &  ${lfto_df["a"].median():.2f}\pm {lfto_df["a"].std():.2f}$ & ${msp_lfto_df["a"].median():.2f}\pm {msp_lfto_df["a"].std():.2f}$ & ${np_lfto_df["a"].median():.2f}\pm {np_lfto_df["a"].std():.2f} $ \\\\
DTOS &  ${dtos_df["a"].median():.2f}\pm {dtos_df["a"].std():.2f}$ & ${msp_dtos_df["a"].median():.2f}\pm {msp_dtos_df["a"].std():.2f}$ & ${np_dtos_df["a"].median():.2f}\pm {np_dtos_df["a"].std():.2f} $ \\\\
Total & ${df["a"].median():.2f}\pm {df["a"].std():.2f}          $ & ${msp_df["a"].median():.2f}\pm {     msp_df["a"].std():.2f}     $ & ${np_df["a"].median():.2f}\pm {     np_df["a"].std():.2f}      $ \\\\
\hline
''')


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
   :header: "Model", "Total", "%", "MSP", "%", "Normal", "%"

   "simple_power_law",                  "{len(spl_df)}",  "{len(spl_df) /len(df)*100:.1f} %",  "{len(msp_spl_df)}",  "{len(msp_spl_df) /len(msp_df)*100:.1f} %",  "{len(np_spl_df)}", "{len(np_spl_df) /len(np_df)*100:.1f} %"
   "broken_power_law",                  "{len(bpl_df)}",  "{len(bpl_df) /len(df)*100:.1f} %",  "{len(msp_bpl_df)}",  "{len(msp_bpl_df) /len(msp_df)*100:.1f} %",  "{len(np_bpl_df)}", "{len(np_bpl_df) /len(np_df)*100:.1f} %"
   "high_frequency_cut_off_power_law",  "{len(hfto_df)}", "{len(hfto_df)/len(df)*100:.1f} %", "{len(msp_hfto_df)}", "{len(msp_hfto_df)/len(msp_df)*100:.1f} %", "{len(np_hfto_df)}", "{len(np_hfto_df)/len(np_df)*100:.1f} %"
   "low_frequency_turn_over_power_law", "{len(lfto_df)}", "{len(lfto_df)/len(df)*100:.1f} %", "{len(msp_lfto_df)}", "{len(msp_lfto_df)/len(msp_df)*100:.1f} %", "{len(np_lfto_df)}", "{len(np_lfto_df)/len(np_df)*100:.1f} %"
   "double_turn_over_spectrum",         "{len(dtos_df)}", "{len(dtos_df)/len(df)*100:.1f} %", "{len(msp_dtos_df)}", "{len(msp_dtos_df)/len(msp_df)*100:.1f} %", "{len(np_dtos_df)}", "{len(np_dtos_df)/len(np_df)*100:.1f} %"
   "Total",                             "{len(df)}",      "{len(df)     /len(df)*100:.1f} %", "{len(msp_df)}",      "{len(msp_df)     /len(msp_df)*100:.1f} %",      "{len(np_df)}", "{len(np_df)     /len(np_df)*100:.1f} %"

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

def cost(x, y, z):
    return (x - 1) ** 2 + (y - x) ** 2 + (z - 2) ** 2
#cost.errordef = Minuit.LEAST_SQUARES

def make_histogram_plots(all_data, hist_range, label, titles, plotname):
    # Make histogram plots
    n_bins = 20
    colours = [
        'blue',
        'green',
        'orange',
        'purple',
    ]
    n_data = len(all_data) + 1

    fig, axes = plt.subplots(nrows=n_data, figsize=(5, 3*n_data))

    axes[0].hist(all_data, n_bins, density=True, histtype='bar', stacked=True, label=label, color=colours[:n_data-1])
    axes[0].set_title("All spectral indexs")
    axes[0].legend(prop={'size': 10})

    for ai, df_col, colour, title in zip(range(1, n_data), all_data, colours, titles):
        #print(ai, n_data)
        axes[ai].hist(df_col, n_bins, histtype='bar', color=colour, range=hist_range)
        axes[ai].set_title(title)

    fig.tight_layout()
    fig.savefig(plotname)
    plt.close(fig)

# alpha hist
all_indexs = np.array([
    spl_df ["a"],
    hfto_df["a"],
    lfto_df["a"],
    dtos_df["a"],
])
hist_range = (min(df["a"]), max(df["a"]))
titles = [
    'All spectral indexs',
    'Simple power law',
    'High-frequency cut off',
    'Low-frequency turn over',
    'Double turn over spectrum',
]
make_histogram_plots(all_indexs, hist_range, label=["spl", "hfto", "lfto", "dtos"], titles=titles, plotname="spectral_index_histogram.png")


# Make spectral index histogram plots for vc
# ------------------------------------------
hist_range = (np.log10(df["vc"].min()), np.log10(df["vc"].max()))
all_indexs = np.array([
    np.log10(hfto_df["vc"]),
    np.log10(dtos_df["vc"]),
])
titles = [
    'All vc',
    'High-frequency cut off',
    'Double turn over spectrum',
]
make_histogram_plots(all_indexs, hist_range, label=["hfto", "dtos"], titles=titles, plotname="vc_histogram.png")
# just high frequency
fig, ax = plt.subplots()
ax.hist(np.log10(hfto_df["vc"]), 20, histtype='bar', color='blue')
ax.set_title("vc histogram")
# ax.set_xscale('log')
# ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.xaxis.set_major_formatter(
    FuncFormatter(lambda x, _:
        #f'$\mathdefault{{10^{{{x:.2f}}}}}$'
        f'{10**x/1e9:.1f}'
    )
)
ax.set_xlabel(f"$\\nu_c \mathrm{{(GHz)}}$")
ax.set_ylabel(f"#")
fig.tight_layout()
fig.savefig("vc_histogram_just_hfto.png")
plt.close(fig)

# Make spectral index histogram plots for vpeak
# ------------------------------------------
n_bins = 20
hist_range = (np.log10(df["vpeak"].min()), np.log10(df["vpeak"].max()))

all_indexs = np.array([
    np.log10(lfto_df["vpeak"]),
    np.log10(dtos_df["vpeak"]),
])
titles = [
    'All vpeak',
    'Low-frequency turn over',
    'Double turn over spectrum',
]

make_histogram_plots(all_indexs, hist_range, label=["lfto", "dtos"], titles=titles, plotname="vpeak_histogram.png")

# # Remove unessiary columns
# del df["Pulsar"]
# # del df["Model"]
# del df["Probability Best"]
# del df["N data flux"]

# # Remove broken power law stuff
# del df["vb"]
# del df["a1"]
# del df["a2"]
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
    if col_name == "ATNF Fdot":
        log_df[col_name] = np.abs(log_df[col_name])
        log_df[col_name + " log"] = np.log10(log_df[col_name])
    elif "ATNF" in col_name or col_name in ("L400 (mJy kpc^2)", "L1400 (mJy kpc^2)", "Age (Yr)"):
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

def line(x, g, c):
    return g * x + c

def power_law(x, a, c):
    return c*x**a

def yline(x, g, xmean, ymean):
    return ymean + g * (x - xmean)

def xline(y, g, ymean, xmean):
    return xmean + g * (y - ymean)

def spearmanr_ci(x,y,alpha=0.05):
    ''' calculate spearmanr correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''

    r, p = stats.spearmanr(x, y, nan_policy="omit")
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi

def line_fit(
        this_df,
        xcol,
        ycol,
        raw_xcol,
        raw_ycol,
        lax=None,
        label=None,
    ):
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
    msp_x    = np.array(msp_raw_x)
    msp_y    = np.array(msp_raw_y)
    msp_yerr = np.array(msp_raw_yerr)
    np_x     = np.array(np_raw_x)
    np_y     = np.array(np_raw_y)
    np_yerr  = np.array(np_raw_yerr)
    msp_raw_x    = np.array(msp_raw_x)
    msp_raw_y    = np.array(msp_raw_y)
    msp_raw_yerr = np.array(msp_raw_yerr)
    np_raw_x     = np.array(np_raw_x)
    np_raw_y     = np.array(np_raw_y)
    np_raw_yerr  = np.array(np_raw_yerr)
    if "v" in ycol:
        # Convert to MHz
        # y    /= 10**6
        # yerr /= 10**6
        raw_y    /= 10**6
        raw_yerr /= 10**6
        msp_y    /= 10**6
        msp_yerr /= 10**6
        np_y     /= 10**6
        np_yerr  /= 10**6
        msp_raw_y    /= 10**6
        msp_raw_yerr /= 10**6
        np_raw_y     /= 10**6
        np_raw_yerr  /= 10**6
    if "Age" in xcol:
        # Convert to Myr
        # y    /= 10**6
        # yerr /= 10**6
        raw_x    /= 10**6
        msp_x    /= 10**6
        np_x     /= 10**6
        msp_raw_x    /= 10**6
        np_raw_x     /= 10**6
    # print(f"\nCorrelations of x: {xcol} and y: {ycol}. N: {len(x)}")
    # print(f"-------------------------------------------------")
    # print(f"pandas:        {corr_matrix[xcol][ycol]:6.3f}")
    # print(f"numpy:         {np.corrcoef(x,y)[0][1]:6.3f}")
    # weighted_corr = weighted_correlation(x, y ,yerr)
    rho, pval, lo, hi = spearmanr_ci(x, y)
    if abs(rho) >= 0.4 and pval < 0.01:
        weights_str = f"{{\\bf {rho:.2f}}} ({pval:.1e}, {len(x)})"
    else:
        weights_str = f"{rho:.2f} ({pval:.1e}, {len(x)})"
    # print(f"weighted corr: {weighted_correlation(x, y ,yerr):6.3f}")
    # print(f"spearmanr:     {rho:6.3f}")

    if lax is not None:
        f, ax = plt.subplots()

        # fit y line
        # least_squares = LeastSquares(x, y, yerr, yline)
        # xmean=np.mean(x)
        # ymean=np.mean(y)
        # m = Minuit(least_squares, g=0, xmean=xmean, ymean=ymean)
        # m.fixed["xmean"] = True
        # m.fixed["ymean"] = True


        # if 'log' in xcol and 'log' in ycol:
        #     # least_squares = LeastSquares(raw_x, raw_y, raw_yerr, power_law)
        #     least_squares = LeastSquares(raw_x, raw_y, 0.1 / np.log(10) , power_law)
        #     m = Minuit(least_squares, a=0, c=0)
        #     m.migrad()  # finds minimum of least_squares function
        #     m.hesse()   # accurately computes uncertaintiesax):
        #     # print(ycol, xcol)
        #     # print(m)
        #     # exit()
        #     fit_x = np.logspace(np.log10(min(raw_x)), np.log10(max(raw_x)), num=len(raw_x))
        #     fitted_line, fitted_line_cov = propagate(lambda p: power_law(fit_x, *p), m.values, m.covariance)
        #     fitted_line_err = abs(np.diag(fitted_line_cov) ** 0.5)
        # else:
        #     # least_squares = LeastSquares(x, y, yerr, line)
        #     least_squares = LeastSquares(x, y, y*0.1, line)
        #     least_squares.loss = huber_loss_function
        #     m = Minuit(least_squares, g=0, c=-1)
        #     m.migrad()  # finds minimum of least_squares function
        #     m.hesse()   # accurately computes uncertaintiesax):
        #     fit_x = np.logspace(np.log10(min(x)), np.log10(max(x)), num=len(x))
        #     fitted_line, fitted_line_cov = propagate(lambda p: line(fit_x, *p), m.values, m.covariance)
        #     fitted_line_err = abs(np.diag(fitted_line_cov) ** 0.5)
        #     if "log" in ycol:
        #         fitted_line = 10**fitted_line
        #         fitted_line_err = fitted_line * np.log(10) * fitted_line_err
        #         #print(max(fitted_line_err/fitted_line))
        #     if "v" in ycol:
        #         # Convert to MHz
        #         fitted_line     /= 10**6
        #         fitted_line_err /= 10**6
        #         #print(max(fitted_line_err/fitted_line))
        # # least_squares = LeastSquares(x, y, yerr, cost)
        # # m = Minuit(least_squares, y=0, z=0)
        # # m.migrad()
        # # m.draw_mncontour("x", "y", cl=(0.68, 0.9, 0.99))
        # #print(f"error percent: {np.mean(fitted_line_err/fitted_line)}")
        # ax.plot( fit_x, fitted_line, label="fit", c='orange')
        # lax.plot(fit_x, fitted_line, label="fit", c='orange')
        # ax.fill_between( fit_x, fitted_line - fitted_line_err, fitted_line + fitted_line_err, facecolor="C1", alpha=alpha)
        # lax.fill_between(fit_x, fitted_line - fitted_line_err, fitted_line + fitted_line_err, facecolor="C1", alpha=alpha)

        capsize = 1.5
        errorbar_linewidth = 0.7
        marker_border_thickness = 0.5
        markersize = 3.5
        # Add normal pulsar data
        colour = "b"
        ecolour = 'gray'
        alpha = 0.6
        # if "Period" not in xcol and ycol != "vpeak log":
        # if 'vc' in ycol and 'Age' in xcol:
        #     print(list(np_x))
        #     print(list(np_y))
        #     print(list(np_yerr))
        #     exit()
        (_, caps, _) = ax.errorbar(np_raw_x, np_raw_y, np.array(np_raw_yerr) / 2.,
            fmt="o",
            markeredgewidth=marker_border_thickness,
            elinewidth=errorbar_linewidth,
            capsize=capsize,
            markersize=markersize,
            label="Slow pulsars",
            ecolor=ecolour,
            c=colour,
            alpha=alpha,
        )
        for cap in caps:
            cap.set_markeredgewidth(errorbar_linewidth)
        # else:
        #     ax.set_xlim([np_raw_x[0], 0.03])
        (_, caps, _) = lax.errorbar(np_raw_x, np_raw_y, np.array(np_raw_yerr) / 2.,
            fmt="o",
            markeredgewidth=marker_border_thickness,
            elinewidth=errorbar_linewidth,
            capsize=capsize,
            markersize=markersize,
            label="Slow pulsars",
            ecolor=ecolour,
            c=colour,
            alpha=alpha,
        )
        colour = "g"
        for cap in caps:
            cap.set_markeredgewidth(errorbar_linewidth)
        if len(msp_raw_x) > 0:
            # Add MSP data
            (_, caps, _) = ax.errorbar(msp_raw_x, msp_raw_y, np.array(msp_raw_yerr) / 2.,
                fmt='^',
                markeredgewidth=marker_border_thickness,
                elinewidth=errorbar_linewidth,
                capsize=capsize,
                markersize=markersize,
                label="MSPs",
                ecolor=ecolour,
                c=colour,
                alpha=alpha,
            )
            for cap in caps:
                cap.set_markeredgewidth(errorbar_linewidth)
            (_, caps, _) = lax.errorbar(msp_raw_x, msp_raw_y, np.array(msp_raw_yerr) / 2.,
                fmt="^",
                markeredgewidth=marker_border_thickness,
                elinewidth=errorbar_linewidth,
                capsize=capsize,
                markersize=markersize,
                label="MSPs",
                ecolor=ecolour,
                c=colour,
                alpha=alpha,
            )
            for cap in caps:
                cap.set_markeredgewidth(errorbar_linewidth)

        # plot fit line and error bar
        # print(x)
        # print(y)
        # fitted_line, fitted_line_cov = propagate(lambda p: yline(x, *p), m.values, m.covariance)
        # print(fitted_line)
        # print(fitted_line_err)

        if ycol == "vc log" and xcol == "ATNF Spin Frequency (Hz) log":
            # Display a theory line
            fit_x = np.logspace(np.log10(min(raw_x)), np.log10(max(raw_x)))
            fit_y    = 1.4*10**9 * (fit_x)**0.46 / 10**6
            fit_y_lu = 1.4*10**9 * (fit_x)**0.28 / 10**6
            fit_y_hu = 1.4*10**9 * (fit_x)**0.64 / 10**6
            #print(fit_y)
            # fit_y = fit_y / max(fit_y) * max(raw_y)
            # print(fit_y)
            ax.plot( fit_x, fit_y, label='theory', color='purple')
            lax.plot(fit_x, fit_y, label='theory', color='purple')
            ax.fill_between( fit_x, fit_y_lu, fit_y_hu, facecolor='purple', alpha=alpha)
            lax.fill_between(fit_x, fit_y_lu, fit_y_hu, facecolor='purple', alpha=alpha)


        if ycol == "vpeak log":
            #print(vpeak_x, vpeak_y, vpeak_yerr)
            # Add vpeak limits
            colour = 'orange'
            if "Period" in xcol:
                (_, caps, _) = ax.errorbar(
                    vpeak_x, vpeak_y, vpeak_yerr,
                    uplims=True,
                    fmt="^",
                    markeredgewidth=marker_border_thickness,
                    elinewidth=errorbar_linewidth,
                    capsize=capsize,
                    markersize=markersize,
                    label="MSPs",
                    ecolor=ecolour,
                    c=colour,
                    alpha=alpha,
                )
                for cap in caps:
                    cap.set_markeredgewidth(errorbar_linewidth)
            # (_, caps, _) = lax.errorbar(
            #     vpeak_x, vpeak_y, vpeak_yerr,
            #     uplims=True,
            #     fmt="o",
            #     markeredgewidth=marker_border_thickness,
            #     elinewidth=errorbar_linewidth,
            #     capsize=capsize,
            #     markersize=markersize,
            #     label="MSPs",
            #     ecolor=ecolour,
            #     c=colour,
            # )
            # for cap in caps:
            #     cap.set_markeredgewidth(errorbar_linewidth)

        # Labels
        ax.set_xlabel( f"{math_names_x[xcol]}")
        lax.set_xlabel(f"{math_names_x[xcol]}")
        ax.set_ylabel( f"{math_names_y[ycol]}")
        lax.set_ylabel(f"{math_names_y[ycol]}")
        ax.legend( title=f"Corr=${rho:.2f}_{{-{rho-lo:.2f}}}^{{+{hi-rho:.2f}}}$")
        lax.legend(title=f"Corr=${rho:.2f}_{{-{rho-lo:.2f}}}^{{+{hi-rho:.2f}}}$")
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
        #     f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {len(x) - m.nfit}",
        # ]
        # for p, v, e in zip(m.parameters, m.values, m.errors):
        #     fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")

        # f.legend(title="\n".join(fit_info))
        ax.set_title(f"{label}")
        f.tight_layout()
        f.savefig(f'corr_plots/corr_line_{ycol}_{xcol.replace("/", "_")}_{label}.png'.replace(" ", "_"), dpi=300)
        plt.close(f)
    return weights_str

# Make individual correlation plots
# df, x, y
corr_plots = [
    (log_df, "ATNF E_dot (ergs/s) log", "vpeak"),
]
xcols = [
    # "ATNF Period (s) log",
    "ATNF Spin Frequency (Hz) log",
    "ATNF Fdot log",
    "ATNF Pdot log",
    "ATNF DM log",
    # "ATNF B_surf (G) log",
    "ATNF B_LC (G) log",
    "Age (Yr) log",
    "ATNF E_dot (ergs/s) log",
    "L400 (mJy kpc^2) log",
    "L1400 (mJy kpc^2) log",
]
ycols = [
        "a"         ,
        # "c"         ,
        # "vb"        ,
        # "a1"        ,
        # "a2"        ,
        "vpeak log"     ,
        "vc log"        ,
        # "beta"      ,
]

math_names_x = {
    "ATNF Period (s) log": "P (s)",
    "ATNF Pdot log": "$\dot{P}$ (s s$^{-1}$)",
    "ATNF Spin Frequency (Hz) log": "$\\tilde{\\nu}$ (Hz)",
    "ATNF Fdot log": "$\left| \dot{\\tilde{\\nu}} \\right|$ (Hz)",
    "ATNF DM log": "DM (pc cm$^{-3}$)",
    "ATNF B_surf (G) log": "$B_{surf}$ (G)",
    "ATNF B_LC (G) log": "$B_{LC}$ (G)",
    "ATNF E_dot (ergs/s) log": "$\dot{E}$ (ergs/s)",
    "L400 (mJy kpc^2) log": "$L_{400}$ (mJy kpc$^2$)",
    "L1400 (mJy kpc^2) log": "$L_{1400}$ (mJy kpc$^2$)",
    "Age (Yr) log": "$\\tau$ (MYr)",
}
math_names_y = {
        "a": "$\\alpha$",
        "vpeak log": "$\\nu_{peak}$ (MHz)",
        "vc log": "$\\nu_{c}$ (MHz)",
}
nx = len(xcols)
ny = len(ycols)
sfig, saxes = plt.subplots(ny, nx, figsize=(5*nx, 5*ny))
lfig, laxes = plt.subplots(ny, nx, figsize=(5*nx, 5*ny))
bfig, baxes = plt.subplots(ny, nx, figsize=(5*nx, 5*ny))
ifig, iaxes = plt.subplots(ny, nx, figsize=(5*nx, 5*ny))
mfig, maxes = plt.subplots(ny, nx, figsize=(5*nx, 5*ny))
sfig, saxes = plt.subplots(ny, nx, figsize=(5*nx, 5*ny))
corr_matrix = log_df.corr()
# print(list(df[df.notnull()]["ATNF Fdot"]))
# print(list(log_df[log_df.notnull()]["ATNF Fdot log"]))
# exit()
for ya, ycol in enumerate(ycols):
    weights = []
    ycol_df = log_df[log_df[ycol].notnull()]
    if ycol == "a":
        ycol_df = ycol_df[ycol_df["Model"] == "simple_power_law"]
    nbinary   = len(ycol_df[ycol_df["ANTF Binary (type)"].notnull()])
    nisolated = len(ycol_df[ycol_df["ANTF Binary (type)"].isnull()])
    nmsp      = len(ycol_df[ycol_df["ATNF Period (s)"] < msp_cutoff])
    nslow     = len(ycol_df[ycol_df["ATNF Period (s)"] >= msp_cutoff])
    print("")
    print("\\\\")
    print(f"\multicolumn{{6}}{{c}}{{ ${math_names_y[ycol]}$ Correlation Coefficient }} \\\\")
    print("\hline")
    print("set & all & in binary & isolated & MSP & slow \\\\")
    print(f"\#pulsars & {len(ycol_df)} & {nbinary} & {nisolated} & {nmsp} & {nslow}\\\\")
    print("\hline")
    print("$log_{10}(x)$ & $r_s (p, N)$ & $r_s (p, N)$ & $r_s (p, N)$ & $r_s (p, N)$ & $r_s (p, N)$ \\\\")
    print("\hline")
    for xa, xcol in enumerate(xcols):
        if ycol == "beta":
            this_df = log_df_sync
        else:
            this_df = log_df
        if ycol == "a":
            # only use simple power law
            this_df = this_df[this_df["Model"] == "simple_power_law"]
            # Filter out outliers
            this_df = this_df[this_df["a"] < 0]
            this_df = this_df[this_df["a"] > -4]
        if ycol == "vc log":
            # only use high frequency cut off
            this_df = this_df[this_df["Model"] == "high_frequency_cut_off_power_law"]
            this_df = this_df[this_df["Pulsar"] != 'J1917+1353']
        # Filter out low luminosity
        if xcol == "L400 (mJy kpc^2) log":
            this_df = this_df[this_df["L400 (mJy kpc^2)"] > 3]
        if xcol == "L1400 (mJy kpc^2) log":
            this_df = this_df[this_df["L1400 (mJy kpc^2)"] > 0.1]
        # Make
        if "log" in xcol:
            raw_xcol = xcol[:-4]
        else:
            raw_xcol = xcol
        if "log" in ycol:
            raw_ycol = ycol[:-4]
        else:
            raw_ycol = ycol

        sub_weights = []
        binary_df = this_df[this_df["ANTF Binary (type)"].notnull()]
        isolated_df = this_df[this_df["ANTF Binary (type)"].isnull()]
        msp_df  = this_df[this_df["ATNF Period (s)"] < msp_cutoff]
        slow_df = this_df[this_df["ATNF Period (s)"] >= msp_cutoff]

        df_axes_pairs = [
            (this_df,     laxes, 'All Pulsars'),
            (binary_df,   baxes, 'Only Binary Pulsars'),
            (isolated_df, iaxes, 'Only Isolated Pulsars'),
            (msp_df,      maxes, 'Only MSPs'),
            (slow_df,     saxes, 'Only Slow Pulsars'),
        ]

        # print(f"line:          {weighted_line_corr:6.3f}")
        for sub_df, sub_axes, label in df_axes_pairs:
            weight_str = weighted_line_corr = line_fit(
                sub_df,
                xcol,
                ycol,
                raw_xcol,
                raw_ycol,
                lax=sub_axes[ya][xa],
                label=label,
            )
            sub_weights.append(weight_str)
        print(f'${math_names_x[xcol]}$ & {" & ".join(sub_weights)} \\\\')

        sns.regplot(data=this_df, x=xcol, y=ycol, ax=saxes[ya][xa])
    print("\hline")
    #print(f'{ycol} & {" & ".join(weights)} \\')
sfig.savefig(f'corr_plots/coor_all.png', dpi=300)
lfig.tight_layout()
lfig.savefig(f'corr_plots/corr_all_weighted.png', dpi=300)
bfig.savefig(f'corr_plots/corr_b_weighted.png', dpi=300)
ifig.savefig(f'corr_plots/corr_i_weighted.png', dpi=300)
mfig.savefig(f'corr_plots/corr_m_weighted.png', dpi=300)
sfig.savefig(f'corr_plots/corr_s_weighted.png', dpi=300)
