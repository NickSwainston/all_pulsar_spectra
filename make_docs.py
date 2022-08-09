import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
   smart_gallery


Fit Summary
-----------
.. csv-table::
   :header: "Model", "Pulsar Count"

   "simple_power_law", "{len(spl_df)}"
   "broken_power_law", "{len(bpl_df)}"
   "log_parabolic_spectrum", "{len(lps_df)}"
   "high_frequency_cut_off_power_law", "{len(hfto_df)}"
   "low_frequency_turn_over_power_law", "{len(lfto_df)}"
   "double_turn_over_spectrum", "{len(dtos_df)}"
   "Total", "{len(df)}"

Analysis Summary
----------------
.. csv-table::
   :header: "Parameter", "Mean"

   "spectral index", "{df["a"].mean():.2f}±{df["a"].std():.2f}"
   "Peak Frequency (GHz)", "{df["vpeak"].mean()/1e9:.2f}±{df["vpeak"].std()/1e9:.2f}"
   "Cut off frequency (GHz)", "{df["vc"].mean()/1e9:.2f}±{df["vc"].std()/1e9:.2f}"
   "Beta", "{df["beta"].mean():.2f}±{df["beta"].std():.2f}"

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




# Remove unessiary columns
del df["Pulsar"]
del df["Model"]
del df["Probability Best"]
del df["Min freq (MHz)"]
del df["Max freq (MHz)"]
del df["N data flux"]

# Remove broken power law stuff
del df["vb"]
del df["a1"]
del df["a2"]
for col_name in df.keys():
    if "u_" in col_name or col_name.endswith("_c"):
        del df[col_name]

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

log_df = df
# convert to log data
for col_name in log_df.keys():
    if "ATNF" in col_name or "_v" in col_name:
        log_df[col_name + " log"] = np.log10(log_df[col_name])
        del log_df[col_name]

# Plot correlation
corr = log_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True,
            cmap='seismic', mask=mask,
            vmin=-1, vmax=1,)
corr.to_csv(f"pulsar_coor_log.csv")
plt.tight_layout(pad=0.5)
plt.savefig("pulsa_coor_log.png")

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
]
ycols = [
        "a"         ,
        # "c"         ,
        # "vb"        ,
        # "a1"        ,
        # "a2"        ,
        "vc"        ,
        "vpeak"     ,
        "beta"      ,
]
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
print(axes)
print(max(log_df["vpeak"]))

for xa, xcol in enumerate(xcols):
    for ya, ycol in enumerate(ycols):
        # x = []
        # y = []
        # for xi, yi in zip(list(log_df[xcol]), list(log_df[ycol])):
        #     if not ( np.isnan(xi) or np.isnan(yi) ):
        #         x.append(xi)
        #         y.append(yi)
        # print(np.cov(x,y))
        sns.regplot(log_df[xcol], log_df[ycol], ax=axes[ya][xa])
fig.savefig(f'coor_all.png')
