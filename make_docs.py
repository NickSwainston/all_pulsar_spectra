import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy import stats

import pulsar_spectra
from pulsar_spectra.catalogue import collect_catalogue_fluxes

cat_list = collect_catalogue_fluxes()

docs_dir = f"{os.path.dirname(os.path.realpath(__file__))}/docs"

# Read and organise data
# -----------------------------------------------------------------------------

capsize = 1.5
errorbar_linewidth = 0.7
marker_border_thickness = 0.5
markersize = 3.5
# Add normal pulsar data
colour = "b"
ecolour = "gray"
alpha = 0.6

# Read in the fits
df = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/all_pulsar_fits.csv")

# Grab model specific data frames
spl_df = df[df["Model"] == "simple_power_law"]
bpl_df = df[df["Model"] == "broken_power_law"]
hfto_df = df[df["Model"] == "high_frequency_cut_off_power_law"]
lfto_df = df[df["Model"] == "low_frequency_turn_over_power_law"]
dtos_df = df[df["Model"] == "double_turn_over_spectrum"]
nomod_df = df[df["Model"] == ""]
smart_df = df[df["SMART"]]

# query = QueryATNF(params=['P0', 'P1', 'ASSOC', 'BINARY', 'TYPE', 'P1_I'])
# fig = query.ppdot()
# fig.savefig("ppdot_all.png")
# query = QueryATNF(psrs=list(df["Pulsar"]), params=['P0', 'P1', 'ASSOC', 'BINARY', 'TYPE', 'P1_I'])
# fig = query.ppdot()
# fig.savefig("ppdot_ours.png")
# query = QueryATNF(psrs=list(hfto_df["Pulsar"]), params=['P0', 'P1', 'ASSOC', 'BINARY', 'TYPE', 'P1_I'])
# fig = query.ppdot()
# fig.savefig("ppdot_hfco.png")
# query = QueryATNF(psrs=list(lfto_df["Pulsar"]), params=['P0', 'P1', 'ASSOC', 'BINARY', 'TYPE', 'P1_I'])
# fig = query.ppdot()
# fig.savefig("ppdot_lfto.png")
# exit()

msp_cutoff = 0.03  # seconds

# MSPs
msp_spl_df = spl_df[spl_df["ATNF Period (s)"] < msp_cutoff]
msp_bpl_df = bpl_df[bpl_df["ATNF Period (s)"] < msp_cutoff]
msp_hfto_df = hfto_df[hfto_df["ATNF Period (s)"] < msp_cutoff]
msp_lfto_df = lfto_df[lfto_df["ATNF Period (s)"] < msp_cutoff]
msp_dtos_df = dtos_df[dtos_df["ATNF Period (s)"] < msp_cutoff]
msp_df = df[df["ATNF Period (s)"] < msp_cutoff]

# Normal pulsars
np_spl_df = spl_df[spl_df["ATNF Period (s)"] >= msp_cutoff]
np_bpl_df = bpl_df[bpl_df["ATNF Period (s)"] >= msp_cutoff]
np_hfto_df = hfto_df[hfto_df["ATNF Period (s)"] >= msp_cutoff]
np_lfto_df = lfto_df[lfto_df["ATNF Period (s)"] >= msp_cutoff]
np_dtos_df = dtos_df[dtos_df["ATNF Period (s)"] >= msp_cutoff]
np_df = df[df["ATNF Period (s)"] >= msp_cutoff]


# Output summary in latex format
print(f"""
Model & Total & \% & MSP & \% & Normal & \% \\\\

SPL   & {len(spl_df)}  & {len(spl_df) / len(df) * 100:.1f} \% & {len(msp_spl_df)}  & {len(msp_spl_df) / len(msp_df) * 100:.1f} \% &  {len(np_spl_df)} & {len(np_spl_df) / len(np_df) * 100:.1f} \% \\\\
BPL   & {len(bpl_df)}  & {len(bpl_df) / len(df) * 100:.1f} \% & {len(msp_bpl_df)}  & {len(msp_bpl_df) / len(msp_df) * 100:.1f} \% &  {len(np_bpl_df)} & {len(np_bpl_df) / len(np_df) * 100:.1f} \% \\\\
HFCO  & {len(hfto_df)} & {len(hfto_df) / len(df) * 100:.1f} \% & {len(msp_hfto_df)} & {len(msp_hfto_df) / len(msp_df) * 100:.1f} \% & {len(np_hfto_df)} & {len(np_hfto_df) / len(np_df) * 100:.1f} \% \\\\
LFTO  & {len(lfto_df)} & {len(lfto_df) / len(df) * 100:.1f} \% & {len(msp_lfto_df)} & {len(msp_lfto_df) / len(msp_df) * 100:.1f} \% & {len(np_lfto_df)} & {len(np_lfto_df) / len(np_df) * 100:.1f} \% \\\\
DTOS  & {len(dtos_df)} & {len(dtos_df) / len(df) * 100:.1f} \% & {len(msp_dtos_df)} & {len(msp_dtos_df) / len(msp_df) * 100:.1f} \% & {len(np_dtos_df)} & {len(np_dtos_df) / len(np_df) * 100:.1f} \% \\\\
Total & {len(df)}      & {len(df) / len(df) * 100:.1f} \% & {len(msp_df)}      & {len(msp_df) / len(msp_df) * 100:.1f} \% &      {len(np_df)} & {len(np_df) / len(np_df) * 100:.1f} \%)\\\\
""")

print(np.std(spl_df["a"], ddof=1) / np.sqrt(np.size(spl_df["a"])))

print(f"""
\hline
Model & All Mean & MSP Mean & Normal Mean \\\\
\hline
SPL &   ${spl_df["a"].mean():.2f} \pm {spl_df["a"].std():.2f} $ & ${msp_spl_df["a"].mean():.2f} \pm {msp_spl_df["a"].std():.2f}$ & $ {np_spl_df["a"].mean():.2f}\pm {np_spl_df["a"].std():.2f}$ \\\\
HFCO &  ${hfto_df["a"].mean():.2f}\pm {hfto_df["a"].std():.2f}$ & ${msp_hfto_df["a"].mean():.2f}\pm {msp_hfto_df["a"].std():.2f}$ & ${np_hfto_df["a"].mean():.2f}\pm {np_hfto_df["a"].std():.2f}$ \\\\
LFTO &  ${lfto_df["a"].mean():.2f}\pm {lfto_df["a"].std():.2f}$ & ${msp_lfto_df["a"].mean():.2f}\pm {msp_lfto_df["a"].std():.2f}$ & ${np_lfto_df["a"].mean():.2f}\pm {np_lfto_df["a"].std():.2f}$ \\\\
DTOS &  ${dtos_df["a"].mean():.2f}\pm {dtos_df["a"].std():.2f}$ & ${msp_dtos_df["a"].mean():.2f}\pm {msp_dtos_df["a"].std():.2f}$ & ${np_dtos_df["a"].mean():.2f}\pm {np_dtos_df["a"].std():.2f}$ \\\\
Total & ${df["a"].mean():.2f}     \pm {df["a"].std():.2f}     $ & ${msp_df["a"].mean():.2f}     \pm {msp_df["a"].std():.2f}$ & $     {np_df["a"].mean():.2f}\pm {np_df["a"].std():.2f}$ \\\\
\hline
\\\\
\hline
Model & All Median & MSP Median & Normal Median \\\\
\hline
SPL &   ${spl_df["a"].median():.2f} \pm {spl_df["a"].std():.2f} $ & ${msp_spl_df["a"].median():.2f} \pm {msp_spl_df["a"].std():.2f}$ & ${np_spl_df["a"].median():.2f} \pm {np_spl_df["a"].std():.2f} $ \\\\
HFCO &  ${hfto_df["a"].median():.2f}\pm {hfto_df["a"].std():.2f}$ & ${msp_hfto_df["a"].median():.2f}\pm {msp_hfto_df["a"].std():.2f}$ & ${np_hfto_df["a"].median():.2f}\pm {np_hfto_df["a"].std():.2f} $ \\\\
LFTO &  ${lfto_df["a"].median():.2f}\pm {lfto_df["a"].std():.2f}$ & ${msp_lfto_df["a"].median():.2f}\pm {msp_lfto_df["a"].std():.2f}$ & ${np_lfto_df["a"].median():.2f}\pm {np_lfto_df["a"].std():.2f} $ \\\\
DTOS &  ${dtos_df["a"].median():.2f}\pm {dtos_df["a"].std():.2f}$ & ${msp_dtos_df["a"].median():.2f}\pm {msp_dtos_df["a"].std():.2f}$ & ${np_dtos_df["a"].median():.2f}\pm {np_dtos_df["a"].std():.2f} $ \\\\
Total & ${df["a"].median():.2f}\pm {df["a"].std():.2f}     $ & ${msp_df["a"].median():.2f}     \pm {msp_df["a"].std():.2f}$ & ${np_df["a"].median():.2f}     \pm {np_df["a"].std():.2f} $ \\\\
\hline
""")


# Set up docs
# -----------------------------------------------------------------------------

# Record summary results on homepage
with open(f"{docs_dir}/index.rst", "w") as file:
    file.write(f'''
Pulsar Spectra all pulsars fit results
======================================

The following is the result of fitting all pulsars with more than four flux density measurements in version
of pulsar_spectra. If using any of the data, please cite `Swainston et al. (2022) <https://ui.adsabs.harvard.edu/abs/2022PASA...39...56S/abstract>`_
and `Nicholas Swainston's thesis <https://catalogue.curtin.edu.au/discovery/search?vid=61CUR_INST:CUR_ALMA>`_ (link will be updated once it is published).
Chapter 6 of the thesis analyised version {pulsar_spectra.__version__}.

.. toctree::
    :maxdepth: 1
    :caption: Spectral Fit Summaries:

    spectral_index_summary
    vpeak_summary
    vc_summary
    suggested_campaigns


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

    "simple_power_law",                  "{len(spl_df)}",  "{len(spl_df) / len(df) * 100:.1f} %",  "{len(msp_spl_df)}",  "{len(msp_spl_df) / len(msp_df) * 100:.1f} %",  "{len(np_spl_df)}", "{len(np_spl_df) / len(np_df) * 100:.1f} %"
    "broken_power_law",                  "{len(bpl_df)}",  "{len(bpl_df) / len(df) * 100:.1f} %",  "{len(msp_bpl_df)}",  "{len(msp_bpl_df) / len(msp_df) * 100:.1f} %",  "{len(np_bpl_df)}", "{len(np_bpl_df) / len(np_df) * 100:.1f} %"
    "high_frequency_cut_off_power_law",  "{len(hfto_df)}", "{len(hfto_df) / len(df) * 100:.1f} %", "{len(msp_hfto_df)}", "{len(msp_hfto_df) / len(msp_df) * 100:.1f} %", "{len(np_hfto_df)}", "{len(np_hfto_df) / len(np_df) * 100:.1f} %"
    "low_frequency_turn_over_power_law", "{len(lfto_df)}", "{len(lfto_df) / len(df) * 100:.1f} %", "{len(msp_lfto_df)}", "{len(msp_lfto_df) / len(msp_df) * 100:.1f} %", "{len(np_lfto_df)}", "{len(np_lfto_df) / len(np_df) * 100:.1f} %"
    "double_turn_over_spectrum",         "{len(dtos_df)}", "{len(dtos_df) / len(df) * 100:.1f} %", "{len(msp_dtos_df)}", "{len(msp_dtos_df) / len(msp_df) * 100:.1f} %", "{len(np_dtos_df)}", "{len(np_dtos_df) / len(np_df) * 100:.1f} %"
    "Total",                             "{len(df)}",      "{len(df) / len(df) * 100:.1f} %", "{len(msp_df)}",      "{len(msp_df) / len(msp_df) * 100:.1f} %",      "{len(np_df)}", "{len(np_df) / len(np_df) * 100:.1f} %"

Analysis Summary
----------------
.. csv-table::
    :header: "Parameter", "All Mean", "MSP Mean", "Normal Mean"

    "spectral index",          "{df["a"].mean():.2f}±{df["a"].std():.2f}",                 "{msp_df["a"].mean():.2f}±{msp_df["a"].std():.2f}",                 "{np_df["a"].mean():.2f}±{np_df["a"].std():.2f}"
    "Peak Frequency (GHz)",    "{df["vpeak"].mean() / 1e9:.2f}±{df["vpeak"].std() / 1e9:.2f}", "{msp_df["vpeak"].mean() / 1e9:.2f}±{msp_df["vpeak"].std() / 1e9:.2f}", "{np_df["vpeak"].mean() / 1e9:.2f}±{np_df["vpeak"].std() / 1e9:.2f}"
    "Cut off frequency (GHz)", "{df["vc"].mean() / 1e9:.2f}±{df["vc"].std() / 1e9:.2f}",       "{msp_df["vc"].mean() / 1e9:.2f}±{msp_df["vc"].std() / 1e9:.2f}",       "{np_df["vc"].mean() / 1e9:.2f}±{np_df["vc"].std() / 1e9:.2f}"
    "Beta",                    "{df["beta"].mean():.2f}±{df["beta"].std():.2f}",           "{msp_df["beta"].mean():.2f}±{msp_df["beta"].std():.2f}",           "{np_df["beta"].mean():.2f}±{np_df["beta"].std():.2f}"

Single Power Law Results
------------------------
.. csv-table::
    :header: "Pulsar", "a"

''')
    for index, row in spl_df.iterrows():
        data_str = f'    ":ref:`{row["Pulsar"]}`", '
        for val, error in [("a", "u_a")]:
            if "v" in val:
                data_str += f'"{int(row[val] / 1e6):d}±{int(row[error] / 1e6):d}", '
            else:
                data_str += f'"{row[val]:.2f}±{row[error]:.2f}", '
        file.write(f"{data_str[:-2]}\n")

    file.write("""


Broken Power Law Results
------------------------
.. csv-table::
    :header: "Pulsar", "vb (MHz)", "a1", "a2"

""")
    for index, row in bpl_df.iterrows():
        data_str = f'    ":ref:`{row["Pulsar"]}`", '
        for val, error in [("vb", "u_vb"), ("a1", "u_a1"), ("a2", "u_a2")]:
            if "v" in val:
                data_str += f'"{int(row[val] / 1e6):d}±{int(row[error] / 1e6):d}", '
            else:
                data_str += f'"{row[val]:.2f}±{row[error]:.2f}", '
        file.write(f"{data_str[:-2]}\n")

    file.write("""


Low Frequency Turn Over Results
-------------------------------
.. csv-table::
    :header: "Pulsar", "vpeak (MHz)", "a", "beta"

""")
    for index, row in lfto_df.iterrows():
        data_str = f'    ":ref:`{row["Pulsar"]}`", '
        for val, error in [("vpeak", "u_vpeak"), ("a", "u_a"), ("beta", "u_beta")]:
            if "v" in val:
                data_str += f'"{int(row[val] / 1e6):d}±{int(row[error] / 1e6):d}", '
            else:
                data_str += f'"{row[val]:.2f}±{row[error]:.2f}", '
        file.write(f"{data_str[:-2]}\n")

    file.write("""


High Frequency Cut Off Results
------------------------------
.. csv-table::
    :header: "Pulsar", "vc (MHz)", "a"

""")
    for index, row in hfto_df.iterrows():
        data_str = f'    ":ref:`{row["Pulsar"]}`", '
        for val, error in [("vc", "u_vc"), ("a", "u_a")]:
            if "v" in val:
                data_str += f'"{int(row[val] / 1e6):d}±{int(row[error] / 1e6):d}", '
            else:
                data_str += f'"{row[val]:.2f}±{row[error]:.2f}", '
        file.write(f"{data_str[:-2]}\n")

    file.write("""


Double Turn Over Spectrum Results
---------------------------------
.. csv-table::
    :header: "Pulsar", "vc (MHz)", "vpeak (MHz)", "a", "beta"

""")
    for index, row in dtos_df.iterrows():
        data_str = f'    ":ref:`{row["Pulsar"]}`", '
        for val, error in [
            ("vc", "u_vc"),
            ("vpeak", "u_vpeak"),
            ("a", "u_a"),
            ("beta", "u_beta"),
        ]:
            if "v" in val:
                data_str += f'"{int(row[val] / 1e6):d}±{int(row[error] / 1e6):d}", '
            else:
                data_str += f'"{row[val]:.2f}±{row[error]:.2f}", '
        file.write(f"{data_str[:-2]}\n")


# Set up the gallerys
# -----------------------------------------------------------------------------
with open(f"{docs_dir}/spl_gallery.rst", "w") as file:
    file.write("""
Simple Power Law Gallery
========================

""")
    for index, row in spl_df.iterrows():
        file.write(f"""

.. _{row["Pulsar"]}:

{row["Pulsar"]}
{"-" * len(row["Pulsar"])}
.. image:: best_fits/{row["Pulsar"]}_fit.png
    :width: 800
""")

with open(f"{docs_dir}/bpl_gallery.rst", "w") as file:
    file.write("""
Broken Power Law Gallery
========================

""")
    for index, row in bpl_df.iterrows():
        file.write(f"""

.. _{row["Pulsar"]}:

{row["Pulsar"]}
{"-" * len(row["Pulsar"])}
.. image:: best_fits/{row["Pulsar"]}_fit.png
    :width: 800
""")

with open(f"{docs_dir}/lfto_gallery.rst", "w") as file:
    file.write("""
Low Frequency Turn Over Gallery
===============================

""")
    for index, row in lfto_df.iterrows():
        file.write(f"""

.. _{row["Pulsar"]}:

{row["Pulsar"]}
{"-" * len(row["Pulsar"])}
.. image:: best_fits/{row["Pulsar"]}_fit.png
    :width: 800
""")

with open(f"{docs_dir}/hfco_gallery.rst", "w") as file:
    file.write("""
High Frequency Cut Off Gallery
==============================

""")
    for index, row in hfto_df.iterrows():
        file.write(f"""

.. _{row["Pulsar"]}:

{row["Pulsar"]}
{"-" * len(row["Pulsar"])}
.. image:: best_fits/{row["Pulsar"]}_fit.png
    :width: 800
""")

with open(f"{docs_dir}/dtos_gallery.rst", "w") as file:
    file.write("""
Double Turn Over Spectrum Gallery
=================================

""")
    for index, row in dtos_df.iterrows():
        file.write(f"""

.. _{row["Pulsar"]}:

{row["Pulsar"]}
{"-" * len(row["Pulsar"])}
.. image:: best_fits/{row["Pulsar"]}_fit.png
    :width: 800
""")

with open(f"{docs_dir}/smart_gallery.rst", "w") as file:
    file.write("""
SMART Gallery
=============

All pulsar detections from the SMART pulsar survey (these will be in other galleries).

""")
    for index, row in smart_df.iterrows():
        file.write(f"""

{row["Pulsar"]}
{"-" * len(row["Pulsar"])}
.. image:: best_fits/{row["Pulsar"]}_fit.png
    :width: 800
""")


with open(f"{docs_dir}/msp_gallery.rst", "w") as file:
    file.write("""
MSP Gallery
===========

All millisecond pulsar detections (these will be in other galleries).

""")
    for index, row in msp_df.iterrows():
        file.write(f"""

{row["Pulsar"]}
{"-" * len(row["Pulsar"])}
.. image:: best_fits/{row["Pulsar"]}_fit.png
    :width: 800
""")


# Based on estimated flux density and available measurement frequency, recomend pulsars that should be measured in a flux density campaign
low_freq_camp = pd.DataFrame(
    columns=[
        "Pulsar",
        "Model",
        "Min freq (MHz)",
        "S150 (mJy)",
        "S300 (mJy)",
    ]
)
high_freq_camp = pd.DataFrame(
    columns=[
        "Pulsar",
        "Model",
        "Max freq (MHz)",
        "S5000 (mJy)",
        "S10000 (mJy)",
    ]
)
min_flux_low = 1  # mJy
max_freq_low = 300  # MHz
min_flux_high = 0.01  # mJy
min_freq_high = 5000  # MHz
for index, row in df.iterrows():
    pulsar = row["Pulsar"]
    # model
    if row["Model"] == "simple_power_law":
        model = "SPL"
    elif row["Model"] == "broken_power_law":
        model = "BPL"
    elif row["Model"] == "log_parabolic_spectrum":
        model = "LPS"
    elif row["Model"] == "high_frequency_cut_off_power_law":
        model = "HFCO"
    elif row["Model"] == "low_frequency_turn_over_power_law":
        model = "LFTO"
    elif row["Model"] == "double_turn_over_spectrum":
        model = "DTOS"
    # freqs
    min_freq = float(row["Min freq (MHz)"])
    max_freq = float(row["Max freq (MHz)"])
    # fluxs
    s150 = float(row["S150 (mJy)"])
    s300 = float(row["S300 (mJy)"])
    s5000 = float(row["S5000 (mJy)"])
    s10000 = float(row["S10000 (mJy)"])
    u_s150 = float(row["u_S150 (mJy)"])
    u_s300 = float(row["u_S300 (mJy)"])
    u_s5000 = float(row["u_S5000 (mJy)"])
    u_s10000 = float(row["u_S10000 (mJy)"])

    # Check if worth following up at low freq
    # print(pulsar, min_freq, s150, s300)
    # print(min_freq > 300, s150 > min_flux_low, s300 > min_flux_low)
    if min_freq > max_freq_low and (s150 > min_flux_low and s300 > min_flux_low):
        low_freq_camp = pd.concat(
            [
                low_freq_camp,
                pd.Series(
                    {
                        "Pulsar": pulsar,
                        "Model": model,
                        "Min freq (MHz)": min_freq,
                        "S150 (mJy)": s150,
                        "S300 (mJy)": s300,
                        "u_S150 (mJy)": u_s150,
                        "u_S300 (mJy)": u_s300,
                    }
                )
                .to_frame()
                .T,
            ],
            ignore_index=True,
        )
    # Check if worth following up at high freq
    # print(pulsar, max_freq, s5000, s10000)
    # print(max_freq < 5000, s5000 > min_flux_high, s10000 > min_flux_high)
    if max_freq < min_freq_high and (s5000 > min_flux_high and s10000 > min_flux_high):
        high_freq_camp = pd.concat(
            [
                high_freq_camp,
                pd.Series(
                    {
                        "Pulsar": pulsar,
                        "Model": model,
                        "Max freq (MHz)": max_freq,
                        "S5000 (mJy)": s5000,
                        "S10000 (mJy)": s10000,
                        "u_S5000 (mJy)": u_s5000,
                        "u_S10000 (mJy)": u_s10000,
                    }
                )
                .to_frame()
                .T,
            ],
            ignore_index=True,
        )

# print(low_freq_camp)
# print(high_freq_camp)
low_freq_camp.to_csv("low_freq_camp.csv", index=False)
high_freq_camp.to_csv("high_freq_camp.csv", index=False)

# Write them to the webpage
with open(f"{docs_dir}/suggested_campaigns.rst", "w") as file:
    file.write(f"""
Suggested Campaigns
===================

Based on estimated flux densities and lack of available measurement frequency, the following pulsars are recommended inclusions in a flux density measurement campaign.

Low Frequency Campaign
----------------------

The following {len(low_freq_camp)} pulsars had no flux density measurements below {min_freq_high} MHz and
an estimated flux density of greater than {min_flux_low} mJy at both 150 or 300 MHz.
The CSV file can be found `here <https://github.com/NickSwainston/all_pulsar_spectra/blob/{pulsar_spectra.__version__}/low_freq_camp.csv>`__.

.. csv-table::
    :header: "Pulsar", "Model", "Min freq (MHz)", "S150 (mJy)", "S300 (mJy)"

""")
    print("Pulsar & Model & Min freq (MHz) & S150 (mJy) & S300 (mJy) \\\\")
    print("\hline \hline")
    for index, row in low_freq_camp.iterrows():
        data_str = f'    ":ref:`{row["Pulsar"]}`", "{row["Model"]}", "{row["Min freq (MHz)"]}", '
        for val, error in [
            ("S150 (mJy)", "u_S150 (mJy)"),
            ("S300 (mJy)", "u_S300 (mJy)"),
        ]:
            data_str += f'"{row[val]:.1f}±{row[error]:.1f}", '
        file.write(f"{data_str[:-2]}\n")
        print(
            f"{row['Pulsar']} & {row['Model']} & {int(row['Min freq (MHz)'])} & {row['S150 (mJy)']:.1f} $\\pm$ {row['u_S150 (mJy)']:.1f} & {row['S300 (mJy)']:.1f} $\\pm$ {row['S300 (mJy)']:.1f} \\\\"
        )

    file.write(f"""

High Frequency Campaign
-----------------------

The following {len(high_freq_camp)} pulsars had no flux density measurements above {max_freq_low} MHz and
an estimated flux density of greater than {min_flux_high} mJy both either 5 or 10 GHz.
The CSV file can be found `here <https://github.com/NickSwainston/all_pulsar_spectra/blob/{pulsar_spectra.__version__}/high_freq_camp.csv>`__.

.. csv-table::
    :header: "Pulsar", "Model", "Max freq (MHz)", "S5000 (mJy)", "S10000 (mJy)"

""")
    print("Pulsar & Model & Max freq (MHz) & S5000 (mJy) & S10000 (mJy) \\\\")
    print("\hline \hline")
    for index, row in high_freq_camp.iterrows():
        data_str = f'    ":ref:`{row["Pulsar"]}`", "{row["Model"]}", "{row["Max freq (MHz)"]}", '
        for val, error in [
            ("S5000 (mJy)", "u_S5000 (mJy)"),
            ("S10000 (mJy)", "u_S10000 (mJy)"),
        ]:
            data_str += f'"{row[val]:.2f}±{row[error]:.2f}", '
        file.write(f"{data_str[:-2]}\n")
        print(
            f"{row['Pulsar']} & {row['Model']} & {int(row['Max freq (MHz)'])} & {row['S5000 (mJy)']:.2f} $\\pm$ {row['u_S5000 (mJy)']:.2f} & {row['S10000 (mJy)']:.2f} $\\pm$ {row['S10000 (mJy)']:.2f} \\\\"
        )


# Make some summary histograms
# -----------------------------------------------------------------------------


def make_histogram_plots(all_data, hist_range, label, titles, plotname, xlabel):
    # Make histogram plots
    n_bins = 20
    colours = [
        "blue",
        "green",
        "orange",
        "purple",
    ]
    characters = ["a)", "b)", "c)", "d)", "e)"]
    n_data = len(all_data) + 1

    fig, axes = plt.subplots(nrows=n_data, figsize=(5, 3 * n_data))

    axes[0].hist(
        all_data,
        n_bins,
        density=True,
        histtype="bar",
        stacked=True,
        label=label,
        color=colours[: n_data - 1],
    )
    axes[0].text(0.1, 0.8, characters[0], transform=axes[0].transAxes, size=20)
    axes[0].set_title(titles[0])
    axes[0].legend(prop={"size": 10})
    axes[0].set_ylabel("Probability Density")
    axes[0].set_xlabel(f"${xlabel}$")

    for ai, df_col, colour, title in zip(
        range(1, n_data), all_data, colours, titles[1:]
    ):
        # print(ai, n_data)
        axes[ai].hist(df_col, n_bins, histtype="bar", color=colour, range=hist_range)
        axes[ai].text(0.1, 0.8, characters[ai], transform=axes[ai].transAxes, size=20)
        axes[ai].set_title(title)
        axes[ai].set_ylabel("#")
        axes[ai].set_xlabel(f"${xlabel}$")

    fig.tight_layout()
    fig.savefig(plotname)
    plt.close(fig)


# Alpha histogram
all_indexs = [
    spl_df["a"],
    hfto_df["a"],
    lfto_df["a"],
    dtos_df["a"],
]
hist_range = (df["a"].min(), df["a"].max())
titles = [
    "All models",
    "Simple power law",
    "High-frequency cut-off",
    "Low-frequency turn-over",
    "Double turn-over spectrum",
]
make_histogram_plots(
    all_indexs,
    hist_range,
    label=["SPL", "HFCO", "LFTO", "DTOS"],
    titles=titles,
    plotname=f"{docs_dir}/histograms/spectral_index_histogram.png",
    xlabel="\\alpha",
)


# Vc histogram
hist_range = (np.log10(df["vc"].min()), np.log10(df["vc"].max()))
all_indexs = [
    np.log10(hfto_df["vc"]),
    np.log10(dtos_df["vc"]),
]
titles = [
    "All models",
    "High-frequency cut-off",
    "Double turn-over spectrum",
]
make_histogram_plots(
    all_indexs,
    hist_range,
    label=["HFCO", "DTOS"],
    titles=titles,
    plotname=f"{docs_dir}/histograms/vc_histogram.png",
    xlabel="\\nu_{\\mathrm{c}}",
)

# Vpeak histogram
hist_range = (np.log10(df["vpeak"].min()), np.log10(df["vpeak"].max()))
all_indexs = [
    np.log10(lfto_df["vpeak"]),
    np.log10(dtos_df["vpeak"]),
]
titles = [
    "All models",
    "Low-frequency turn-over",
    "Double turn-over spectrum",
]
make_histogram_plots(
    all_indexs,
    hist_range,
    label=["LFTO", "DTOS"],
    titles=titles,
    plotname=f"{docs_dir}/histograms/vpeak_histogram.png",
    xlabel="\\nu_{\\mathrm{peak}}",
)

# MSP vs slow pulsar histograms
hist_range = (np_spl_df["a"].min(), np_spl_df["a"].max())
all_indexs = [
    msp_spl_df["a"],
    np_spl_df["a"],
]
titles = [
    "MSPs and slow pulsars",
    "MSPs",
    "Slow pulsars",
]
make_histogram_plots(
    all_indexs,
    hist_range,
    label=["MSP", "Slow pulsar"],
    titles=titles,
    plotname=f"{docs_dir}/histograms/msp_spectral_index_histogram.png",
    xlabel="\\alpha",
)


log_df = df  # [df["beta"] < 2.05]
# convert to log data
for col_name in log_df.keys():
    if col_name == "ATNF Fdot":
        log_df[col_name] = np.abs(log_df[col_name])
        log_df[col_name + " log"] = np.log10(log_df[col_name])
    elif "ATNF" in col_name or col_name in (
        "L400 (mJy kpc^2)",
        "L1400 (mJy kpc^2)",
        "Age (Yr)",
    ):
        log_df[col_name + " log"] = np.log10(log_df[col_name])
        # del log_df[col_name]
    elif col_name.startswith("v"):
        log_df[col_name + " log"] = np.log10(log_df[col_name])
        log_df["u_" + col_name + " log"] = log_df["u_" + col_name] / log_df[col_name]
        # del log_df[col_name]

log_df_sync = log_df[log_df["beta"] < 2.05]

# Calculate correlation coefficients and make plots
# -----------------------------------------------------------------------------


def spearmanr_ci(x, y, alpha=0.05):
    """Calculate spearmanr correlation along with the confidence interval using scipy and numpy
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
    """

    r, p = stats.spearmanr(x, y, nan_policy="omit")
    r_z = np.arctanh(r)
    se = 1 / np.sqrt(x.size - 3)
    z = stats.norm.ppf(1 - alpha / 2)
    lo_z, hi_z = r_z - z * se, r_z + z * se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi


def plot_correlations(
    this_df,
    xcol,
    ycol,
    raw_xcol,
    raw_ycol,
    lax=None,
    label=None,
):
    # Set up data
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
    # Extract data and split into MSP and Slow pulsars
    for xi, yi, yerri, raw_xi, raw_yi, raw_yerri, period, min_freq in zip(
        list(this_df[xcol]),
        list(this_df[ycol]),
        list(this_df["u_" + ycol]),
        list(this_df[raw_xcol]),
        list(this_df[raw_ycol]),
        list(this_df["u_" + raw_ycol]),
        list(this_df["ATNF Period (s)"]),
        list(this_df["Min freq (MHz)"]),
    ):
        if (
            (not np.isnan(xi))
            and (not np.isnan(yi))
            and (not np.isnan(yerri))
            and (not np.isinf(xi))
        ):
            x.append(xi)
            y.append(yi)
            yerr.append(yerri)
            raw_x.append(raw_xi)
            raw_y.append(raw_yi)
            raw_yerr.append(raw_yerri)
            # MSP check
            if period < msp_cutoff:
                # MSP
                msp_x.append(xi)
                msp_y.append(yi)
                msp_yerr.append(yerri)
                msp_raw_x.append(raw_xi)
                msp_raw_y.append(raw_yi)
                msp_raw_yerr.append(raw_yerri)
            else:
                # Normal (slow)
                np_x.append(xi)
                np_y.append(yi)
                np_yerr.append(yerri)
                np_raw_x.append(raw_xi)
                np_raw_y.append(raw_yi)
                np_raw_yerr.append(raw_yerri)
        elif ycol == "vpeak log":
            vpeak_x.append(raw_xi)
            vpeak_y.append(min_freq * 1e6)
            vpeak_yerr.append(min_freq * 1e6 * 0.1)

    # Convert everything to nupmy arrays
    x = np.array(x)
    y = np.array(y)
    yerr = np.array(yerr)
    raw_x = np.array(raw_x)
    raw_y = np.array(raw_y)
    raw_yerr = np.array(raw_yerr)
    msp_x = np.array(msp_raw_x)
    msp_y = np.array(msp_raw_y)
    msp_yerr = np.array(msp_raw_yerr)
    np_x = np.array(np_raw_x)
    np_y = np.array(np_raw_y)
    np_yerr = np.array(np_raw_yerr)
    msp_raw_x = np.array(msp_raw_x)
    msp_raw_y = np.array(msp_raw_y)
    msp_raw_yerr = np.array(msp_raw_yerr)
    np_raw_x = np.array(np_raw_x)
    np_raw_y = np.array(np_raw_y)
    np_raw_yerr = np.array(np_raw_yerr)

    # Convert to common units
    if "v" in ycol:
        # Convert to GHz
        raw_y /= 10**9
        raw_yerr /= 10**9
        msp_y /= 10**9
        msp_yerr /= 10**9
        np_y /= 10**9
        np_yerr /= 10**9
        msp_raw_y /= 10**9
        msp_raw_yerr /= 10**9
        np_raw_y /= 10**9
        np_raw_yerr /= 10**9
    if "Age" in xcol:
        # Convert to Myr
        raw_x /= 10**6
        msp_x /= 10**6
        np_x /= 10**6
        msp_raw_x /= 10**6
        np_raw_x /= 10**6

    # Calculate spearman correlation coefficient
    rho, pval, lo, hi = spearmanr_ci(x, y)
    if abs(rho) >= 0.4 and pval < 0.01 / 105:
        weights_str = f"{{\\bf {rho:5.2f}}} ({pval:.1e}, {len(x):3d})"
    else:
        weights_str = f"{rho:5.2f} ({pval:.1e}, {len(x):3d})"

    if lax is not None:
        f, ax = plt.subplots()

        capsize = 1.5
        errorbar_linewidth = 0.7
        marker_border_thickness = 0.5
        markersize = 3.5
        # Add normal pulsar data
        colour = "b"
        ecolour = "gray"
        alpha = 0.6
        (_, caps, _) = ax.errorbar(
            np_raw_x,
            np_raw_y,
            np.array(np_raw_yerr) / 2.0,
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

        (_, caps, _) = lax.errorbar(
            np_raw_x,
            np_raw_y,
            np.array(np_raw_yerr) / 2.0,
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
            (_, caps, _) = ax.errorbar(
                msp_raw_x,
                msp_raw_y,
                np.array(msp_raw_yerr) / 2.0,
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

            (_, caps, _) = lax.errorbar(
                msp_raw_x,
                msp_raw_y,
                np.array(msp_raw_yerr) / 2.0,
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

        if ycol == "vc log" and xcol == "ATNF Spin Frequency (Hz) log":
            # Display a theory line from https://ui.adsabs.harvard.edu/abs/2013Ap%26SS.345..169K/abstract
            fit_x = np.logspace(np.log10(min(raw_x)), np.log10(max(raw_x)))
            fit_y = 1.4 * 10**9 * (fit_x) ** 0.46 / 10**9
            fit_y_lu = 1.4 * 10**9 * (fit_x) ** 0.28 / 10**9
            fit_y_hu = 1.4 * 10**9 * (fit_x) ** 0.64 / 10**9
            ax.plot(fit_x, fit_y, label="theory", color="purple")
            lax.plot(fit_x, fit_y, label="theory", color="purple")
            ax.fill_between(fit_x, fit_y_lu, fit_y_hu, facecolor="purple", alpha=alpha)
            lax.fill_between(fit_x, fit_y_lu, fit_y_hu, facecolor="purple", alpha=alpha)

        # Labels
        ax.set_xlabel(f"{math_names_x[xcol]}")
        lax.set_xlabel(f"{math_names_x[xcol]}")
        ax.set_ylabel(f"{math_names_y[ycol]}")
        lax.set_ylabel(f"{math_names_y[ycol]}")
        ax.legend(title=f"Corr=${rho:.2f}_{{-{rho - lo:.2f}}}^{{+{hi - rho:.2f}}}$")
        lax.legend(title=f"Corr=${rho:.2f}_{{-{rho - lo:.2f}}}^{{+{hi - rho:.2f}}}$")
        if "log" in xcol:
            ax.set_xscale("log")
            lax.set_xscale("log")
            ax.xaxis.set_major_formatter(
                FuncFormatter(
                    lambda x, p:
                    # '$\mathdefault{10^{%i}}$' % x
                    # if x > 1e5 else
                    "%g" % x
                )
            )
            lax.xaxis.set_major_formatter(
                FuncFormatter(
                    lambda x, p: "$\mathdefault{10^{%i}}$" % np.log10(x)
                    if x > 1e5
                    else "%g" % x
                )
            )
        if "log" in ycol:
            ax.set_yscale("log")
            lax.set_yscale("log")
            ax.yaxis.set_major_formatter(
                FuncFormatter(
                    lambda x, p: "$\mathdefault{10^{%i}}$" % x if x > 1e5 else "%g" % x
                )
            )
            lax.yaxis.set_major_formatter(
                FuncFormatter(
                    lambda x, p: "$\mathdefault{10^{%i}}$" % np.log10(x)
                    if x > 1e5
                    else "%g" % x
                )
            )
        ax.tick_params(which="both", direction="in", top=1, right=1)
        lax.tick_params(which="both", direction="in", top=1, right=1)
        f.tight_layout()
        f.savefig(
            f"{docs_dir}/correlations/corr_line_{ycol}_{xcol.replace('/', '_')}_{label}.png".replace(
                " ", "_"
            ),
            dpi=300,
        )
        plt.close(f)
    return weights_str


# A quick period vs fdot plot

np_df = log_df[log_df["ATNF Period (s)"] >= msp_cutoff]
f, ax = plt.subplots()
print(np_df["ATNF Period (s) log"])
print(np_df["ATNF Fdot log"])
ax.scatter(
    list(np_df["ATNF Period (s)"]),
    list(np_df["ATNF Fdot"]),
    s=markersize,
    c=colour,
    alpha=alpha,
)
ax.set_xlabel("P (s)")
ax.set_ylabel("$\left| \dot{\\tilde{\\nu}} \\right|$ (s$^{-2}$)")
ax.set_yscale("log")
ax.set_xscale("log")
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda x, p: "$\mathdefault{10^{%i}}$" % x if x > 1e5 else "%g" % x)
)
ax.tick_params(which="both", direction="in", top=1, right=1)
f.tight_layout()
f.savefig("nudot_p_check.png".replace(" ", "_"), dpi=300)
plt.close(f)


#  Some dm covariance plot checks
lfto_np_df = lfto_df[lfto_df["ATNF Period (s)"] >= msp_cutoff]
f, ax = plt.subplots()
geo_freq = []
for pulsar in lfto_np_df["Pulsar"]:
    freqs, bands, fluxs, flux_errs, refs = cat_list[pulsar]
    v0_Hz = 10 ** ((np.log10(min(freqs)) + np.log10(max(freqs))) / 2) * 1e6
    geo_freq.append(v0_Hz)
ax.scatter(
    geo_freq,
    list(lfto_np_df["ATNF DM"]),
    s=markersize,
    c=colour,
    alpha=alpha,
)
ax.set_xlabel("$\\nu_O$ (Hz)")
ax.set_ylabel("DM (pc cm$^{-3}$)")
ax.set_yscale("log")
ax.set_xscale("log")
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda x, p: "$\mathdefault{10^{%i}}$" % x if x > 1e5 else "%g" % x)
)
ax.tick_params(which="both", direction="in", top=1, right=1)
f.tight_layout()
f.savefig("v0_DM_check.png".replace(" ", "_"), dpi=300)
plt.close(f)

f, ax = plt.subplots()
ax.scatter(
    geo_freq,
    list(lfto_np_df["vpeak"]),
    s=markersize,
    c=colour,
    alpha=alpha,
)
ax.set_xlabel("$\\nu_O$ (Hz)")
ax.set_ylabel("$\\nu_{\\mathrm{peak}}$ (Hz)")
ax.set_yscale("log")
ax.set_xscale("log")
ax.tick_params(which="both", direction="in", top=1, right=1)
f.tight_layout()
f.savefig("v0_vpeak_check.png".replace(" ", "_"), dpi=300)
plt.close(f)

f, ax = plt.subplots()
markersize = 10
freq_min = []
for pulsar in lfto_np_df["Pulsar"]:
    freqs, bands, fluxs, flux_errs, refs = cat_list[pulsar]
    v0_Hz = min(freqs) * 1e6
    freq_min.append(v0_Hz)
ax.scatter(
    list(lfto_np_df["ATNF DM"]),
    freq_min,
    s=markersize,
    c=colour,
    alpha=alpha,
    label="LFTO",
)
spl_np_df = spl_df[spl_df["ATNF Period (s)"] >= msp_cutoff]
freq_min = []
for pulsar in spl_np_df["Pulsar"]:
    freqs, bands, fluxs, flux_errs, refs = cat_list[pulsar]
    v0_Hz = min(freqs) * 1e6
    freq_min.append(v0_Hz)
ax.scatter(
    list(spl_np_df["ATNF DM"]), freq_min, s=markersize, c="g", alpha=alpha, label="SPL"
)
ax.set_xlabel("DM (pc cm$^{-3}$)")
ax.set_ylabel("$\\nu_{\\mathrm{min}}$ (Hz)")
ax.set_yscale("log")
ax.set_xscale("log")
ax.tick_params(which="both", direction="in", top=1, right=1)
ax.legend()
f.tight_layout()
f.savefig("vmin_DM_check.png".replace(" ", "_"), dpi=300)
plt.close(f)


fig = plt.figure()
ax = fig.add_subplot(projection="3d")
freq_min = []
for pulsar in lfto_np_df["Pulsar"]:
    freqs, bands, fluxs, flux_errs, refs = cat_list[pulsar]
    v0_Hz = min(freqs) * 1e6
    freq_min.append(v0_Hz)
ax.scatter(
    list(lfto_np_df["ATNF DM"]),
    freq_min,
    list(lfto_np_df["vpeak"]),
    s=markersize,
    c=colour,
    alpha=alpha,
    label="LFTO",
)
ax.set_xlabel("DM (pc cm$^{-3}$)")
ax.set_ylabel("$\\nu_{\\mathrm{min}}$ (Hz)")
ax.set_zlabel("$\\nu_{\\mathrm{peak}}$ (Hz)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_zscale("log")
ax.tick_params(which="both", direction="in", top=1, right=1)
ax.legend()
f.tight_layout()
f.savefig("vmin_DM_check.png".replace(" ", "_"), dpi=300)
# f.show()
plt.close(f)


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
    "a",
    # "c"         ,
    # "vb"        ,
    # "a1"        ,
    # "a2"        ,
    "vpeak log",
    "vc log",
    # "beta"      ,
]

math_names_x = {
    # "ATNF Period (s) log": "P (s)",
    "ATNF Spin Frequency (Hz) log": "$\\tilde{\\nu}$ (GHz)",
    "ATNF Pdot log": "$\dot{P}$ (s s$^{-1}$)",
    "ATNF Fdot log": "$\left| \dot{\\tilde{\\nu}} \\right|$ (s$^{-2}$)",
    "ATNF DM log": "DM (pc cm$^{-3}$)",
    # "ATNF B_surf (G) log": "$B_{surf}$ (G)",
    "ATNF B_LC (G) log": "$B_{LC}$ (G)",
    "ATNF E_dot (ergs/s) log": "$\dot{E}$ (ergs/s)",
    "L400 (mJy kpc^2) log": "$L_{400}$ (mJy kpc$^2$)",
    "L1400 (mJy kpc^2) log": "$L_{1400}$ (mJy kpc$^2$)",
    "Age (Yr) log": "$\\tau$ (MYr)",
}
math_names_y = {
    "a": "$\\alpha$",
    "vpeak log": "$\\nu_{\\mathrm{peak}}$ (GHz)",
    "vc log": "$\\nu_{\\mathrm{c}}$ (GHz)",
}

correlation_tables = {
    "a": [],
    "vpeak log": [],
    "vc log": [],
}

nx = len(xcols)
ny = len(ycols)
# Make figures and axis for each pulsar type
sfig, saxes = plt.subplots(ny, nx, figsize=(5 * nx, 5 * ny))
lfig, laxes = plt.subplots(ny, nx, figsize=(5 * nx, 5 * ny))
bfig, baxes = plt.subplots(ny, nx, figsize=(5 * nx, 5 * ny))
ifig, iaxes = plt.subplots(ny, nx, figsize=(5 * nx, 5 * ny))
mfig, maxes = plt.subplots(ny, nx, figsize=(5 * nx, 5 * ny))
sfig, saxes = plt.subplots(ny, nx, figsize=(5 * nx, 5 * ny))

# Loop over fit parameters
for ya, ycol in enumerate(ycols):
    weights = []
    ycol_df = log_df[log_df[ycol].notnull()]
    if ycol == "a":
        ycol_df = ycol_df[ycol_df["Model"] == "simple_power_law"]
    nbinary = len(ycol_df[ycol_df["ANTF Binary (type)"].notnull()])
    nisolated = len(ycol_df[ycol_df["ANTF Binary (type)"].isnull()])
    nmsp = len(ycol_df[ycol_df["ATNF Period (s)"] < msp_cutoff])
    nslow = len(ycol_df[ycol_df["ATNF Period (s)"] >= msp_cutoff])

    # Output latex correlation table
    print("")
    print("\\\\")
    print(
        f"\multicolumn{{6}}{{c}}{{ {math_names_y[ycol]} Correlation Coefficient }} \\\\"
    )
    print("\hline")
    print("set & all & in binary & isolated & MSP & slow \\\\")
    print(
        f"\#pulsars & {len(ycol_df)} & {nbinary} & {nisolated} & {nmsp} & {nslow}\\\\"
    )
    print("\hline")
    print(
        "$log_{10}(x)$ & $r_s (p, N)$ & $r_s (p, N)$ & $r_s (p, N)$ & $r_s (p, N)$ & $r_s (p, N)$ \\\\"
    )
    # print("$log_{10}(x)$ & $r_s$ & $r_s$ & $r_s$ & $r_s$ & $r_s$ \\\\")
    print("\hline")
    # Record table for sphinx output
    correlation_tables[ycol].append(
        "+------------------------------------------+--------------------------+--------------------------+--------------------------+--------------------------+--------------------------+"
    )
    correlation_tables[ycol].append(
        "|                                      set |                      all |                in binary |                 isolated |                      MSP |                     slow |"
    )
    correlation_tables[ycol].append(
        "+------------------------------------------+--------------------------+--------------------------+--------------------------+--------------------------+--------------------------+"
    )
    correlation_tables[ycol].append(
        f"|                                 #pulsars |                      {len(ycol_df):3d} |                      {nbinary:3d} |                      {nisolated:3d} |                      {nmsp:3d} |                      {nslow:3d} |"
    )
    correlation_tables[ycol].append(
        "+------------------------------------------+--------------------------+--------------------------+--------------------------+--------------------------+--------------------------+"
    )
    correlation_tables[ycol].append(
        "|                :math:`{\\bf log_{10}(x)}` | :math:`{\\bf r_s (p, N)}` | :math:`{\\bf r_s (p, N)}` | :math:`{\\bf r_s (p, N)}` | :math:`{\\bf r_s (p, N)}` | :math:`{\\bf r_s (p, N)}` |"
    )
    correlation_tables[ycol].append(
        "+==========================================+==========================+==========================+==========================+==========================+==========================+"
    )

    # Loop over pulsar paramters
    for xa, xcol in enumerate(xcols):
        # Filter out some fits
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
            this_df = this_df[this_df["Pulsar"] != "J1917+1353"]
        # Filter out low luminosity
        if xcol == "L400 (mJy kpc^2) log":
            this_df = this_df[this_df["L400 (mJy kpc^2)"] > 3]
        if xcol == "L1400 (mJy kpc^2) log":
            this_df = this_df[this_df["L1400 (mJy kpc^2)"] > 0.1]

        # Remove log from col
        if "log" in xcol:
            raw_xcol = xcol[:-4]
        else:
            raw_xcol = xcol
        if "log" in ycol:
            raw_ycol = ycol[:-4]
        else:
            raw_ycol = ycol

        # Loop over each pulsar types
        sub_weights = []
        binary_df = this_df[this_df["ANTF Binary (type)"].notnull()]
        isolated_df = this_df[this_df["ANTF Binary (type)"].isnull()]
        msp_df = this_df[this_df["ATNF Period (s)"] < msp_cutoff]
        slow_df = this_df[this_df["ATNF Period (s)"] >= msp_cutoff]
        df_axes_pairs = [
            (this_df, laxes, "All Pulsars"),
            (binary_df, baxes, "Only Binary Pulsars"),
            (isolated_df, iaxes, "Only Isolated Pulsars"),
            (msp_df, maxes, "Only MSPs"),
            (slow_df, saxes, "Only Slow Pulsars"),
        ]
        plt.rcParams.update({"font.size": 18})
        for sub_df, sub_axes, label in df_axes_pairs:
            weight_str = weighted_line_corr = plot_correlations(
                sub_df,
                xcol,
                ycol,
                raw_xcol,
                raw_ycol,
                lax=sub_axes[ya][xa],
                label=label,
            )
            sub_weights.append(weight_str)

        # Output latex results
        print(f"{math_names_x[xcol].split('(')[0]} & {' & '.join(sub_weights)} \\\\")
        # Replace latex bold with sphinx bold
        sphinx_weights = []
        for weight in sub_weights:
            if "bf" in weight:
                new_weight = weight.replace("{\\bf ", "").replace("}", "")
                if new_weight.startswith(" "):
                    sphinx_weights.append(f" **{new_weight[1:]}**")
                else:
                    sphinx_weights.append(f"**{new_weight}**")
            else:
                sphinx_weights.append(f"  {weight}  ")
        mathed_name = f":math:`{math_names_x[xcol].split(' (')[0].replace('$', '')}`"
        correlation_tables[ycol].append(
            f"| {' ' * (40 - len(mathed_name))}{mathed_name} | {' | '.join(sphinx_weights)} |"
        )
        correlation_tables[ycol].append(
            "+------------------------------------------+--------------------------+--------------------------+--------------------------+--------------------------+--------------------------+"
        )
    print("\hline")
    # correlation_tables[ycol].append(f'+------------------------------------------+--------------------------+--------------------------+--------------------------+--------------------------+--------------------------+')
    print("\n".join(correlation_tables[ycol]))

# Save plots
sfig.tight_layout()
sfig.savefig(f"{docs_dir}/correlations/coor_all.png", dpi=300)
lfig.tight_layout()
lfig.savefig(f"{docs_dir}/correlations/corr_all_weighted.png", dpi=300)
bfig.tight_layout()
bfig.savefig(f"{docs_dir}/correlations/corr_b_weighted.png", dpi=300)
ifig.tight_layout()
ifig.savefig(f"{docs_dir}/correlations/corr_i_weighted.png", dpi=300)
mfig.tight_layout()
mfig.savefig(f"{docs_dir}/correlations/corr_m_weighted.png", dpi=300)
sfig.tight_layout()
sfig.savefig(f"{docs_dir}/correlations/corr_s_weighted.png", dpi=300)

pulsar_types = [
    "All Pulsars",
    "Only Binary Pulsars",
    "Only Isolated Pulsars",
    "Only MSPs",
    "Only Slow Pulsars",
]

# Set up the spectral property summaries
# -----------------------------------------------------------------------------
# Spectral index
with open(
    f"{docs_dir}/spectral_index_summary.rst",
    "w",
) as file:
    file.write("""
Spectral Index Summary
======================

""")
    for line in correlation_tables["a"]:
        file.write(f"{line}\n")

    file.write(f'''

Spectral Index Mean Summary
---------------------------
.. csv-table::
    :header: "Model", "All Mean", "MSP Mean", "Normal Mean"

    "simple_power_law",                  "{spl_df["a"].mean():.2f}±{spl_df["a"].std():.2f}",   "{msp_spl_df["a"].mean():.2f}±{msp_spl_df["a"].std():.2f}",  "{np_spl_df["a"].mean():.2f}±{np_spl_df["a"].std():.2f}"
    "high_frequency_cut_off_power_law",  "{hfto_df["a"].mean():.2f}±{hfto_df["a"].std():.2f}", "{msp_hfto_df["a"].mean():.2f}±{msp_hfto_df["a"].std():.2f}", "{np_hfto_df["a"].mean():.2f}±{np_hfto_df["a"].std():.2f}"
    "low_frequency_turn_over_power_law", "{lfto_df["a"].mean():.2f}±{lfto_df["a"].std():.2f}", "{msp_lfto_df["a"].mean():.2f}±{msp_lfto_df["a"].std():.2f}", "{np_lfto_df["a"].mean():.2f}±{np_lfto_df["a"].std():.2f}"
    "double_turn_over_spectrum",         "{dtos_df["a"].mean():.2f}±{dtos_df["a"].std():.2f}", "{msp_dtos_df["a"].mean():.2f}±{msp_dtos_df["a"].std():.2f}", "{np_dtos_df["a"].mean():.2f}±{np_dtos_df["a"].std():.2f}"
    "Total",                             "{df["a"].mean():.2f}±{df["a"].std():.2f}",           "{msp_df["a"].mean():.2f}±{msp_df["a"].std():.2f}",      "{np_df["a"].mean():.2f}±{np_df["a"].std():.2f}"

Spectral Index Median Summary
-----------------------------
.. csv-table::
    :header: "Model", "All Median", "MSP Median", "Normal Median"

    "simple_power_law",                  "{spl_df["a"].median():.2f}±{spl_df["a"].std():.2f}",   "{msp_spl_df["a"].median():.2f}±{msp_spl_df["a"].std():.2f}",  "{np_spl_df["a"].median():.2f}±{np_spl_df["a"].std():.2f}"
    "high_frequency_cut_off_power_law",  "{hfto_df["a"].median():.2f}±{hfto_df["a"].std():.2f}", "{msp_hfto_df["a"].median():.2f}±{msp_hfto_df["a"].std():.2f}", "{np_hfto_df["a"].median():.2f}±{np_hfto_df["a"].std():.2f}"
    "low_frequency_turn_over_power_law", "{lfto_df["a"].median():.2f}±{lfto_df["a"].std():.2f}", "{msp_lfto_df["a"].median():.2f}±{msp_lfto_df["a"].std():.2f}", "{np_lfto_df["a"].median():.2f}±{np_lfto_df["a"].std():.2f}"
    "double_turn_over_spectrum",         "{dtos_df["a"].median():.2f}±{dtos_df["a"].std():.2f}", "{msp_dtos_df["a"].median():.2f}±{msp_dtos_df["a"].std():.2f}", "{np_dtos_df["a"].median():.2f}±{np_dtos_df["a"].std():.2f}"
    "Total",                             "{df["a"].median():.2f}±{df["a"].std():.2f}",           "{msp_df["a"].median():.2f}±{msp_df["a"].std():.2f}",      "{np_df["a"].median():.2f}±{np_df["a"].std():.2f}"

Spectral Index Histogram
------------------------

.. image:: histograms/spectral_index_histogram.png
    :width: 800

''')
    for pulsar_param in math_names_x.keys():
        file.write(f"""
:math:`{math_names_x[pulsar_param].replace("$", "").split(" (")[0]}` Correlations
{(len(math_names_x[pulsar_param].replace("$", "").split(" (")[0]) + 21) * "-"}

""")
        for pulsar_type in pulsar_types:
            file.write(f"""
{pulsar_type}
{len(pulsar_type) * "^"}

.. image:: correlations/corr_line_a_{pulsar_param.replace(" ", "_").replace("/", "_")}_{pulsar_type.replace(" ", "_")}.png
    :width: 800
""")

# vpeak
with open(
    f"{os.path.dirname(os.path.realpath(__file__))}/docs/vpeak_summary.rst", "w"
) as file:
    file.write("""
:math:`\\nu_{\mathrm{peak}}` Summary
====================================

""")
    for line in correlation_tables["vpeak log"]:
        file.write(f"{line}\n")

    file.write("""

:math:`\\nu_{\mathrm{peak}}` Histogram
--------------------------------------

.. image:: histograms/vpeak_histogram.png
    :width: 800

""")
    for pulsar_param in math_names_x.keys():
        file.write(f"""
:math:`{math_names_x[pulsar_param].replace("$", "").split(" (")[0]}` Correlations
{(len(math_names_x[pulsar_param].replace("$", "").split(" (")[0]) + 21) * "-"}

""")
        for pulsar_type in pulsar_types:
            file.write(f"""
{pulsar_type}
{len(pulsar_type) * "^"}

.. image:: correlations/corr_line_vpeak_log_{pulsar_param.replace(" ", "_").replace("/", "_")}_{pulsar_type.replace(" ", "_")}.png
    :width: 800
""")


# v_c
with open(f"{docs_dir}/vc_summary.rst", "w") as file:
    file.write("""
:math:`\\nu_{\mathrm{c}}` Summary
================================+

""")
    for line in correlation_tables["vc log"]:
        file.write(f"{line}\n")

    file.write("""


:math:`\\nu_{\mathrm{c}}` Histogram
-----------------------------------

.. image:: histograms/vc_histogram.png
    :width: 800

""")
    for pulsar_param in math_names_x.keys():
        file.write(f"""
:math:`{math_names_x[pulsar_param].replace("$", "").split(" (")[0]}` Correlations
{(len(math_names_x[pulsar_param].replace("$", "").split(" (")[0]) + 21) * "-"}

""")
        for pulsar_type in pulsar_types:
            file.write(f"""
{pulsar_type}
{len(pulsar_type) * "^"}

.. image:: correlations/corr_line_vc_log_{pulsar_param.replace(" ", "_").replace("/", "_")}_{pulsar_type.replace(" ", "_")}.png
    :width: 800
""")
