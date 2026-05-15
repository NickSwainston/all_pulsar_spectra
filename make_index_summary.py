import pandas as pd
import os
import numpy as np
import argparse
from pathlib import Path

import pulsar_spectra


def make_index_summary(output_dir, msp_cutoff=0.03):
    docs_dir = f"{output_dir}/docs"

    # Read in the fits
    df = pd.read_csv(f"{output_dir}/all_pulsar_fits.csv")

    # Adjust for all units
    # c_mJy,u_c_mJy,
    df["c_Jy"] = df["c_mJy"] / 1e3
    df["u_c_Jy"] = df["u_c_mJy"] / 1e3
    # vb_MHz,u_vb_MHz,
    df["vb_Hz"] = df["vb_MHz"] * 1e6
    df["u_vb_Hz"] = df["u_vb_MHz"] * 1e6
    df["vb_GHz"] = df["vb_MHz"] / 1e3
    df["u_vb_GHz"] = df["u_vb_MHz"] / 1e3
    # vc_MHz,u_vc_MHz,
    df["vc_Hz"] = df["vc_MHz"] * 1e6
    df["u_vc_Hz"] = df["u_vc_MHz"] * 1e6
    df["vc_GHz"] = df["vc_MHz"] / 1e3
    df["u_vc_GHz"] = df["u_vc_MHz"] / 1e3
    # vpeak_MHz,u_vpeak_MHz,
    df["vpeak_Hz"] = df["vpeak_MHz"] * 1e6
    df["u_vpeak_Hz"] = df["u_vpeak_MHz"] * 1e6
    df["vpeak_GHz"] = df["vpeak_MHz"] / 1e3
    df["u_vpeak_GHz"] = df["u_vpeak_MHz"] / 1e3

    # convert to log data
    for col_name in df.keys():
        if col_name == "ATNF Fdot":
            df[col_name] = np.abs(df[col_name])
            df[col_name + " log"] = np.log10(df[col_name])
        elif "ATNF" in col_name or col_name in (
            "Age (Yr)",
        ):
            df[col_name + " log"] = np.log10(df[col_name])
        elif col_name in (
            "L400 (mJy kpc^2)",
            "L1400 (mJy kpc^2)",
        ):
            df[col_name + " log"] = np.log10(df[col_name])
        elif col_name.startswith("v") and col_name.endswith("_Hz"):
            df[col_name + " log"] = np.log10(df[col_name])
            df["u_" + col_name + " log"] = df["u_" + col_name] / df[col_name]

    # dump for other functions to use
    df.to_csv(f"{output_dir}/all_pulsar_fits_log.csv", index=False)

    # Grab model specific data frames
    spl_df = df[df["Model"] == "simple_power_law"]
    bpl_df = df[df["Model"] == "broken_power_law"]
    hfto_df = df[df["Model"] == "high_frequency_cut_off_power_law"]
    lfto_df = df[df["Model"] == "low_frequency_turn_over_power_law"]
    dtos_df = df[df["Model"] == "double_turn_over_spectrum"]

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
    with open(f"{docs_dir}/index.rst", "w", encoding="utf-8") as file:
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
    "Peak Frequency (GHz)",    "{df["vpeak_GHz"].mean():.2f}±{df["vpeak_GHz"].std():.2f}", "{msp_df["vpeak_GHz"].mean():.2f}±{msp_df["vpeak_GHz"].std():.2f}", "{np_df["vpeak_GHz"].mean():.2f}±{np_df["vpeak_GHz"].std():.2f}"
    "Cut off frequency (GHz)", "{df["vc_GHz"].mean():.2f}±{df["vc_GHz"].std():.2f}",       "{msp_df["vc_GHz"].mean():.2f}±{msp_df["vc_GHz"].std():.2f}",       "{np_df["vc_GHz"].mean():.2f}±{np_df["vc_GHz"].std():.2f}"
    "Beta",                    "{df["beta"].mean():.2f}±{df["beta"].std():.2f}",           "{msp_df["beta"].mean():.2f}±{msp_df["beta"].std():.2f}",           "{np_df["beta"].mean():.2f}±{np_df["beta"].std():.2f}"

Single Power Law Results
------------------------
.. csv-table::
    :header: "Pulsar", "a"

''')
        for _, row in spl_df.iterrows():
            data_str = f'    ":ref:`{row["Pulsar"]}`", '
            for val, error in [("a", "u_a")]:
                if "v" in val:
                    data_str += f'"{int(row[val]):d}±{int(row[error]):d}", '
                else:
                    data_str += f'"{row[val]:.2f}±{row[error]:.2f}", '
            file.write(f"{data_str[:-2]}\n")

        file.write("""


Broken Power Law Results
------------------------
.. csv-table::
    :header: "Pulsar", "vb (MHz)", "a1", "a2"

""")
        for _, row in bpl_df.iterrows():
            data_str = f'    ":ref:`{row["Pulsar"]}`", '
            for val, error in [("vb_MHz", "u_vb_MHz"), ("a1", "u_a1"), ("a2", "u_a2")]:
                if "v" in val:
                    data_str += f'"{int(row[val]):d}±{int(row[error]):d}", '
                else:
                    data_str += f'"{row[val]:.2f}±{row[error]:.2f}", '
            file.write(f"{data_str[:-2]}\n")

        file.write("""


Low Frequency Turn Over Results
-------------------------------
.. csv-table::
    :header: "Pulsar", "vpeak (MHz)", "a", "beta"

""")
        for _, row in lfto_df.iterrows():
            data_str = f'    ":ref:`{row["Pulsar"]}`", '
            for val, error in [("vpeak_MHz", "u_vpeak_MHz"), ("a", "u_a"), ("beta", "u_beta")]:
                if "v" in val:
                    data_str += f'"{int(row[val]):d}±{int(row[error]):d}", '
                else:
                    data_str += f'"{row[val]:.2f}±{row[error]:.2f}", '
            file.write(f"{data_str[:-2]}\n")

        file.write("""


High Frequency Cut Off Results
------------------------------
.. csv-table::
    :header: "Pulsar", "vc (MHz)", "a"

""")
        for _, row in hfto_df.iterrows():
            data_str = f'    ":ref:`{row["Pulsar"]}`", '
            for val, error in [("vc_MHz", "u_vc_MHz"), ("a", "u_a")]:
                if "v" in val:
                    data_str += f'"{int(row[val]):d}±{int(row[error]):d}", '
                else:
                    data_str += f'"{row[val]:.2f}±{row[error]:.2f}", '
            file.write(f"{data_str[:-2]}\n")

        file.write("""


Double Turn Over Spectrum Results
---------------------------------
.. csv-table::
    :header: "Pulsar", "vc (MHz)", "vpeak (MHz)", "a", "beta"

""")
        for _, row in dtos_df.iterrows():
            data_str = f'    ":ref:`{row["Pulsar"]}`", '
            for val, error in [
                ("vc_MHz", "u_vc_MHz"),
                ("vpeak_MHz", "u_vpeak_MHz"),
                ("a", "u_a"),
                ("beta", "u_beta"),
            ]:
                if "v" in val:
                    data_str += f'"{int(row[val]):d}±{int(row[error]):d}", '
                else:
                    data_str += f'"{row[val]:.2f}±{row[error]:.2f}", '
            file.write(f"{data_str[:-2]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process pulsars and optionally specify output directory."
    )
    parser.add_argument(
        "-o", "--output_dir",
        default=os.path.dirname(os.path.realpath(__file__)),
        help="Directory to save output (default: current directory)"
    )
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    if args.output_dir is None or args.output_dir == "None":
        args.output_dir = script_dir
    # If output_dir is relative, make it absolute relative to script_dir
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(script_dir, args.output_dir)
    make_index_summary(args.output_dir)