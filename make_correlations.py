import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import argparse
from pathlib import Path
from matplotlib.ticker import FuncFormatter


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
    "vpeak_Hz log": "$\\nu_{\\mathrm{peak}}$ (Hz)",
    "vc_Hz log": "$\\nu_{\\mathrm{c}}$ (Hz)",
}

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
    docs_dir,
    lax=None,
    label=None,
    msp_cutoff=0.03,
):
    # Filter to rows with valid x, y, and y-error
    valid_mask = (
        this_df[xcol].notna()
        & this_df[ycol].notna()
        & this_df["u_" + ycol].notna()
        & ~np.isinf(this_df[xcol])
    )
    df_valid = this_df[valid_mask]

    x = df_valid[xcol].values
    y = df_valid[ycol].values
    raw_x = df_valid[raw_xcol].values

    # Split into MSP and slow pulsars
    msp_mask = df_valid["ATNF Period (s)"] < msp_cutoff
    msp_df = df_valid[msp_mask]
    np_df = df_valid[~msp_mask]

    msp_raw_x = msp_df[raw_xcol].values.copy()
    msp_raw_y = msp_df[raw_ycol].values.copy()
    msp_raw_yerr = msp_df["u_" + raw_ycol].values.copy()
    np_raw_x = np_df[raw_xcol].values.copy()
    np_raw_y = np_df[raw_ycol].values.copy()
    np_raw_yerr = np_df["u_" + raw_ycol].values.copy()

    if "Age" in xcol:
        # Convert to Myr
        for arr in (raw_x, msp_raw_x, np_raw_x):
            arr /= 10**6

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

        if ycol == "vc_Hz log" and xcol == "ATNF Spin Frequency (Hz) log":
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


def make_correlations(output_dir, msp_cutoff=0.03):
    # Calculate correlation coefficients and make plots
    # -----------------------------------------------------------------------------
    docs_dir = f"{output_dir}/docs"
    # Read in the fits
    df = pd.read_csv(f"{output_dir}/all_pulsar_fits_log.csv")

    df_sync = df[df["beta"] < 2.05]

    # Grab model specific data frames
    spl_df = df[df["Model"] == "simple_power_law"]
    hfto_df = df[df["Model"] == "high_frequency_cut_off_power_law"]
    lfto_df = df[df["Model"] == "low_frequency_turn_over_power_law"]
    dtos_df = df[df["Model"] == "double_turn_over_spectrum"]

    # MSPs
    msp_spl_df = spl_df[spl_df["ATNF Period (s)"] < msp_cutoff]
    msp_hfto_df = hfto_df[hfto_df["ATNF Period (s)"] < msp_cutoff]
    msp_lfto_df = lfto_df[lfto_df["ATNF Period (s)"] < msp_cutoff]
    msp_dtos_df = dtos_df[dtos_df["ATNF Period (s)"] < msp_cutoff]
    msp_df = df[df["ATNF Period (s)"] < msp_cutoff]

    # Normal pulsars
    np_spl_df = spl_df[spl_df["ATNF Period (s)"] >= msp_cutoff]
    np_hfto_df = hfto_df[hfto_df["ATNF Period (s)"] >= msp_cutoff]
    np_lfto_df = lfto_df[lfto_df["ATNF Period (s)"] >= msp_cutoff]
    np_dtos_df = dtos_df[dtos_df["ATNF Period (s)"] >= msp_cutoff]
    np_df = df[df["ATNF Period (s)"] >= msp_cutoff]

    # Make individual correlation plots
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
        "vpeak_Hz log",
        "vc_Hz log",
        # "beta"      ,
    ]


    correlation_tables = {
        "a": [],
        "vpeak_Hz log": [],
        "vc_Hz log": [],
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
        ycol_df = df[df[ycol].notnull()]
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
        print("\\hline")
        print("set & all & in binary & isolated & MSP & slow \\\\")
        print(
            f"\#pulsars & {len(ycol_df)} & {nbinary} & {nisolated} & {nmsp} & {nslow}\\\\"
        )
        print("\\hline")
        print(
            "$log_{10}(x)$ & $r_s (p, N)$ & $r_s (p, N)$ & $r_s (p, N)$ & $r_s (p, N)$ & $r_s (p, N)$ \\\\"
        )
        # print("$log_{10}(x)$ & $r_s$ & $r_s$ & $r_s$ & $r_s$ & $r_s$ \\\\")
        print("\\hline")
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
                this_df = df_sync.copy()
            else:
                this_df = df.copy()
            if ycol == "a":
                # only use simple power law
                this_df = this_df[this_df["Model"] == "simple_power_law"]
                # Filter out outliers
                this_df = this_df[this_df["a"] < 0]
                this_df = this_df[this_df["a"] > -4]
            if ycol == "vc_Hz log":
                # only use high frequency cut off
                this_df = this_df[this_df["Model"].isin([
                    "high_frequency_cut_off_power_law",
                    "double_turn_over_spectrum",
                ])]
            if ycol == "vpeak_Hz log":
                # only use models that fit a spectral peak
                this_df = this_df[this_df["Model"].isin([
                    "low_frequency_turn_over_power_law",
                    "double_turn_over_spectrum",
                ])]
            # Filter out low luminosity
            if xcol == "L400 (mJy kpc^2) log":
                this_df = this_df[this_df["L400 (mJy kpc^2)"] > 1]
            if xcol == "L1400 (mJy kpc^2) log":
                this_df = this_df[this_df["L1400 (mJy kpc^2)"] > 0.1]

            raw_ycol = ycol.replace(" log", "")
            raw_xcol = xcol.replace(" log", "")

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
                print(f"Plotting {label} {ycol=}/{raw_ycol=} vs {xcol=}/{raw_xcol=} with {len(sub_df)} pulsars")
                print(f"{min(sub_df[raw_xcol])=} {max(sub_df[raw_xcol])=}")
                print(f"{min(sub_df[raw_ycol])=} {max(sub_df[raw_ycol])=}")
                weight_str = plot_correlations(
                    sub_df,
                    xcol,
                    ycol,
                    raw_xcol,
                    raw_ycol,
                    docs_dir,
                    lax=sub_axes[ya][xa],
                    label=label,
                    msp_cutoff=msp_cutoff,
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
        print("\\hline")
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
    with open(f"{docs_dir}/spectral_index_summary.rst", "w", encoding="utf-8") as file:
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
        f"{docs_dir}/vpeak_summary.rst", "w", encoding="utf-8"
    ) as file:
        file.write("""
:math:`\\nu_{\mathrm{peak}}` Summary
====================================

""")
        for line in correlation_tables["vpeak_Hz log"]:
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
=================================

""")
        for line in correlation_tables["vc_Hz log"]:
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
    make_correlations(args.output_dir)