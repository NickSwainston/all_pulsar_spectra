import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


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

def make_histograms(output_dir):
    docs_dir = f"{output_dir}/docs"

    # Read in the fits
    df = pd.read_csv(f"{output_dir}/all_pulsar_fits_log.csv")

    # Grab model specific data frames
    spl_df = df[df["Model"] == "simple_power_law"]
    hfto_df = df[df["Model"] == "high_frequency_cut_off_power_law"]
    lfto_df = df[df["Model"] == "low_frequency_turn_over_power_law"]
    dtos_df = df[df["Model"] == "double_turn_over_spectrum"]

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
    hist_range = (df["vc_Hz log"].min(), df["vc_Hz log"].max())
    all_indexs = [
        hfto_df["vc_Hz log"],
        dtos_df["vc_Hz log"],
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
    hist_range = (df["vpeak_Hz log"].min(), df["vpeak_Hz log"].max())
    all_indexs = [
        lfto_df["vpeak_Hz log"],
        dtos_df["vpeak_Hz log"],
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
    hist_range = (df["a"].min(), df["a"].max())
    all_indexs = [
        spl_df["a"],
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
    make_histograms(args.output_dir)