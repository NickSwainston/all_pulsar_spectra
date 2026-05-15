import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy import stats
import argparse
from pathlib import Path

import pulsar_spectra
from pulsar_spectra.catalogue import collect_catalogue_fluxes

from make_index_summary import make_index_summary
from make_pulsar_gallery import make_pulsar_gallery
from make_suggested_campaigns import make_suggested_campaigns
from make_misc_plots import make_misc_plots
from make_histograms import make_histograms
from make_correlations import make_correlations

cat_list = collect_catalogue_fluxes()




def main(output_dir):

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    docs_dir = f"{output_dir}/docs"
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(f"{docs_dir}/best_fits", exist_ok=True)
    os.makedirs(f"{docs_dir}/correlations", exist_ok=True)
    os.makedirs(f"{docs_dir}/histograms", exist_ok=True)

    msp_cutoff = 0.03

    make_index_summary(output_dir, msp_cutoff=msp_cutoff)
    make_pulsar_gallery(output_dir, msp_cutoff=msp_cutoff)
    make_suggested_campaigns(output_dir)
    make_misc_plots(output_dir, msp_cutoff=msp_cutoff)
    make_histograms(output_dir)
    make_correlations(output_dir, msp_cutoff=msp_cutoff)


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

    main(args.output_dir)
