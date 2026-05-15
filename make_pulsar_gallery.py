import pandas as pd
import os
import argparse
from pathlib import Path


def make_pulsar_gallery(output_dir, msp_cutoff=0.03):
    docs_dir = f"{output_dir}/docs"

    # Read in the fits
    df = pd.read_csv(f"{output_dir}/all_pulsar_fits_log.csv")

    # Grab model specific data frames
    spl_df = df[df["Model"] == "simple_power_law"]
    bpl_df = df[df["Model"] == "broken_power_law"]
    hfto_df = df[df["Model"] == "high_frequency_cut_off_power_law"]
    lfto_df = df[df["Model"] == "low_frequency_turn_over_power_law"]
    dtos_df = df[df["Model"] == "double_turn_over_spectrum"]

    smart_df = df[df["SMART"]]

    # MSPs
    msp_df = df[df["ATNF Period (s)"] < msp_cutoff]

    # Set up the gallerys
    # -----------------------------------------------------------------------------
    with open(f"{docs_dir}/spl_gallery.rst", "w", encoding="utf-8") as file:
        file.write("""
Simple Power Law Gallery
========================

""")
        for _, row in spl_df.iterrows():
            file.write(f"""

.. _{row["Pulsar"]}:

{row["Pulsar"]}
{"-" * len(row["Pulsar"])}
.. image:: best_fits/{row["Pulsar"]}_fit.png
    :width: 800
""")

    with open(f"{docs_dir}/bpl_gallery.rst", "w", encoding="utf-8") as file:
        file.write("""
Broken Power Law Gallery
========================

""")
        for _, row in bpl_df.iterrows():
            file.write(f"""

.. _{row["Pulsar"]}:

{row["Pulsar"]}
{"-" * len(row["Pulsar"])}
.. image:: best_fits/{row["Pulsar"]}_fit.png
    :width: 800
""")

    with open(f"{docs_dir}/lfto_gallery.rst", "w", encoding="utf-8") as file:
        file.write("""
Low Frequency Turn Over Gallery
===============================

""")
        for _, row in lfto_df.iterrows():
            file.write(f"""

.. _{row["Pulsar"]}:

{row["Pulsar"]}
{"-" * len(row["Pulsar"])}
.. image:: best_fits/{row["Pulsar"]}_fit.png
    :width: 800
""")

    with open(f"{docs_dir}/hfco_gallery.rst", "w", encoding="utf-8") as file:
        file.write("""
High Frequency Cut Off Gallery
==============================

""")
        for _, row in hfto_df.iterrows():
            file.write(f"""

.. _{row["Pulsar"]}:

{row["Pulsar"]}
{"-" * len(row["Pulsar"])}
.. image:: best_fits/{row["Pulsar"]}_fit.png
    :width: 800
""")

    with open(f"{docs_dir}/dtos_gallery.rst", "w", encoding="utf-8") as file:
        file.write("""
Double Turn Over Spectrum Gallery
=================================

""")
        for _, row in dtos_df.iterrows():
            file.write(f"""

.. _{row["Pulsar"]}:

{row["Pulsar"]}
{"-" * len(row["Pulsar"])}
.. image:: best_fits/{row["Pulsar"]}_fit.png
    :width: 800
""")

    with open(f"{docs_dir}/smart_gallery.rst", "w", encoding="utf-8") as file:
        file.write("""
SMART Gallery
=============

All pulsar detections from the SMART pulsar survey (these will be in other galleries).

""")
        for _, row in smart_df.iterrows():
            file.write(f"""

{row["Pulsar"]}
{"-" * len(row["Pulsar"])}
.. image:: best_fits/{row["Pulsar"]}_fit.png
    :width: 800
""")


    with open(f"{docs_dir}/msp_gallery.rst", "w", encoding="utf-8") as file:
        file.write("""
MSP Gallery
===========

All millisecond pulsar detections (these will be in other galleries).

""")
        for _, row in msp_df.iterrows():
            file.write(f"""

{row["Pulsar"]}
{"-" * len(row["Pulsar"])}
.. image:: best_fits/{row["Pulsar"]}_fit.png
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
    make_pulsar_gallery(args.output_dir)