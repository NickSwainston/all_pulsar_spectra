import pandas as pd
import os
import argparse
from pathlib import Path

import pulsar_spectra

def make_suggested_campaigns(output_dir):
    docs_dir = f"{output_dir}/docs"

    # Read in the fits
    df = pd.read_csv(f"{output_dir}/all_pulsar_fits_log.csv")

    model_short_name_map = {
        "simple_power_law": "SPL",
        "broken_power_law": "BPL",
        "log_parabolic_spectrum": "LPS",
        "high_frequency_cut_off_power_law": "HFCO",
        "low_frequency_turn_over_power_law": "LFTO",
        "double_turn_over_spectrum": "DTOS",
    }

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
    for _, row in df.iterrows():
        pulsar = row["Pulsar"]
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
                            "Model": model_short_name_map[row["Model"]],
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
                            "Model": model_short_name_map[row["Model"]],
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
    with open(f"{docs_dir}/suggested_campaigns.rst", "w", encoding="utf-8") as file:
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
        print("\\hline \\hline")
        for _, row in low_freq_camp.iterrows():
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
        print("\\hline \\hline")
        for _, row in high_freq_camp.iterrows():
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
    make_suggested_campaigns(args.output_dir)