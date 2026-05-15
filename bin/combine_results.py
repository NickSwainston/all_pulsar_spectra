#!/usr/bin/env python3
"""Merge per-pulsar quick-fit YAML files into a single analysis CSV.

Adds ATNF catalogue properties and evaluates the best-fit model at four
reference frequencies (150, 300, 5000, 10000 MHz) plus luminosities at
400 and 1400 MHz where a DM distance is available.
"""

import argparse
import glob

import numpy as np
import pandas as pd
import psrqpy

from pulsar_spectra.analysis import estimate_flux_density
from pulsar_spectra.catalogue import collect_catalogue_fluxes
from pulsar_spectra.spectral_fit import load_results_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "yaml_files",
        nargs="*",
        help="YAML result files from quick-fit (globs accepted)",
    )
    parser.add_argument("-o", "--output", default="all_pulsar_fits.csv")
    args = parser.parse_args()

    # Expand any shell globs that were passed literally (e.g. from Nextflow)
    yaml_paths = []
    for pattern in args.yaml_files:
        expanded = glob.glob(pattern)
        yaml_paths.extend(expanded if expanded else [pattern])

    if not yaml_paths:
        raise SystemExit("No YAML files found.")

    # Load all fit results
    fit_results = {}
    for path in yaml_paths:
        data = load_results_yaml(path)
        fit_results.update(data)

    pulsars = list(fit_results.keys())
    print(f"Loaded results for {len(pulsars)} pulsars")

    # Single ATNF batch query
    atnf_params = ["PSRJ", "P0", "P1", "F0", "F1", "DM", "BSURF", "EDOT", "BINARY", "AGE", "DIST"]
    query_df = psrqpy.QueryATNF(psrs=pulsars, params=atnf_params).pandas
    atnf = {row["PSRJ"]: row for _, row in query_df.iterrows()}

    # Catalogue fluxes – for min/max frequency and SMART flag
    cat_dict = collect_catalogue_fluxes()

    rows = []
    for pulsar, result in fit_results.items():
        params = result.params
        p_errs = result.param_errs

        # --- ATNF properties ---
        q = atnf.get(pulsar, {})
        period  = q.get("P0")
        pdot    = q.get("P1")
        f0      = q.get("F0")
        f1      = q.get("F1")
        dm      = q.get("DM")
        bsurf   = q.get("BSURF")
        edot    = q.get("EDOT")
        binary  = q.get("BINARY")
        age     = q.get("AGE")
        dist    = q.get("DIST")

        b_lc = None
        if bsurf is not None and period is not None:
            try:
                if not (np.isnan(float(bsurf)) or np.isnan(float(period))):
                    b_lc = float(bsurf) * 9.2 * (10**12 * float(period) ** 3)
            except (TypeError, ValueError):
                pass

        # --- Catalogue metadata ---
        cat_entry = cat_dict.get(pulsar)
        if cat_entry is not None:
            freq_all = cat_entry[0]
            ref_all  = cat_entry[-1]   # refs are always the last element
            min_freq = float(min(freq_all)) if freq_all else None
            max_freq = float(max(freq_all)) if freq_all else None
            smart_pulsar = "Bhat_2022" in ref_all
        else:
            min_freq = max_freq = None
            smart_pulsar = False

        # --- Flux density estimates ---
        s150 = u_s150 = s300 = u_s300 = s5000 = u_s5000 = s10000 = u_s10000 = None
        l400 = l1400 = None
        print(result)
        if result.model is not None:
            fluxes, errs = estimate_flux_density([150, 300, 400, 1400, 5000, 10000], result)
            s150,   u_s150   = float(fluxes[0]), float(errs[0])
            s300,   u_s300   = float(fluxes[1]), float(errs[1])
            s400  = float(fluxes[2])
            s1400 = float(fluxes[3])
            s5000,  u_s5000  = float(fluxes[4]), float(errs[4])
            s10000, u_s10000 = float(fluxes[5]), float(errs[5])
            try:
                dist_f = float(dist)
                if not np.isnan(dist_f):
                    l400  = s400  * dist_f ** 2
                    l1400 = s1400 * dist_f ** 2
            except (TypeError, ValueError) as e:
                print(f"Error calculating luminosity for {pulsar}: {e}")

        rows.append({
            "Pulsar":                  pulsar,
            "ATNF Period (s)":         period,
            "ATNF Pdot":               pdot,
            "ATNF Spin Frequency (Hz)": f0,
            "ATNF Fdot":               f1,
            "ATNF DM":                 dm,
            "ATNF B_surf (G)":         bsurf,
            "ATNF B_LC (G)":           b_lc,
            "ATNF E_dot (ergs/s)":     edot,
            "ANTF Binary (type)":      binary,
            "Model":                   result.model,
            "Probability Best":        result.p_best,
            "Min freq (MHz)":          min_freq,
            "Max freq (MHz)":          max_freq,
            "N data flux":             result.n_data,
            "Bandwidth fit?":          result.band_bool,
            "L400 (mJy kpc^2)":        l400,
            "L1400 (mJy kpc^2)":       l1400,
            "S150 (mJy)":              s150,
            "u_S150 (mJy)":            u_s150,
            "S300 (mJy)":              s300,
            "u_S300 (mJy)":            u_s300,
            "S5000 (mJy)":             s5000,
            "u_S5000 (mJy)":           u_s5000,
            "S10000 (mJy)":            s10000,
            "u_S10000 (mJy)":          u_s10000,
            "Age (Yr)":                age,
            # Model-specific parameters (None where not applicable)
            "a":       params.get("a"),
            "u_a":     p_errs.get("a"),
            "c_mJy":       params.get("c_mJy"),
            "u_c_mJy":     p_errs.get("c_mJy"),
            "vb_MHz":      params.get("vb_MHz"),
            "u_vb_MHz":    p_errs.get("vb_MHz"),
            "a1":      params.get("a1"),
            "u_a1":    p_errs.get("a1"),
            "a2":      params.get("a2"),
            "u_a2":    p_errs.get("a2"),
            "vc_MHz":      params.get("vc_MHz"),
            "u_vc_MHz":    p_errs.get("vc_MHz"),
            "vpeak_MHz":   params.get("vpeak_MHz"),
            "u_vpeak_MHz": p_errs.get("vpeak_MHz"),
            "beta":    params.get("beta"),
            "u_beta":  p_errs.get("beta"),
            "SMART":   smart_pulsar,
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
