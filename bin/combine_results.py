#!/usr/bin/env python3
"""Merge per-pulsar quick-fit YAML files into a single analysis CSV.

Adds ATNF catalogue properties and evaluates the best-fit model at four
reference frequencies (150, 300, 5000, 10000 MHz) plus luminosities at
400 and 1400 MHz where a DM distance is available.
"""

import argparse
import glob
import inspect

import numpy as np
import pandas as pd
import psrqpy
import yaml

from pulsar_spectra.catalogue import collect_catalogue_fluxes
from pulsar_spectra.models import model_settings


def _flux_mJy(model_name, params, freq_MHz):
    """Evaluate model flux at *freq_MHz* (MHz) → mJy.

    Converts params from user units (MHz / mJy) back to internal units
    (Hz / Jy) before calling the model function, then returns mJy.
    Returns None if the model is unknown or evaluation fails.
    """
    model_dict = model_settings()
    if model_name not in model_dict:
        return None
    model_func = model_dict[model_name][0]
    # Parameter names of the model function, excluding the first arg ('v')
    sig_keys = list(inspect.signature(model_func).parameters.keys())[1:]
    internal_vals = []
    for p in sig_keys:
        val = params[p]
        if p.startswith("v"):
            internal_vals.append(val * 1e6)   # MHz → Hz
        elif p == "c":
            internal_vals.append(val / 1e3)   # mJy → Jy
        else:
            internal_vals.append(val)
    v_Hz = np.array([freq_MHz * 1e6])
    try:
        flux_Jy = model_func(v_Hz, *internal_vals)
        return float(flux_Jy[0] * 1e3)        # Jy → mJy
    except Exception:
        return None


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
        with open(path) as fh:
            data = yaml.safe_load(fh)
        if data:
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
        model  = result.get("model")
        params = result.get("params") or {}
        p_errs = result.get("param_errs") or {}

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
        s150 = s300 = s5000 = s10000 = None
        l400 = l1400 = None
        if model is not None:
            s150   = _flux_mJy(model, params, 150)
            s300   = _flux_mJy(model, params, 300)
            s5000  = _flux_mJy(model, params, 5000)
            s10000 = _flux_mJy(model, params, 10000)
            try:
                dist_f = float(dist)
                if not np.isnan(dist_f):
                    s400  = _flux_mJy(model, params, 400)
                    s1400 = _flux_mJy(model, params, 1400)
                    if s400  is not None:
                        l400  = s400  * dist_f ** 2
                    if s1400 is not None:
                        l1400 = s1400 * dist_f ** 2
            except (TypeError, ValueError):
                pass

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
            "Model":                   model,
            "Probability Best":        result.get("p_best"),
            "Min freq (MHz)":          min_freq,
            "Max freq (MHz)":          max_freq,
            "N data flux":             result.get("n_data"),
            "Bandwidth fit?":          result.get("band_bool"),
            "L400 (mJy kpc^2)":        l400,
            "L1400 (mJy kpc^2)":       l1400,
            "S150 (mJy)":              s150,
            "S300 (mJy)":              s300,
            "S5000 (mJy)":             s5000,
            "S10000 (mJy)":            s10000,
            "Age (Yr)":                age,
            # Model-specific parameters (None where not applicable)
            "a":       params.get("a"),
            "u_a":     p_errs.get("a"),
            "c":       params.get("c"),
            "u_c":     p_errs.get("c"),
            "vb":      params.get("vb"),
            "u_vb":    p_errs.get("vb"),
            "a1":      params.get("a1"),
            "u_a1":    p_errs.get("a1"),
            "a2":      params.get("a2"),
            "u_a2":    p_errs.get("a2"),
            "vc":      params.get("vc"),
            "u_vc":    p_errs.get("vc"),
            "vpeak":   params.get("vpeak"),
            "u_vpeak": p_errs.get("vpeak"),
            "beta":    params.get("beta"),
            "u_beta":  p_errs.get("beta"),
            "SMART":   smart_pulsar,
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
