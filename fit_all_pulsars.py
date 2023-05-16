import pandas as pd
import os
import matplotlib.pyplot as plt
import psrqpy
import shutil
import torch.multiprocessing as mp
from functools import partial
from tqdm import tqdm
import glob
import numpy as np

from pulsar_spectra.spectral_fit import find_best_spectral_fit, estimate_flux_density
from pulsar_spectra.catalogue import collect_catalogue_fluxes
from pulsar_spectra.analysis import calc_log_parabolic_spectrum_max_freq


cat_dict = collect_catalogue_fluxes()
query = psrqpy.QueryATNF().pandas

results_record = []

#for output csv
output_df = pd.DataFrame(
    columns=[
        "Pulsar",
        "ATNF Period (s)",
        "ATNF Pdot",
        "ATNF Spin Frequency (Hz)",
        "ATNF Fdot",
        "ATNF DM",
        "ATNF B_surf (G)",
        "ATNF B_LC (G)",
        "ATNF E_dot (ergs/s)",
        "ANTF Binary (type)",
        "Offset (degrees)",
        "Model",
        "Probability Best",
        "N data flux",
        "Min freq (MHz)",
        "Max freq (MHz)",
        "Bandwidth fit?",
        "L400 (mJy kpc^2)",
        "L1400 (mJy kpc^2)",
        "S150 (mJy)",
        "u_S150 (mJy)",
        "S300 (mJy)",
        "u_S300 (mJy)",
        "S5000 (mJy)",
        "u_S5000 (mJy)",
        "S10000 (mJy)",
        "u_S10000 (mJy)",
        "Age (Yr)",
        "a"         ,
        "u_a"       ,
        "c"         ,
        "u_c"       ,
        "vb"        ,
        "u_vb"      ,
        "a1"        ,
        "u_a1"      ,
        "a2"        ,
        "u_a2"      ,
        "vc"        ,
        "u_vc"      ,
        "vpeak"     ,
        "u_vpeak"   ,
        "beta"      ,
        "u_beta"    ,
        "lps_a"     ,
        "lps_u_a"   ,
        "lps_b"     ,
        "lps_u_b"   ,
        "lps_c"     ,
        "lps_u_c"   ,
        "SMART",
    ]
)


def fit_and_plot(pulsar):
    # Set up plot
    scale_figure = 0.9
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.5*scale_figure,4*scale_figure))

    a       = None
    u_a     = None
    c       = None
    u_c     = None
    vb      = None
    u_vb    = None
    a1      = None
    u_a1    = None
    a2      = None
    u_a2    = None
    vc      = None
    u_vc    = None
    vpeak   = None
    u_vpeak = None
    beta    = None
    u_beta  = None
    lps_a   = None
    lps_u_a = None
    lps_b   = None
    lps_u_b = None
    lps_c   = None
    lps_u_c = None
    l400 = None
    l1400 = None
    s150 = None
    u_s150 = None
    s300 = None
    u_s300 = None
    s5000 = None
    u_s5000 = None
    s10000 = None
    u_s10000 = None
    char_age = None


    freq_all, bands_all, flux_all, flux_err_all, ref_all = cat_dict[pulsar]
    query_id = list(query['PSRJ']).index(pulsar)

    if 'Bhat_2022' in ref_all:
        smart_pulsar = True
    else:
        smart_pulsar = False

    print(pulsar)
    models, iminuit_results, fit_infos, p_best, band_bool = find_best_spectral_fit(
        pulsar, freq_all, bands_all, flux_all, flux_err_all, ref_all,
        plot_best=True,# axis=ax
    )

    if models is not None:
        fit_loc = glob.glob(f"{pulsar}_*_fit.png")
        shutil.move(fit_loc[0], f"{os.path.dirname(os.path.realpath(__file__))}/docs/best_fits/{pulsar}_fit.png")

        # Calculate luminosity at 2 frequencies
        dist = query["DIST"][query_id]
        if not np.isnan(dist):
            s400, s400_err  = estimate_flux_density(400,  models,  iminuit_results)
            s1400, s1400_err = estimate_flux_density(1400, models,  iminuit_results)
            l400  = s400  * dist**2
            l1400 = s1400 * dist**2

        # Estimate flux desnity at 4 frequencies
        s150  , u_s150   = estimate_flux_density(150,    models,  iminuit_results)
        s300  , u_s300   = estimate_flux_density(300,    models,  iminuit_results)
        s5000 , u_s5000  = estimate_flux_density(5000,   models,  iminuit_results)
        s10000, u_s10000 = estimate_flux_density(10000,  models,  iminuit_results)


        # record model specific bits
        if models == "simple_power_law":
            a = iminuit_results.values["a"]
            u_a = iminuit_results.errors["a"]
            c = iminuit_results.values["c"]
            u_c = iminuit_results.errors["c"]
        elif models == "broken_power_law":
            #vb, a1, a2, b
            vb = iminuit_results.values["vb"]
            u_vb = iminuit_results.errors["vb"]
            a1 = iminuit_results.values["a1"]
            u_a1 = iminuit_results.errors["a1"]
            a2 = iminuit_results.values["a2"]
            u_a2 = iminuit_results.errors["a2"]
            c = iminuit_results.values["c"]
            u_c = iminuit_results.errors["c"]
        elif models == "log_parabolic_spectrum":
            lps_a = iminuit_results.values["a"]
            lps_u_a = iminuit_results.errors["a"]
            lps_b = iminuit_results.values["b"]
            lps_u_b = iminuit_results.errors["b"]
            lps_c = iminuit_results.values["c"]
            lps_u_c = iminuit_results.errors["c"]
            # Calculate the peak frequency
            vpeak, u_vpeak = calc_log_parabolic_spectrum_max_freq(
                iminuit_results.values["a"],
                iminuit_results.values["b"],
                iminuit_results.values["v0"],
                iminuit_results.errors["a"],
                iminuit_results.errors["b"],
                iminuit_results.covariance[0][1],
            )
        elif models == "high_frequency_cut_off_power_law":
            vc = iminuit_results.values["vc"]
            u_vc = iminuit_results.errors["vc"]
            a = iminuit_results.values["a"]
            u_a = iminuit_results.errors["a"]
            c = iminuit_results.values["c"]
            u_c = iminuit_results.errors["c"]
        elif models == "low_frequency_turn_over_power_law":
            #  vc, a, b, beta
            vpeak = iminuit_results.values["vpeak"]
            u_vpeak = iminuit_results.errors["vpeak"]
            a = iminuit_results.values["a"]
            u_a = iminuit_results.errors["a"]
            c = iminuit_results.values["c"]
            u_c = iminuit_results.errors["c"]
            beta = iminuit_results.values["beta"]
            u_beta = iminuit_results.errors["beta"]
        elif models == "low_frequency_turn_over_power_law":
            #  vc, a, b, beta
            vpeak = iminuit_results.values["vpeak"]
            u_vpeak = iminuit_results.errors["vpeak"]
            a = iminuit_results.values["a"]
            u_a = iminuit_results.errors["a"]
            c = iminuit_results.values["c"]
            u_c = iminuit_results.errors["c"]
            beta = iminuit_results.values["beta"]
            u_beta = iminuit_results.errors["beta"]
        elif models == "double_turn_over_spectrum":
            #  vc, a, b, beta
            vc = iminuit_results.values["vc"]
            u_vc = iminuit_results.errors["vc"]
            vpeak = iminuit_results.values["vpeak"]
            u_vpeak = iminuit_results.errors["vpeak"]
            a = iminuit_results.values["a"]
            u_a = iminuit_results.errors["a"]
            c = iminuit_results.values["c"]
            u_c = iminuit_results.errors["c"]
            beta = iminuit_results.values["beta"]
            u_beta = iminuit_results.errors["beta"]
    min_freq = min(cat_dict[pulsar][0])
    max_freq = max(cat_dict[pulsar][0])

    # Output data which will re recorded as a CSV later
    return {
        "Pulsar":pulsar,
        "ATNF Period (s)":query["P0"][query_id],
        "ATNF Pdot":query["P1"][query_id],
        "ATNF Spin Frequency (Hz)":query["F0"][query_id],
        "ATNF Fdot":query["F1"][query_id],
        "ATNF DM":query["DM"][query_id],
        "ATNF B_surf (G)":query["BSURF"][query_id],
        "ATNF B_LC (G)":query["BSURF"][query_id] * 9.2 * (10**12 * query["P0"][query_id]**3),
        "ATNF E_dot (ergs/s)":query["EDOT"][query_id],
        "ANTF Binary (type)":query["BINARY"][query_id],
        "Model":models,
        "Probability Best":p_best,
        "Min freq (MHz)":min_freq,
        "Max freq (MHz)":max_freq,
        "N data flux": len(flux_all),
        "Bandwidth fit?": band_bool,
        "L400 (mJy kpc^2)": l400,
        "L1400 (mJy kpc^2)": l1400,
        "S150 (mJy)": s150,
        "u_S150 (mJy)": u_s150,
        "S300 (mJy)": s300,
        "u_S300 (mJy)": u_s300,
        "S5000 (mJy)": s5000,
        "u_S5000 (mJy)": u_s5000,
        "S10000 (mJy)": s10000,
        "u_S10000 (mJy)": u_s10000,
        "Age (Yr)" : query["AGE"][query_id],
        "a"       : a,
        "u_a"     : u_a,
        "c"       : c,
        "u_c"     : u_c,
        "vb"      : vb,
        "u_vb"    : u_vb,
        "a1"      : a1,
        "u_a1"    : u_a1,
        "a2"      : a2,
        "u_a2"    : u_a2,
        "vc"      : vc,
        "u_vc"    : u_vc,
        "vpeak"   : vpeak,
        "u_vpeak" : u_vpeak,
        "beta"    : beta,
        "u_beta"  : u_beta,
        "lps_a"     : lps_a,
        "lps_u_a"   : lps_u_a,
        "lps_b"     : lps_b,
        "lps_u_b"   : lps_u_b,
        "lps_c"     : lps_c,
        "lps_u_c"   : lps_u_c,
        "SMART":smart_pulsar,
    }


pulsars_to_process = []
for pulsar in cat_dict.keys():
    # Skip pulsars without enough data
    if len(cat_dict[pulsar][0]) < 4:
        continue
    pulsars_to_process.append(pulsar)

# Set up CPU multiprocessing
pbar = tqdm(pulsars_to_process)
# freeze params/function as object
fc_ = partial(fit_and_plot)
# set number of processes
p = mp.Pool(8)
# runs mp with params on pbar
results = list(p.imap(fc_, pbar))
p.close()
p.join()

# Dump to csv
df = pd.DataFrame(results)
df.to_csv('all_pulsar_fits.csv', index=False)