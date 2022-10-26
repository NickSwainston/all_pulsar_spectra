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

smart_pulsars = ['J0030+0451', 'J0034-0534', 'J0034-0721', 'J0038-2501', 'J0051+0423', 'J0133-6957', 'J0134-2937', 'J0151-0635', 'J0152-1637', 'J0206-4028', 'J0255-5304', 'J0304+1932', 'J0401-7608', 'J0418-4154', 'J0437-4715', 'J0450-1248', 'J0452-1759', 'J0459-0210', 'J0514-4408', 'J0520-2553', 'J0525+1115', 'J0528+2200', 'J0534+2200', 'J0600-5756', 'J0601-0527', 'J0614+2229', 'J0624-0424', 'J0630-2834', 'J0636-4549', 'J0702-4956', 'J0729-1448', 'J0729-1836', 'J0737-3039A', 'J0742-2822', 'J0749-4247', 'J0758-1528', 'J0820-1350', 'J0820-3921', 'J0820-4114', 'J0823+0159', 'J0826+2637', 'J0835-4510', 'J0837+0610', 'J0837-4135', 'J0838-3947', 'J0842-4851', 'J0855-3331', 'J0856-6137', 'J0902-6325', 'J0904-7459', 'J0905-6019', 'J0907-5157', 'J0908-1739', 'J0922+0638', 'J0924-5302', 'J0924-5814', 'J0942-5552', 'J0942-5657', 'J0943+1631', 'J0944-1354', 'J0946+0951', 'J0953+0755', 'J0955-5304', 'J0959-4809', 'J1003-4747', 'J1012-2337', 'J1018-1642', 'J1022+1001', 'J1034-3224', 'J1041-1942', 'J1057-5226', 'J1059-5742', 'J1112-6926', 'J1116-4122', 'J1121-5444', 'J1123-4844', 'J1123-6651', 'J1136+1551', 'J1136-5525', 'J1141-6545', 'J1146-6030', 'J1202-5820', 'J1224-6407', 'J1225-5556', 'J1239-6832', 'J1240-4124', 'J1257-1027', 'J1300+1240', 'J1311-1228', 'J1312-5402', 'J1313+0931', 'J1320-5359', 'J1328-4357', 'J1332-3032', 'J1335-3642', 'J1340-6456', 'J1355-5153', 'J1418-3921', 'J1430-6623', 'J1440-6344', 'J1453-6413', 'J1455-3330', 'J1456-6843', 'J1507-4352', 'J1510-4422', 'J1527-3931', 'J1534-5334', 'J1536-4948', 'J1543+0929', 'J1543-0620', 'J2048-1616', 'J2108-3429', 'J2145-0750', 'J2155-3118', 'J2222-0137', 'J2234+2114', 'J2241-5236', 'J2248-0101', 'J2317+2149', 'J2324-6054', 'J2325-0530', 'J2330-2005', 'J2336-01', 'J2354-22']

cat_dict = collect_catalogue_fluxes()#use_atnf=False)
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
    # plt.tight_layout(pad=2.5)
    # plt.savefig(f"{pulsar}_fit.png", bbox_inches='tight', dpi=300)

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

    # Record data for csv
    #output_df = output_df.append({
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
    #, ignore_index=True)


pulsars_to_process = []
for pulsar in cat_dict.keys():
    # Skip smart pulsars and pulsars without enough data
    # if pulsar in smart_pulsars:
    #     continue
    if len(cat_dict[pulsar][0]) < 4:
        continue
    pulsars_to_process.append(pulsar)
#pulsars_to_process = ['J0034-0534', 'J0835-4510', 'J1141-6545', 'J1751-4657', 'J0953+0755']
# pulsars_to_process = ['J1136+1551']
# Pulsars with large bandwidths
#pulsars_to_process = ['J1721-3532', 'J1048-5832', 'J0908-4913', 'J1644-4559', 'J1740-3015', 'J0738-4042', 'J0437-4715', 'J1935+1616', 'J2048-1616', 'J2321+6024', 'J1622-4950', 'J0358+5413', 'J2022+5154', 'J1752-2806', 'J2022+2854', 'J1645-0317', 'J1709-1640', 'J0528+2200', 'J0332+5434', 'J2018+2839', 'J0814+7429', 'J0826+2637', 'J0953+0755', 'J1136+1551', 'J1239+2453', 'J1932+1059', 'J1745-2900', 'J0835-4510']
#pulsars_to_process = ["J0151-0635"]

pbar = tqdm(pulsars_to_process)
# freeze params/function as object
fc_ = partial(fit_and_plot)
# set number of processes
p = mp.Pool(8)
# runs mp with params on pbar
#results = p.imap(fc_, pbar)
results = list(p.imap(fc_, pbar))
print("done")
#print(results)
# close out and join processes
p.close()
print("closed")
p.join()
print("joined")

# results = []
# for pulsar in pulsars_to_process:
#     results.append(fit_and_plot(pulsar))

df = pd.DataFrame(results)
print('start dump')
df.to_csv('all_pulsar_fits.csv', index=False)
print('dumped')