import pandas as pd
import os
import matplotlib.pyplot as plt
import psrqpy
import shutil
from sympy import reduce_abs_inequalities
import torch.multiprocessing as mp
from functools import partial
from tqdm import tqdm
import glob

from pulsar_spectra.spectral_fit import find_best_spectral_fit
from pulsar_spectra.catalogue import collect_catalogue_fluxes
from pulsar_spectra.models import calc_log_parabolic_spectrum_max_freq

smart_pulsars = ['J0030+0451', 'J0034-0534', 'J0034-0721', 'J0038-2501', 'J0051+0423', 'J0133-6957', 'J0134-2937', 'J0151-0635', 'J0152-1637', 'J0206-4028', 'J0255-5304', 'J0304+1932', 'J0401-7608', 'J0418-4154', 'J0437-4715', 'J0450-1248', 'J0452-1759', 'J0459-0210', 'J0514-4408', 'J0520-2553', 'J0525+1115', 'J0528+2200', 'J0534+2200', 'J0600-5756', 'J0601-0527', 'J0614+2229', 'J0624-0424', 'J0630-2834', 'J0636-4549', 'J0702-4956', 'J0729-1448', 'J0729-1836', 'J0737-3039A', 'J0742-2822', 'J0749-4247', 'J0758-1528', 'J0820-1350', 'J0820-3921', 'J0820-4114', 'J0823+0159', 'J0826+2637', 'J0835-4510', 'J0837+0610', 'J0837-4135', 'J0838-3947', 'J0842-4851', 'J0855-3331', 'J0856-6137', 'J0902-6325', 'J0904-7459', 'J0905-6019', 'J0907-5157', 'J0908-1739', 'J0922+0638', 'J0924-5302', 'J0924-5814', 'J0942-5552', 'J0942-5657', 'J0943+1631', 'J0944-1354', 'J0946+0951', 'J0953+0755', 'J0955-5304', 'J0959-4809', 'J1003-4747', 'J1012-2337', 'J1018-1642', 'J1022+1001', 'J1034-3224', 'J1041-1942', 'J1057-5226', 'J1059-5742', 'J1112-6926', 'J1116-4122', 'J1121-5444', 'J1123-4844', 'J1123-6651', 'J1136+1551', 'J1136-5525', 'J1141-6545', 'J1146-6030', 'J1202-5820', 'J1224-6407', 'J1225-5556', 'J1239-6832', 'J1240-4124', 'J1257-1027', 'J1300+1240', 'J1311-1228', 'J1312-5402', 'J1313+0931', 'J1320-5359', 'J1328-4357', 'J1332-3032', 'J1335-3642', 'J1340-6456', 'J1355-5153', 'J1418-3921', 'J1430-6623', 'J1440-6344', 'J1453-6413', 'J1455-3330', 'J1456-6843', 'J1507-4352', 'J1510-4422', 'J1527-3931', 'J1534-5334', 'J1536-4948', 'J1543+0929', 'J1543-0620', 'J2048-1616', 'J2108-3429', 'J2145-0750', 'J2155-3118', 'J2222-0137', 'J2234+2114', 'J2241-5236', 'J2248-0101', 'J2317+2149', 'J2324-6054', 'J2325-0530', 'J2330-2005', 'J2336-01', 'J2354-22']

cat_dict = collect_catalogue_fluxes()
query = psrqpy.QueryATNF().pandas

results_record = []

#for output csv
output_df = pd.DataFrame(
    columns=[
        "Pulsar",
        "ATNF Period (s)",
        "ATNF DM",
        "ATNF B_surf (G)",
        "ATNF E_dot (ergs/s)",
        "Offset (degrees)",
        "Model",
        "Probability Best",
        "N data flux",
        "Min freq (MHz)",
        "Max freq (MHz)",
        "pl_a",
        "pl_u_a",
        "pl_c"      ,
        "pl_u_c"      ,
        "bpl_vb"    ,
        "bpl_u_vb"    ,
        "bpl_a1"    ,
        "bpl_u_a1"    ,
        "bpl_a2"    ,
        "bpl_u_a2"    ,
        "bpl_c"     ,
        "bpl_u_c"     ,
        "lps_a"     ,
        "lps_u_a"     ,
        "lps_b"     ,
        "lps_u_b"     ,
        "lps_c"     ,
        "lps_u_c"     ,
        "lps_v_peak",
        "lps_u_v_peak",
        "hfco_vc"   ,
        "hfco_u_vc"   ,
        "hfto_a"    ,
        "hfto_u_a"    ,
        "hfco_c"    ,
        "hfco_u_c"    ,
        "lfto_vpeak"   ,
        "lfto_u_vpeak" ,
        "lfto_a"    ,
        "lfto_u_a"    ,
        "lfto_c"    ,
        "lfto_u_c"    ,
        "lfto_beta" ,
        "lfto_u_beta" ,
        "dtos_vc"   ,
        "dtos_u_vc"   ,
        "dtos_vpeak"   ,
        "dtos_u_vpeak" ,
        "dtos_a"    ,
        "dtos_u_a"    ,
        "dtos_c"    ,
        "dtos_u_c"    ,
        "dtos_beta" ,
        "dtos_u_beta" ,
    ]
)


def fit_and_plot(pulsar):
    # Set up plot
    scale_figure = 0.9
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.5*scale_figure,4*scale_figure))

    pl_a      = None
    pl_u_a      = None
    pl_c      = None
    pl_u_c      = None
    bpl_vb    = None
    bpl_u_vb    = None
    bpl_a1    = None
    bpl_u_a1    = None
    bpl_a2    = None
    bpl_u_a2    = None
    bpl_c     = None
    bpl_u_c     = None
    lps_a     = None
    lps_u_a     = None
    lps_b     = None
    lps_u_b     = None
    lps_c     = None
    lps_u_c     = None
    lps_v_peak= None
    lps_u_v_peak= None
    hfco_vc   = None
    hfco_u_vc   = None
    hfco_a   = None
    hfco_u_a   = None
    hfco_c    = None
    hfco_u_c    = None
    lfto_vpeak   = None
    lfto_u_vpeak   = None
    lfto_a    = None
    lfto_u_a    = None
    lfto_c    = None
    lfto_u_c    = None
    lfto_beta = None
    lfto_u_beta = None
    dtos_vc   = None
    dtos_u_vc   = None
    dtos_vpeak   = None
    dtos_u_vpeak   = None
    dtos_a    = None
    dtos_u_a    = None
    dtos_c    = None
    dtos_u_c    = None
    dtos_beta = None
    dtos_u_beta = None

    freq_all, flux_all, flux_err_all, ref_all = cat_dict[pulsar]
    query_id = list(query['PSRJ']).index(pulsar)

    models, iminuit_results, fit_infos, p_best, p_catagory = find_best_spectral_fit(
        pulsar, freq_all, flux_all, flux_err_all, ref_all,
        plot_best=True, alternate_style=True,# axis=ax
    )
    # plt.tight_layout(pad=2.5)
    # plt.savefig(f"{pulsar}_fit.png", bbox_inches='tight', dpi=300)

    if models is not None:
        fit_loc = glob.glob(f"{pulsar}_*_fit.png")
        shutil.move(fit_loc[0], f"{os.path.dirname(os.path.realpath(__file__))}/docs/best_fits/{pulsar}_fit.png")

        # record model specific bits
        if models == "simple_power_law":
            pl_a = iminuit_results.values["a"]
            pl_u_a = iminuit_results.errors["a"]
            pl_c = iminuit_results.values["c"]
            pl_u_c = iminuit_results.errors["c"]
        elif models == "broken_power_law":
            #vb, a1, a2, b
            bpl_vb = iminuit_results.values["vb"]
            bpl_u_vb = iminuit_results.errors["vb"]
            bpl_a1 = iminuit_results.values["a1"]
            bpl_u_a1 = iminuit_results.errors["a1"]
            bpl_a2 = iminuit_results.values["a2"]
            bpl_u_a2 = iminuit_results.errors["a2"]
            bpl_c = iminuit_results.values["c"]
            bpl_u_c = iminuit_results.errors["c"]
        elif models == "log_parabolic_spectrum":
            lps_a = iminuit_results.values["a"]
            lps_u_a = iminuit_results.errors["a"]
            lps_b = iminuit_results.values["b"]
            lps_u_b = iminuit_results.errors["b"]
            lps_c = iminuit_results.values["c"]
            lps_u_c = iminuit_results.errors["c"]
            # Calculate the peak frequency
            v_peak, u_v_peak = calc_log_parabolic_spectrum_max_freq(
                iminuit_results.values["a"],
                iminuit_results.values["b"],
                iminuit_results.values["v0"],
                iminuit_results.errors["a"],
                iminuit_results.errors["b"],
                iminuit_results.covariance[0][1],
            )
            lps_v_peak = v_peak
            lps_u_v_peak = u_v_peak
            #print(f"vpeak: {v_peak/1e6:6.2f} +/- {u_v_peak/1e6:6.2f}")
        elif models == "high_frequency_cut_off_power_law":
            hfco_vc = iminuit_results.values["vc"]
            hfco_u_vc = iminuit_results.errors["vc"]
            hfco_a = iminuit_results.values["a"]
            hfco_u_a = iminuit_results.errors["a"]
            hfco_c = iminuit_results.values["c"]
            hfco_u_c = iminuit_results.errors["c"]
        elif models == "low_frequency_turn_over_power_law":
            #  vc, a, b, beta
            lfto_vpeak = iminuit_results.values["vpeak"]
            lfto_u_vpeak = iminuit_results.errors["vpeak"]
            lfto_a = iminuit_results.values["a"]
            lfto_u_a = iminuit_results.errors["a"]
            lfto_c = iminuit_results.values["c"]
            lfto_u_c = iminuit_results.errors["c"]
            lfto_beta = iminuit_results.values["beta"]
            lfto_u_beta = iminuit_results.errors["beta"]
        elif models == "low_frequency_turn_over_power_law":
            #  vc, a, b, beta
            lfto_vpeak = iminuit_results.values["vpeak"]
            lfto_u_vpeak = iminuit_results.errors["vpeak"]
            lfto_a = iminuit_results.values["a"]
            lfto_u_a = iminuit_results.errors["a"]
            lfto_c = iminuit_results.values["c"]
            lfto_u_c = iminuit_results.errors["c"]
            lfto_beta = iminuit_results.values["beta"]
            lfto_u_beta = iminuit_results.errors["beta"]
        elif models == "double_turn_over_spectrum":
            #  vc, a, b, beta
            dtos_vc = iminuit_results.values["vc"]
            dtos_u_vc = iminuit_results.errors["vc"]
            dtos_vpeak = iminuit_results.values["vpeak"]
            dtos_u_vpeak = iminuit_results.errors["vpeak"]
            dtos_a = iminuit_results.values["a"]
            dtos_u_a = iminuit_results.errors["a"]
            dtos_c = iminuit_results.values["c"]
            dtos_u_c = iminuit_results.errors["c"]
            dtos_beta = iminuit_results.values["beta"]
            dtos_u_beta = iminuit_results.errors["beta"]
    min_freq = min(cat_dict[pulsar][0])
    max_freq = max(cat_dict[pulsar][0])

    # Record data for csv
    #output_df = output_df.append({
    return {
        "Pulsar":pulsar,
        "ATNF Period (s)":query["P0"][query_id],
        "ATNF DM":query["DM"][query_id],
        "ATNF B_surf (G)":query["BSURF"][query_id],
        "ATNF E_dot (ergs/s)":query["EDOT"][query_id],
        "Model":models,
        "Probability Best":p_best,
        "Min freq (MHz)":min_freq,
        "Max freq (MHz)":max_freq,
        "N data flux": len(flux_all),
        "pl_a"      : pl_a     ,
        "pl_u_a"      : pl_u_a     ,
        "pl_c"      : pl_c     ,
        "pl_u_c"      : pl_u_c     ,
        "bpl_vb"    : bpl_vb   ,
        "bpl_u_vb"    : bpl_u_vb   ,
        "bpl_a1"    : bpl_a1   ,
        "bpl_u_a1"    : bpl_u_a1   ,
        "bpl_a2"    : bpl_a2   ,
        "bpl_u_a2"    : bpl_u_a2   ,
        "bpl_c"     : bpl_c    ,
        "bpl_u_c"     : bpl_u_c    ,
        "lps_a"     : lps_a    ,
        "lps_u_a"     : lps_u_a    ,
        "lps_b"     : lps_b    ,
        "lps_u_b"     : lps_u_b    ,
        "lps_c"     : lps_c    ,
        "lps_u_c"     : lps_u_c    ,
        "lps_v_peak": lps_v_peak,
        "lps_u_v_peak": lps_u_v_peak,
        "hfco_vc"   : hfco_vc  ,
        "hfco_u_vc"   : hfco_u_vc  ,
        "hfco_a"   : hfco_a  ,
        "hfco_u_a"   : hfco_u_a  ,
        "hfco_c"    : hfco_c   ,
        "hfco_u_c"    : hfco_u_c   ,
        "lfto_vpeak"   : lfto_vpeak  ,
        "lfto_u_vpeak" : lfto_u_vpeak,
        "lfto_a"    : lfto_a   ,
        "lfto_u_a"    : lfto_u_a   ,
        "lfto_c"    : lfto_c   ,
        "lfto_u_c"    : lfto_u_c   ,
        "lfto_beta" : lfto_beta,
        "lfto_u_beta" : lfto_u_beta,
        "lfto_vpeak"   : lfto_vpeak  ,
        "lfto_u_vpeak" : lfto_u_vpeak,
        "lfto_a"    : lfto_a   ,
        "lfto_u_a"    : lfto_u_a   ,
        "lfto_c"    : lfto_c   ,
        "lfto_u_c"    : lfto_u_c   ,
        "lfto_beta" : lfto_beta,
        "lfto_u_beta" : lfto_u_beta,
        "dtos_vc"   : dtos_vc  ,
        "dtos_u_vc"   : dtos_u_vc  ,
        "dtos_vpeak"   : dtos_vpeak  ,
        "dtos_u_vpeak" : dtos_u_vpeak,
        "dtos_a"    : dtos_a   ,
        "dtos_u_a"    : dtos_u_a   ,
        "dtos_c"    : dtos_c   ,
        "dtos_u_c"    : dtos_u_c   ,
        "dtos_beta" : dtos_beta,
        "dtos_u_beta" : dtos_u_beta,
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

pbar = tqdm(pulsars_to_process)
# freeze params/function as object
fc_ = partial(fit_and_plot)
# set number of processes
p = mp.Pool(8)
# runs mp with params on pbar
#results = p.imap(fc_, pbar)
results = list(p.imap(fc_, pbar))
#print(results)
# close out and join processes
p.close()
p.join()

# results = []
# for pulsar in pulsars_to_process:
#     results.append(fit_and_plot(pulsar))

df = pd.DataFrame(results)
df.to_csv('all_pulsar_fits.csv', index=False)