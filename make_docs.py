import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
import numpy as np
import seaborn as sns
from scipy import stats
import math
from sympy import sec
from jacobi import propagate

from iminuit import Minuit
from iminuit.cost import LeastSquares

df = pd.read_csv('all_pulsar_fits.csv')

# grab model specific data frames
spl_df  = df[df["Model"] == "simple_power_law"]
bpl_df  = df[df["Model"] == "broken_power_law"]
lps_df  = df[df["Model"] == "log_parabolic_spectrum"]
hfto_df = df[df["Model"] == "high_frequency_cut_off_power_law"]
lfto_df = df[df["Model"] == "low_frequency_turn_over_power_law"]
dtos_df = df[df["Model"] == "double_turn_over_spectrum"]
nomod_df = df[df["Model"] == ""]
smart_df = df[df["SMART"]]
all_to_df = df[df["vpeak"].notnull()]
sync_df = all_to_df[all_to_df["beta"] < 2.05]
print(len(sync_df))
thermal_df = all_to_df[all_to_df["beta"] >= 2.05]
print(len(thermal_df))

if len(lps_df) == 0:
    del df["lps_a"]
    del df["lps_u_a"]
    del df["lps_b"]
    del df["lps_u_b"]
    del df["lps_c"]
    del df["lps_u_c"]


# Set up the docs

# Record summary results on homepage
with open(f'{os.path.dirname(os.path.realpath(__file__))}/docs/index.rst', 'w') as file:
    file.write(f'''
Pulsar Spectra all pulsars fit results
======================================

.. toctree::
   :maxdepth: 1
   :caption: Spectral Model Gallerys:

   spl_gallery
   bpl_gallery
   lfto_gallery
   hfco_gallery
   dtos_gallery
   smart_gallery


Fit Summary
-----------
.. csv-table::
   :header: "Model", "Pulsar Count"

   "simple_power_law", "{len(spl_df)}"
   "broken_power_law", "{len(bpl_df)}"
   "log_parabolic_spectrum", "{len(lps_df)}"
   "high_frequency_cut_off_power_law", "{len(hfto_df)}"
   "low_frequency_turn_over_power_law", "{len(lfto_df)}"
   "double_turn_over_spectrum", "{len(dtos_df)}"
   "Total", "{len(df)}"

Analysis Summary
----------------
.. csv-table::
   :header: "Parameter", "Mean"

   "spectral index", "{df["a"].mean():.2f}±{df["a"].std():.2f}"
   "Peak Frequency (GHz)", "{df["vpeak"].mean()/1e9:.2f}±{df["vpeak"].std()/1e9:.2f}"
   "Cut off frequency (GHz)", "{df["vc"].mean()/1e9:.2f}±{df["vc"].std()/1e9:.2f}"
   "Beta", "{df["beta"].mean():.2f}±{df["beta"].std():.2f}"

Single Power Law Results
------------------------
.. csv-table::
   :header: "Pulsar", "a"

''')
    for index, row in spl_df.iterrows():
        data_str = f'   ":ref:`{row["Pulsar"]}`", '
        for val, error in [("a", "u_a")]:
            if "v" in val:
                data_str += f'"{int(row[val]/1e6):d}±{int(row[error]/1e6):d}", '
            else:
                data_str += f'"{row[val]:.2f}±{row[error]:.2f}", '
        file.write(f'{data_str[:-2]}\n')

    file.write(f'''


Broken Power Law Results
------------------------
.. csv-table::
   :header: "Pulsar", "vb (MHz)", "a1", "a2"

''')
    for index, row in bpl_df.iterrows():
        data_str = f'   ":ref:`{row["Pulsar"]}`", '
        for val, error in [("vb", "u_vb"), ("a1", "u_a1"), ("a2","u_a2")]:
            if "v" in val:
                data_str += f'"{int(row[val]/1e6):d}±{int(row[error]/1e6):d}", '
            else:
                data_str += f'"{row[val]:.2f}±{row[error]:.2f}", '
        file.write(f'{data_str[:-2]}\n')

    file.write(f'''


Low Frequency Turn Over Results
-------------------------------
.. csv-table::
   :header: "Pulsar", "vpeak (MHz)", "a", "beta"

''')
    for index, row in lfto_df.iterrows():
        data_str = f'   ":ref:`{row["Pulsar"]}`", '
        for val, error in [("vpeak", "u_vpeak"), ("a", "u_a"), ("beta", "u_beta")]:
            if "v" in val:
                data_str += f'"{int(row[val]/1e6):d}±{int(row[error]/1e6):d}", '
            else:
                data_str += f'"{row[val]:.2f}±{row[error]:.2f}", '
        file.write(f'{data_str[:-2]}\n')

    file.write(f'''


High Frequency Cut Off Results
------------------------------
.. csv-table::
   :header: "Pulsar", "vc (MHz)", "a"

''')
    for index, row in hfto_df.iterrows():
        data_str = f'   ":ref:`{row["Pulsar"]}`", '
        for val, error in [("vc", "u_vc"), ("a", "u_a")]:
            if "v" in val:
                data_str += f'"{int(row[val]/1e6):d}±{int(row[error]/1e6):d}", '
            else:
                data_str += f'"{row[val]:.2f}±{row[error]:.2f}", '
        file.write(f'{data_str[:-2]}\n')

    file.write(f'''


Double Turn Over Spectrum Results
---------------------------------
.. csv-table::
   :header: "Pulsar", "vc (MHz)", "vpeak (MHz)", "a", "beta"

''')
    for index, row in dtos_df.iterrows():
        data_str = f'   ":ref:`{row["Pulsar"]}`", '
        for val, error in [("vc", "u_vc"), ("vpeak", "u_vpeak"), ("a", "u_a"), ("beta", "u_beta")]:
            if "v" in val:
                data_str += f'"{int(row[val]/1e6):d}±{int(row[error]/1e6):d}", '
            else:
                data_str += f'"{row[val]:.2f}±{row[error]:.2f}", '
        file.write(f'{data_str[:-2]}\n')

    file.write(f'''


No Model Results
----------------
.. csv-table::
   :header: "Pulsar", "N data"

''')
    for index, row in nomod_df.iterrows():
        file.write(f'   ":ref:`{row["Pulsar"]}`", "{row["N data flux"]}"')


# Set up the gallerys
with open(f'{os.path.dirname(os.path.realpath(__file__))}/docs/spl_gallery.rst', 'w') as file:
    file.write(f'''
Simple Power Law Gallery
========================

''')
    for index, row in spl_df.iterrows():
        file.write(f'''

.. _{row["Pulsar"]}:

{row["Pulsar"]}
{"-"*len(row["Pulsar"])}
.. image:: best_fits/{row["Pulsar"]}_fit.png
  :width: 800
''')

with open(f'{os.path.dirname(os.path.realpath(__file__))}/docs/bpl_gallery.rst', 'w') as file:
    file.write(f'''
Broken Power Law Gallery
========================

''')
    for index, row in bpl_df.iterrows():
        file.write(f'''

.. _{row["Pulsar"]}:

{row["Pulsar"]}
{"-"*len(row["Pulsar"])}
.. image:: best_fits/{row["Pulsar"]}_fit.png
  :width: 800
''')

with open(f'{os.path.dirname(os.path.realpath(__file__))}/docs/lfto_gallery.rst', 'w') as file:
    file.write(f'''
Low Frequency Turn Over Gallery
===============================

''')
    for index, row in lfto_df.iterrows():
        file.write(f'''

.. _{row["Pulsar"]}:

{row["Pulsar"]}
{"-"*len(row["Pulsar"])}
.. image:: best_fits/{row["Pulsar"]}_fit.png
  :width: 800
''')

with open(f'{os.path.dirname(os.path.realpath(__file__))}/docs/hfco_gallery.rst', 'w') as file:
    file.write(f'''
High Frequency Cut Off Gallery
==============================

''')
    for index, row in hfto_df.iterrows():
        file.write(f'''

.. _{row["Pulsar"]}:

{row["Pulsar"]}
{"-"*len(row["Pulsar"])}
.. image:: best_fits/{row["Pulsar"]}_fit.png
  :width: 800
''')

with open(f'{os.path.dirname(os.path.realpath(__file__))}/docs/dtos_gallery.rst', 'w') as file:
    file.write(f'''
Double Turn Over Spectrum Gallery
=================================

''')
    for index, row in dtos_df.iterrows():
        file.write(f'''

.. _{row["Pulsar"]}:

{row["Pulsar"]}
{"-"*len(row["Pulsar"])}
.. image:: best_fits/{row["Pulsar"]}_fit.png
  :width: 800
''')

with open(f'{os.path.dirname(os.path.realpath(__file__))}/docs/smart_gallery.rst', 'w') as file:
    file.write(f'''
SMART Gallery
=============

All pulsar detections from the SMART pulsar survey (these will be in other galleries).

''')
    for index, row in smart_df.iterrows():
        file.write(f'''

.. _{row["Pulsar"]}:

{row["Pulsar"]}
{"-"*len(row["Pulsar"])}
.. image:: best_fits/{row["Pulsar"]}_fit.png
  :width: 800
''')




# Remove unessiary columns
del df["Pulsar"]
del df["Model"]
del df["Probability Best"]
del df["Min freq (MHz)"]
del df["Max freq (MHz)"]
del df["N data flux"]

# Remove broken power law stuff
del df["vb"]
del df["a1"]
del df["a2"]
# for col_name in df.keys():
#     if "u_" in col_name or col_name.endswith("_c"):
#         del df[col_name]

f, ax = plt.subplots(figsize=(10, 5))
# Plot correlations
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True,
            cmap='seismic', mask=mask,
            vmin=-1, vmax=1,)
corr.to_csv(f"pulsar_coor.csv")
plt.tight_layout(pad=0.5)
plt.savefig("pulsar_coor.png")
plt.clf()

log_df = df[df["beta"] < 2.05]
# convert to log data
for col_name in log_df.keys():
    if "ATNF" in col_name or col_name in ("L400 (mJy kpc^2)", "L1400 (mJy kpc^2)", "Age (Yr)"):
        log_df[col_name + " log"] = np.log10(log_df[col_name])
        del log_df[col_name]
    elif col_name.startswith("v"):
        log_df[col_name + " log"] = np.log10(log_df[col_name])
        log_df["u_" + col_name + " log"] = log_df["u_" + col_name] / log_df[col_name]
        del log_df[col_name]

print(log_df.keys())

# Plot correlation
corr = log_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True,
            cmap='seismic', mask=mask,
            vmin=-1, vmax=1,)
corr.to_csv(f"pulsar_coor_log.csv")
plt.tight_layout(pad=0.5)
plt.savefig("pulsa_coor_log.png")

def weighted_correlation(x, y, yerr):
    """Weighted correlation based on:
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Weighted_correlation_coefficient
    """
    # weights based on y uncertainty
    w = 1 / yerr**2
    # weighted mean
    mx = np.sum( w * x ) / np.sum( w )
    my = np.sum( w * y ) / np.sum( w )
    # weighted covariance
    cov_xy = np.sum( w * (x - mx) * (y - my) ) / np.sum( w )
    cov_xx = np.sum( w * (x - mx) * (x - mx) ) / np.sum( w )
    cov_yy = np.sum( w * (y - my) * (y - my) ) / np.sum( w )
    # weighted correlation
    corr = cov_xy / np.sqrt( cov_xx * cov_yy )
    return corr

def yline(x, g, xmean, ymean):
    return ymean + g * (x - xmean)

def xline(y, g, ymean, xmean):
    return xmean + g * (y - ymean)

def line_fit(data_x, data_y, data_yerr, xcol, ycol, lax, weighted_corr):
    xmean=np.mean(data_x)
    ymean=np.mean(data_y)

    # fit y line
    least_squares = LeastSquares(data_x, data_y, data_yerr, yline)
    m = Minuit(least_squares, g=0, xmean=xmean, ymean=ymean)  # starting values for α and β
    m.fixed["xmean"] = True
    m.fixed["ymean"] = True
    m.migrad()  # finds minimum of least_squares function
    m.hesse()   # accurately computes uncertaintiesax):

    # # fit x line
    # least_squares = LeastSquares(data_y, data_x, data_x*0.001, xline)
    # mx = Minuit(least_squares, g=0, xmean=xmean, ymean=ymean)  # starting values for α and β
    # mx.fixed["xmean"] = True
    # mx.fixed["ymean"] = True
    # mx.migrad()  # finds minimum of least_squares function
    # mx.hesse()   # accurately computes uncertainties

    # m1 = my.values["g"]
    # m2 = mx.values["g"]

    # # Delta from the difference between regression lines
    # delta = math.atan( (m1 - m2) / (1 + m1 * m2) )

    # Delta from fot product
    delta = math.acos( np.dot(data_x, data_y) / ( np.linalg.norm(data_x) * np.linalg.norm(data_y) ) )
    # print(delta)
    #r = float(sec(delta) - math.tan(delta))
    r = (1 - np.sqrt(1 - np.cos(delta))) / np.cos(delta)
    #print(r)

    capsize = 1.5
    errorbar_linewidth = 0.7
    marker_border_thickness = 0.5
    # draw data and fitted line
    f, ax = plt.subplots()
    ax.errorbar( data_x, data_y, data_yerr, fmt="o", label="data")
    (_, caps, _) = lax.errorbar(data_x, data_y, data_yerr, fmt="o", label="data",
        markeredgewidth=marker_border_thickness,
        elinewidth=errorbar_linewidth,
        capsize=capsize,
        markersize=3,
    )
    for cap in caps:
        cap.set_markeredgewidth(errorbar_linewidth)

    # plot fit line and error bar
    fitted_line, fitted_line_cov = propagate(lambda p: yline(data_x, *p), m.values, m.covariance)
    fitted_line_err = abs(np.diag(fitted_line_cov) ** 0.5)
    print(f"error percent: {np.mean(fitted_line_err/fitted_line)}")
    ax.plot( data_x, fitted_line, label="fit")
    lax.plot(data_x, fitted_line, label="fit")
    ax.fill_between( data_x, fitted_line - fitted_line_err, fitted_line + fitted_line_err, facecolor="C1", alpha=0.5)
    lax.fill_between(data_x, fitted_line - fitted_line_err, fitted_line + fitted_line_err, facecolor="C1", alpha=0.5)
    # print(fitted_line)
    # print(fitted_line_err)

    # Labels
    ax.set_xlabel(xcol)
    lax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    lax.set_ylabel(ycol)
    ax.legend( title=f"Corr={weighted_corr:.3f}")
    lax.legend(title=f"Corr={weighted_corr:.3f}")
    if 'log' in xcol:
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, p:
                '$\mathdefault{10^{%i}}$' % x))
        lax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, p:
                '$\mathdefault{10^{%i}}$' % x))
        # ax.get_xaxis().set_major_formatter(FormatStrFormatter('%g'))
        # lax.get_xaxis().set_major_formatter(FormatStrFormatter('%g'))
    if 'log' in ycol:
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, p:
                '$\mathdefault{10^{%i}}$' % x))
        lax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, p:
                '$\mathdefault{10^{%i}}$' % x))
        ax.get_yaxis().set_major_formatter(FormatStrFormatter('%g'))
        lax.get_yaxis().set_major_formatter(FormatStrFormatter('%g'))
    ax.tick_params(which='both', direction='in', top=1, right=1)
    lax.tick_params(which='both', direction='in', top=1, right=1)

    # # display legend with some fit info
    # fit_info = [
    #     f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {len(data_x) - m.nfit}",
    # ]
    # for p, v, e in zip(m.parameters, m.values, m.errors):
    #     fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")

    # f.legend(title="\n".join(fit_info))
    f.savefig(f"corr_line_{xcol}_{ycol}.png".replace(" ", "_").replace("/", "_"))
    return r

# Make individual correlation plots
# df, x, y
corr_plots = [
    (log_df, "ATNF E_dot (ergs/s) log", "vpeak"),
]
xcols = [
    "ATNF Period (s) log",
    "ATNF DM log",
    "ATNF B_surf (G) log",
    "ATNF E_dot (ergs/s) log",
    "L400 (mJy kpc^2) log",
    "L1400 (mJy kpc^2) log",
    "Age (Yr) log",
]
ycols = [
        "a"         ,
        # "c"         ,
        # "vb"        ,
        # "a1"        ,
        # "a2"        ,
        "vc log"        ,
        "vpeak log"     ,
        "beta"      ,
]
nx = len(xcols)
ny = len(ycols)
sfig, saxes = plt.subplots(ny, nx, figsize=(5*nx, 5*ny))
lfig, laxes = plt.subplots(ny, nx, figsize=(5*nx, 5*ny))
corr_matrix = log_df.corr()
for xa, xcol in enumerate(xcols):
    for ya, ycol in enumerate(ycols):
        x = []
        y = []
        yerr = []
        for xi, yi, yerri in zip(list(log_df[xcol]), list(log_df[ycol]), list(log_df["u_"+ycol])):
            if (not np.isnan(xi)) and ( not np.isnan(yi) ) and ( not np.isnan(yerri) ) and (not np.isinf(xi)):
                x.append(xi)
                y.append(yi)
                yerr.append(yerri)
        x = np.array(x)
        y = np.array(y)
        yerr = np.array(yerr)
        print(f"\nCorrelations of x: {xcol} and y: {ycol}")
        print(f"-------------------------------------------------")
        print(f"pandas:        {corr_matrix[xcol][ycol]:6.3f}")
        print(f"numpy:         {np.corrcoef(x,y)[0][1]:6.3f}")
        weighted_corr = weighted_correlation(x, y ,yerr)
        print(f"weighted corr: {weighted_correlation(x, y ,yerr):6.3f}")
        rho, pval = stats.spearmanr(x, y, nan_policy="omit")
        print(f"spearmanr:     {rho:6.3f}")

        weighted_line_corr = line_fit(x, y, yerr, xcol, ycol, laxes[ya][xa], weighted_corr)
        print(f"line:          {weighted_line_corr:6.3f}")

        sns.regplot(data=log_df, x=xcol, y=ycol, ax=saxes[ya][xa])

sfig.savefig(f'coor_all.png', dpi=300)
lfig.savefig(f'corr_all_weighted.png', dpi=300)
