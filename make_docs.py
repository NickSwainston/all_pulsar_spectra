import pandas as pd
import os

df = pd.read_csv('all_pulsar_fits.csv')

# grab model specific data frames
spl_df  = df[df["Model"] == "simple_power_law"]
bpl_df  = df[df["Model"] == "broken_power_law"]
lps_df  = df[df["Model"] == "log_parabolic_spectrum"]
hfto_df = df[df["Model"] == "high_frequency_cut_off_power_law"]
lfto_df = df[df["Model"] == "low_frequency_turn_over_power_law"]
dtos_df = df[df["Model"] == "double_turn_over_spectrum"]
nomod_df = df[df["Model"] == ""]

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
   lps_gallery
   lfto_gallery
   hfco_gallery
   dtos_gallery

Single Power Law Results
------------------------
.. csv-table::
   :header: "Pulsar", "a"

''')
    for index, row in spl_df.iterrows():
        data_str = f'   "{row["Pulsar"]}", '
        for val, error in [("pl_a", "pl_u_a")]:
            if "_v" in val:
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
        data_str = f'   "{row["Pulsar"]}", '
        for val, error in [("bpl_vb", "bpl_u_vb"), ("bpl_a1", "bpl_u_a1"), ("bpl_a2","bpl_u_a2")]:
            if "_v" in val:
                data_str += f'"{int(row[val]/1e6):d}±{int(row[error]/1e6):d}", '
            else:
                data_str += f'"{row[val]:.2f}±{row[error]:.2f}", '
        file.write(f'{data_str[:-2]}\n')

    file.write(f'''


Log Parabolic Spectrum Results
------------------------------
.. csv-table::
   :header: "Pulsar", "vpeak (MHz)", "a", "b", "c"

''')
    for index, row in lps_df.iterrows():
        data_str = f'   "{row["Pulsar"]}", '
        for val, error in [("lps_v_peak", "lps_u_v_peak"), ("lps_a", "lps_u_a"), ("lps_b", "lps_u_b"), ("lps_c", "lps_u_c")]:
            if "_v" in val:
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
        data_str = f'   "{row["Pulsar"]}", '
        for val, error in [("lfto_vpeak", "lfto_u_vpeak"), ("lfto_a", "lfto_u_a"), ("lfto_beta", "lfto_u_beta")]:
            if "_v" in val:
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
        data_str = f'   "{row["Pulsar"]}", '
        for val, error in [("hfco_vc", "hfco_u_vc"), ("hfco_a", "hfco_u_a")]:
            if "_v" in val:
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
        data_str = f'   "{row["Pulsar"]}", '
        for val, error in [("dtos_vc", "dtos_u_vc"), ("dtos_vpeak", "dtos_u_vpeak"), ("dtos_a", "dtos_u_a"), ("dtos_beta", "dtos_u_beta")]:
            if "_v" in val:
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
        file.write(f'   "{row["Pulsar"]}", "{row["N data flux"]}"')


# Set up the gallerys
with open(f'{os.path.dirname(os.path.realpath(__file__))}/docs/spl_gallery.rst', 'w') as file:
    file.write(f'''
Simple Power Law Gallery
========================

''')
    for index, row in spl_df.iterrows():
        file.write(f'''

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

{row["Pulsar"]}
{"-"*len(row["Pulsar"])}
.. image:: best_fits/{row["Pulsar"]}_fit.png
  :width: 800
''')

with open(f'{os.path.dirname(os.path.realpath(__file__))}/docs/lps_gallery.rst', 'w') as file:
    file.write(f'''
Log Parabolic Spectrum Gallery
==============================

''')
    for index, row in lps_df.iterrows():
        file.write(f'''

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

{row["Pulsar"]}
{"-"*len(row["Pulsar"])}
.. image:: best_fits/{row["Pulsar"]}_fit.png
  :width: 800
''')

