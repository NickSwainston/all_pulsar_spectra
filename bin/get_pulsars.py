#!/usr/bin/env python3
"""Print all pulsar J-names that have at least 4 flux measurements, one per line."""

from pulsar_spectra.catalogue import collect_catalogue_fluxes

cat_dict = collect_catalogue_fluxes()
for pulsar, entry in cat_dict.items():
    freq_all = entry[0]
    if len(freq_all) >= 4:
        print(pulsar)
