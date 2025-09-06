# Version comes from env or defaults
VERSION ?= 2.1.0
PWD := $(shell pwd)

# Default target
all: compile_docs

# Build image with version tag
build_$(VERSION): Dockerfile
	docker build --network host -t nickswainston/all_pulsar_spectra:$(VERSION) --build-arg PS_VERSION=$(VERSION) .
	touch $@   # mark this target as "done" (creates a stamp file)

# Run the spectral fits with multithreading
fit_all_pulsars: build_$(VERSION) fit_all_pulsars.py
	docker run --rm --network host -v $(PWD):/root nickswainston/all_pulsar_spectra:$(VERSION) python /root/fit_all_pulsars.py

# Process the results to make plots, tables and doc pages
make_docs: fit_all_pulsars all_pulsar_fits.csv make_docs.py
	docker run --rm --network host -v $(PWD):/root nickswainston/all_pulsar_spectra:$(VERSION) python /root/make_docs.py

compile_docs: make_docs docs/index.rst
	uv venv .venv  --allow-existing && uv pip install -r docs/requirements.txt && uv run sphinx-build docs html
