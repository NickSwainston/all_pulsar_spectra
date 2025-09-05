# Version comes from env or defaults
VERSION ?= 2.1.0
PWD := $(shell pwd)

# Default target
all: fit_all_pulsars

# Build image with version tag
build_$(VERSION): Dockerfile
	docker build --network host -t nickswainston/all_pulsar_spectra:$(VERSION) --build-arg PS_VERSION=$(VERSION) .
	touch $@   # mark this target as "done" (creates a stamp file)

# Run inside the container
fit_all_pulsars: build_$(VERSION) fit_all_pulsars.py
	docker run --rm --network host -v $(PWD):/root nickswainston/all_pulsar_spectra:$(VERSION) python /root/fit_all_pulsars.py