# Define pulsar_spectra version as a build argument with a default value
ARG PS_VERSION=2.1.0

FROM nickswainston/pulsar_spectra:${PS_VERSION}

RUN uv pip install tqdm
