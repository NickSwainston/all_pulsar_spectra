#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

params.all         = false
params.pulsars     = ""
params.outdir      = "results"
params.loglvl      = "INFO"
params.version     = "2.1.0"
params.method      = "maximum-likelihood"
params.plot_type   = "best"
params.legend      = "raw"
params.likelihood  = "t"
params.cpus        = 1  // cores per fit job; also passed as --npool to quick-fit

// ---------------------------------------------------------------------------
// Process 1: find which pulsars to fit
// Runs only when --all is given; skipped when --pulsars provides the list.
// ---------------------------------------------------------------------------
process GET_PULSARS {
    output:
    path "pulsars.txt"

    script:
    """
    get_pulsars.py > pulsars.txt
    """
}

// ---------------------------------------------------------------------------
// Process 2: fit one pulsar and write a results YAML
// ---------------------------------------------------------------------------
process FIT_PULSAR {
    tag "$pulsar"
    cpus params.cpus
    publishDir "${params.outdir}/plots", mode: 'copy', pattern: "*.png"

    input:
    val pulsar

    output:
    path "${pulsar}_result.yaml" , emit: yaml
    path "${pulsar}_*.png"       , emit: plots, optional: true

    script:
    """
    quick-fit \
        -p ${pulsar} \
        -o ${pulsar}_result.yaml \
        -L ${params.loglvl} \
        -m ${params.method} \
        -t ${params.plot_type} \
        -s ${params.legend} \
        -l ${params.likelihood} \
        -n ${task.cpus}
    """
}

// ---------------------------------------------------------------------------
// Process 3: merge all YAMLs into a single CSV
// ---------------------------------------------------------------------------
process COMBINE_RESULTS {
    publishDir params.outdir, mode: 'copy'

    input:
    path yaml_files

    output:
    path "all_pulsar_fits.csv"

    script:
    """
    combine_results.py *.yaml -o all_pulsar_fits.csv
    """
}

// ---------------------------------------------------------------------------
// Workflow
// ---------------------------------------------------------------------------
workflow {
    if (params.pulsars) {
        // Explicit list passed as --pulsars J0437-4715,J0738-4042,...
        pulsars_ch = channel.from(params.pulsars.tokenize(','))
    } else if (params.all) {
        // Discover all pulsars with >= 4 flux measurements
        pulsars_ch = GET_PULSARS()
            | splitText()
            | map { line -> line.trim() }
    } else {
        error("Provide either --all to fit all pulsars with >=4 fluxes, or --pulsars <comma-separated list>")
    }

    fit_ch = FIT_PULSAR(pulsars_ch)
    COMBINE_RESULTS(fit_ch.yaml.collect())
}
