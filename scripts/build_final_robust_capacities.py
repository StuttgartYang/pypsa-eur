
import logging
from _helpers import configure_logging

import pypsa
import numpy as np

from pathlib import Path
from vresutils.benchmark import memory_logger
from solve_network import solve_network, prepare_network
from six import iteritems
import pandas as pd
import os
from build_robust_capacities import calculate_nodal_capacities
logger = logging.getLogger(__name__)

idx = pd.IndexSlice


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_robust_capacities', network='elec', simpl='',
                           clusters='5', ll='copt', opts='Co2L-24H', capacitiy_years='2013')
        network_dir = os.path.join('..', 'results', 'networks','robust_capacities_extra_generator')
    else:
        network_dir = os.path.join('results', 'networks', 'robust_capacities_extra_generator')
    configure_logging(snakemake)

    def expand_from_wildcard(key):
        w = getattr(snakemake.wildcards, key)
        return snakemake.config["scenario"][key] if w == "all" else [w]

    tmpdir = snakemake.config['solving'].get('tmpdir')
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)

    networks_dict = {(capacity_year) :
        os.path.join(network_dir, f'elec_s{simpl}_'
                                  f'{clusters}_ec_l{l}_{opts}_{capacity_year}.nc')
                     for capacity_year in snakemake.config["scenario"]["capacity_years"]
                     for simpl in snakemake.config["scenario"]["simpl"]
                     for clusters in snakemake.config["scenario"]["clusters"]
                     for l in snakemake.config["scenario"]["ll"]
                     for opts in snakemake.config["scenario"]["opts"]}
    print(networks_dict)

    nodal_capacities = calculate_nodal_capacities(networks_dict)
    nodal_capacities["robust_capacities"] = nodal_capacities.mean(axis=1)
    nodal_capacities.to_csv(snakemake.output[0])
