# SPDX-FileCopyrightText: : 2017-2020 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Solves linear optimal dispatch in hourly resolution
using the capacities of previous capacity expansion in rule :mod:`solve_network`.

Relevant Settings
-----------------

.. code:: yaml

    solving:
        tmpdir:
        options:
            formulation:
            clip_p_max_pu:
            load_shedding:
            noisy_costs:
            nhours:
            min_iterations:
            max_iterations:
        solver:
            name:
            (solveroptions):

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`solving_cf`

Inputs
------

- ``networks/elec_s{simpl}_{clusters}.nc``: confer :ref:`cluster`
- ``results/networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: confer :ref:`solve`

Outputs
-------

- ``results/networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_op.nc``: Solved PyPSA network for optimal dispatch including optimisation results

Description
-----------

"""

import logging
from _helpers import configure_logging

import pypsa
import numpy as np

from pathlib import Path
from vresutils.benchmark import memory_logger
from solve_network import solve_network, prepare_network
import pandas as pd

logger = logging.getLogger(__name__)

def set_parameters_from_optimized(n, nodal_capacities):

    links_dc_i = n.links.index[n.links.p_nom_extendable]
    links_capacities = nodal_capacities.loc['links']
    n.links.loc[links_dc_i, 'p_nom'] = links_capacities.loc[links_dc_i, "robust_capacities"]
    n.links.loc[links_dc_i, 'p_nom_extendable'] = False

    gen_extend_i = n.generators.index[n.generators.p_nom_extendable]
    gen_capacities = nodal_capacities.loc['generators']
    # gen_extend_i_exclude_biomass = [elem for i, elem in enumerate(gen_extend_i) if elem not in biomass_extend_index]
    n.generators.loc[gen_extend_i, 'p_nom'] = gen_capacities.loc[gen_extend_i, "robust_capacities"]
    n.generators.loc[gen_extend_i, 'p_nom_extendable'] = False


    stor_extend_i = n.storage_units.index[n.storage_units.p_nom_extendable]
    stor_capacities = nodal_capacities.loc['storage_units']
    n.storage_units.loc[stor_extend_i, 'p_nom'] = stor_capacities.loc[stor_extend_i, "robust_capacities"]
    n.storage_units.loc[stor_extend_i, 'p_nom_extendable'] = False

    stores_extend_i = n.stores.index[n.stores.e_nom_extendable]
    stores_capacities = nodal_capacities.loc['stores']
    n.stores.loc[stores_extend_i, 'e_nom'] = stores_capacities.loc[stores_extend_i, "robust_capacities"]
    n.stores.loc[stores_extend_i, 'e_nom_extendable'] = False

    return n

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('solve_operations_network', network='elec',
                                  simpl='', clusters='5', ll='copt', opts='')
    configure_logging(snakemake)

    tmpdir = snakemake.config['solving'].get('tmpdir')
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)

    n = pypsa.Network(snakemake.input.unprepared)
    robust_capacities = pd.read_csv(snakemake.input.capacities,index_col=list(range(2)),header=[0])

    n = set_parameters_from_optimized(n, robust_capacities)
   # del n_optim

    config = snakemake.config
    opts = snakemake.wildcards.opts.split('-')
    #config['solving']['options']['skip_iterations'] = False
    config['solving']['options']['skip_iterations'] = True

    fn = getattr(snakemake.log, 'memory', None)
    with memory_logger(filename=fn, interval=30.) as mem:
        n = prepare_network(n, solve_opts=snakemake.config['solving']['options'])
        n = solve_network(n, config, solver_dir=tmpdir,
                          solver_log=snakemake.log.solver, opts=opts)
        n.export_to_netcdf(snakemake.output[0])

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
