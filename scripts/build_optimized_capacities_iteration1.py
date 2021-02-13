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
from six import iteritems
import pandas as pd
import os

logger = logging.getLogger(__name__)

idx = pd.IndexSlice

opt_name = {"Store": "e", "Line" : "s", "Transformer" : "s"}
capacity_years = ['2014','2015']

def assign_carriers(n):
    if "carrier" not in n.lines:
        n.lines["carrier"] = "AC"


def assign_locations(n):
    for c in n.iterate_components(n.one_port_components|n.branch_components):
        ifind = pd.Series(c.df.index.str.find(" ",start=4),c.df.index)
        for i in ifind.unique():
            names = ifind.index[ifind == i]
            if i == -1:
                c.df.loc[names,'location'] = ""
            else:
                c.df.loc[names,'location'] = names.str[:i]

def calculate_nodal_capacities(networks_dict):
    #Beware this also has extraneous locations for country (e.g. biomass) or continent-wide (e.g. fossil gas/oil) stuff
    # networks_dict = {(year):'results/networks/{year}/elec_s_40_ec_lv1.0_Co2L0p0-3H_storage_units.nc' \
    #                          .format(year=year) for year in capacity_years}
    columns = list(networks_dict.keys())
    nodal_capacities = pd.DataFrame(columns=columns)
    lines_capacities = pd.DataFrame()

    for label, filename in iteritems(networks_dict):
        if not os.path.exists(filename):
            continue
        n = pypsa.Network(filename)
        assign_carriers(n)
        assign_locations(n)

        for c in n.iterate_components(n.branch_components ^ {"Transformer"}| n.controllable_one_port_components ^ {"Load"}):
            #nodal_capacities_c = c.df.groupby(["location", "carrier"])[opt_name.get(c.name, "p") + "_nom_opt"].sum()
            nodal_capacities_c = c.df[opt_name.get(c.name, "p") + "_nom_opt"]
           # print([(c.list_name,) + t for t in nodal_capacities_c.index])
            index = pd.MultiIndex.from_tuples([(c.list_name,t) for t in nodal_capacities_c.index])
            nodal_capacities = nodal_capacities.reindex(index | nodal_capacities.index)
            nodal_capacities.loc[index, label] = nodal_capacities_c.values
        # df = pd.concat([nodal_capacities, df], axis=1)
    nodal_capacities.to_csv("notebook/data/nodal_capacities.csv")
    return nodal_capacities


def set_parameters_from_optimized(n, networks_dict):
    nodal_capacities = calculate_nodal_capacities(networks_dict)


    # lines_typed_i = n.lines.index[n.lines.type != '']
    # n.lines.loc[lines_typed_i, 'num_parallel'] = \
    #     n_optim.lines['num_parallel'].reindex(lines_typed_i, fill_value=0.)
    # n.lines.loc[lines_typed_i, 's_nom'] = (
    #     np.sqrt(3) * n.lines['type'].map(n.line_types.i_nom) *
    #     n.lines.bus0.map(n.buses.v_nom) * n.lines.num_parallel)
    #
    # lines_untyped_i = n.lines.index[n.lines.type == '']
    # for attr in ('s_nom', 'r', 'x'):
    #     n.lines.loc[lines_untyped_i, attr] = n_optim.lines[attr].reindex(lines_untyped_i, fill_value=0.)
    # n.lines['s_nom_extendable'] = False

    # lines_extend_i = n.lines.index[n.lines.s_nom_extendable]
    # lines_capacities = nodal_capacities.loc['lines']
    # print(lines_capacities)
    # lines_capacities = lines_capacities.reset_index(level=[1]).reindex(lines_extend_i, fill_value=0.)
    # n.lines.loc[lines_extend_i, 's_nom_max'] = lines_capacities.loc[lines_extend_i,:].max(axis=1)
    # n.lines.loc[lines_extend_i, 's_nom_min'] = lines_capacities.loc[lines_extend_i, :].mean(axis=1)
    # n.lines.loc[lines_extend_i, 's_nom_extendable'] = False

    links_dc_i = n.links.index[n.links.carrier == 'DC']
    links_capacities = nodal_capacities.loc['links']
    n.links.loc[links_dc_i, 'p_nom_max'] = links_capacities.loc[links_dc_i,:].max(axis=1)
    n.links.loc[links_dc_i, 'p_nom_min'] = links_capacities.loc[links_dc_i,:].mean(axis=1)
   # n.links.loc[links_dc_i, 'p_nom_extendable'] = False

    gen_extend_i = n.generators.index[n.generators.p_nom_extendable]
    gen_capacities = nodal_capacities.loc['generators']
    #gen_capacities = gen_capacities.reset_index(level=[1]).reindex(gen_extend_i, fill_value=0.)
    n.generators.loc[gen_extend_i, 'p_nom_max'] = gen_capacities.loc[gen_extend_i,:].max(axis=1)
    n.generators.loc[gen_extend_i, 'p_nom_min'] = gen_capacities.loc[gen_extend_i,:].mean(axis=1)
   # n.generators.loc[gen_extend_i, 'p_nom_extendable'] = False

    stor_extend_i = n.storage_units.index[n.storage_units.p_nom_extendable]
    stor_capacities = nodal_capacities.loc['storage_units']
    n.storage_units.loc[stor_extend_i, 'p_nom_max'] = stor_capacities.loc[stor_extend_i,:].max(axis=1)
    n.storage_units.loc[stor_extend_i, 'p_nom_min'] = stor_capacities.loc[stor_extend_i, :].mean(axis=1)
   # n.storage_units.loc[stor_extend_i, 'p_nom_extendable'] = False

    stores_extend_i = n.stores.index[n.stores.e_nom_extendable]
    stores_capacities = nodal_capacities.loc['stores']
    n.stores.loc[stores_extend_i, 'e_nom_max'] = stores_capacities.loc[stores_extend_i,:].max(axis=1)
    n.stores.loc[stores_extend_i, 'e_nom_min'] = stores_capacities.loc[stores_extend_i, :].mean(axis=1)
   # n.stores.loc[stores_extend_i, 'e_nom_extendable'] = False
    return n

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_optimized_capacities_iteration1', network='elec', simpl='',
                           clusters='5', ll='copt', opts='Co2L-24H', capacitiy_years='2013')
        network_dir = os.path.join('..', 'results', 'networks')
    else:
        network_dir = os.path.join('results', 'networks')
    configure_logging(snakemake)

    def expand_from_wildcard(key):
        w = getattr(snakemake.wildcards, key)
        return snakemake.config["scenario"][key] if w == "all" else [w]

    tmpdir = snakemake.config['solving'].get('tmpdir')
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
    if snakemake.wildcards.ll.endswith("all"):
        ll = snakemake.config["scenario"]["ll"]
        if len(snakemake.wildcards.ll) == 4:
            ll = [l for l in ll if l[0] == snakemake.wildcards.ll[0]]
    else:
        ll = [snakemake.wildcards.ll]
    networks = snakemake.input.solved_networks

    networks_dict = {(capacity_year) :
        os.path.join(network_dir, 'iteration0', f'elec_s{simpl}_'
                                  f'{clusters}_ec_l{l}_{opts}_{capacity_year}')
                     for capacity_year in snakemake.config["scenario"]["capacity_years"]
                     for simpl in expand_from_wildcard("simpl")
                     for clusters in expand_from_wildcard("clusters")
                     for l in ll
                     for opts in expand_from_wildcard("opts")}
    print(networks_dict)
    #
    #
    n = pypsa.Network(snakemake.input.unprepared)
    # n_optim = pypsa.Network(snakemake.input.optimized)
    n = set_parameters_from_optimized(n, networks_dict)
    #del n_optim

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

