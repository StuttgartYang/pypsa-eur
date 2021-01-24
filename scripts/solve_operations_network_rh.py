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

import numpy as np
import pandas as pd
import re

import pypsa
from pypsa.linopf import (get_var, define_constraints, linexpr, join_exprs,
                          network_lopf, ilopf)
from vresutils.benchmark import memory_logger
from solve_network import solve_network, prepare_network, extra_functionality

logger = logging.getLogger(__name__)

def set_parameters_from_optimized(n, n_optim):
    lines_typed_i = n.lines.index[n.lines.type != '']
    n.lines.loc[lines_typed_i, 'num_parallel'] = \
        n_optim.lines['num_parallel'].reindex(lines_typed_i, fill_value=0.)
    n.lines.loc[lines_typed_i, 's_nom'] = (
        np.sqrt(3) * n.lines['type'].map(n.line_types.i_nom) *
        n.lines.bus0.map(n.buses.v_nom) * n.lines.num_parallel)

    lines_untyped_i = n.lines.index[n.lines.type == '']
    for attr in ('s_nom', 'r', 'x'):
        n.lines.loc[lines_untyped_i, attr] = \
            n_optim.lines[attr].reindex(lines_untyped_i, fill_value=0.)
    n.lines['s_nom_extendable'] = False

    links_dc_i = n.links.index[n.links.carrier == 'DC']
    n.links.loc[links_dc_i, 'p_nom'] = \
        n_optim.links['p_nom_opt'].reindex(links_dc_i, fill_value=0.)
    n.links.loc[links_dc_i, 'p_nom_extendable'] = False

    gen_extend_i = n.generators.index[n.generators.p_nom_extendable]
    n.generators.loc[gen_extend_i, 'p_nom'] = \
        n_optim.generators['p_nom_opt'].reindex(gen_extend_i, fill_value=0.)
    n.generators.loc[gen_extend_i, 'p_nom_extendable'] = False

    stor_extend_i = n.storage_units.index[n.storage_units.p_nom_extendable]
    n.storage_units.loc[stor_extend_i, 'p_nom'] = \
        n_optim.storage_units['p_nom_opt'].reindex(stor_extend_i, fill_value=0.)
    n.storage_units.loc[stor_extend_i, 'p_nom_extendable'] = False

    stores_extend_i = n.stores.index[n.stores.e_nom_extendable]
    n.stores.loc[stores_extend_i, 'e_nom'] = n_optim.stores['e_nom_opt'].reindex(stores_extend_i,fill_value=0.)
    n.stores.loc[stores_extend_i, 'e_nom_extendable'] = False

    n.stores.e_cyclic = False
    n.storage_units.cyclic_state_of_charge=False

    return n


# def get_h2_bid(n, snapshots, node, link_charge, link_discharge):
#     λ = n.buses_t.marginal_price.loc[snapshots, node+ " H2"].mean()
#     charge_efficiency = n.links.efficiency[node + link_charge]
#     discharge_efficiency = n.links.efficiency[node + link_discharge]
#
#     mu = n.stores_t.mu_state_of_charge.loc[snapshots, node + " H2"].sum()
#     #mu = n.stores_t.mu_state_of_charge.loc[snapshots, node + " H2"].mean()
#
#     charging_bid = charge_efficiency * (λ + mu)
#     discharging_bid = (λ + mu) / discharge_efficiency
#     return charging_bid, discharging_bid

def get_h2_bid(n, snapshots, node, link_charge, link_discharge):
    λ = n.buses_t.marginal_price.loc[snapshots, node+ " H2"]
    charge_efficiency = n.links.efficiency[node + link_charge]
    discharge_efficiency = n.links.efficiency[node + link_discharge]
    charging_bid = charge_efficiency * λ
    discharging_bid = λ / discharge_efficiency
    return charging_bid, discharging_bid


def get_hydro_bid(n, snapshots, node):
    λ = n.storage_units_t.mu_state_of_charge.loc[snapshots, node + " hydro"].mean()
    discharge_efficiency = n.storage_units.loc[node + " hydro", "efficiency_dispatch"]
    mu_lower = n.storage_units_t.mu_lower_state_of_charge.loc[snapshots, node + " hydro"].sum()
    mu_upper = n.storage_units_t.mu_upper_state_of_charge.loc[snapshots, node + " hydro"].sum()
    discharging_bid = (λ + (mu_lower - mu_upper)) / discharge_efficiency
    return discharging_bid


def get_store_demand_bid(n,  carrier, snapshots, node, link_charger, link_discharger):
    λ = n.buses_t.marginal_price.loc[:, node + " " + carrier]
    # if carrier == "hydro":
    #     n.links.efficiency[node + link_charger] = 1
    charge_efficiency = n.links.efficiency[node + link_charger]
    discharge_efficiency = n.links.efficiency[node + link_discharger]
    if node + " " + carrier in n.stores_t.mu_lower_e.columns:
        mu_lower = n.stores_t.mu_lower_e.loc[snapshots, node + " " + carrier]
    else: mu_lower =0
    if node + " " + carrier in n.stores_t.mu_upper_e.columns:
        mu_upper = n.stores_t.mu_upper_e.loc[snapshots, node + " " + carrier]
    else: mu_upper=0
    charging_bid = charge_efficiency * λ
    discharging_bid = λ / discharge_efficiency
    return charging_bid, discharging_bid

# def get_hydro_bid(n, snapshots, node):
#     λ = n.buses_t.marginal_price.loc[snapshots, node].mean()
#     discharge_efficiency = n.storage_units.loc[node + " hydro", "efficiency_dispatch"]
#     mu = n.storage_units_t.mu_state_of_charge.loc[snapshots, node + " hydro"].sum()
#     discharging_bid = (λ + mu) / discharge_efficiency
#     return discharging_bid

def solve_network_rh(n, config, solver_log=None, opts='', period='', **kwargs):
    solver_options = config['solving']['solver'].copy()
    solver_name = solver_options.pop('name')
    cf_solving = config['solving']['options']
    track_iterations = cf_solving.get('track_iterations', False)
    min_iterations = cf_solving.get('min_iterations', 4)
    max_iterations = cf_solving.get('max_iterations', 6)

    # add to network for extra_functionality
    n.config = config
    n.opts = opts
    freq = 1
    length = len(n.snapshots)
    if period == "3h":
        loop_num = length - 1
        kept = 1
        overlap = 0

    elif period == "2w":
        window = 14 * 24 // freq
        overlap = 7 * 24 // freq
        kept = window - overlap
        loop_num = length // kept
    else:
        print( "period should be 3h or 2w")

    nodelist = n.buses[n.buses.carrier == "AC"].index


    for i in range(loop_num):
        # set initial state of charge
        snapshots = n.snapshots[i * kept:(i + 1) * kept + overlap]
        print(snapshots)
        for carrier in n.stores.carrier.unique():
            if carrier == "H2":
                link_charger = " H2 Electrolysis"
                link_discharger = " H2 Fuel Cell"
            else:
                link_charger = " " + carrier + " charger"
                link_discharger = " " + carrier + " discharger"
            for node in nodelist:
                if node + " " + carrier in n_optim.buses.index:
                    charging_bid, discharging_bid = get_store_demand_bid(n_optim, carrier, snapshots, node,
                                                                         link_charger=link_charger,
                                                                         link_discharger=link_discharger)
                    n.links_t.marginal_cost[node + link_charger] = 0
                    n.links_t.marginal_cost.loc[snapshots, node + link_charger] = - charging_bid
                    n.links_t.marginal_cost[node + link_discharger] = 0
                    n.links_t.marginal_cost.loc[snapshots, node + link_discharger] = discharging_bid
        if i == 0:
            n.stores.e_initial = n_optim.stores_t.e.iloc[0]
            n.storage_units.state_of_charge_initial = n_optim.storage_units_t.state_of_charge.iloc[-1]
        else:

            n.stores.e_initial = n.stores_t.e.iloc[i * kept - 1]
            n.storage_units.state_of_charge_initial = n.storage_units_t.state_of_charge.iloc[i * kept - 1]
        if cf_solving.get('skip_iterations', False):
            network_lopf(n, snapshots, solver_name=solver_name, solver_options=solver_options, keep_shadowprices=True,
                         extra_functionality=extra_functionality, **kwargs)
        else:
            ilopf(n, snapshots, solver_name=solver_name, solver_options=solver_options,
                  track_iterations=track_iterations,
                  min_iterations=min_iterations,
                  max_iterations=max_iterations,
                  extra_functionality=extra_functionality, **kwargs)


    return n


if __name__ == "__main__":
    from _helpers import mock_snakemake
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('solve_operations_network_rh', network='elec',
                                   simpl='', clusters='40', ll='v1.0', opts='Co2L0p0-3H', period='2w')
    configure_logging(snakemake)

    tmpdir = snakemake.config['solving'].get('tmpdir')
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)

    n = pypsa.Network(snakemake.input.unprepared)
    n_optim = pypsa.Network(snakemake.input.optimized)
    # n = pypsa.Network("../networks/2014/elec_s_40_ec_lv1.0_Co2L0p0-3H.nc")
    # n_optim = pypsa.Network("../results/networks/2014/elec_s_40_ec_lv1.0_Co2L0p0-3H_op.nc")

    n = set_parameters_from_optimized(n, n_optim)
    #del n_optim

    config = snakemake.config
    opts = snakemake.wildcards.opts.split('-')
    period = snakemake.wildcards.period
    #config['solving']['options']['skip_iterations'] = False
    config['solving']['options']['skip_iterations'] = True

    fn = getattr(snakemake.log, 'memory', None)
    with memory_logger(filename=fn, interval=30.) as mem:
        n = prepare_network(n, solve_opts=snakemake.config['solving']['options'])
        n = solve_network_rh(n, config, solver_dir=tmpdir,
                          solver_log=snakemake.log.solver, opts=opts, period = period)
        n.export_to_netcdf(snakemake.output[0])

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
