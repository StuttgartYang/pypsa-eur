# SPDX-FileCopyrightText: : 2017-2020 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Solves linear optimal power flow for a network iteratively while updating reactances.

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
            skip_iterations:
            track_iterations:
        solver:
            name:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`electricity_cf`, :ref:`solving_cf`, :ref:`plotting_cf`

Inputs
------

- ``networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: confer :ref:`prepare`

Outputs
-------

- ``results/networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: Solved PyPSA network including optimisation results

    .. image:: ../img/results.png
        :scale: 40 %

Description
-----------

Total annual system costs are minimised with PyPSA. The full formulation of the
linear optimal power flow (plus investment planning
is provided in the
`documentation of PyPSA <https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#linear-optimal-power-flow>`_.
The optimization is based on the ``pyomo=False`` setting in the :func:`network.lopf` and  :func:`pypsa.linopf.ilopf` function.
Additionally, some extra constraints specified in :mod:`prepare_network` are added.

Solving the network in multiple iterations is motivated through the dependence of transmission line capacities and impedances.
As lines are expanded their electrical parameters change, which renders the optimisation bilinear even if the power flow
equations are linearized.
To retain the computational advantage of continuous linear programming, a sequential linear programming technique
is used, where in between iterations the line impedances are updated.
Details (and errors made through this heuristic) are discussed in the paper

- Fabian Neumann and Tom Brown. `Heuristics for Transmission Expansion Planning in Low-Carbon Energy System Models <https://arxiv.org/abs/1907.10548>`_), *16th International Conference on the European Energy Market*, 2019. `arXiv:1907.10548 <https://arxiv.org/abs/1907.10548>`_.

.. warning::
    Capital costs of existing network components are not included in the objective function,
    since for the optimisation problem they are just a constant term (no influence on optimal result).

    Therefore, these capital costs are not included in ``network.objective``!

    If you want to calculate the full total annual system costs add these to the objective value.

.. tip::
    The rule :mod:`solve_all_networks` runs
    for all ``scenario`` s in the configuration file
    the rule :mod:`solve_network`.

"""

import logging
from _helpers import configure_logging

import numpy as np
import pandas as pd
import re

import pypsa
from pypsa.linopf import (get_var, define_constraints, linexpr, join_exprs,
                          network_lopf, ilopf)

from pathlib import Path
from vresutils.benchmark import memory_logger

from pyomo.environ import Constraint

logger = logging.getLogger(__name__)

def leap_year(y):
    if y % 400 == 0:
        return True
    if y % 100 == 0:
        return False
    if y % 4 == 0:
        return True
    else:
        return False

def cut_snapshots(n):
    df = pd.DataFrame(n.snapshots)
    df['datetime'] = pd.to_datetime(df['name'])
    df = df.set_index('datetime')
    df.drop(['name'], axis=1, inplace=True)
    date = str(n.snapshots[0].year) + "-02-29"
    to_drop = df[date]
    df.drop(to_drop.index, axis=0, inplace=True)
    n.set_snapshots(df.index)
    return n


def replace_su(network, su_to_replace):
    """Replace the storage unit su_to_replace with a bus for the energy
    carrier, two links for the conversion of the energy carrier to and from electricity,
    a store to keep track of the depletion of the energy carrier and its
    CO2 emissions, and a variable generator for the storage inflow.

    Because the energy size and power size are linked in the storage unit by the max_hours,
    extra functionality must be added to the LOPF to implement this constraint."""

    su = network.storage_units.loc[su_to_replace]

    bus_name = "{} {}".format(su["bus"], su["carrier"])

    link_1_name = "{} discharger".format(su_to_replace, su["carrier"])

    link_2_name = "{} charger".format(su_to_replace, su["carrier"])

    store_name = "{} store {}".format(su_to_replace, su["carrier"])

    gen_name = "{} inflow".format(su_to_replace)

    network.add("Bus",
                bus_name,
                carrier=su["carrier"])

    # dispatch link
    network.add("Link",
                link_1_name,
                bus0=bus_name,
                bus1=su["bus"],
                carrier=su["carrier"] + " discharger",
                capital_cost=su["capital_cost"] * su["efficiency_dispatch"],
                p_nom=su["p_nom"] / su["efficiency_dispatch"],
                p_nom_extendable=su["p_nom_extendable"],
                p_nom_max=su["p_nom_max"] / su["efficiency_dispatch"],
                p_nom_min=su["p_nom_min"] / su["efficiency_dispatch"],
                p_max_pu=su["p_max_pu"],
                marginal_cost=su["marginal_cost"] * su["efficiency_dispatch"],
                efficiency=su["efficiency_dispatch"])


    # store link
    network.add("Link",
                link_2_name,
                bus1=bus_name,
                bus0=su["bus"],
                carrier=su["carrier"] + " charger",
                p_nom=su["p_nom"],
                p_nom_extendable=su["p_nom_extendable"],
                p_nom_max=su["p_nom_max"],
                p_nom_min=su["p_nom_min"],
                p_max_pu=-su["p_min_pu"],
                efficiency=su["efficiency_store"])
    network.links.loc[link_1_name, 'carrier'] = su["carrier"] + " discharger"
    network.links.loc[link_2_name, 'carrier'] = su["carrier"] + " charger"

    if su_to_replace in network.storage_units_t.state_of_charge_set.columns and (
            ~pd.isnull(network.storage_units_t.state_of_charge_set[su_to_replace])).any():
        e_max_pu = pd.Series(data=1., index=network.snapshots)
        e_min_pu = pd.Series(data=0., index=network.snapshots)
        non_null = ~pd.isnull(network.storage_units_t.state_of_charge_set[su_to_replace])
        e_max_pu[non_null] = network.storage_units_t.state_of_charge_set[su_to_replace][non_null]
        e_min_pu[non_null] = network.storage_units_t.state_of_charge_set[su_to_replace][non_null]
    else:
        e_max_pu = 1.
        e_min_pu = 0.

    network.add("Store",
                store_name,
                bus=bus_name,
                carrier=su["carrier"],
                e_nom=su["p_nom"] * su["max_hours"],
                e_nom_min=su["p_nom_min"] / su["efficiency_dispatch"] * su["max_hours"],
                e_nom_max=su["p_nom_max"] / su["efficiency_dispatch"] * su["max_hours"],
                e_nom_extendable=su["p_nom_extendable"],
                e_max_pu=e_max_pu,
                e_min_pu=e_min_pu,
                standing_loss=su["standing_loss"],
                e_cyclic=su['cyclic_state_of_charge'],
                e_initial=su['state_of_charge_initial'])
    network.stores.loc[store_name, 'carrier'] = su["carrier"]

    #     network.add("Carrier",
    #                 "rain",
    #                 co2_emissions=0.)

    # inflow from a variable generator, which can be curtailed (i.e. spilled)
    if su_to_replace in network.storage_units_t.inflow.columns:
        inflow_max = network.storage_units_t.inflow[su_to_replace].max()

        if inflow_max == 0.:
            inflow_pu = 0.
        else:
            inflow_pu = network.storage_units_t.inflow[su_to_replace] / inflow_max
    else:
        inflow_max = 0.
        inflow_pu = 0.

    network.add("Generator",
                gen_name,
                bus=bus_name,
                carrier=su["carrier"],
                p_nom=inflow_max,
                p_max_pu=inflow_pu)
    network.generators.loc[gen_name, 'carrier'] = su["carrier"]

    if su["p_nom_extendable"]:
        ratio2 = su["max_hours"]
        ratio1 = ratio2 * su["efficiency_dispatch"]

        def extra_functionality(network, snapshots):
            model = network.model
            model.store_fix_1 = Constraint(
                rule=lambda model: model.store_e_nom[store_name] == model.link_p_nom[link_1_name] * ratio1)
            model.store_fix_2 = Constraint(
                rule=lambda model: model.store_e_nom[store_name] == model.link_p_nom[link_2_name] * ratio2)

    else:
        extra_functionality = None

    network.remove("StorageUnit", su_to_replace)

    return bus_name, link_1_name, link_2_name, store_name, gen_name, extra_functionality

def prepare_network(n, solve_opts):
    if leap_year(n.snapshots[0].year):
        n = cut_snapshots(n)
    for index in n.storage_units.index:
            replace_su(n, index)

    if 'clip_p_max_pu' in solve_opts:
        for df in (n.generators_t.p_max_pu, n.storage_units_t.inflow):
            df.where(df>solve_opts['clip_p_max_pu'], other=0., inplace=True)

    if solve_opts.get('load_shedding'):
        n.add("Carrier", "Load")
        n.madd("Generator", n.buses.index, " load",
               bus=n.buses.index,
               carrier='load',
               #sign=1e-3, # Adjust sign to measure p and p_nom in kW instead of MW
               marginal_cost=1e6, # Eur/kWh
               # intersect between macroeconomic and surveybased
               # willingness to pay
               # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
               p_nom=1e9 # kW
               )

    if solve_opts.get('noisy_costs'):
        for t in n.iterate_components(n.one_port_components):
            #if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if 'marginal_cost' in t.df:
                t.df['marginal_cost'] += (1e-2 + 2e-3 *
                                          (np.random.random(len(t.df)) - 0.5))

        for t in n.iterate_components(['Line', 'Link']):
            t.df['capital_cost'] += (1e-1 +
                2e-2*(np.random.random(len(t.df)) - 0.5)) * t.df['length']

    if solve_opts.get('nhours'):
        nhours = solve_opts['nhours']
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760./nhours
    # ### replace AC load with AC load from 2013
    network_ac_load = pypsa.Network('networks/2013/elec_s_40_ec_lv1.0_Co2L0p0-3H.nc')
    #electric_nodes = n.loads.index[n.loads.carrier == "electricity"]
    # n.loads_t.p_set[electric_nodes] = n.loads_t.p_set[electric_nodes] - electric_heat_supply.groupby(level=1, axis=1).sum()[electric_nodes]
    load_index = n.loads_t.p_set.index
    network_ac_load.loads_t.p_set.index = load_index
    n.loads_t.p_set.loc[load_index, :] = network_ac_load.loads_t.p_set.loc[load_index, :]

    return n


def add_CCL_constraints(n, config):
    agg_p_nom_limits = config['electricity'].get('agg_p_nom_limits')

    try:
        agg_p_nom_minmax = pd.read_csv(agg_p_nom_limits,
                                       index_col=list(range(2)))
    except IOError:
        logger.exception("Need to specify the path to a .csv file containing "
                          "aggregate capacity limits per country in "
                          "config['electricity']['agg_p_nom_limit'].")
    logger.info("Adding per carrier generation capacity constraints for "
                "individual countries")

    gen_country = n.generators.bus.map(n.buses.country)
    # cc means country and carrier
    p_nom_per_cc = (pd.DataFrame(
                    {'p_nom': linexpr((1, get_var(n, 'Generator', 'p_nom'))),
                    'country': gen_country, 'carrier': n.generators.carrier})
                    .dropna(subset=['p_nom'])
                    .groupby(['country', 'carrier']).p_nom
                    .apply(join_exprs))
    minimum = agg_p_nom_minmax['min'].dropna()
    if not minimum.empty:
        minconstraint = define_constraints(n, p_nom_per_cc[minimum.index],
                                           '>=', minimum, 'agg_p_nom', 'min')
    maximum = agg_p_nom_minmax['max'].dropna()
    if not maximum.empty:
        maxconstraint = define_constraints(n, p_nom_per_cc[maximum.index],
                                           '<=', maximum, 'agg_p_nom', 'max')


def add_EQ_constraints(n, o, scaling=1e-1):
    float_regex = "[0-9]*\.?[0-9]+"
    level = float(re.findall(float_regex, o)[0])
    if o[-1] == 'c':
        ggrouper = n.generators.bus.map(n.buses.country)
        lgrouper = n.loads.bus.map(n.buses.country)
        sgrouper = n.storage_units.bus.map(n.buses.country)
    else:
        ggrouper = n.generators.bus
        lgrouper = n.loads.bus
        sgrouper = n.storage_units.bus
    load = n.snapshot_weightings @ \
           n.loads_t.p_set.groupby(lgrouper, axis=1).sum()
    inflow = n.snapshot_weightings @ \
             n.storage_units_t.inflow.groupby(sgrouper, axis=1).sum()
    inflow = inflow.reindex(load.index).fillna(0.)
    rhs = scaling * ( level * load - inflow )
    lhs_gen = linexpr((n.snapshot_weightings * scaling,
                       get_var(n, "Generator", "p").T)
              ).T.groupby(ggrouper, axis=1).apply(join_exprs)
    lhs_spill = linexpr((-n.snapshot_weightings * scaling,
                         get_var(n, "StorageUnit", "spill").T)
                ).T.groupby(sgrouper, axis=1).apply(join_exprs)
    lhs_spill = lhs_spill.reindex(lhs_gen.index).fillna("")
    lhs = lhs_gen + lhs_spill
    define_constraints(n, lhs, ">=", rhs, "equity", "min")


def add_BAU_constraints(n, config):
    mincaps = pd.Series(config['electricity']['BAU_mincapacities'])
    lhs = (linexpr((1, get_var(n, 'Generator', 'p_nom')))
           .groupby(n.generators.carrier).apply(join_exprs))
    define_constraints(n, lhs, '>=', mincaps[lhs.index], 'Carrier', 'bau_mincaps')


def add_SAFE_constraints(n, config):
    peakdemand = (1. + config['electricity']['SAFE_reservemargin']) *\
                  n.loads_t.p_set.sum(axis=1).max()
    conv_techs = config['plotting']['conv_techs']
    exist_conv_caps = n.generators.query('~p_nom_extendable & carrier in @conv_techs')\
                       .p_nom.sum()
    ext_gens_i = n.generators.query('carrier in @conv_techs & p_nom_extendable').index
    lhs = linexpr((1, get_var(n, 'Generator', 'p_nom')[ext_gens_i])).sum()
    rhs = peakdemand - exist_conv_caps
    define_constraints(n, lhs, '>=', rhs, 'Safe', 'mintotalcap')


def add_battery_constraints(n):
    nodes = n.buses.index[n.buses.carrier == "battery"]
    if nodes.empty or ('Link', 'p_nom') not in n.variables.index:
        return
    link_p_nom = get_var(n, "Link", "p_nom")
    lhs = linexpr((1,link_p_nom[nodes + " charger"]),
                  (-n.links.loc[nodes + " discharger", "efficiency"].values,
                   link_p_nom[nodes + " discharger"].values))
    define_constraints(n, lhs, "=", 0, 'Link', 'charger_ratio')


def extra_functionality(n, snapshots):
    """
    Collects supplementary constraints which will be passed to ``pypsa.linopf.network_lopf``.
    If you want to enforce additional custom constraints, this is a good location to add them.
    The arguments ``opts`` and ``snakemake.config`` are expected to be attached to the network.
    """
    opts = n.opts
    config = n.config
    if 'BAU' in opts and n.generators.p_nom_extendable.any():
        add_BAU_constraints(n, config)
    if 'SAFE' in opts and n.generators.p_nom_extendable.any():
        add_SAFE_constraints(n, config)
    if 'CCL' in opts and n.generators.p_nom_extendable.any():
        add_CCL_constraints(n, config)
    for o in opts:
        if "EQ" in o:
            add_EQ_constraints(n, o)
    add_battery_constraints(n)

    # for index in n.storage_units.index:
    #     su = n.storage_units.loc[index]
    #     link_1_name = "{} converter {} to AC".format(index, su["carrier"])
    #
    #     link_2_name = "{} converter AC to {}".format(index, su["carrier"])
    #
    #     if su["p_nom_extendable"]:
    #         ratio2 = su["max_hours"]
    #         ratio1 = ratio2 * su["efficiency_dispatch"]
    #
    #         def extra_functionality(n, snapshots):
    #             model = n.model
    #             model.store_fix_1 = Constraint(
    #                 rule=lambda model: model.store_e_nom[store_name] == model.link_p_nom[link_1_name] * ratio1)
    #             model.store_fix_2 = Constraint(
    #                 rule=lambda model: model.store_e_nom[store_name] == model.link_p_nom[link_2_name] * ratio2)




def solve_network(n, config, solver_log=None, opts='', **kwargs):
    solver_options = config['solving']['solver'].copy()
    solver_name = solver_options.pop('name')
    cf_solving = config['solving']['options']
    track_iterations = cf_solving.get('track_iterations', False)
    min_iterations = cf_solving.get('min_iterations', 4)
    max_iterations = cf_solving.get('max_iterations', 6)

    # add to network for extra_functionality
    n.config = config
    n.opts = opts

    if cf_solving.get('skip_iterations', False):
        network_lopf(n, solver_name=solver_name, solver_options=solver_options, keep_shadowprices=True,
                     extra_functionality=extra_functionality, **kwargs)
    else:
        ilopf(n, solver_name=solver_name, solver_options=solver_options,
              track_iterations=track_iterations,
              min_iterations=min_iterations,
              max_iterations=max_iterations,
              extra_functionality=extra_functionality, **kwargs)
    return n


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('solve_network', network='elec', simpl='',
                                  clusters='5', ll='copt', opts='Co2L-BAU-CCL-24H')
    configure_logging(snakemake)

    tmpdir = snakemake.config['solving'].get('tmpdir')
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
    opts = snakemake.wildcards.opts.split('-')
    solve_opts = snakemake.config['solving']['options']

    fn = getattr(snakemake.log, 'memory', None)
    with memory_logger(filename=fn, interval=30.) as mem:
        n = pypsa.Network(snakemake.input[0])
        n = prepare_network(n, solve_opts)
        n.export_to_netcdf("elec.nc")
        n = solve_network(n, config=snakemake.config, solver_dir=tmpdir,
                          solver_log=snakemake.log.solver, opts=opts)
        n.export_to_netcdf(snakemake.output[0])

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
