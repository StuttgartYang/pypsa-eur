

import logging
from _helpers import configure_logging

import pypsa
import numpy as np

from pathlib import Path
from vresutils.benchmark import memory_logger
from solve_network import solve_network, prepare_network
from build_optimized_capacities_iteration1 import calculate_nodal_capacities
from add_electricity import load_costs, load_powerplants, attach_conventional_generators, _add_missing_carriers_from_costs
from six import iteritems
import pandas as pd
import os

logger = logging.getLogger(__name__)

idx = pd.IndexSlice

opt_name = {"Store": "e", "Line" : "s", "Transformer" : "s"}

def add_extra_generator(n, solve_opts):
    extra_generator = solve_opts.get('extra_generator')

    if extra_generator == 'load_shedding':
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
    else:
        Nyears = n.snapshot_weightings.sum() / 8760.
        costs = "data/costs.csv"
        costs = load_costs(Nyears, tech_costs = costs, config = snakemake.config['costs'], elec_config = snakemake.config['electricity'])
        ppl = load_powerplants(ppl_fn='resources/powerplants.csv')
        carriers = extra_generator

        _add_missing_carriers_from_costs(n, costs, carriers)

        ppl = (ppl.query('carrier in @carriers').join(costs, on='carrier')
               .rename(index=lambda s: 'C' + str(s)))

        logger.info('Adding {} generators with capacities [MW] \n{}'
                    .format(len(ppl), ppl.groupby('carrier').p_nom.sum()))

        n.madd("Generator", ppl.index,
               carrier=ppl.carrier,
               bus=ppl.bus,
               p_nom=ppl.p_nom,
               efficiency=ppl.efficiency,
               marginal_cost=ppl.marginal_cost,
               capital_cost=0)

        logger.warning(f'Capital costs for conventional generators put to 0 EUR/MW.')

    return n



def set_parameters_from_optimized(n, networks_dict, solve_opts):
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
    n.links.loc[links_dc_i, 'p_nom'] = links_capacities.loc[links_dc_i,:].mean(axis=1)
    n.links.loc[links_dc_i, 'p_nom_extendable'] = False

    #
    gen_extend_i = n.generators.index[n.generators.p_nom_extendable]
    gen_capacities = nodal_capacities.loc['generators']
    gen_extend_i_exclude_biomass = [elem for i, elem in enumerate(gen_extend_i) if elem not in biomass_extend_index]
    n.generators.loc[gen_extend_i, 'p_nom'] = gen_capacities.loc[gen_extend_i,:].mean(axis=1)
    n.generators.loc[gen_extend_i, 'p_nom_extendable'] = False
    extra_generator = solve_opts.get('extra_generator')
    if extra_generator in snakemake.config["electricity"]["conventional_carriers"]:
        generator_extend_index = n.generators.index[n.generators.carrier == 'biomass']
        n.generators.loc[generator_extend_index, 'p_nom_extendable'] = True


    stor_extend_i = n.storage_units.index[n.storage_units.p_nom_extendable]
    stor_capacities = nodal_capacities.loc['storage_units']
    n.storage_units.loc[stor_extend_i, 'p_nom'] = stor_capacities.loc[stor_extend_i, :].mean(axis=1)
    n.storage_units.loc[stor_extend_i, 'p_nom_extendable'] = False

    stores_extend_i = n.stores.index[n.stores.e_nom_extendable]
    stores_capacities = nodal_capacities.loc['stores']
    n.stores.loc[stores_extend_i, 'e_nom'] = stores_capacities.loc[stores_extend_i, :].mean(axis=1)
    n.stores.loc[stores_extend_i, 'e_nom_extendable'] = False
    return n

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_robust_capacities_extra_generator_iteration4', network='elec', simpl='',
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

    networks_dict = {(capacity_year) :
        os.path.join(network_dir,'iteration3', f'elec_s{simpl}_'
                                  f'{clusters}_ec_l{l}_{opts}_{capacity_year}.nc')
                     for capacity_year in snakemake.config["scenario"]["capacity_years"]
                     for simpl in expand_from_wildcard("simpl")
                     for clusters in expand_from_wildcard("clusters")
                     for l in ll
                     for opts in expand_from_wildcard("opts")}

    print(networks_dict)
    configure_logging(snakemake)

    tmpdir = snakemake.config['solving'].get('tmpdir')
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)

    n = pypsa.Network(snakemake.input.unprepared)
    #add_extra_generator(n, snakemake.config['solving']['options'])
    n = set_parameters_from_optimized(n, networks_dict, snakemake.config['solving']['options'])
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

