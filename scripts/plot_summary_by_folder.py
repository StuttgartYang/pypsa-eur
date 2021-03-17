# SPDX-FileCopyrightText: : 2017-2020 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Plots energy and cost summaries for solved networks.

Relevant Settings
-----------------

Inputs
------

Outputs
-------

Description
-----------

"""

import os
import logging
from _helpers import configure_logging

import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def rename_techs(label):
    if "H2" in label:
        label = "hydrogen storage"
    elif label == "solar":
        label = "solar PV"
    elif label == "offwind-ac":
        label = "offshore wind ac"
    elif label == "offwind-dc":
        label = "offshore wind dc"
    elif label == "onwind":
        label = "onshore wind"
    elif label == "ror":
        label = "hydroelectricity"
    elif label == "hydro":
        label = "hydroelectricity"
    elif label == "PHS":
        label = "hydroelectricity"
    elif "battery" in label:
        label = "battery storage"

    return label


preferred_order = pd.Index(["transmission lines","hydroelectricity","hydro reservoir","run of river","pumped hydro storage","onshore wind","offshore wind ac", "offshore wind dc","solar PV","solar thermal","OCGT","hydrogen storage","battery storage"])


def plot_costs(infn, fn=None):

    ## For now ignore the simpl header
    cost_df = pd.read_csv(infn,index_col=list(range(3)),header=[0])

    df = cost_df.groupby(cost_df.index.get_level_values(2)).sum()

    #convert to billions
    df = df/1e9

    df = df.groupby(df.index.map(rename_techs)).sum()

    to_drop = df.index[df.max(axis=1) < snakemake.config['plotting']['costs_threshold']]

    df = df.drop(to_drop)


    new_index = (preferred_order&df.index).append(df.index.difference(preferred_order))

    #new_columns = df.sum().sort_values().index
    new_columns = df.columns.sort_values()

    fig, ax = plt.subplots()
    fig.set_size_inches((8,8))

    df.loc[new_index,new_columns].T.plot(kind="bar",ax=ax,stacked=True,color=[snakemake.config['plotting']['tech_colors'][i] for i in new_index])


    handles,labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    ax.set_ylim([0,snakemake.config['plotting']['costs_max']])

    ax.set_ylabel("System Cost [EUR billion per year]")

    ax.set_xlabel("")

    ax.grid(axis="y")

    ax.legend(handles,labels,ncol=4,loc="upper left")
    cost_total = df.loc[new_index,new_columns].sum(axis=0)
    for i, v in enumerate(cost_total):
        ax.text(i,v, str("%.2f" %v))
    ax.title.set_text("Total costs are " + str("%.2f" %cost_total.sum()))

    fig.tight_layout()

    if fn is not None:
        fig.savefig(fn, transparent=True)


def plot_energy(infn, fn=None):

    energy_df = pd.read_csv(infn, index_col=list(range(2)),header=[0])

    df = energy_df.groupby(energy_df.index.get_level_values(1)).sum()

    #convert MWh to TWh
    df = df/1e6

    df = df.groupby(df.index.map(rename_techs)).sum()

    to_drop = df.index[df.abs().max(axis=1) < snakemake.config['plotting']['energy_threshold']]

    df = df.drop(to_drop)
    new_index = (preferred_order&df.index).append(df.index.difference(preferred_order))

    new_columns = df.columns.sort_values()

    fig, ax = plt.subplots()
    fig.set_size_inches((12,8))

    df.loc[new_index,new_columns].T.plot(kind="bar",ax=ax,stacked=True,color=[snakemake.config['plotting']['tech_colors'][i] for i in new_index])

    handles,labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    ax.set_ylim([snakemake.config['plotting']['energy_min'],snakemake.config['plotting']['energy_max']])

    ax.set_ylabel("Energy [TWh/a]")

    ax.set_xlabel("")

    ax.grid(axis="y")

    ax.legend(handles,labels,ncol=4,loc="upper left")
    energy_total = df.loc[new_index,new_columns].sum(axis=0)

    fig.tight_layout()

    if fn is not None:
        fig.savefig(fn, transparent=True)

def plot_capacities():
    capacity_df = pd.read_csv(snakemake.input.capacities,index_col=list(range(2)),header=[0])
    df = capacity_df.groupby(capacity_df.index.get_level_values(1)).sum()
    # convert MWh to TWh
    df = df / 1e6
    df = df.groupby(df.index.map(rename_techs)).sum()
    if "load" in df.index:
       df = df.drop("load", axis=0)
    new_index = (preferred_order&df.index).append(df.index.difference(preferred_order))
    # new_columns = df.sum().sort_values().index
    new_columns = df.columns.sort_values()
    capacity_total = df[df.loc[new_index, new_columns] > 0].sum(axis=0)

    fig, ax = plt.subplots()
    # fig.set_size_inches((12,8))
    df.loc[new_index,new_columns].T.plot(kind="bar",ax=ax,stacked=True,color=[snakemake.config['plotting']['tech_colors'][i] for i in new_index])

    handles,labels = ax.get_legend_handles_labels()
    handles.reverse()
    labels.reverse()
    ax.set_ylabel("Capacities")

    ax.set_xlabel("")

    ax.grid(axis="y")

    ax.legend(handles,labels,ncol=1,loc="upper left", bbox_to_anchor=(1,1))
    ticklabel=ax.get_xticklabels()
    ax.set_xticklabels(ticklabel)

    for i, v in enumerate(capacity_total):
        ax.text(i, v, str("%.2f" % v))

    fig.tight_layout()

    fig.savefig(snakemake.output.capacities,transparent=True)

    # standard_dev = df.loc[new_index, new_columns].std(axis=1)
    # plt.plot(standard_dev)
    # for i, v in enumerate(standard_dev):
    #     plt.text(i, v + 10, str("%.2f" % v), rotation="45")
    # plt.ylabel('Standard Devidation of Capacities in 30 years')
    # plt.xticks(rotation=45, wrap=True)
    #
    # plt.savefig("../results/summary/postnetworks/graphs/without/capacities_standard_deviation.jpg")
    # plt.close()

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from vresutils import Dict
        import yaml
        snakemake = Dict()
        with open('../config.yaml') as f:
            snakemake.config = yaml.load(f)
        snakemake.input = Dict()
        snakemake.output = Dict()


        # for item in ["costs","energy","capacities","original_curtailment","load_shedding"]:
        for item in ["costs", "energy","capacities" ]:
            snakemake.input[item] = '../results/summary/csvs/'+snakemake.config['make_summary']['iteration']+'/{item}.csv'.format(item=item)
            snakemake.output[item] = '../results/summary/graphs/'+snakemake.config['make_summary']['iteration']+'/{item}_'.format(item=item)+snakemake.config['make_summary']['iteration']+'.jpg'

    print("here", snakemake.output["costs"])
    plot_costs(snakemake.input["costs"], snakemake.output["costs"])


    plot_energy(snakemake.input["energy"], snakemake.output["energy"])
    plot_capacities()


