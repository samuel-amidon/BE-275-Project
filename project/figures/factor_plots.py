import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import gridspec, pyplot as plt


def factorHeatmaps(cp, lab1, lab2, lab3, dim_names, gene_val_lim=0.1, sort_samples=True):
    """
    Plots heatmaps of the factors of a CP decomposition
    Parameters:
        cp: CP decomposition
        lab1: name of first variable (samples IDs or sample classifications)
        lab2: name of second variable (genes)
        lab3: name of third variable (cell types)
        dim_names: names of the dimensions
        gene_val_lim: threshold for the gene values to be plotted
    """

    if sort_samples:
        order = {"NL": 0, "AD": 1, "SER": 2, "UNC": 3}
        sorted_indices = np.argsort([order[lab] for lab in lab1])
        lab1 = np.array(lab1)[sorted_indices]
        cp[1][0] = cp[1][0][sorted_indices]

    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(1, 3)
    axes = [plt.subplot(gs[dim]) for dim in range(3)]

    factors = [pd.DataFrame(cp[1][dim],
                        columns=[f"Comp. {comp+1}" for comp in np.arange(1, len(cp[0])+1)],
                        index=[lab1, lab2, lab3][dim])
                for dim in range(3)]

    factors[1] = factors[1][(factors[1] > gene_val_lim).any(axis=1) | (factors[1] < -gene_val_lim).any(axis=1)]

    for dim in range(3):
        if len(factors[dim].index) > 100:
            sns.heatmap(factors[dim], ax=axes[dim], cmap="vlag", center=0, cbar=True,
                        xticklabels=[str(comp) for comp in np.arange(1, len(cp[0])+1)], 
                        yticklabels=factors[dim].index if not dim == 1 else False)
        else:
            sns.heatmap(factors[dim], ax=axes[dim], cmap="vlag", center=0, cbar=True,
                        xticklabels=[str(comp) for comp in np.arange(1, len(cp[0])+1)], 
                        yticklabels=factors[dim].index)
        axes[dim].set_title(dim_names[dim])
        axes[dim].set_xlabel("Components")
        axes[dim].tick_params(axis='both', which='major', labelsize=7)

    plt.tight_layout()

    plt.savefig("/Users/samuelamidon/Desktop/BE275 Project Supp/Data_0.01cutoff/factor_heatmaps.png")