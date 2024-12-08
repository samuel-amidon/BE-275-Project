import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import gridspec, pyplot as plt

def makeFigure(cp, var1, var2, var3, axes_names, gene_val_lim):
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(1, 3, wspace=0.5)
    axes = [plt.subplot(gs[dim]) for dim in range(3)]

    var1 = [f"Spec {i}" for i in range(len(cp[1][0]))]

    factors = [pd.DataFrame(cp[1][dim],
                        columns=[f"Comp. {comp+1}" for comp in np.arange(1, len(cp[0])+1)],
                        index=[var1, var2, var3][dim])
                for dim in range(3)]

    # factors[1] = factors[1].iloc[:2000, :]
    factors[1] = factors[1][(factors[1] > gene_val_lim).any(axis=1)]

    for dim in range(3):
        # sns.heatmap(factors[dim], ax=axes[dim], cmap="vlag", center=0, cbar=True,
        #             xticklabels=[str(comp) for comp in np.arange(1, len(cp[0])+1)], 
        #             yticklabels=factors[dim].index if dim == 2 else False)
        sns.heatmap(factors[dim], ax=axes[dim], cmap="vlag", center=0, cbar=True,
                    xticklabels=[str(comp) for comp in np.arange(1, len(cp[0])+1)], 
                    yticklabels=factors[dim].index if not dim == 0 else False)
        axes[dim].set_title(axes_names[dim])
        axes[dim].set_xlabel("Components")

    plt.savefig("/Users/samuelamidon/Desktop/BE275 Project Supp/Data_0.01cutoff/factor_heatmaps.png")