import numpy as np
import matplotlib.pyplot as plt

def meanExpression(adat, gene, cell_type):
    """
    Plots the mean expression of a gene and cell type across samples
    Parameters:
        adat: AnnData object
        gene: gene name
        cell_type: cell type
    """
    col_mask = adat.var_names == gene
    row_mask = adat.obs["Cell_Type"] == cell_type

    data = adat.X[row_mask, col_mask]
    labs = adat.obs["HTAN Specimen ID"][row_mask]

    data_agg = np.zeros(len(labs.unique()))
    for i, lab in enumerate(labs.unique()):
        mask = labs == lab
        data_agg[i] = np.mean(data[mask])

    plt.bar(np.arange(len(data_agg)), data_agg)
    plt.xlabel('Sample')
    plt.ylabel('Mean Expression')
    plt.title(f'Mean Expression of {gene} in {cell_type}')
    plt.xticks(np.arange(len(data_agg)), labs.unique(), rotation=90, fontsize=5)

    plt.savefig("/Users/samuelamidon/Desktop/BE275 Project Supp/Data_0.01cutoff/mean_expression.png")
