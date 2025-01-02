import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def geneDistributions(adata, cell_type, genes):
    """
    Plots the distribution of certain genes for one cell type across patients
    Parameters:
        adata: AnnData object
        cell_type: cell type to calculate distributions for
        genes: list of genes to calculate distributions for
    """

    cell_data = adata[adata.obs['Cell_Type'] == cell_type]
    num_genes = len(genes)
    fig, axes = plt.subplots(int(np.ceil(num_genes/2)), 2, figsize=(20, 0.5 * num_genes))
    axes = axes.flatten()
    
    print(cell_data)

    for i, gene in enumerate(genes):
        print(cell_data[:, gene].X.flatten())
        sns.histplot(cell_data[:, gene].X.flatten(), kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {gene} in {cell_type} cells')
        axes[i].set_xlabel(gene)
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()

    plt.savefig("/Users/samuelamidon/Desktop/BE275 Project Supp/Data_0.01cutoff/gene_distributions.png")