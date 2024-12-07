import numpy as np
import pandas as pd
import scanpy as sc

def normalize(
    data: np.ndarray
) -> np.ndarray:
    """
    Normalize cells by total counts
    Do batch correction by scaling genes by total expression for that batch

    Parameters:
        :data: array with raw gene expression data
            should be a subset of original data by batch
    Returns:
        :data_norm: array with normalized gene expression data
    """

    # print(data)

    ## Cell Normalization
    cell_sums = np.sum(data, axis=1)
    nonzero_cell_mask = cell_sums > 0
    cell_counts = cell_sums / np.median(cell_sums[nonzero_cell_mask])

    data_rowscaled = np.zeros_like(data)
    data_rowscaled[nonzero_cell_mask, :] = data[nonzero_cell_mask, :] / cell_counts[nonzero_cell_mask, None]


    ## Batch Corrections
    # print(data_rowscaled)
    # print(data_rowscaled.shape)

    # nonzero_gene_mask = np.sum(data_rowscaled, axis=0) > 0
    # gene_means = np.mean(data_rowscaled[:, nonzero_gene_mask], axis=0)
    # gene_counts = gene_means * data_rowscaled.shape[0]

    # print(nonzero_gene_mask)
    # print(nonzero_gene_mask.shape)
    # print(gene_means)
    # print(gene_means.shape)
    # print(gene_counts)
    # print(gene_counts.shape)

    # data_norm = np.zeros_like(data_rowscaled)
    # data_norm[:, nonzero_gene_mask] = data_rowscaled[:, nonzero_gene_mask] / gene_counts[None, :]

    data_norm = data_rowscaled

    return data_norm


def prune_genes(
    data: np.ndarray,
    min_expression: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prune genes with low expression
    Parameters:
        :data: normalized gene expression data
        :min_expression: minimum mean expression to keep gene
    Returns:
        :pruned_data: gene expression data with pruned genes
        :gene_mask: boolean mask of genes to keep
    """
    
    gene_mask = data.mean(axis=0) > min_expression
    pruned_data = data[:, gene_mask]

    # print(data)
    # print(data.shape)
    # print(gene_mask)
    # print(gene_mask.shape)
    # print(pruned_data)
    # print(pruned_data.shape)

    return pruned_data, gene_mask


def pseudobulk(
    adata: sc.AnnData,
    groupby: str = "HTAN Specimen ID",
    batch_group: str = "HTAN Specimen ID",
    cell_type_col: str = "Cell_Type",
    min_expression: float = 0.1,
    pseudobulk_method: str = "mean",
) -> sc.AnnData:
    """
    Compute pseudobulk expression for each group in adata
    Parameters:
        :adata: AnnData object with gene expression data in the form of RAW cell counts
        :groupby: column to group data by for pseudobulk
        :batch_group: column to group data by for batch
        :cell_type_col: column with cell type information
        :min_expression: minimum mean expression to keep gene
        :pseudobulk_method: method to compute pseudobulk expression (mean or median)
    Returns:
        :pseudobulk_adata: AnnData object with pseudobulk expression
    """

    data_raw = adata.raw.X


    ## Cell Normalization and Batch Correction
    data_norm = np.zeros(data_raw.shape)

    for obs in adata.obs[batch_group].unique(): #Batch correction is obsolete, can remove for loop
        obs_mask = adata.obs[batch_group] == obs
        obs_mask = obs_mask.values

        data_norm[obs_mask, :] = normalize(data_raw[obs_mask, :])

    print(data_norm)
    print(data_norm.shape)
    print(np.sum(data_norm, axis=0))
    print(np.max(data_norm, axis=0))
    print(np.min(np.max(data_norm, axis=0)))
    print(np.mean(data_norm, axis=0))
    print(np.min(np.mean(data_norm, axis=0)))
    print(np.max(np.mean(data_norm, axis=0)))
    print(np.median(np.mean(data_norm, axis=0)))


    ## Gene Pruning
    data_prune, gene_mask = prune_genes(data_norm, min_expression=min_expression)

    print(data_prune)
    print(data_prune.shape)
    
    adata_processed = sc.AnnData(
        X = data_prune,
        obs = adata.obs.copy(),
        var = adata.var.iloc[gene_mask]
    )

    print(adata_processed)

    data = adata_processed.X

    print(data)
    print(data.shape)

    pseudobulk_data = np.zeros((0, data.shape[1]))
    pseudobulk_group = []
    pseudobulk_cell_type = [] #Probably want to save other patient observation columns as well for factor interpretation
    num_empty_groups = 0

    assert groupby in adata.obs.columns, f"Column '{groupby}' not found in data"
    assert cell_type_col in adata.obs.columns, f"Column '{cell_type_col}' not found in data"

    groups = adata.obs[groupby].unique()
    cell_types = adata.obs[cell_type_col].unique()


    ## Pseudobulking
    for group in groups:
        for cell_type in cell_types:
            group_mask = (adata_processed.obs[groupby] == group) & (adata_processed.obs[cell_type_col] == cell_type)
            group_data = data[group_mask]

            if group_data.shape[0] == 0:
                num_empty_groups += 1
                continue

            if pseudobulk_method == "mean":
                aggregate_data = group_data.mean(axis=0)

            pseudobulk_data = np.vstack((pseudobulk_data, aggregate_data))
            pseudobulk_group = np.append(pseudobulk_group, group)
            pseudobulk_cell_type = np.append(pseudobulk_cell_type, cell_type)

    print(pseudobulk_data)
    print(pseudobulk_data.shape)

    assert pseudobulk_data.shape[0] == (len(adata.obs[groupby].unique()) * len(adata.obs[cell_type_col].unique()) - num_empty_groups), "Pseudobulk data was incorrectly aggregated"

    pseudobulk_adata = sc.AnnData(
        X = pseudobulk_data,
        obs = pd.DataFrame(
            {
                groupby: pseudobulk_group,
                cell_type_col: pseudobulk_cell_type
            }
        ),
        var = adata_processed.var.copy()
    )

    print(pseudobulk_adata)

    return pseudobulk_adata