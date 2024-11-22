import numpy as np
import pandas as pd
import scanpy as sc


def prune_genes(
    adata: sc.AnnData,
    min_expression: float = 0.1
) -> sc.AnnData:
    """
    Prune genes with low expression
    Parameters:
        :adata: AnnData object with gene expression data
        :min_expression: minimum mean expression to keep gene
    Returns:
        :pruned_adata: AnnData object with pruned gene expression data
    """
    
    data = adata.X

    gene_mask = data.mean(axis=0) > min_expression
    pruned_data = data[:, gene_mask]

    pruned_adata = sc.AnnData(
        X = pruned_data,
        obs = adata.obs.copy(),
        var = adata.var.iloc[gene_mask]
    )

    return pruned_adata


def normalize(
    adata: sc.AnnData,
    norm_group: str = "HTAN Specimen ID",
    norm_method: str = "mean"
) -> sc.AnnData:
    """
    Normalize gene expression data by batch/experiment
    Parameters:
        :adata: AnnData object with gene expression data
        :norm_group: column to group data by for normalization
        :norm_method: method to normalize data (mean or median)
    Returns:
        :normalized_adata: AnnData object with normalized gene expression data
    """

    data = adata.X

    for obs in adata.obs[norm_group].unique():
        obs_mask = adata.obs[norm_group] == obs
        obs_data = data[obs_mask]

        if all(obs_data.sum(axis=0) == 0):
            data[obs_mask] = obs_data
            continue

        if norm_method == "mean":
            obs_data = obs_data / obs_data.mean(axis=0)
        elif norm_method == "median":
            obs_data = obs_data / np.median(obs_data, axis=0)

        data[obs_mask] = obs_data

    normalized_adata = sc.AnnData(
        X = data,
        obs = adata.obs.copy(),
        var = adata.var.copy()
    )

    return normalized_adata


def pseudobulk(
    adata: sc.AnnData,
    groupby: str = "HTAN Specimen ID",
    method: str = "mean",
    cell_type_col: str = "Cell_Type"
) -> sc.AnnData:
    """
    Compute pseudobulk expression for each group in adata
    Parameters:
        :adata: AnnData object with gene expression data
        :groupby: column to group data by for pseudobulk
        :method: method to aggregate data (mean or median)
        :cell_type_col: column with cell type information
    Returns:
        :pseudobulk_adata: AnnData object with pseudobulk expression
    """

    assert groupby in adata.obs.columns, f"Column '{groupby}' not found in data"
    assert cell_type_col in adata.obs.columns, f"Column '{cell_type_col}' not found in data"

    groups = adata.obs[groupby].unique()
    cell_types = adata.obs[cell_type_col].unique()

    pruned_adata = prune_genes(adata, min_expression=0.05)

    normalized_adata = normalize(pruned_adata, norm_group=groupby, norm_method="mean")

    data = normalized_adata.X

    pseudobulk_data = np.zeros((0, data.shape[1]))
    pseudobulk_group = []
    pseudobulk_cell_type = [] #Probably want to save other observation columns as well for factor interpretation

    for group in groups:
        for cell_type in cell_types:
            group_mask = (pruned_adata.obs[groupby] == group) & (pruned_adata.obs[cell_type_col] == cell_type)
            group_data = data[group_mask]

            if group_data.shape[0] == 0:
                continue

            if method == "mean":
                aggregate_data = group_data.mean(axis=0)

            pseudobulk_data = np.vstack((pseudobulk_data, aggregate_data))
            pseudobulk_group = np.append(pseudobulk_group, group)
            pseudobulk_cell_type = np.append(pseudobulk_cell_type, cell_type)

    assert pseudobulk_data.shape[0] == len(adata.obs[groupby].unique()) * len(adata.obs[cell_type_col].unique()), "Pseudobulk data was incorrectly aggregated"

    pseudobulk_adata = sc.AnnData(
        X = pseudobulk_data,
        obs = pd.DataFrame(
            {
                groupby: pseudobulk_group,
                cell_type_col: pseudobulk_cell_type
            }
        ),
        var = pruned_adata.var.copy()
    )

    return pseudobulk_adata