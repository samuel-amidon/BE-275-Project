import numpy as np
import pandas as pd
import scanpy as sc
import numba as nb

@nb.njit(parallel=True)
def normalize(
    data: np.ndarray,
    norm_method: str = "median"
) -> np.ndarray:
    """
    Normalize gene expression data by batch/experiment
    Parameters:
        :adata: AnnData object with raw or modified gene expression data
        :norm_group: column to group data by for normalization
        :norm_method: method to normalize data (mean or median)
        :raw: type of data to normalize - raw (True) or modified (False)
    Returns:
        :normalized_adata: AnnData object with normalized gene expression data
    """

    data_norm = np.zeros(data.shape)

    for i in range(data.shape[1]):
        if all(data[:, i] == 0):
            data_norm[:, i] = data[:, i]
            continue

        if norm_method == "mean":
            data_norm[:, i] = data[:, i] / data[:, i].mean()
        elif norm_method == "median": #Should take in raw data in which case each element is a cell count
            data_nonzero = data[data[:, i] > 0, i]
            data_norm[:, i] = data[:, i] / np.median(data_nonzero)

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

    return pruned_data, gene_mask


def pseudobulk(
    adata: sc.AnnData,
    groupby: str = "HTAN Specimen ID",
    norm_group: str = "HTAN Specimen ID",
    cell_type_col: str = "Cell_Type",
    min_expression: float = 0.1,
    pseudobulk_method: str = "mean",
    norm_method: str = "median"
) -> sc.AnnData:
    """
    Compute pseudobulk expression for each group in adata
    Parameters:
        :adata: AnnData object with gene expression data in the form of RAW cell counts
        :groupby: column to group data by for pseudobulk
        :norm_group: column to group data by for normalization
        :min_expression: minimum mean expression to keep gene
        :cell_type_col: column with cell type information
        :pseudobulk_method: method to compute pseudobulk expression (mean or median)
        :norm_method: method to normalize data (mean or median)
    Returns:
        :pseudobulk_adata: AnnData object with pseudobulk expression
    """

    data_raw = adata.raw.X

    data_norm = np.zeros(data_raw.shape)

    for obs in adata.obs[norm_group].unique():
        obs_mask = adata.obs[norm_group] == obs
        obs_mask = obs_mask.values

        data_norm[obs_mask, :] = normalize(data_raw[obs_mask, :], norm_method=norm_method)

    data_prune, gene_mask = prune_genes(data_norm, min_expression=min_expression)

    adata_processed = sc.AnnData(
        X = data_prune,
        obs = adata.obs.copy(),
        var = adata.var.iloc[gene_mask]
    )

    data = adata_processed.X

    pseudobulk_data = np.zeros((0, data.shape[1]))
    pseudobulk_group = []
    pseudobulk_cell_type = [] #Probably want to save other patient observation columns as well for factor interpretation
    num_empty_groups = 0

    assert groupby in adata.obs.columns, f"Column '{groupby}' not found in data"
    assert cell_type_col in adata.obs.columns, f"Column '{cell_type_col}' not found in data"

    groups = adata.obs[groupby].unique()
    cell_types = adata.obs[cell_type_col].unique()

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

    return pseudobulk_adata