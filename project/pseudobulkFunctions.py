import numpy as np
import pandas as pd
import scanpy as sc

# adat = sc.read_h5ad("Data/VUMC_HTAN_DIS_EPI_V2.h5ad")
# print(adat.X)
# print(adat.obs)
# print(adat.var)
# print(adat.obs_names)
# print(adat.var_names)
# print(adat.shape)
# print(adat.X.shape)

def psuedobulk(
    adata: sc.AnnData,
    groupby: str = "Sample",
    method: str = "mean"
) -> sc.AnnData:
    """
    Compute pseudobulk expression for each group in adata
    """
    
    cell_type_col = "Cell_Type"

    groups = adata.obs[groupby].unique()
    cell_types = adata.obs[cell_type_col].unique()

    data = adata.X

    pseudobulk_data = []
    pseudobulk_group = []
    pseudobulk_cell_type = []

    for group in groups:
        for cell_type in cell_types:
            group_mask = (adata.obs[groupby] == group) & (adata.obs[cell_type_col] == cell_type)
            group_data = data[group_mask]

            if method == "mean":
                aggregate_data = group_data.mean(axis=0)

            pseudobulk_data = np.append(pseudobulk_data, aggregate_data)
            pseudobulk_group = np.append(pseudobulk_group, group)
            pseudobulk_cell_type = np.append(pseudobulk_cell_type, cell_type)


    pseudobulk_adata = sc.AnnData(
        X = pseudobulk_data,
        obs = pd.DataFrame(
            {
                groupby: pseudobulk_group,
                cell_type_col: pseudobulk_cell_type
            }
        ),
        var = adata.var.copy()
    )

    return pseudobulk_adata

