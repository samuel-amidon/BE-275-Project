import numpy as np
import scanpy as sc
from tensorly.decomposition import parafac

def tensor_restructure(
    adata: sc.AnnData
) -> np.ndarray:
    """
    Restructure 2D AnnData object into a 3D numpy array
    Parameters:
        :adata: pseudobulked data with shape samples + cell_types x genes
    Returns:
        :data: numpy array with shape samples x genes x cell_types
    """

    data = adata.X
    n_genes = len(adata.var)
    n_samples = len(adata.obs["HTAN Specimen ID"].unique())
    n_cell_types = len(adata.obs["Cell_Type"].unique())
    restructured_data = np.zeros((n_samples, n_genes, n_cell_types))
    
    for i, sample in enumerate(adata.obs["HTAN Specimen ID"].unique()):
        for j, cell_type in enumerate(adata.obs["Cell_Type"].unique()):
            mask = (adata.obs["HTAN Specimen ID"] == sample) & (adata.obs["Cell_Type"] == cell_type)
            restructured_data[i, :, j] = data[mask]

    assert restructured_data.shape == (n_samples, n_genes, n_cell_types), "Data shape is incorrect"

    return restructured_data


def tensor_factorization(
    tdata: np.ndarray,
    rank: int = 10
) -> np.ndarray:
    """
    Perform tensor factorization on 3D tensor data
    Parameters:
        :tdata: 3D numpy array with shape samples x genes x cell_types
        :rank: rank of the tensor factorization
    Returns:
        :factors: factors of the tensor factorization
    """

    factors = parafac(tdata, rank=rank)

    return factors