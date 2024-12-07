import numpy as np
import scanpy as sc
from tensorly.decomposition import parafac
from tensorly.tenalg import outer
from tensorly.cp_tensor import CPTensor

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
            
            if np.sum(mask) == 0:
                continue

            restructured_data[i, :, j] = data[mask]

    assert restructured_data.shape == (n_samples, n_genes, n_cell_types), "Data shape is incorrect"

    return restructured_data


def reconstruct_manual(
    cp: CPTensor
) -> np.ndarray:
    """ 
    Manually reconstruct tensor from CP tensor
    Parameters:
        :cp: CPD object
    Returns:
        :reconstruction: reconstructed tensor
    """
    reconstruction = np.zeros((len(cp[1][0][:, 0]), len(cp[1][1][:, 0]), len(cp[1][2][:, 0])))
    for i in range(len(cp[0])):
        reconstruction += outer([cp[1][0][:, i], cp[1][1][:, i], cp[1][2][:, i]])

    return reconstruction


def reconstruction_error(
    original: np.ndarray,
    cp: CPTensor,
    manual_check: bool = False
) -> float:
    """
    Compute reconstruction error of tensor decomposition
    Parameters:
        :original: original data tensor
        :cp: CPD object
        :manual_check: determines whether manual reconstruction is performed
    Returns:
        :error_auto: built-in reconstruction error
        :error_manual: reconstruction error from manual method
    """
    
    reconstruction_auto = cp.to_tensor()

    assert original.shape == reconstruction_auto.shape

    error_auto = np.sum((original - reconstruction_auto)**2) / np.sum(original**2)
    
    if not manual_check:
        return error_auto
    else:
        reconstruction_manual = reconstruct_manual(cp)
        error_manual = np.sum((original - reconstruction_manual)**2) / np.sum(original**2)

        assert original.shape == reconstruction_manual.shape

        return error_auto, error_manual