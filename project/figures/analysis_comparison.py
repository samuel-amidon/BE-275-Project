import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA


def analysisComparison(cp_errors, tensor, nmf=False, nmf_errors=None):
    """
    Plots the error of CPD, PCA, and NMF as a function of the model sizes
    Parameters:
        cp_errors: list of CPD errors
        tensor: tensor data
        nmf: whether to include NMF
        nmf_errors: list of NMF errors
    """
    
    cp_pd = pd.DataFrame(columns=["Size", "Error"])
    rank = len(cp_errors)
    for i in range(rank):
        cp_pd.loc[len(cp_pd)+1] = [(i+1) * np.sum(tensor.shape), cp_errors[i]]

    pca_pd = pd.DataFrame(columns=["Size", "Error"])
    tensor_flat = tensor.reshape(tensor.shape[0], -1)
    pca = PCA(n_components=tensor_flat.shape[0])
    pca.fit(tensor_flat)
    pca_errors = 1 - np.cumsum(pca.explained_variance_ratio_)
    pca_sizes = np.sum(tensor_flat.shape) * np.arange(1, tensor_flat.shape[0] + 1)
    for i in range(len(pca_errors)):
        pca_pd.loc[len(pca_pd)+1] = [pca_sizes[i], pca_errors[i]]

    if nmf:
        nmf_pd = pd.DataFrame(columns=["Size", "Error"])
        nmf_sizes = np.sum(tensor_flat.shape) * np.arange(1, len(nmf_errors) + 1)
        for i in range(len(nmf_errors)):
            nmf_pd.loc[len(nmf_pd)+1] = [nmf_sizes[i], nmf_errors[i]]

    plt.plot(cp_pd["Size"], cp_pd["Error"], "o-", label="CPD")
    plt.plot(pca_pd["Size"], pca_pd["Error"], "o--", label="PCA")
    if nmf:
        plt.plot(nmf_pd["Size"], nmf_pd["Error"], "o-.", label="NMF")
    plt.xlabel("Size")
    plt.ylabel("Error")
    plt.xlim(0, 2e6)
    plt.legend()

    plt.savefig("/Users/samuelamidon/Desktop/BE275 Project Supp/Data_0.01cutoff/analysis_comparison.png")

    