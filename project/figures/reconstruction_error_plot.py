import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def errorPlot(errors, overlay_svd=False, svd_errors=None):
    """
    Plots the reconstruction error of a CP decomposition over components
    Parameters:
        errors: either a list of errors or a list of lists of errors
    """

    if not type(errors[0]) == list:
        plt.plot(range(1, len(errors)+1), errors, marker="o")
    else:
        markers = ["o", "s", "D", "^"]
        marker_sizes = [6, 5, 3, 4]
        line_styles = ["-", "--", ":", "-."]
        for i in range(len(errors)):
            plt.plot(range(1, len(errors[i])+1), errors[i], marker=markers[i], markersize=marker_sizes[i], linestyle=line_styles[i], label=f"Random Init. {i+1}")
            plt.legend()
            plt.xticks(range(1, len(errors[i])+1))

    if overlay_svd:
        plt.plot(range(1, len(svd_errors)+1), svd_errors, marker="*", label="SVD Init.")
        plt.legend()

    plt.xlabel("Component Number")
    plt.ylabel("Reconstruction Error")
    plt.ylim(0.0, 1.0)
    plt.title("Reconstruction Error vs Component Number")

    plt.savefig("/Users/samuelamidon/Desktop/BE275 Project Supp/Data_0.01cutoff/reconstruction_error_all.png")