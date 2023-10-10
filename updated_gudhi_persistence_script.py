import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Turn off usetex mode to avoid requiring TeX
plt.rcParams["text.usetex"] = False

# Given image array
image_array = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0.4, 0, 0.2, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
)


def plot_persistence_diagram(persistence):
    """Plots the persistence diagram."""
    gd.plot_persistence_diagram(persistence)
    plt.title("Persistence Diagram")
    plt.show()


def plot_persistence_barcode(persistence):
    """Plots the persistence barcode."""
    gd.plot_persistence_barcode(persistence)
    plt.title("Persistence Barcode")
    plt.show()


def plot_image_array(image_array):
    """Plots the grayscale image based on the image array."""
    plt.imshow(image_array, cmap="gray")
    plt.title("Original Image")
    plt.colorbar()
    plt.show()


def compute_and_visualize_persistent_homology(image_data):
    """Computes and visualizes the persistent homology for the given image data."""
    cc = gd.CubicalComplex(
        dimensions=image_data.shape, top_dimensional_cells=image_data.flatten()
    )
    persistence = cc.persistence()

    # Visualization
    plot_image_array(image_data)
    plot_persistence_diagram(persistence)
    plot_persistence_barcode(persistence)

    return persistence


# Calling the function with the image_array to visualize the results
compute_and_visualize_persistent_homology(image_array)
