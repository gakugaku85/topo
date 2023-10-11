import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from gudhi.wasserstein import wasserstein_distance

# Turn off usetex mode to avoid requiring TeX
plt.rcParams["text.usetex"] = False

# Given image array
# image_array = np.array(
#     [
#         [1, 1, 1, 1, 1, 1],
#         [1, 1, 1, 1, 1, 1],
#         [1, 0.4, 0, 0.2, 1, 1],
#         [1, 1, 0, 1, 1, 1],
#         [1, 1, 1, 1, 1, 1],
#         [1, 1, 1, 1, 1, 1],
#     ]
# )

image_array = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0.4, 0, 0, 0.2, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ]
)

# image_array_2 = np.array(
#     [
#         [1, 1, 1, 1, 1, 1],
#         [1, 0, 0, 0, 0, 1],
#         [1, 0.6, 1, 1, 0.8, 1],
#         [1, 0, 1, 1, 0, 1],
#         [1, 0, 0, 0, 0, 1],
#         [1, 1, 1, 1, 1, 1],
#     ]
# )

image_array_2 = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0.8, 0, 0, 0.3, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ]
)


def plot_persistence_diagram(persistence, output_file_name="output"):
    """Plots the persistence diagram."""
    plt.clf()
    print(persistence)
    gd.plot_persistence_diagram(persistence)
    plt.title("Persistence Diagram")
    plt.savefig("test_result/" + output_file_name + "_diagram.png")
    # plt.show()


def plot_persistence_barcode(persistence, output_file_name="output"):
    """Plots the persistence barcode."""
    plt.clf()
    gd.plot_persistence_barcode(persistence)
    plt.title("Persistence Barcode")
    plt.savefig("test_result/" + output_file_name + "_barcode.png")
    # plt.show()


def plot_image_array(image_array, output_file_name="output"):
    """Plots the grayscale image based on the image array."""
    plt.clf()
    plt.imshow(image_array, cmap="gray", interpolation="nearest")
    plt.axis("off")
    plt.savefig(
        "test_result/" + output_file_name + ".png",
        dpi=300,
        bbox_inches="tight",
    )


def persistent_homology(image_data, output_file_name="output"):
    """Computes and visualizes the persistent homology for the given image data."""
    cc = gd.CubicalComplex(
        dimensions=image_data.shape, top_dimensional_cells=image_data.flatten()
    )
    persistence = cc.persistence()

    for idx, (birth, death) in enumerate(persistence):
        if death[1] == float("inf"):
            persistence[idx] = (birth, (death[0], image_array.max()))

    # Visualization
    plot_image_array(image_data, output_file_name)
    plot_persistence_diagram(persistence, output_file_name)
    plot_persistence_barcode(persistence, output_file_name)

    return persistence


def convert_to_numpy_array(diagram):
    """Converts the persistence diagram to the expected numpy.ndarray format."""
    diag_array = []
    for point in diagram:
        diag_array.append(list(point[1]))
    return np.array(diag_array)


def create_simplicial_complex(image_array):
    sc = gd.SimplexTree()
    for i in range(len(image_array)):
        for j in range(len(image_array[i])):
            if image_array[i][j] > 0:
                sc.insert([i, j])  # 単体の頂点を挿入
    return sc


# Calling the function with the image_array to visualize the results
# print("Image Array:")
# print(image_array)

image_persistent = persistent_homology(image_array)
convert_image_array = 1 - image_array
conv_persistent = persistent_homology(convert_image_array, "convert")

# Compute the bottleneck distance between the two diagrams
image_persistent_array = convert_to_numpy_array(image_persistent)
conv_persistent_array = convert_to_numpy_array(conv_persistent)
b_distance = gd.bottleneck_distance(image_persistent_array, conv_persistent_array)
w_distance = wasserstein_distance(image_persistent_array, conv_persistent_array)
# print("Bottleneck Distance:%.2f " % b_distance)
# print("Wasserstein Distance:%.2f " % w_distance)
