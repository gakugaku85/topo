import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from gudhi.wasserstein import wasserstein_distance

from gudhi_distance import plot_diagram

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
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0.5, 0.5, 0, 1, 0],
        [0, 1, 0, 0.5, 0.5, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
)

image_array_2 = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0.3, 0.5, 0.5, 0, 1, 0],
        [0, 1, 0, 0.5, 0.5, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
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
    gd.plot_persistence_barcode(persistence, max_intervals=0, inf_delta=100)
    plt.xlim(0, 1)
    plt.ylim(-1, len(persistence))
    plt.xticks(ticks=np.linspace(0, 1, 6), labels=np.round(np.linspace(1, 0, 6), 2))
    plt.yticks([])
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
        dimensions=image_data.shape, top_dimensional_cells=1 - image_data.flatten()
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


def plot_diagram(a, b, name="ab"):
    plt.clf()
    plt.scatter(a[:, 0], a[:, 1], c="red", label="a")
    plt.scatter(b[:, 0], b[:, 1], c="blue", label="b")
    plt.title("Persistence Diagram")
    plt.legend(bbox_to_anchor=(1, 0), loc="lower right", borderaxespad=1)
    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.plot([0, 1], [0, 1], c="r")  # 対角線

    bottleneck_distance = gd.bottleneck_distance(a, b)
    cost, matchings = wasserstein_distance(a, b, matching=True)
    message = "Bottleneck distance = " + "%.3f" % bottleneck_distance
    plt.text(0.7, 0.3, message, ha="center", fontsize=10, color="black")
    message = "Wasserstein distance = " + "%.3f" % cost
    plt.text(0.7, 0.25, message, ha="center", fontsize=10, color="black")

    for i in range(len(a)):
        plt.text(
            a[i, 0] + 0.02,
            a[i, 1],
            f"({a[i, 0]:.1f}, {a[i, 1]:.1f})",
            fontsize=8,
            color="black",
        )
    for i in range(len(b)):
        plt.text(
            b[i, 0] + 0.02,
            b[i, 1],
            f"({b[i, 0]:.1f}, {b[i, 1]:.1f})",
            fontsize=8,
            color="black",
        )

    for match in matchings:
        if match[0] != -1 and match[1] != -1:
            plt.plot(
                [a[match[0], 0], b[match[1], 0]],
                [a[match[0], 1], b[match[1], 1]],
                linestyle="--",
                c="gray",
            )
        else:
            if match[0] == -1:
                x = (b[match[1], 0] + b[match[1], 1]) / 2
                y = -x + b[match[1], 0] + b[match[1], 1]
                plt.plot(
                    [(b[match[1], 0]), x],
                    [b[match[1], 1], y],
                    linestyle="--",
                    c="gray",
                )
                plt.text(
                    x + 0.02,
                    y,
                    f"({x:.1f}, {y:.1f})",
                    fontsize=8,
                    color="black",
                )
            else:
                x = (a[match[0], 0] + a[match[0], 1]) / 2
                y = -x + a[match[0], 0] + a[match[0], 1]
                plt.plot(
                    [a[match[0], 0], x],
                    [a[match[0], 1], y],
                    linestyle="--",
                    c="gray",
                )
                plt.text(
                    x + 0.02,
                    y,
                    f"({x:.1f}, {y:.1f})",
                    fontsize=8,
                    color="black",
                )
    plt.savefig("test_result/" + name + "_diagram.png")
    plt.clf()


image_persistent = persistent_homology(image_array)
convert_image_array = 1 - image_array
conv_persistent = persistent_homology(image_array_2, "2")
conv_persistent = persistent_homology(convert_image_array, "convert")

# Compute the bottleneck distance between the two diagrams
image_persistent_array = convert_to_numpy_array(image_persistent)
conv_persistent_array = convert_to_numpy_array(conv_persistent)
plot_diagram(image_persistent_array, conv_persistent_array, "distance")
# b_distance = gd.bottleneck_distance(image_persistent_array, conv_persistent_array)
# w_distance = wasserstein_distance(image_persistent_array, conv_persistent_array)
# print("Bottleneck Distance:%.2f " % b_distance)
# print("Wasserstein Distance:%.2f " % w_distance)
