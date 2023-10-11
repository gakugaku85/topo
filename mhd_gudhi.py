import os

import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image

# Turn off usetex mode to avoid requiring TeX
plt.rcParams["text.usetex"] = False


def plot_image_array(image_array, output_file_name="output"):
    """Plots the grayscale image based on the image array."""
    plt.clf()
    plt.imshow(image_array, cmap="gray", interpolation="nearest")
    plt.axis("off")
    plt.savefig(
        "result/" + output_file_name + ".png",
    )


def plot_persistence_barcode(persistence, output_file_name="output"):
    """Plots the persistence barcode."""
    plt.clf()
    gd.plot_persistence_barcode(persistence)
    plt.title("Persistence Barcode")
    plt.savefig("result/" + output_file_name + "_barcode.png")


def plot_persistence_diagram(persistence, output_file_name="output"):
    """Plots the persistence diagram."""
    plt.clf()
    gd.plot_persistence_diagram(persistence)
    plt.title("Persistence Diagram")
    plt.savefig("result/" + output_file_name + "_diagram.png")


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
        if point[1][1] == float(
            "inf"
        ):  # Replace inf with a large value for visualization
            diag_array.append([point[1][0], point[1][0] + 10])
        else:
            diag_array.append(list(point[1]))
    return np.array(diag_array)


name_lists = ["lr", "sr", "hr"]
bottle_neck_lists = []
for curDir, dirs, files in os.walk("data/mhd"):
    for file in files:
        if file.endswith(".mhd"):
            print(file)
            image = sitk.ReadImage(os.path.join(curDir, file))
            image_array = sitk.GetArrayFromImage(image)
            persistents = [0, 0, 0]
            for i in range(name_lists.__len__()):
                cropped_image = image_array[0:64, i * 64 : (i + 1) * 64]
                persistents[i] = persistent_homology(
                    cropped_image, file.split(".")[0] + "_" + name_lists[i]
                )
            persistents[1] = convert_to_numpy_array(persistents[1])
            persistents[2] = convert_to_numpy_array(persistents[2])
            bottleneck_distance = gd.bottleneck_distance(
                persistents[1], persistents[2]
            )
            bottle_neck_lists.append((file.split(".")[0], bottleneck_distance))

pd.DataFrame(bottle_neck_lists).to_csv("result/csv/bottle_neck.csv")
