import os

import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from PIL import Image

# Turn off usetex mode to avoid requiring TeX
plt.rcParams["text.usetex"] = False


def compute_persistence_features(image_data, output_file_name):
    # Convert image data to cubical complex
    # print(output_file_name)
    cubical_complex = gd.CubicalComplex(
        dimensions=image_data.shape, top_dimensional_cells=image_data.flatten()
    )

    # Compute persistence
    persistence = cubical_complex.persistence()

    persistence_ones = [
        (birth_val, death_val)
        for dim, (birth_val, death_val) in persistence
        if dim == 1
    ]
    persistence_zeros = [
        (birth_val, death_val)
        for dim, (birth_val, death_val) in persistence
        if dim == 0
    ]

    plt.figure()
    for idx, (birth, death) in enumerate(persistence_ones):
        plt.plot([birth, death], [idx, idx], color="blue")
    offset = len(persistence_ones)
    for idx, (birth, death) in enumerate(persistence_zeros):
        plt.plot([birth, death], [idx + offset, idx + offset], color="red")
    plt.yticks(
        range(len(persistence)),
        ["1"] * len(persistence_ones) + ["0"] * len(persistence_zeros),
    )
    plt.gca().invert_xaxis()
    plt.xlabel("Value")
    plt.ylabel("Dimension")
    plt.title("Persistence Barcode")
    plt.gca().invert_yaxis()  # Inverting the y-axis for barcode
    plt.savefig("result/" + output_file_name + "_barcode.png")

    # Plot persistence diagram with swapped axes
    plt.figure()
    plt.scatter(
        [x[0] for x in persistence_ones],
        [x[1] for x in persistence_ones],
        color="blue",
        label="1",
    )
    plt.scatter(
        [x[0] for x in persistence_zeros],
        [x[1] for x in persistence_zeros],
        color="red",
        label="0",
    )
    plt.plot([0, 256], [0, 256], c="r")  # 対角線
    plt.legend()
    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.title("Persistence Diagram")
    plt.savefig("result/" + output_file_name + "_diagram.png")


name_lists = ["lr", "sr", "hr"]
for curDir, dirs, files in os.walk("data/mhd"):
    # print(files)
    for file in files:
        if file.endswith(".mhd"):
            print(file)
            image = sitk.ReadImage(os.path.join(curDir, file))
            image_array = sitk.GetArrayFromImage(image)
            for i in range(3):
                cropped_image = image_array[0:64, i * 64 : (i + 1) * 64]
                compute_persistence_features(
                    cropped_image, file.split(".")[0] + f"_{name_lists[i]}"
                )
