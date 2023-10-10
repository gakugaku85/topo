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


# Compute persistence diagram and barcode using gudhi with corrected axes swapping
def compute_persistence_features(image_data, name):
    # Convert image data to cubical complex
    cubical_complex = gd.CubicalComplex(
        dimensions=image_data.shape, top_dimensional_cells=image_data.flatten()
    )

    # Compute persistence
    persistence = cubical_complex.persistence()

    print(persistence)

    # Separate persistence values based on dimension
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

    # Plot persistence barcode with 0's and 1's separated
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
    plt.savefig("test_result/barcode" + name + ".png")
    # plt.show()

    # Plot persistence diagram with colors based on given result
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
    plt.legend()
    plt.plot([0, 1], [0, 1], c="r")  # 対角線
    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.title("Persistence Diagram")
    plt.savefig("test_result/diagram" + name + ".png")
    # plt.show()


# Execute the function
print(image_array)
compute_persistence_features(image_array, "original")
plt.clf()
plt.imshow(image_array, cmap="gray", interpolation="nearest")
plt.axis("off")  # 軸を非表示にする
plt.savefig("test_result/output.png", dpi=300, bbox_inches="tight")

# 画像を反転
convert_image_array = 1 - image_array
print(convert_image_array)
compute_persistence_features(convert_image_array, "convert")
plt.clf()
plt.imshow(convert_image_array, cmap="gray", interpolation="nearest")
plt.axis("off")  # 軸を非表示にする
plt.savefig("test_result/output_convert.png", dpi=300, bbox_inches="tight")
