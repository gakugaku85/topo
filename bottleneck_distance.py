import gudhi as gd
import numpy as np

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


def compute_persistent_homology(image_data):
    """Computes the persistent homology for the given image data."""
    cc = gd.CubicalComplex(
        dimensions=image_data.shape, top_dimensional_cells=image_data.flatten()
    )
    persistence = cc.persistence()
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


# Compute the persistence diagrams for the image_array
persistence1 = compute_persistent_homology(image_array)
persistence2 = compute_persistent_homology(image_array)

# Convert to the expected format
persistence1_array = convert_to_numpy_array(persistence1)
persistence2_array = convert_to_numpy_array(persistence2)

# Compute the bottleneck distance between the two diagrams
bottleneck_distance = gd.bottleneck_distance(
    persistence1_array, persistence2_array
)
print("Bottleneck Distance:", bottleneck_distance)
