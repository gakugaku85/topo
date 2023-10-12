import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from gudhi.wasserstein import wasserstein_distance

# a = np.array([[10.0, 18.0], [5.0, 17.0], [1.0, 15.0], [2.0, 14.0]])
# b = np.array([[10.0, 20.0], [5.0, 14.0], [3.0, 13.0]])

b = np.array([[10.0, 18.0], [5.0, 17.0], [1.0, 15.0], [2.0, 13.0]])
a = np.array([[10.0, 14.0], [1.0, 13.0], [3.0, 13.0], [5.0, 14.0], [10.0, 20.0]])


# plot_diagram(a,b)
plt.clf()
plt.scatter(a[:, 0], a[:, 1], c="red", label="a")
plt.scatter(b[:, 0], b[:, 1], c="blue", label="b")
plt.title("Persistence Diagram")
plt.legend(bbox_to_anchor=(1, 0), loc="lower right", borderaxespad=1)
plt.xlabel("Birth")
plt.ylabel("Death")
plt.plot([0, 20], [0, 20], c="r")  # 対角線

bottleneck_distance = gd.bottleneck_distance(a, b)
cost, matchings = wasserstein_distance(a, b, matching=True)
wasserstein_dis = wasserstein_distance(a, b)
message = "Bottleneck distance = " + "%.3f" % bottleneck_distance
plt.text(15, 5, message, ha="center", fontsize=10, color="black")
message = "Wasserstein distance = " + "%.3f" % cost
plt.text(15, 4, message, ha="center", fontsize=10, color="black")

for match in matchings:
    if match[0] != -1 and match[1] != -1:
        plt.plot(
            [a[match[0], 0], b[match[1], 0]],
            [a[match[0], 1], b[match[1], 1]],
            linestyle="--",
            c="gray",
        )
    else:
        print("innlo")
        if match[0] == -1:
            plt.plot(
                [b[match[1], 0], b[match[1], 0]],
                [b[match[1], 1], b[match[1], 0]],
                linestyle="--",
                c="gray",
            )
        else:
            plt.plot(
                [a[match[0], 0], a[match[0], 0]],
                [a[match[0], 1], a[match[0], 0]],
                linestyle="--",
                c="gray",
            )

plt.savefig("test_result/" + "ab" + "_diagram.png")
plt.clf()

print(f"Wasserstein distance value = {cost:.2f}")
print(f"matchings = {matchings}")
dgm1_to_diagonal = matchings[matchings[:, 1] == -1, 0]
dgm2_to_diagonal = matchings[matchings[:, 0] == -1, 1]
off_diagonal_match = np.delete(matchings, np.where(matchings == -1)[0], axis=0)

for i, j in off_diagonal_match:
    print(f"point {i} in dgm1 is matched to point {j} in dgm2")
for i in dgm1_to_diagonal:
    print(f"point {i} in dgm1 is matched to the diagonal")
for j in dgm2_to_diagonal:
    print(f"point {j} in dgm2 is matched to the diagonal")
