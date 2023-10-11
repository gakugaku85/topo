import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from gudhi.wasserstein import wasserstein_distance

a = np.array([[10.0, 18.0], [5.0, 17.0], [1.0, 15.0], [2.0, 14.0]])
b = np.array([[10.0, 20.0], [5.0, 14.0], [3.0, 13.0]])

message = "Bottleneck distance = " + "%.1f" % gd.bottleneck_distance(
    [[0.0, 0.0], [0.0, 5.0]], [[0.0, 13.0], [0.0, 16.0]]
)
print(message)

# plot_diagram(a,b)
plt.clf()
plt.scatter(a[:, 0], a[:, 1], c="red", label="a")
plt.scatter(b[:, 0], b[:, 1], c="blue", label="b")
plt.title("Persistence Diagram")
plt.legend(bbox_to_anchor=(1, 0), loc="lower right", borderaxespad=1)
plt.xlabel("Birth")
plt.ylabel("Death")
plt.plot([0, 20], [0, 20], c="r")  # 対角線
if len(a) >= len(b):
    for i in range(len(a)):
        if len(b) > i:
            plt.plot(
                [a[i, 0], b[i, 0]], [a[i, 1], b[i, 1]], linestyle="--", c="gray"
            )
        else:
            plt.plot(
                [a[i, 0], a[i, 0]], [a[i, 1], a[i, 0]], linestyle="--", c="gray"
            )  # 垂直な線

if len(b) > len(a):
    for i in range(len(b)):
        if len(a) > i:
            plt.plot(
                [a[i, 0], b[i, 0]], [a[i, 1], b[i, 1]], linestyle="--", c="gray"
            )
        else:
            plt.plot(
                [b[i, 0], b[i, 0]], [b[i, 1], b[i, 0]], linestyle="--", c="gray"
            )

bottleneck_distance = gd.bottleneck_distance(a, b)
wasserstein_distance = wasserstein_distance(np.array(a), np.array(b))
message = "Bottleneck distance = " + "%.3f" % bottleneck_distance
plt.text(15, 5, message, ha="center", fontsize=10, color="black")
message = "Wasserstein distance = " + "%.3f" % wasserstein_distance
plt.text(15, 4, message, ha="center", fontsize=10, color="black")

plt.savefig("test_result/" + "ab" + "_diagram.png")
plt.clf()
