import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

points = {"blue": [[2, 4, 2], [1, 3, 1], [2, 3, 5], [3, 2, 6], [2, 1, 7]],
          "red": [[5, 6, 2], [4, 5, 1], [4, 6, 4], [6, 6, 1], [5, 4, 8]]}

new_point = [8, 8, 3]


def distance(p, q):
    return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))


class KNN:

    def __init__(self, k=3):
        self.k = k
        self.point = None

    def fit(self, points):
        self.points = points

    def predict(self, new_point):
        distances = []

        for category in self.points:
            for point in self.points[category]:
                dist = distance(point, new_point)
                distances.append([dist, category])

        categories = [category[1] for category in sorted(distances)[:self.k]]
        res = Counter(categories).most_common(1)[0][0]
        return res


classifier = KNN()
classifier.fit(points)
print(classifier.predict(new_point))

fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(projection="3d")
ax.grid(True, color="gray")
ax.set_facecolor("black")
ax.figure.set_facecolor("#121212")
ax.tick_params(axis="x", color="white")
ax.tick_params(axis="y", color="white")

for point in points['blue']:
    ax.scatter(point[0], point[1], point[2], color="blue", s=50)

for point in points['red']:
    ax.scatter(point[0], point[1], point[2], color="red", s=50)


new_class = classifier.predict(new_point)
color = "orange" if new_class == "red" else "cyan"
ax.scatter(new_point[0], new_point[1], new_point[2], color=color,
           marker="*", s=100, zorder=100)


for point in points['blue']:
    ax.plot([new_point[0], point[0]], [new_point[1], point[1]], [new_point[2], point[2]],
            color='cyan', linestyle="--", linewidth=1)


for point in points['red']:
    ax.plot([new_point[0], point[0]], [new_point[1], point[1]], [new_point[2], point[2]],
            color='orange', linestyle="--", linewidth=1)

plt.show()
