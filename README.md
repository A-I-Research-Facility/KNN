# 🧠 K-Nearest Neighbors (KNN) in 3D - Python Implementation

This project is a simple implementation of the **K-Nearest Neighbors (KNN)** classification algorithm using Python. It visualizes data points in **3D space** and shows how a new data point is classified based on its proximity to labeled examples.

---

## 📌 What is KNN?

**KNN** is a supervised machine learning algorithm used for **classification** and **regression**. It works by:

1. Calculating the distance between a new data point and all existing points.
2. Selecting the `k` nearest neighbors.
3. Assigning the class that occurs most frequently among those neighbors.

---

## ✅ What This Code Does

-   Creates a labeled dataset with two classes: **blue** and **red** points in 3D.
-   Uses the **KNN algorithm** to predict the class of a new 3D point.
-   Visualizes:

    -   All original points
    -   The new test point
    -   The distances between the new point and each existing point

---

## 📦 Requirements

Install the required packages:

```bash
pip install numpy matplotlib
```

numpy: math operations
matplotlib: plot the graph

---

## 🧠 How the Code Works (Step-by-Step)

### 1. 🗃️ Define the Dataset

```python
points = {
  "blue": [[2, 4, 2], [1, 3, 1], [2, 3, 5], [3, 2, 6], [2, 1, 7]],
  "red": [[5, 6, 2], [4, 5, 1], [4, 6, 4], [6, 6, 1], [5, 4, 8]]
}
new_point = [8, 8, 3]
```

-   `points` is a dictionary where each key is a class (label) and each value is a list of 3D coordinates.
-   `new_point` is the data point whose class we want to predict.

---

### 2. 📏 Calculate Distance

```python
def distance(p, q):
    return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))
```

-   Calculates **Euclidean distance** between two points `p` and `q`.
-   This is the default metric for proximity in KNN.

**Formula**:

$$
\text{distance}(p, q) = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2 + (p_3 - q_3)^2}
$$

---

### 3. 🧠 Implement KNN Classifier

```python
class KNN:
    def __init__(self, k=3):
        self.k = k
```

-   `k` is the number of nearest neighbors to consider (default is 3).

```python
    def fit(self, points):
        self.points = points
```

-   `fit()` just stores the training data.

```python
    def predict(self, new_point):
        distances = []
        for category in self.points:
            for point in self.points[category]:
                dist = distance(point, new_point)
                distances.append([dist, category])
```

-   For each point in the dataset, compute the distance to the `new_point` and store the label.

```python
        categories = [category[1] for category in sorted(distances)[:self.k]]
        res = Counter(categories).most_common(1)[0][0]
        return res
```

-   Sort the distances and take the **top k nearest neighbors**.
-   Count the class labels and return the **most frequent class**.

---

### 4. 🧪 Make Prediction

```python
classifier = KNN()
classifier.fit(points)
print(classifier.predict(new_point))
```

-   Initializes and fits the classifier.
-   Prints the predicted class for the `new_point`.

---

### 5. 📊 Visualize in 3D

```python
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(projection="3d")
```

-   Sets up a 3D plot using `matplotlib`.

```python
for point in points['blue']:
    ax.scatter(point[0], point[1], point[2], color="blue", s=50)

for point in points['red']:
    ax.scatter(point[0], point[1], point[2], color="red", s=50)
```

-   Plots all labeled points by color.

```python
new_class = classifier.predict(new_point)
color = "orange" if new_class == "red" else "cyan"
ax.scatter(new_point[0], new_point[1], new_point[2], color=color, marker="*", s=100)
```

-   Predicts and plots the new point in **cyan** (if classified as "blue") or **orange** (if "red").

---

### 6. 🔗 Draw Lines for Distances

```python
for point in points['blue']:
    ax.plot([...], color='cyan', linestyle="--", linewidth=1)
for point in points['red']:
    ax.plot([...], color='orange', linestyle="--", linewidth=1)
```

-   These lines show the visual distance between the `new_point` and every point in the dataset.

---

## 🧪 Output Example

The output in your console will be the predicted class label:

```bash
red
```

And the 3D plot will show:

-   Red and blue labeled points
-   A new star-shaped point for the input
-   Dashed lines representing distances

---

## 📈 Example Visualization

![alt text](/assets/result.png)

```
[blue points]    [red points]    [*new point]
     ●                ●                  ★
     ●                ●
     ●                ●
```

---

## 💡 Key Concepts Covered

-   Distance metrics (Euclidean)
-   Nearest neighbor voting
-   Basic object-oriented programming (OOP)
-   3D data visualization using `matplotlib`

---

## 🔧 Customize

-   Change `k` in `KNN(k=3)` to experiment with different neighbor counts.
-   Add more categories (e.g., "green", "yellow") to test multi-class classification.
-   Replace Euclidean distance with Manhattan or cosine similarity.

---

## 📁 File Structure

```
knn/
│
├── main.py        # Main Python script
├── README.md        # This explanation
```

---
