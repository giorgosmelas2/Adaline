import numpy as np
import matplotlib.pyplot as plt
from adaline import adaline
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=150, n_features=2, n_clusters_per_class=1, n_informative=2, n_redundant=0, random_state=1)
y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

adaline_model = adaline(n_iter=150, learning_rate=0.001)
adaline_model.fit(X_train, y_train)

y_pred = adaline_model.predict(X_test)


def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    plt.contourf(xx, yy, z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolor='k', cmap=plt.cm.Paired)
    plt.title('Adaline Decision Boundary')
    plt.show()

plot_decision_boundary(X_test, y_test, adaline_model)