import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Generišem slučajne tačke (X1, X2, y)
np.random.seed(42)
X1 = np.random.rand(100, 1) * 10
X2 = np.random.rand(100, 1) * 5
y = 2 * X1 + 3 * X2 + 5 + np.random.randn(100, 1)

# Kreiram model višestruke linearne regresije
model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model.fit(np.hstack((X1, X2)), y)

# Pravim predviđanja za nove vrednosti X1 i X2
X1_new = np.array([[0], [10]])
X2_new = np.array([[0], [5]])
y_pred = model.predict(np.hstack((X1_new, X2_new)))

# Prikazujem podatke i regresionu površinu
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1, X2, y, label="Podaci")
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("y")
ax.set_title("Višestruka Linearna Regresija")
ax.plot_surface(X1_new, X2_new, y_pred, color='red', alpha=0.5, label="Regresiona Površina")
plt.legend()
plt.show()
