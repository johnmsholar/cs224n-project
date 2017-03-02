# Generate Sample Dataset in 2-dimensional space
# Class 1 is a Gaussian Centered Around (0, -5)
# Class 2 is a Gaussian Centered Around (0, 5)

centroid_1 = np.array([0, -1])
centroid_2 = np.array([0, 1])
cov = np.array([
  [1, 0],
  [0, 1]
])
size = 500
x1, y1 = np.random.multivariate_normal(centroid_1, cov, size).T
x2, y2 = np.random.multivariate_normal(centroid_2, cov, size).T
labels_1 = np.concatenate([np.array([[1, 0]]) for _ in range(size)], axis=0)
labels_2 = np.concatenate([np.array([[0, 1]]) for _ in range(size)], axis=0)
x = np.concatenate([x1, x2], axis = 0).reshape((-1, 1))
y = np.concatenate([y1, y2], axis = 0).reshape((-1, 1))
all_data = np.concatenate([x, y], axis=1)
all_labels = np.concatenate([labels_1, labels_2], axis=0)

# Split example data into train and test sets
train_data, test_data, train_labels, test_labels = train_test_split(
  all_data, all_labels, test_size=0.2, random_state=42
)

# Plot data
plt.plot(x1, y1, 'x')
plt.plot(x2, y2, 'x')
plt.axis('equal')
plt.savefig('plot.png', bbox_inches='tight')
