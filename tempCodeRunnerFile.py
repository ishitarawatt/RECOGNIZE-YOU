import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import numpy as np
import os
import cv2

# Helper function to plot a gallery of portraits
def plot_gallery(images, titles, h, w, n_rows=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_rows))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_rows * n_col):
        plt.subplot(n_rows, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks([])
        plt.yticks([])

# Data loading and preprocessing
dir_name = "dataset/faces/"
Y = []
X = []
target_names = []
person_id = 0
h = w = 300
n_samples = 0
class_names = []

for person_name in os.listdir(dir_name):
    dir_path = os.path.join(dir_name, person_name)
    class_names.append(person_name)
    for image_name in os.listdir(dir_path):
        image_path = os.path.join(dir_path, image_name)
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray, (h, w))
        X.append(resized_image.flatten())
        n_samples += 1
        Y.append(person_id)
    person_id += 1

# Transform lists to numpy arrays
Y = np.array(Y)
X = np.array(X)
target_names = np.array(class_names)

n_features = X.shape[1]
n_classes = len(class_names)

print("Number of samples:", n_samples)
print("Number of features:", n_features)
print("Number of classes:", n_classes)

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Apply PCA
n_components = 150
print(f"Extracting the top {n_components} eigenfaces from {X_train.shape[0]} faces")

pca = PCA(n_components=n_components, whiten=True, svd_solver='randomized').fit(X_train)
eigenfaces = pca.components_.reshape((n_components, h, w))

# Plot the most significant eigenfaces
eigenfaces_titles = [f"Eigenface {i}" for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenfaces_titles, h, w)
plt.show()

# Projecting the input data on the eigenfaces orthonormal basis
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(X_train_pca.shape, X_test_pca.shape)

# Apply LDA to the PCA-transformed data
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_pca, Y_train)

X_train_lda = lda.transform(X_train_pca)
X_test_lda = lda.transform(X_test_pca)

# Training with a multi-layer perceptron
clf = MLPClassifier(random_state=1, hidden_layer_sizes=(10, 10), max_iter=1000, verbose=True).fit(X_train_lda, Y_train)
print("Model weights:")
model_info = [coef.shape for coef in clf.coefs_]
print(model_info)

# Predict on the test set
Y_pred = []
y_prob = []

for test_face in X_test_lda:
    prob = clf.predict_proba([test_face])[0]
    class_id = np.argmax(prob)
    Y_pred.append(class_id)
    y_prob.append(np.max(prob))

Y_pred = np.array(Y_pred)

# Prepare the titles for the test results
prediction_titles = []
true_positive = 0
for i in range(Y_pred.shape[0]):
    true_name = class_names[Y_test[i]]
    pred_name = class_names[Y_pred[i]]
    result = f'Pred: {pred_name}, Prob: {str(y_prob[i])[:3]}\nTrue: {true_name}'
    prediction_titles.append(result)
    if true_name == pred_name:
        true_positive += 1

# Calculate accuracy
accuracy = true_positive * 100 / Y_pred.shape[0]
print("Accuracy:", accuracy)

# Plot the result
plot_gallery(X_test, prediction_titles, h, w)
plt.show()
