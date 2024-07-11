import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as im
from IPython.display import Image, display

folder_path = "C:\\Users\\Nimansh\\Desktop\\SHIV NADAR\\COURSE MATERIALS\\SEM-2\\MAT 161\\Folder\\faces"

# Resize images to img_size
img_size = (64, 64)
faces = []
for i in range(20):
    image_path = folder_path + str(i) + ".jpg"
    image_path = str(image_path)
    # Convert images to grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize
    img = cv2.resize(img, img_size)
    faces.append(img.flatten())
faces = np.array(faces)
faces_mat = np.vstack(faces)
faces_mat.shape

# Mean face of the 20 input faces.
mean = np.mean(faces_mat, axis=0)
plt.imshow(mean.reshape(64, 64), cmap='gray')

# Calculate the norm and covariance matrices
training_norm = faces_mat - mean
covariance_mat = np.cov(training_norm.T)
covariance_mat.shape

# Calculate the eigen vactors
eigen_vectors, eigen_values, _=np.linalg.svd(covariance_mat)
eigen_vectors.shape

# Sort the eigenvectors and eigenvalues in descending order of eigenvalues
idx = np.argsort(eigen_values)[::-1]
eigen_values = eigen_values[idx]
eigen_vectors = eigen_vectors[:, idx]

# Keep the top k eigenfaces
k = 10
eigen_faces = eigen_vectors[:, :k]
for i in range(k):
    image = eigen_faces[:, i].reshape(64, 64)
    plt.imshow(image, cmap='gray')
    plt.show()

# Trained features
trained_features = np.dot(training_norm, eigen_faces)

image_path = "/content/drive/MyDrive/faces/21.jpg"

img_size = (64, 64)
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, img_size)
face = np.array(img.flatten())

test_norm = face-mean
test_features = np.dot(test_norm, eigen_faces)
# Compare the test feature vector with the trained feature vectors using Euclidean distance
distances = []
for vec in trained_features:
    dist = np.linalg.norm(test_features - vec) # Compute Euclidean distance
    distances.append(dist)
print(distances)

# Find the index of the smallest distance
min_idx = np.argmin(distances)

# Find the label of the closest match
label = f"{min_idx}.jpg"

output_path = folder_path + label

# Display the closest match
plt.imshow(im.open(output_path))

#Check if the output is a match or not
max_distance = 1000
if distances[min_idx] < max_distance:
    print(f"The person closely matches")
else:
    print(f"The person does not match")



# from google.colab import drive
# drive.mount('/content/drive')

"""This code will load 10 face images from the `face_` directory, convert them to grayscale, and calculate the eigenfaces.
It will then project a new face image onto the eigenface space and find the closest face in the training set.
The name of the person in the closest face will be printed to the console."""
