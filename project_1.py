# -*- coding: utf-8 -*-
"""Project 1.ipynb

Hadley Dixon,, MATH 373, 4/4/2024

Original file is located at
    https://colab.research.google.com/drive/1_MsmMfVA1-h-CjsTOLDUYeKQmFny7rdl
"""

#%%

# INSTALL PACKAGES

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.model_selection
import sklearn.datasets

#%%

# Sigmoid
def sigmoid(u):
  return 1/(1+(np.exp(-u)))
  
# Binary Cross Entropy
def bce(p,q):
  return -p * np.log(q) - (1-p) * np.log(1-q)

# Average Cross Entropy
def avg_cross_entropy(beta, X, y):
  y_pred = sigmoid(X @ beta)
  L = np.mean(bce(y, y_pred))
  return L

# Gradient Descent
def grad_L(beta, X, y):
  # Assumes that X has been augmented
  N = len(X)
  grad = (1/N) * X.T @ (sigmoid(X @ beta) - y)
  return grad

# Logistic regression using gradient descent, by hand
def Logistic_Regression_Gradient(learning_rate, iterations, X, y):
  # Assumes that X has been augmented
    N_train, d = X.shape
    beta = np.zeros(d)
    
    ace_vals = []
    for i in range(iterations):
        beta -= learning_rate * grad_L(beta, X, y)
      
        ace = avg_cross_entropy(beta, X, y)
        ace_vals.append(ace)
    
    return beta, ace_vals

# Logistic regression using stochastic gradient descent, by hand
def Logistic_Regression_StochGradient(num_epochs, batch_size, learning_rate, X, y):
    # Assumes that X has been augmented
    N, d = X.shape
    ace_vals_SG = []
    beta = np.zeros(d)
    
    for ep in range(num_epochs):        
        prm = np.random.permutation(N)
        start_idx = 0
        while start_idx < N:
            stop_idx = start_idx + batch_size
            batch_idxs= prm[start_idx:stop_idx]
            X_batch = X[batch_idxs]
            y_batch = y[batch_idxs]
            
            # Compute gradient
            beta -= learning_rate * grad_L(beta, X_batch, y_batch)
            
            start_idx = stop_idx
        
        ace = avg_cross_entropy(beta, X, y)
        ace_vals_SG.append(ace)
        
    return beta, ace_vals_SG

# Compute accuracy
def accuracy(beta, X, y):
    y_pred_val = sigmoid(X @ beta)
    accuracy = np.mean(np.round(y_pred_val) == y)
    return accuracy

#%%%

# BREAST CANCER & GRADIENT DESCENT

print("\n------------------------------------------------------------------")
print("Using Gradient Descent (Breast Cancer)")
print("------------------------------------------------------------------")

# LOAD & SPLIT DATA

dataset1 = sk.datasets.load_breast_cancer()
X_full = dataset1.data
y_full = dataset1.target
X_train, X_val, y_train, y_val = sk.model_selection.train_test_split(X_full, y_full, random_state = 123)

# STANDARDIZE 

mu = np.mean(X_train, axis=0)
s = np.std(X_train, axis=0)
X_train = (X_train - mu) / s
X_val = (X_val - mu) / s

# AUGMENT MATRICES
X_train = np.insert(X_train, 0, 1, axis=1)
X_val = np.insert(X_val, 0, 1, axis=1)

# TRAIN MODEL & TEST
beta1, ace_vals1 = Logistic_Regression_Gradient(1, 100, X_train, y_train)
beta2, ace_vals2 = Logistic_Regression_Gradient(.1, 100, X_train, y_train)
beta3, ace_vals3 = Logistic_Regression_Gradient(.01, 100, X_train, y_train)

# # PLOT COST FUNCTION VS. ITERACTION

plt.figure()
plt.plot(ace_vals1, color="blue", label="1")
plt.plot(ace_vals2, color="green", label="0.1")
plt.plot(ace_vals3, color="red", label="0.01")
plt.legend(title="learning rates")
plt.xlabel("Iterations")
plt.ylabel("Average Cross Entropy")
plt.title("Breast Cancer: Cost Function vs. Iteration (gradient descent)")
plt.show()

# CLASSIFICATION ACCURACY

accuracy_grad1 = accuracy(beta1, X_val, y_val)
accuracy_grad2 = accuracy(beta2, X_val, y_val)
accuracy_grad3 = accuracy(beta3, X_val, y_val)
print("\nI analyzed the the convergence of my Cost Functions using 3 different learning rates, over iterations varying from 1-300 iterations, and observed how the graph responds.")
print("\nThe learning rate that yields the most accurate predictions of the validation data is 1.")
print('\nAccuracy when learning rate is 1 (validation): ', accuracy_grad1)
print('Accuracy when learning rate is .1 (validation): ', accuracy_grad2)
print('Accuracy when learning rate is .01 (validation): ', accuracy_grad3)
print("\nAbout 40 iterations are required until the gradient descent method has converged, using a learning rate of 1.")

#%%

# BREAST CANCER & STOCHASTIC GRADIENT DESCENT

print("\n------------------------------------------------------------------")
print("Using Stochastic Gradient Descent (Breast Cancer)")
print("------------------------------------------------------------------")

# LOAD & SPLIT DATA

dataset2 = dataset1
X_full = dataset2.data
y_full = dataset2.target
X_train, X_val, y_train, y_val = sk.model_selection.train_test_split(X_full, y_full, random_state = 123)

# STANDARDIZE 

mu = np.mean(X_train, axis=0)
s = np.std(X_train, axis=0)
X_train = (X_train - mu) / s
X_val = (X_val - mu) / s

# AUGMENT MATRICES
X_train = np.insert(X_train, 0, 1, axis=1)
X_val = np.insert(X_val, 0, 1, axis=1)
            
# TRAIN MODEL & TEST
            
beta_SG1, ace_vals_SG1 = Logistic_Regression_StochGradient(100, 16, 1, X_train, y_train)
beta_SG2, ace_vals_SG2 = Logistic_Regression_StochGradient(100, 16, 0.1, X_train, y_train)
beta_SG3, ace_vals_SG3 = Logistic_Regression_StochGradient(100, 16, 0.01, X_train, y_train)

# PLOT COST FUNCTION VS. EPOCH

plt.figure()
plt.plot(ace_vals_SG1, color="blue", label="1")
plt.plot(ace_vals_SG2, color="green", label="0.1")
plt.plot(ace_vals_SG3, color="red", label="0.01")
plt.legend(title="learning rates")
plt.xlabel("Epochs")
plt.ylabel("Average Cross Entropy")
plt.title("Breast Cancer: Cost Function vs. Epoch (stochastic gradient descent)")
plt.show()

# CLASSIFICATION ACCURACY

accuracy_SG1 = accuracy(beta_SG1, X_val, y_val)
accuracy_SG2 = accuracy(beta_SG2, X_val, y_val)
accuracy_SG3 = accuracy(beta_SG3, X_val, y_val)
print("\nI analyzed the the convergence of my Cost Functions using 3 different learning rates, over iterations varying from 1-100 epochs, and observed how the graph responds.")
print("\nThe learning rate that yields the most accurate predictions of the validation data is 0.1.")
print('\nAccuracy when learning rate is 1 (validation): ', accuracy_SG1)
print('Accuracy when learning rate is 0.1 (validation): ', accuracy_SG2)
print('Accuracy when learning rate is .01 (validation): ', accuracy_SG3)
print("\nAbout 25 epochs are required until the gradient descent method has converged, using a learning rate of 0.1.")

#%%

# BREAST CANCER: GRADIENT DESCENT VS. STOCHASTIC GRADIENT DESCENT

plt.figure()
plt.plot(ace_vals1, color="blue", label="GD, 1")
plt.plot(ace_vals_SG2, color="green", label="SGD, 0.1")
plt.legend(title="Method, LR")
plt.xlabel("Epochs")
plt.ylabel("Average Cross Entropy")
plt.title("Breast Cancer: Gradient Descent vs. Stochastic Gradient Descent")
plt.show()

print("\n------------------------------------------------------------------")
print("GRADIENT DESCENT VS. STOCHASTIC GRADIENT DESCENT (Breast Cancer)")
print("------------------------------------------------------------------")
print("\nThe graph demonstrates that the cost function converges faster when using stochastic gradient descent. Both methods find a beta vector that minimizes the cost function to the point of convergence.")
print('\nClassification accuracy when learning rate is 1, using gradient descent (validation): ', accuracy_grad1)
print('Classification accuracy when learning rate is .1, using stochastic gradient descent (validation): ', accuracy_SG2)

#%%

# MNIST & STOCHASTIC GRADIENT DESCENT

print("\n------------------------------------------------------------------")
print("Using Stochastic Gradient Descent (MNSIT)")
print("------------------------------------------------------------------")

# ALTERNATE LOAD METHOD

# import pandas as pd
# df = pd.read_csv('/Users/hadleydixon/Desktop/MATH 373/digit-recognizer/train.csv').values
# y_full = df[:, 0]
# X_full = df[:, 1:] / 255.0

# LOAD & SPLIT DATA 

dataset = sk.datasets.fetch_openml('mnist_784')

X_full = dataset.data.astype(int).values / 255
y_full = dataset.target.astype(int).values

y_full = (y_full ==5).astype(int) # Convert multiclass into binary "5 vs. not 5"

X_train, X_val, y_train, y_val = sk.model_selection.train_test_split(X_full, y_full, test_size = 10000)

# AUGMENT MATRICES

X_train = np.insert(X_train, 0, 1, axis=1)
X_val = np.insert(X_val, 0, 1, axis=1)
X_val_non_augmented = X_val[:,1:]

#%%

# TRAIN MODEL & TEST
beta_SG4, ace_vals_SG4 = Logistic_Regression_StochGradient(100, 16, 0.01, X_train, y_train)
beta_SG5, ace_vals_SG5 = Logistic_Regression_StochGradient(100, 16, 0.001, X_train, y_train)
beta_SG6, ace_vals_SG6 = Logistic_Regression_StochGradient(100, 16, 0.0001, X_train, y_train)

# PLOT COST FUNCTION VS. EPOCH

plt.figure()
plt.plot(ace_vals_SG4, color="blue", label="0.01")
plt.plot(ace_vals_SG5, color="green", label="0.001")
plt.plot(ace_vals_SG6, color="red", label="0.0001")
plt.legend(title="learning rates")
plt.title("MNIST: Cost Function vs. Epoch (stochastic gradient descent)")
plt.show()

# CLASSIFICATION ACCURACY

accuracy_SG4 = accuracy(beta_SG4, X_val, y_val)
accuracy_SG5 = accuracy(beta_SG5, X_val, y_val)
accuracy_SG6 = accuracy(beta_SG6, X_val, y_val)
print("\nI analyzed the the convergence of my Cost Functions using 3 different learning rates, over iterations varying from 1-100 epochs, and observed how the graph responds.")
print("\nThe learning rate that yields the most accurate predictions of the validation data is 0.01.")
print('\nAccuracy when learning rate is 0.01 (validation): ', accuracy_SG4)
print('Accuracy when learning rate is 0.001 (validation): ', accuracy_SG5)
print('Accuracy when learning rate is .0001 (validation): ', accuracy_SG6)
print("\nAbout 25 epochs are required until the gradient descent method has converged, using a learning rate of 0.01.")

#%%

# TOP 8 MOST CONFUSING IMAGES

best_beta = beta_SG4
predictions = sigmoid(X_val @ best_beta)
confusion = bce(y_val, predictions)

top8_idxs = np.argpartition(confusion, -8)[-8:]

def DisplayImages(images, rows, cols):
    figure, ax = plt.subplots(rows,cols)
    
    for i, ax in enumerate(ax.flatten()):
        img = images[i]
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()
  
def ConfusedImages(X, indices):
    imgs = []
    for idx in indices:
        img = X[idx]
        img = img.reshape((28,28))
        imgs.append(img)
    return imgs

confused_images = ConfusedImages(X_val_non_augmented, top8_idxs)
DisplayImages(confused_images, 2, 4)

print("\nAbove are the top 8 most confusing MNIST images for my model, which was built using stochastic gradient with a learning rate of 0.01. In this context, 'confusion' means that my model was confident in its prediction, yet incorrectly labeled the image nonetheless (either a false positive or a false negative).")
print("\nI chose these 8 images by first calculating a prediction vector with X_val and the beta vector from my best performing stochastic gradient descent model (+ sigmoid). Then, I calculated the binary cross entropy for each component of my prediction vector, against the ground-truth labels of y_val.")
print("By identifying the 8 indices with the greatest binary cross entropy, I am able to identify the most 'confident' false positives and false negatives.")