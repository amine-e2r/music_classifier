import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("SpotifyFeatures.csv")
print("Number of samples (rows):", df.shape[0])
print("Number of features (columns):", df.shape[1])

df = df[(df["genre"] == "Pop") | (df["genre"] == "Classical")]

#We create the binary categories
df["Pop"] = (df["genre"] == "Pop").astype(int)
df["Classical"] = (df["genre"] == "Classical").astype(int)

print("Number of Pop samples:",df[(df["Pop"] == 1)].shape[0])
print("Number of Classical samples:",df[(df["Classical"] == 1)].shape[0])

df = df[["Pop","Classical","liveness", "loudness"]]

#We create the arrays
#Target vector
y = df["Pop"].to_numpy()
#Featurs matrix
X = df[["liveness", "loudness"]].to_numpy()

#Find the indexs of each genre
pop_index = np.where(y==1)[0]
classical_index = np.where(y==0)[0]

#Index for the split
split_pop = int(0.8 * len(pop_index))
split_classical = int(0.8 * len(classical_index))

#Create the training set
X_train = np.vstack([X[pop_index[:split_pop]], X[classical_index[:split_classical]]])
y_train = np.concatenate([y[pop_index[:split_pop]], y[classical_index[:split_classical]]])

#Create the test set
X_test = np.vstack([X[pop_index[split_pop:]], X[classical_index[split_classical:]]])
y_test = np.concatenate([y[pop_index[split_pop:]], y[classical_index[split_classical:]]])

# scatter pop and classical
plt.scatter(
    df[df["Pop"]==1]["liveness"], 
    df[df["Pop"]==1]["loudness"], 
    alpha=0.5, label="Pop", s=10, c="blue"
)

# Classical samples
plt.scatter(
    df[df["Classical"]==1]["liveness"], 
    df[df["Classical"]==1]["loudness"], 
    alpha=0.5, label="Classical", s=10, c="red"
)

# Labels and legend
plt.xlabel("Liveness")
plt.ylabel("Loudness")
plt.title("Samples loudness vs liveness")
plt.legend()
plt.show()

#Logistic regression
#sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#gradients
def gradient_dw(x, y, w, b):
    y_hat = sigmoid(np.dot(w, x) + b)
    dw = (y_hat - y) * x
    return dw

def gradient_db(x, y, w, b):
    y_hat = sigmoid(np.dot(w, x) + b)
    db = y_hat - y
    return db

def sample_loss(x, y, w, b):
    y_hat = sigmoid(np.dot(w, x) + b)
    return - (y * np.log(y_hat + 1e-15) + (1 - y) * np.log(1 - y_hat + 1e-15))
    # +1e-15 to avoid 0 in the log

def logistic_sgd(X,y,epochs=100,lr=0.1):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)  # weights
    b = 0 #biais
    losses = []
    for epoch in range(epochs):
        #shuffle
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        epoch_loss = 0
        
        
        for i in range(n_samples):
            x_i = X_shuffled[i]
            y_i = y_shuffled[i]
        
            #compute gradents
            dw = gradient_dw(x_i, y_i, w, b)
            db = gradient_db(x_i, y_i, w, b)
            
            #update parameters
            w -= lr * dw
            b -= lr * db

            epoch_loss += sample_loss(x_i, y_i, w, b)
        
        losses.append(epoch_loss / n_samples)
    return w, b, losses

"""
#Visualisation for different lr
learning_rates = [0.001, 0.01, 0.1, 0.5]
epochs = 50  # number of epochs
loss_curves = {}

for lr in learning_rates:
    w, b, losses = logistic_sgd(X_train, y_train, epochs=epochs, lr=lr)
    loss_curves[lr] = losses

plt.figure(figsize=(8,5))
for lr, losses in loss_curves.items():
    plt.plot(range(1, epochs+1), losses, label=f"lr={lr}")
plt.xlabel("Epochs")
plt.ylabel("Average Loss")
plt.title("Training Loss for Different Learning Rates")
plt.legend()
plt.grid(True)
plt.show()
"""
#train + predict

def predict(X, w, b):
    y_hat = sigmoid(np.dot(X, w) + b)
    return (y_hat >= 0.5).astype(int)

w, b, losses = logistic_sgd(X_train, y_train, epochs=10, lr=0.01)

#plot scatter + reglin
# Scatter plot
plt.figure(figsize=(8,6))
plt.scatter(df[df["Pop"]==1]["liveness"], df[df["Pop"]==1]["loudness"], alpha=0.5, s=10, label="Pop",c="blue")
plt.scatter(df[df["Classical"]==1]["liveness"], df[df["Classical"]==1]["loudness"], alpha=0.5, s=10, label="Classical", c="red")

# Decision boundary: w^T x + b = 0
x_vals = np.linspace(df["liveness"].min(), df["liveness"].max(), 100)
y_vals = -(w[0] * x_vals + b) / w[1]

plt.plot(x_vals, y_vals, 'r-', linewidth=2, label="Decision boundary",c="black")

plt.xlabel("Liveness")
plt.ylabel("Loudness")
plt.title("Decision boundary of logistic regression")
plt.legend()
plt.grid(True)
plt.show()



y_train_pred = predict(X_train, w, b)
train_accuracy = np.mean(y_train_pred == y_train)
print("Training set accuracy:", train_accuracy)

y_test_pred = predict(X_test, w, b)
test_accuracy = np.mean(y_test_pred == y_test)
print("Test set accuracy:", test_accuracy)

#problem 3
TP = np.sum((y_test == 1) & (y_test_pred == 1))  # True Positives
TN = np.sum((y_test == 0) & (y_test_pred == 0))  # True Negatives
FP = np.sum((y_test == 0) & (y_test_pred == 1))  # False Positives
FN = np.sum((y_test == 1) & (y_test_pred == 0))  # False Negatives

confusion_matrix = np.array([[TP, FN],
                             [FP, TN]])

print("Confusion Matrix:")
print(confusion_matrix)