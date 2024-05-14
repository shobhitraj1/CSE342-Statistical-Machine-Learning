import numpy as np
import matplotlib.pyplot as plt

dataset = np.load("mnist.npz")

X = []

# Choose 100 samples from each class and create a 784 X 1000 data matrix called X
for i in range(10):
    X.append(dataset["x_train"][dataset["y_train"]==i][:100].reshape(-1, 784))

X = np.array(X)
X = X.reshape(1000, 784)
mean = np.mean(X)
X = X - mean # Centralized X

# Applying PCA
S = np.dot(X.T, X) / 999

eigenvalues, eigenvectors = np.linalg.eig(S)

# Sort them in desc order on the basis of eigenvalues & create matrix U of eigenvectors
idxs = sorted([i for i in range(len(eigenvalues))], key = lambda x: -eigenvalues[x])

U = []
for i in idxs:
    U.append(eigenvectors.T[i])

U = np.array(U).T
# print(U.shape)
# print(X.shape)

# Y = U'X and X_recon = UY
Y = np.dot(U.T, X.T)
# print(Y.shape)
X_recon = np.dot(U, Y).T
# print(X_recon.shape)
MSE = abs(np.square(X-X_recon).mean())
print(f"MSE between X and X_recon is = {MSE}")
with open('results2.txt', 'w') as file:
        file.write("\n")
        file.write(f"MSE between X and X_recon is = {MSE}\n")

p_values = [5,10,20]

for p in p_values:
    Up = U[:, :p]
    # print(Up.shape)
    # print(X.shape)
    # U_P Y = U_P (U_P'X)

    # reshaping image and reconstructing image
    images = np.absolute((np.dot(Up, np.dot(Up.T, X.T)) + mean).reshape(28, 28, -1))
    # print(images.shape)
    # num = 901
    # plt.imshow(images[:, :, num])
    # plt.show()

    plt.figure(figsize=(10, 10))
    for category in range(10):
        start_index = category * 100
        for i in range(5):
            image = images[:, :, start_index + i]
            plt.subplot(10, 5, category * 5 + i + 1)
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.suptitle(f'5 reconstructed samples of the each category for p = {p}', fontsize=16)
    plt.show()


def qda_function(x, log_det_cov, inv_cov, mean):
    constant_term = log_det_cov
    exponent = 0.5 * np.dot(np.dot((x - mean).T, inv_cov), (x - mean))
    return constant_term - exponent

p_values = [5,10,20]

for p in p_values:
    x_test = dataset["x_test"].reshape(-1, 784)
    Up = U[:, :p]

    # Y = U_p'x_test --> projected x_test
    Y = np.absolute(np.dot(Up.T, x_test.T)).T
    # print(Y.shape)
    _lambda = 1e-6

    covariance = []
    means = []
    samples = []
    priors = []

    # new x_train = U_P'(x_train) --> projected x_train
    new_x_train = np.absolute(np.dot(Up.T, dataset["x_train"].reshape(-1, 784).T))

    for i in range(10):
        # current = np.absolute(np.dot(Up.T, X[i*100:(i+1)*100].T) + mean).T
        current = new_x_train.T[dataset["y_train"]==i]
        means.append(np.mean(current, axis=0))
        # print(means[0].shape)
        cov = np.cov(current, rowvar = False)
        # print(cov.shape)
        priors.append(len(current)/len(dataset["x_train"]))
        covariance.append(cov + (_lambda * np.identity(current.shape[1])))

    inv_covariance = []
    for i in range(10):
        inv_covariance.append(np.linalg.inv(covariance[i]))
    det_covariance = []
    for i in range(10):
        sign, const_term = np.linalg.slogdet(covariance[i])
        det_covariance.append(-0.5 * const_term + np.log(priors[i]))

    accuracy = np.zeros(10)
    total_count = np.zeros(10)

    # applying QDA on Y
    for cur in range(len(dataset["y_test"])):
        qda_vals = []
        for i in range(10):
            qda_vals.append(qda_function(Y[cur], det_covariance[i], inv_covariance[i], means[i]))
        if np.argmax(qda_vals) == dataset["y_test"][cur]:
            accuracy[dataset["y_test"][cur]] += 1
        total_count[dataset["y_test"][cur]] += 1

    accurate = np.sum(accuracy)
    counts = np.sum(total_count)
    total_accuracy = accurate / counts
    print(f"\nFor p = {p} :-")
    with open('results2.txt', 'a') as file:
        file.write(f"\nFor p = {p} :-\n")
    for i in range(10):
        print(f"Class {i} accuracy = {accuracy[i]*100/total_count[i]} %")
        with open('results2.txt', 'a') as file:
            file.write(f"Class {i} accuracy = {accuracy[i]*100/total_count[i]} %\n")
    print(f"\nAccuracy for p = {p} = {total_accuracy*100} %")
    with open('results2.txt', 'a') as file:
        file.write(f"\nAccuracy for p = {p} = {total_accuracy*100} %\n")

file.close()
