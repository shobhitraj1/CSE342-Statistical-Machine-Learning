import numpy as np
import matplotlib.pyplot as plt


dataset = np.load("mnist.npz")

# Visualize 5 samples from each class in the train set in form of images
class_samples = {i: [] for i in range(10)}
# print(class_samples)

images_num=0
category=0

# # Normalizing the values between 0 and 1
# x_train_normalized = dataset['x_train'] / 255.0
# x_test_normalized = dataset['x_train'] / 255.0

for category in range(10):
    while(images_num < len(dataset["y_train"])):
        if (category>=10):
            break
        if (len(class_samples[category])>=5):
            images_num=0
            category+=1
        else:
            if(dataset["y_train"][images_num]==category):
                class_samples[category].append(images_num)
        images_num+=1        
print(class_samples)

category=0
plt.figure(figsize=(10, 10))
for category in range(10):
    for j in range(5):
        plt.subplot(10, 5, category*5 + j + 1)
        plt.imshow(dataset["x_train"][class_samples[category][j]], cmap='gray')
        plt.axis('off')
        plt.suptitle('5 samples of the each category', fontsize=16)
plt.show()

# print(dataset["x_train"][0].flatten())
_lambda = 1e-6

covariance = []
means = []
samples = []
priors = []

# Computing mean vector and covariance vector for each class on the basis of train set
for i in range(10):
    current = dataset["x_train"][dataset["y_train"]==i]
    samples.append(len(current))
    # print(len(current))
    # plt.imshow(current[0])
    # plt.show()
    current = current.reshape(-1, 784)
    # print(current.shape)
    mean = np.mean(current, axis=0)
    # print(mean.shape)
    means.append(mean)
    cov = np.cov(current, rowvar=False)
    # print(cov.shape)
    priors.append(len(current)/len(dataset["x_train"]))
    if np.linalg.det(cov) == 0:
        covariance.append(cov + (_lambda * np.identity(current.shape[1])))
    else:
        covariance.append(cov)

    # sum(priors)

# print(covariance)
# Compute and store inverse of covariance, log(determinant of covariance) & log(priors)
# to avoid repeated redundant computation of these values for the 10 classes
inv_covariance = []
for i in range(10):
    inv_covariance.append(np.linalg.inv(covariance[i]))
det_covariance = []
for i in range(10):
    sign, const_term = np.linalg.slogdet(covariance[i])
    det_covariance.append(-0.5 * const_term + np.log(priors[i]))
# log_priors = []
# for i in range(10):
#     log_priors.append(np.log(priors[i]))
accuracy = np.zeros(10)
total_count = np.zeros(10)

with open('results1.txt', 'w') as file:
    file.write('The results of the test samples are :-\n')

x_test = dataset["x_test"].reshape(-1, 784) # Vectorize to 784-d

for cur in range(len(dataset["y_test"])):
    x = x_test[cur]
    qda_vals = []
    for i in range(10):
        log_det_cov = det_covariance[i]
        inv_cov = inv_covariance[i]
        mean = means[i]
        # QDA applied on x_test
        constant_term = log_det_cov # log(det(cov))
        exponent = 0.5 * np.dot(np.dot((x - mean).T, inv_cov), (x - mean)) 
        # prior = log_priors[i]
        qda_vals.append(constant_term - exponent) #+ prior)
    # print(qda_vals)
    print("Sample "+str(cur)+": Predicted: "+str(np.argmax(qda_vals)) +" Actual: "+ str(dataset["y_test"][cur]))
    with open('results1.txt', 'a') as file:
        file.write("Sample "+str(cur)+": Predicted: "+str(np.argmax(qda_vals)) +" Actual: "+ str(dataset["y_test"][cur])+"\n")
    if np.argmax(qda_vals) == dataset["y_test"][cur]:
        accuracy[dataset["y_test"][cur]] += 1
    total_count[dataset["y_test"][cur]] += 1
    if (cur % 500 == 0):
        print("Done: "+str(cur))

total_accuracy = 0

print("Class-wise accuracy :-")
with open('results1.txt', 'a') as file:
        file.write("\n")
        file.write("Class-wise accuracy :-"+"\n")
        file.write("\n")
for i in range(10):
    print("Class "+str(i)+ ": Predicted correctly = "+str(accuracy[i])+"; Total samples in testset = "+str(total_count[i]))
    class_accuracy = (accuracy[i]/total_count[i])*100
    total_accuracy += accuracy[i]
    print("Class "+str(i)+" accuracy = "+str(class_accuracy)+" %")
    with open('results1.txt', 'a') as file:
        file.write("Class "+str(i)+ ": Predicted correctly = "+str(accuracy[i])+"; Total samples in testset = "+str(total_count[i])+"\n")
        file.write("Class "+str(i)+" accuracy = "+str(class_accuracy)+" %"+"\n")
        file.write("\n")

overall_accuracy = (total_accuracy/len(dataset["y_test"]))*100
print("Overall accuracy = "+str(overall_accuracy)+" %")  
with open('results1.txt', 'a') as file:
        file.write("\n")
        file.write("Overall accuracy = "+str(overall_accuracy)+" %"+"\n")
        file.write("\n")
        
file.close()