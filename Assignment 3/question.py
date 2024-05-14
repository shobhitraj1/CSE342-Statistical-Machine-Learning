import numpy as np
from statistics import mode

dataset = np.load("mnist.npz")
x_train = dataset['x_train']
y_train = dataset['y_train']
x_test = dataset['x_test']
y_test = dataset['y_test']

# Choose samples from 0,1 and 2 class in X & store the true label in Y_true
X = []
Y_true = []
for i in range(len(x_train)):
    if (y_train[i] == 0 or y_train[i] == 1 or y_train[i] == 2):
        Y_true.append(y_train[i])
        X.append(x_train[i].flatten())
X_array1 = np.array(X)

# print(X_array.shape)
# print(len(Y_true))

mean = np.mean(X_array1,axis = 0)
# print(mean)
X_array = X_array1 - mean # Centralized X
total_samples = len(Y_true)

# Applying PCA
S = np.dot(X_array.T, X_array) / (total_samples-1)
# print(S.shape)

eigenvalues, eigenvectors = np.linalg.eig(S)

# Sort them in desc order on the basis of eigenvalues & create matrix U of eigenvectors
idxs = sorted([i for i in range(len(eigenvalues))], key = lambda x: -eigenvalues[x])

U = []
for i in idxs:
    U.append(eigenvectors.T[i])
    
U = np.array(U).T
# print(U.shape)
# print(X_array.shape)

p = 10
Up = U[:, :p]
# U_P Y = U_P (U_P'X)
# print(Up.shape)

# Y = U_p'X --> projected x_train i.e. dimensionally reduced dataset
Y = np.real(np.dot(Up.T, X_array.T)).T
# print(Y.shape)
# Y here is the reduced dimension dataset
Y = Y/255 # Normalizing
# print(Y)

# Y --> dimensionally reduced dataset of the x_train with dimensions 18623 X 10
# Y_true --> true labels of samples in Y

# Choose the test samples from 0,1 and 2 class in X_T & store the true label in Y_test_true
X_T =[]
Y_test_true = []
for i in range(len(x_test)):
    if (y_test[i] == 0 or y_test[i] == 1 or y_test[i] == 2):
        Y_test_true.append(y_test[i])
        X_T.append(x_test[i].flatten())
X_Tarray1 = np.array(X_T)
X_Tarray = X_Tarray1 - mean
# print(X_Tarray.shape)
# print(len(Y_test_true))

# Y = U_p'x_test --> projected x_test i.e. dimensionally reduced test dataset
Y_test = np.real(np.dot(Up.T, X_Tarray.T)).T
# print(Y_test.shape)
# Y_test here is the reduced dimension test dataset
Y_test = Y_test/255 # Normalizing
# print(Y_test)

# Y_test --> dimensionally reduced dataset of the x_test with dimensions 3147 X 10
# Y_test_true --> true labels of samples in Y

def calc_split(dataset, true_label):
    # find gini index for first split across all the 10 dimensions
    feature = []
    averages = []
    gini = []
    for k in range(10):
        feature = []
        for i in range(len(dataset)):
            feature.append(dataset[i][k])
        # print(len(feature))
        f = np.array(feature)
        fmid = np.mean(f)
        # print(fmid)
        averages.append(fmid)
        left = {i: 0 for i in range(3)}
        right = {i: 0 for i in range(3)}
        for i in range(len(dataset)):
            if (dataset[i][k] < fmid):
                left[true_label[i]] += 1
            else:
                right[true_label[i]] += 1
                
        # print(left[0]+left[1]+left[2]+right[0]+right[1]+right[2])
        P_left = (left[0]+left[1]+left[2])/(left[0]+left[1]+left[2]+right[0]+right[1]+right[2])
        P_right = (right[0]+right[1]+right[2])/(left[0]+left[1]+left[2]+right[0]+right[1]+right[2])
        P_l0 = left[0]/(left[0]+left[1]+left[2])
        P_l1 = left[1]/(left[0]+left[1]+left[2])
        P_l2 = left[2]/(left[0]+left[1]+left[2])
        P_r0 = right[0]/(right[0]+right[1]+right[2])
        P_r1 = right[1]/(right[0]+right[1]+right[2])
        P_r2 = right[2]/(right[0]+right[1]+right[2])
        gini_left = P_l0*(1-P_l0)+P_l1*(1-P_l1)+P_l2*(1-P_l2)
        gini_right = P_r0*(1-P_r0)+P_r1*(1-P_r1)+P_r2*(1-P_r2)
        total_gini = P_left*gini_left + P_right*gini_right
        gini.append(total_gini)
        
    # print(left)
    # print(right)
    # print(averages)
    # print(gini)
    min_index = gini.index(min(gini))
    return min_index, gini, averages

def make_split(dataset, true_label, split_dim):
    # splitting across the 1st dimension
    feature = []
    gini = []
    left1 = []
    right1 = []
    left_labels = []
    right_labels = []
    for i in range(len(dataset)):
        feature.append(dataset[i][split_dim])
    # print(len(feature))
    f = np.array(feature)
    fmid = np.mean(f)
    # print(fmid)
    left = {i: 0 for i in range(3)}
    right = {i: 0 for i in range(3)}
    for i in range(len(dataset)):
        if (dataset[i][split_dim] < fmid):
            left1.append(dataset[i])
            left_labels.append(true_label[i])
            left[true_label[i]]+=1
        else:
            right1.append(dataset[i])
            right_labels.append(true_label[i])
            right[true_label[i]]+=1
    # print(len(left1))
    # print(len(right1))
    # print(len(left_labels))
    # print(len(right_labels))
    # print(left)
    # print(right)
    left_feature = []
    right_feature = []
    left_averages = []
    right_averages = []
    for k in range(10):
        left_feature = []
        right_feature = []
        for i in range(len(left1)):
            left_feature.append(left1[i][k])
        for j in range(len(right1)):
            right_feature.append(right1[j][k])
        # print(len(left_feature))
        # print(len(right_feature))
        f_left1 = np.array(left_feature)
        fmid_left = np.mean(f_left1)
        f_right1 = np.array(right_feature)
        fmid_right = np.mean(f_right1)
        # print(fmid)
        left_averages.append(fmid_left)
        right_averages.append(fmid_right)
    # print(left_averages)
    # print(right_averages)
    return left1, right1, left_labels, right_labels, left, right

def predict_labels_2left(test_dataset, test_labels, split1, average1, split2, average2):
    correct = {i: 0 for i in range(3)}
    test_samples = {i: 0 for i in range(3)}
    for i in range(len(test_dataset)):
        test_samples[test_labels[i]] += 1
        if (test_dataset[i][split1] < average1[split1]):
            if (test_dataset[i][split2] < average2[split2]):
                if (test_labels[i] == region1_pred):
                    correct[region1_pred] += 1
            else:
                if (test_labels[i] == region2_pred):
                    correct[region2_pred] += 1
        else:
            if (test_labels[i] == region3_pred):
                correct[region3_pred] += 1
                
    total_correct = 0         
    total_samples = 0
    for i in range(len(test_samples)):
        total_correct += correct[i]
        total_samples += test_samples[i]
        print(f"Class {i} accuracy = {correct[i]*100/test_samples[i]} %")
    print(f"Overall accuracy = {total_correct*100/total_samples} %")          

def predict_labels_2right(test_dataset, test_labels, split1, average1, split2, average2):
    correct = {i: 0 for i in range(3)}
    test_samples = {i: 0 for i in range(3)}
    for i in range(len(test_dataset)):
        test_samples[test_labels[i]] += 1
        if (test_dataset[i][split1] < average1[split1]):
            if (test_labels[i] == region1_pred):
                correct[region1_pred] += 1
        else:
            if (test_dataset[i][split2] < average2[split2]):
                if (test_labels[i] == region2_pred):
                    correct[region2_pred] += 1
            else:
                if (test_labels[i] == region3_pred):
                    correct[region3_pred] += 1
                    
    total_correct = 0         
    total_samples = 0
    for i in range(len(test_samples)):
        total_correct += correct[i]
        total_samples += test_samples[i]
        print(f"Class {i} accuracy = {correct[i]*100/test_samples[i]} %")
    print(f"Overall accuracy = {total_correct*100/total_samples} %")  
    
def predicting_left(test_dataset, split1, average1, split2, average2):
    for i in range(len(test_dataset)):
        if (test_dataset[i][split1] < average1[split1]):
            if (test_dataset[i][split2] < average2[split2]):
                predictions[i].append(region1_pred)
            else:
                predictions[i].append(region2_pred)
        else:
            predictions[i].append(region3_pred)
                
def predicting_right(test_dataset, split1, average1, split2, average2):
    for i in range(len(test_dataset)):
        if (test_dataset[i][split1] < average1[split1]):
            predictions[i].append(region1_pred)
        else:
            if (test_dataset[i][split2] < average2[split2]):
                predictions[i].append(region2_pred)
            else:
                predictions[i].append(region3_pred)


# find gini index for first split across all the 10 dimensions & the first dimension with minimum gini index
split1_dim, gini, averages = calc_split(Y, Y_true)
# print(split1_dim)

# Now, make the split across the split1_dim and find the left1 and right1 datasets
left1, right1, left_labels, right_labels, left_count, right_count = make_split(Y, Y_true, split1_dim)

# Now we have 2 spaces 10-D each, left1 and right1, now same procedure again

# find gini index for second split across all the 10 dimensions if chosen left region
split2left_dim, gini_left, left_averages = calc_split(left1, left_labels)
# print(-split2left_dim)
# print(gini_left)
# print(left_averages)

# find gini index for second split across all the 10 dimensions if chosen right region
split2right_dim, gini_right, right_averages = calc_split(right1, right_labels)
# print(split2right_dim)
# print(gini_right)
# print(right_averages)


# Now, we have the minimum gini index for 2nd split on both the left and right region after making the first split
# taking the minimum gini of both the splits & make the 2nd split to obtain the decision tree & making the predictions
split1 = split1_dim
average1 = averages

if (gini_left[split2left_dim] < gini_right[split2right_dim]):
    # 2nd split is in left region across split2left_dim
    left_left1, left_right1, left_left_labels, left_right_labels, left_left_count, left_right_count = make_split(left1, left_labels, split2left_dim)
    
    region1_pred = max(left_left_count, key=lambda k: left_left_count[k]) # Region1 prediction = left_left_count dict mode
    region2_pred = max(left_right_count, key=lambda k: left_right_count[k]) # Region2 prediction = left_right_count dict mode
    region3_pred = max(right_count, key=lambda k: right_count[k]) # Region3 prediction = right_count dict mode
    # print(region1_pred, region2_pred, region3_pred)
    split2 = split2left_dim
    average2 = left_averages
    predict_labels_2left(Y_test, Y_test_true, split1, average1, split2, average2)
else:
    # 2nd split is in right region across split2right_dim
    right_left1, right_right1, right_left_labels, right_right_labels, right_left_count, right_right_count = make_split(right1, right_labels, split2right_dim)

    region1_pred = max(left_count, key=lambda k: left_count[k]) # Region1 prediction = left_count dict mode
    region2_pred = max(right_left_count, key=lambda k: right_left_count[k]) # Region2 prediction = right_left_count dict mode
    region3_pred = max(right_right_count, key=lambda k: right_right_count[k]) # Region3 prediction = right_right_count dict mode
    # print(region1_pred, region2_pred, region3_pred)
    split2 = split2right_dim
    average2 = right_averages
    predict_labels_2right(Y_test, Y_test_true, split1, average1, split2, average2)


# Now, using Bagging

# Create 5 datasets randomly with replacement from Y along with their true labels

DATASETS_5 = []
DATASETS_5_labels = []
DATASETS_TEST = []

for k in range(5):
    d = []
    d_labels = []
    for i in range(len(X_array1)):
        rand_no = np.random.randint(0, len(X_array1))
        d.append(X_array1[rand_no])
        d_labels.append(Y_true[rand_no])
    X_array2 = np.array(d)
    mean = np.mean(X_array2,axis = 0)
    X_array = X_array2 - mean # Centralized X
    total_samples = len(Y_true)

    # Applying PCA
    S = np.dot(X_array.T, X_array) / (total_samples-1)

    eigenvalues, eigenvectors = np.linalg.eig(S)

    # Sort them in desc order on the basis of eigenvalues & create matrix U of eigenvectors
    idxs = sorted([i for i in range(len(eigenvalues))], key = lambda x: -eigenvalues[x])

    U = []
    for i in idxs:
        U.append(eigenvectors.T[i])
        
    U = np.array(U).T

    p = 10
    Up = U[:, :p]

    # Y = U_p'X --> projected x_train i.e. dimensionally reduced dataset
    Y = np.real(np.dot(Up.T, X_array.T)).T
    # Y here is the reduced dimension dataset
    Y = Y/255 # Normalizing
    # print(Y.shape)
    X_Tarray = X_Tarray1 - mean

    # Y = U_p'x_test --> projected x_test i.e. dimensionally reduced test dataset
    Y_test = np.real(np.dot(Up.T, X_Tarray.T)).T
    # print(Y_test.shape)
    # Y_test here is the reduced dimension test dataset
    Y_test = Y_test/255 # Normalizing
    DATASETS_TEST.append(Y_test)
    DATASETS_5.append(Y)
    DATASETS_5_labels.append(d_labels)

# for k in range(5):
#     d = []
#     d_labels = []
#     for i in range(len(Y)):
#         rand_no = np.random.randint(0, len(Y))
#         d.append(Y[rand_no])
#         d_labels.append(Y_true[rand_no])
#     d = np.array(d)
#     DATASETS_5.append(d)
#     DATASETS_5_labels.append(d_labels)

# print(len(DATASETS_5))
# print(len(DATASETS_TEST))
# print(len(DATASETS_TEST[0]))
# print(len(DATASETS_5[0]))
# print(len(DATASETS_5_labels[0]))

predictions = {i: [] for i in range(len(Y_test))}

# Learning trees for all these datasets & get the predictions by each tree in predictions dict

for k in range(5):
    split1_dim, gini, averages = calc_split(DATASETS_5[k], DATASETS_5_labels[k])
    left1, right1, left_labels, right_labels, left_count, right_count = make_split(DATASETS_5[k], DATASETS_5_labels[k], split1_dim)
    split2left_dim, gini_left, left_averages = calc_split(left1, left_labels)
    split2right_dim, gini_right, right_averages = calc_split(right1, right_labels)
    split1 = split1_dim
    average1 = averages

    if (gini_left[split2left_dim] < gini_right[split2right_dim]):
        # 2nd split is in left region across split2left_dim
        left_left1, left_right1, left_left_labels, left_right_labels, left_left_count, left_right_count = make_split(left1, left_labels, split2left_dim)
        
        region1_pred = max(left_left_count, key=lambda k: left_left_count[k]) # Region1 prediction = left_left_count dict mode
        region2_pred = max(left_right_count, key=lambda k: left_right_count[k]) # Region2 prediction = left_right_count dict mode
        region3_pred = max(right_count, key=lambda k: right_count[k]) # Region3 prediction = right_count dict mode
        # print(region1_pred, region2_pred, region3_pred)
        split2 = split2left_dim
        average2 = left_averages
        predicting_left(DATASETS_TEST[k], split1, average1, split2, average2)
    else:
        # 2nd split is in right region across split2right_dim
        right_left1, right_right1, right_left_labels, right_right_labels, right_left_count, right_right_count = make_split(right1, right_labels, split2right_dim)

        region1_pred = max(left_count, key=lambda k: left_count[k]) # Region1 prediction = left_count dict mode
        region2_pred = max(right_left_count, key=lambda k: right_left_count[k]) # Region2 prediction = right_left_count dict mode
        region3_pred = max(right_right_count, key=lambda k: right_right_count[k]) # Region3 prediction = right_right_count dict mode
        # print(region1_pred, region2_pred, region3_pred)
        split2 = split2right_dim
        average2 = right_averages
        predicting_right(DATASETS_TEST[k], split1, average1, split2, average2)
    
    # print(split1, average1, split2, average2)

# print(predictions)
# print(len(predictions))

predicted_labels = []
for i in range(len(predictions)):
    predicted_labels.append(mode(predictions[i]))

# print(predicted_labels)
# print(len(predicted_labels))


correct = {i: 0 for i in range(3)}
test_samples = {i: 0 for i in range(3)}
for i in range(len(Y_test_true)):
    test_samples[Y_test_true[i]] += 1
    if (Y_test_true[i] == predicted_labels[i]):
        correct[Y_test_true[i]] += 1

# print(correct)
# print(test_samples)

print("\nWith Bagging :- ")
total_correct = 0         
total_samples = 0
for i in range(len(test_samples)):
    total_correct += correct[i]
    total_samples += test_samples[i]
    print(f"Class {i} accuracy = {correct[i]*100/test_samples[i]} %")
print(f"Overall accuracy = {total_correct*100/total_samples} %")  