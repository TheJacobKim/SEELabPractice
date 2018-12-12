############################################################################
test_data_file_name = 'sa12_test.choir_dat'  # has to be in the same folder
train_data_file_name = 'sa12_train.choir_dat'
testing_frequency = 1  # where 1 means after every iteration, 2 means after every 2 iteration, ...
boundary = 0.18  # Here you can decide what the boundary is for ternarizing the model
# For example, 0.18 means that all data that is in the range [-0.18*sigma, +0.18*sigma] becomes zero, everything below
# that is -1 and above is +1. If you want to use just binary, then set boundary = 0

############################################################################
import numpy as np
import joblib
import sys
import struct
import os


# This function creates the encoding (base list contains the list of ID's, X is the data to be encoded, D is the
# length of the hypervectors, nFeatures is the number of features
def encode(baseList, X, D, nFeatures):
    encoding_data = np.zeros(D)
    for i in range(0, nFeatures):
        encoding_data += X[i] * baseList[i]
    return encoding_data

# This is the function that opens the .choir_dat data and converts them into a normal list of arrays
def readChoirDat(filename):
    with open(filename, mode='rb') as f:
        nFeatures = struct.unpack('i', f.read(4))[0]
        nClasses = struct.unpack('i', f.read(4))[0]
        X = []
        y = []
        while True:
            newDP = []
            for i in range(nFeatures):
                v_in_bytes = f.read(4)
                if v_in_bytes is None or len(v_in_bytes) == 0:
                    return nFeatures, nClasses, X, y
                v = struct.unpack('f', v_in_bytes)[0]
                newDP.append(v)
            l = struct.unpack('i', f.read(4))[0]
            X.append(newDP)
            y.append(l)
    return nFeatures, nClasses, X, y


print('________________________________________________________')
print('Loading data')
D = 10000  # length of the hypervectors

# If there are already .pkl files with the encoded data present in the folder, we just load them, otherwise, the
# program encodes the data and stores them in the folder for future use
if (os.path.isfile('train_data.pkl') and os.path.isfile('test_data.pkl')):
    print('data ready for importing...')
    train_data = joblib.load('train_data.pkl')
    test_data = joblib.load('test_data.pkl')
    nFeatures, nClasses, x_test, t_test = readChoirDat(test_data_file_name)
    nFeatures2, nClasses2, x_train, t_train = readChoirDat(train_data_file_name)
    trainingNumber = np.shape(t_train)[0]
    testingNumber = np.shape(t_test)[0]
    print('...data imported')
else:
    print('Data will be prepared and saved...')
    nFeatures, nClasses, x_test, t_test = readChoirDat(test_data_file_name)
    nFeatures2, nClasses2, x_train, t_train = readChoirDat(train_data_file_name)
    trainingNumber = np.shape(t_train)[0]
    testingNumber = np.shape(t_test)[0]
    ID = list()  # This is the list of randomly generated IDs
    base = np.zeros(D)  # So this generates a list of zeros with size 10000
    for i in range(0, int(D / 2)):  # Then these for loops fill in 1 and -1, then get randomized.
        base[i] = 1
    for i in range(int(D / 2), D):
        base[i] = -1
    for i in range(nFeatures):
        ID.append(np.random.permutation(base))
    train_data = list()
    # print(ID)
    # Seems like ID is a list with length of numFeatures and random permutations if 1 & -1 array
    # The following for-loops create the encoded data by calling the "encode" function
    for t in range(trainingNumber):
        train_data.append(encode(ID, x_train[t], D, nFeatures))
    test_data = list()
    for t in range(testingNumber):
        test_data.append(encode(ID, x_test[t], D, nFeatures))
    joblib.dump(train_data, open("train_data.pkl", "wb"), compress=True)  # Saving encoded data as .pkl files
    print('Saved pickled train data')
    joblib.dump(test_data, open("test_data.pkl", "wb"), compress=True)  # Saving encoded data as .pkl files
    print('Saved pickled test data')

del x_test
del x_train
print('len t_test', str(len(t_test)))
print('len t_train', str(len(t_train)))
print("# of features: %d" % nFeatures)
print("# of classes: %d" % nClasses)
print('Data loaded and encoded')
print('Boundary =', boundary)
print('__________________________________')

classes = list()  # classes hypervectors
norms = list()  # In order to make the program run faster, we store the norms of all classes hypervectors, so we
# don't have to calculate it all the time
for i in range(nClasses):
    classes.append(np.zeros(D))
    norms.append(np.zeros(D))

### Initial training non-binarized ###
correct = 0
for i in range(0, trainingNumber):
    e = train_data[i]  # this is one encoded data point hypervector
    maxVal = -1
    maxLabel = -1
    currNorm = np.linalg.norm(e)  # Some linear algebra stuff. Norm of matrix
    for m in range(0, len(classes)):
        if np.count_nonzero(classes[m]) == 0:
            continue
        val = np.dot(e, classes[m]) / (currNorm * norms[m])  # cosine of a classes hypervect. with e
        if val > maxVal:
            maxVal = val
            maxLabel = m
    if maxLabel == t_train[i]:  # if matched correctly, we do nothing (just counting)
        correct += 1
    else:  # If matched incorrectly, we update the corresponding classes hypervector (and its norm)
        classes[t_train[i]] += e
        norms[t_train[i]] = np.linalg.norm(classes[t_train[i]])
acc = 1.0 * correct / (i + 1)
print('Done with initial non-binarized training. Acc=%.3f' % acc)

### Testing non-binarized model ###
for m in range(0, len(classes)):
    norms[m] = np.linalg.norm(classes[m])
correct = 0
for i in range(testingNumber):
    e = test_data[i]
    maxVal = -1
    maxLabel = -1
    currNorm = np.linalg.norm(e)
    for m in range(0, len(classes)):
        val = np.dot(e, classes[m]) / (currNorm * norms[m])
        if val > maxVal:
            maxVal = val
            maxLabel = m
    if maxLabel == t_test[i]:  # just testing this time, no updates madee
        correct += 1
acc2 = 1.0 * correct / (i + 1)
print('Accuracy of non-binarized model on test data (after initial training)= %.3f' % acc2)
print('__________________________________')

max_acc = 0
### Retraining non-binarized ###
for training_instance in range(40):
    if training_instance < 20:  # f is just the learning rate, you can adjust it if n
        f = 3.0
    else:
        f = 1.5
    correct = 0
    # now follows the retraining for loop. Same as initial training, just that this time we don't just add
    # to the right class, but also substract from the incorrectly predicted classes hypervecor
    for i in range(trainingNumber):
        e = train_data[i]
        maxVal = -1
        maxLabel = -1
        currNorm = np.linalg.norm(e)
        for m in range(0, len(classes)):
            val = np.dot(e, classes[m]) / (currNorm * norms[m])
            if val > maxVal:
                maxVal = val
                maxLabel = m
        if maxLabel == t_train[i]:
            correct += 1
        else:
            classes[maxLabel] -= f * e
            classes[t_train[i]] += f * e
            norms[t_train[i]] = np.linalg.norm(classes[t_train[i]])
            norms[maxLabel] = np.linalg.norm(classes[maxLabel])
    acc = 1.0 * correct / (i + 1)
    print('Retraining non-binarized. Instance =', str(training_instance), ' Acc=%.3f' % acc)
    for m in range(0, len(classes)):
        norms[m] = np.linalg.norm(classes[m])
    # The next for-loop is just testing what the accuracy after the last retrainign round is (it does not
    # always increase while training)
    correct = 0
    for i in range(testingNumber):
        e = test_data[i]
        maxVal = -1
        maxLabel = -1
        currNorm = np.linalg.norm(e)
        for m in range(0, len(classes)):
            val = np.dot(e, classes[m]) / (currNorm * norms[m])
            if val > maxVal:
                maxVal = val
                maxLabel = m
        if maxLabel == t_test[i]:
            correct += 1
    acc = 1.0 * correct / (i + 1)
    if acc > max_acc:
        max_acc = acc
    print('Accuracy of non-binarized model on test data (during training) = %.3f' % acc)

print('--> Max accuracy of non-binarized model on test data= %.3f <--' % acc)

assert (True)

print('_______________Switching to ternary model___________________')

### Testing ternary model ###
# The next for-loop creates empty zero-valued classes hypervecors for the ternarized model
classes_tern = list()
for i in range(np.shape(classes)[0]):
    classes_tern.append(np.zeros(D))
norm_classes_tern = np.zeros(np.shape(classes)[0])
# Now follows the important step! We will ternarize the model. The way we do is that we keep the floating point classes
# hypervectors (as we need them for retraiing) but create classes_tern hypervectors which will live in a -1,0,+1 space
# The way we decide about what should be -1 and what +1 is by using boundary*std. std (or std_classes in this case) is
# the constant defined multiplied by the standard deviation of that particular classes hypervecor (see below). If a
# value is below -boundary*std_classes, we make it -1, if it is above boundary*std_classes we make it +1. Everything
# else remains zero.
for i in range(np.shape(classes)[0]):
    std_classes = np.std(classes[i])
    for j in range(np.shape(classes)[1]):
        if classes[i][j] <= -boundary * std_classes:
            classes_tern[i][j] = -1
        if classes[i][j] >= boundary * std_classes:
            classes_tern[i][j] = 1
    classes_tern[i] = classes_tern[i].astype(int)
    norm_classes_tern[i] = np.linalg.norm(classes_tern[i])
correct = 0
std_test_data = np.std(test_data)
for i in range(testingNumber):
    # here we ternarize one of the encoded data points which we will use for testing (in this for-loop)
    e_tern = np.copy(test_data[i])  # we use np.copy as we don't want to mess up the test_data variable
    e_tern[abs(e_tern) <= boundary * std_test_data] = 0
    e_tern[e_tern >= boundary * std_test_data] = 1
    e_tern[e_tern <= -boundary * std_test_data] = -1
    e_tern = e_tern.astype(int)
    currNorm = np.linalg.norm(e_tern)
    maxVal = -1
    maxLabel = -1
    for m in range(len(classes_tern)):
        val = np.dot(e_tern, classes_tern[m])
        if val > maxVal:
            maxVal = val
            maxLabel = m
    if maxLabel == t_test[i]:
        correct += 1
acc1 = 1.0 * correct / (i + 1)
print('Accuracy of ternary model on test data (before training) = %.3f' % acc1)
# This accuracy (acc1) is just what we get after making the classes hypervectors, which we previously trained as floating
# point, ternary
print('__________________________________')

# now comes the key part. Here we do the ternary retraining
max_tern_acc = 0
f = 0.03  # learning rate...it has to be much smaller than for floating point retraining
### Training the ternary model ###
for training_instance in range(40):
    # in each retraining step, we have to ternarize the classes hypervectors again, as we update our floating point
    # model with each iteration
    classes_tern = list()
    for i in range(0, np.shape(classes)[0]):
        classes_tern.append(np.zeros(D))
    for i in range(np.shape(classes)[0]):
        std_classes = np.std(classes[i])
        for j in range(np.shape(classes)[1]):
            if classes[i][j] <= -boundary * std_classes:
                classes_tern[i][j] = -1
            if classes[i][j] >= boundary * std_classes:
                classes_tern[i][j] = 1
        classes_tern[i] = classes_tern[i].astype(int)
        norm_classes_tern[i] = np.linalg.norm(classes_tern[i])
    correct = 0
    std_test_data = np.std(test_data)
    for i in range(trainingNumber):
        e_tern = np.copy(train_data[i])  # same story, we ternarize the encoded input data
        e_tern[abs(e_tern) <= boundary * std_test_data] = 0
        e_tern[e_tern >= boundary * std_test_data] = 1
        e_tern[e_tern <= -boundary * std_test_data] = -1
        e_tern = e_tern.astype(int)
        currNorm = np.linalg.norm(e_tern)
        maxVal = -1
        maxLabel = -1
        # as we cannot update the ternary hypervectors, we just update the noramal (foating point) hypervectors
        # The trick is that we make the update of the floating point hypervectors based on the prediction the
        # ternarized hypervectors give us. At the end, we use this updated floating point model to create a new
        # set of ternarized classes hypervectors
        for m in range(0, len(classes_tern)):
            val = np.dot(e_tern, classes_tern[m])  # as this is ternary, no don't need to devide by the norms
            if val > maxVal:
                maxVal = val
                maxLabel = m
        if maxLabel == t_train[i]:
            correct += 1
        else:
            classes[maxLabel] -= f * train_data[i]
            classes[t_train[i]] += f * train_data[i]
    acc = 1.0 * correct / (i + 1)
    print('Retraining binarized. Instance =', str(training_instance), 'Acc=' + str(acc))
    # after each retraining round, we check what the accuracy is
    classes_tern = list()
    for i in range(np.shape(classes)[0]):
        classes_tern.append(np.zeros(D))
    norm_classes_tern = np.zeros(np.shape(classes)[0])
    for i in range(np.shape(classes)[0]):
        std_classes = np.std(classes[i])
        for j in range(np.shape(classes)[1]):
            if classes[i][j] <= -boundary * std_classes:
                classes_tern[i][j] = -1
            if classes[i][j] >= boundary * std_classes:
                classes_tern[i][j] = 1
        classes_tern[i] = classes_tern[i].astype(int)
        norm_classes_tern[i] = np.linalg.norm(classes_tern[i])
    correct = 0
    std_test_data = np.std(test_data)
    for i in range(testingNumber):
        e_tern = np.copy(test_data[i])
        e_tern[abs(e_tern) <= boundary * std_test_data] = 0
        e_tern[e_tern >= boundary * std_test_data] = 1
        e_tern[e_tern <= -boundary * std_test_data] = -1
        e_tern = e_tern.astype(int)
        currNorm = np.linalg.norm(e_tern)
        maxVal = -1
        maxLabel = -1
        for m in range(len(classes_tern)):
            val = np.dot(e_tern, classes_tern[m])  # /(norm_classes_tern[m] * currNorm)
            if val > maxVal:
                maxVal = val
                maxLabel = m
        if maxLabel == t_test[i]:
            correct += 1
    acc = 1.0 * correct / (i + 1)
    print('Accuracy of binarized model on test data (during retraining) = %.3f' % acc)
    if max_tern_acc < acc:
        max_tern_acc = acc

print('__________________________________')
print('Accuracy of non-bin, after initial training    = %.3f' % acc2)
print('Max accuracy non-bin                           = %.3f' % max_acc)
print('Accuracy of ternary, before ternary retraining = %.3f' % acc1)
print('Max accuracy ternary                           = %.3f' % max_tern_acc)
