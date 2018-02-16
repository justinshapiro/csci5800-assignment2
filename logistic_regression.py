import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as cv_metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Custom logistic regression function which produces a model (function), parameter w, and cross-validation metrics
# Input: training_data: used to train a model, refining w
#        num_epochs: the number of times we run gradient descent before settling for a value of w
#        learning_rate:
#        normalize: whether or not we normalize attributes through standardization
#        regularize: whether or not we perform regularization on w. L_2 regularization by default
#        validation_set: if None is provided, we split the training data to perform cross-validation
#                        otherwise, we split the validation set to perform cross-validation
# Output: h(x, w): the logistic regression hypothesis function
#         w: the value of w that we obtained from the training data, in other words: this is the model parameter
#         cross_validation_metrics: object that contains the accuracy, precision, recall and F1-Score of the model
def logistic_regression_model(training_data,
                              num_epochs,
                              learning_rate,
                              normalize=False,
                              min_max_standardized=False,
                              regularize=False,
                              regularization_parameter=None):
    if normalize is False and min_max_standardized is True:
        raise AttributeError("Cannot perform Min/Max standardization if normalize is set to False")

    if regularize is False and regularization_parameter is not None:
        raise AttributeError("regularize must be true to make use of the regularization parameter in regularization")
    elif regularize is True and regularization_parameter is None:
        raise AttributeError("A regularization parameter is required in order to perform regularization")

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def h(x, w, normalize=normalize):
        if normalize:
            if type(x) is not pd.DataFrame:
                x = pd.DataFrame(x).drop(0, axis=1)
            else:
                x = x.drop('bias', axis=1)
            if not min_max_standardized:
                x = pd.DataFrame(StandardScaler().fit_transform(x))
            else:
                x = pd.DataFrame(MinMaxScaler().fit_transform(x))
            x['bias'] = np.ones(x.shape[0])
            x = x[[x.columns.tolist()[-1]] + x.columns.tolist()[:-1]]

        # this prediction function should generalize to batch and single predictions
        if x.shape[0] != w.shape[0]:
            w = np.tile(w, (x.shape[0], 1))

        # obtaining thousands of predictions with a for loop takes HOURS on CPU
        # this is a vectorized implementation of batch prediction that completes in only seconds
        # swap the operands before multiply to ensure the prediction results exit in the resultant matrix's diagonal
        # although the operands are swapped, this will also work for a single prediction
        return sigmoid(np.diagonal(np.matmul(x, np.transpose(w))))  # must have 64-bit Python for this to work

    # perform Gradient Descent (batch version) to find w
    def gradient_descent(X_input, y_input):
        w_val = np.random.uniform(size=(X_input.shape[1],))
        for epoch in np.arange(0, num_epochs):
            error = h(X_input, w_val) - y_input
            loss = np.sum(error ** 2)
            gradient = X_input.T.dot(error)

            if regularize is False:
                w_val -= learning_rate * gradient
            else:
                w_val -= (learning_rate * gradient) - (regularization_parameter * w_val)

        return w_val

    X_train = training_data[0]
    y_train = training_data[1]

    if normalize:
        X_train = X_train.drop('bias', axis=1)
        if not min_max_standardized:
            X_train = pd.DataFrame(StandardScaler().fit_transform(X_train))
        else:
            X_train = pd.DataFrame(MinMaxScaler().fit_transform(X_train))
        X_train['bias'] = np.ones(X_train.shape[0])
        X_train = X_train[[X_train.columns.tolist()[-1]] + X_train.columns.tolist()[:-1]]

    w = gradient_descent(X_train, y_train)

    # perform 10-fold cross-validation to determine how accurate w is for h(x, w)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    kf = KFold(n_splits=10, shuffle=True, random_state=rng_seed)
    for train_index, test_index in kf.split(X_train):
        X_train_new, X_test_new = X_train.as_matrix()[train_index], X_train.as_matrix()[test_index]
        y_train_new, y_test_new = y_train.as_matrix()[train_index], y_train.as_matrix()[test_index]

        w_new = gradient_descent(X_train_new, y_train_new)
        y_pred = h(X_test_new, w_new)

        try:
            current_metrics = cv_metrics(y_test_new, y_pred, average=None)
        except ValueError:
            current_metrics = cv_metrics(y_test_new, y_pred.round(), average=None)
            pass

        try:
            accuracies.append(accuracy_score(y_test_new, y_pred))
        except ValueError:
            accuracies.append(accuracy_score(y_test_new, y_pred.round()))
            pass

        precisions.append(current_metrics[0][0])
        recalls.append(current_metrics[1][0])
        f1_scores.append(current_metrics[2][0])

    # return the model function, model parameter, and cross-validation metrics
    return h, w, CrossValidationMetricsHolder(
        accuracy=accuracies,
        precision=precisions,
        recall=recalls,
        f1_score=f1_scores
    )


# Custom class to containing all the required cross-validation metrics to evaluate model performance
# This class does not perform cross-validation, but only reports metrics in an object
class CrossValidationMetricsHolder(object):
    accuracy = None
    precision = None
    recall = None
    f1_score = None

    def __init__(self, accuracy, precision, recall, f1_score):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score

    def to_list(self):
        return [self.accuracy, self.precision, self.recall, self.f1_score]


# Converts columns with non-numerical values to encoded numerical values, if they exist
# Also returns the number of yes and no targets
def encode_labels_if_needed(dataset):
    # get the columns that do not have numerical values
    non_numerical_columns = []
    for column in dataset.columns:
        for value in dataset[column]:
            if not (type(value) == int or type(value) == float):
                non_numerical_columns.append(column)
                break

    # use scikit-learn's LabelEncoder to translate non-numerical columns to numerical ones
    label_encoder = LabelEncoder()
    for non_numerical_column in non_numerical_columns:
        dataset[non_numerical_column] = label_encoder.fit_transform(dataset[non_numerical_column])

    # obtain a count of yes/no labels for validation
    yes_targets = 0
    no_targets = 0
    for value in dataset['y']:
        if value == 0:
            no_targets += 1
        elif value == 1:
            yes_targets += 1

    return dataset, yes_targets, no_targets


def run_logistic_regression(min_max_scaling=False, regular_scaling=False, regularize=False):
    # From 2: "From the convergence test, choose option 1 with different nEpoch parameters from {100, 500, 1000}"
    epoch_parameters = [100, 500, 1000]

    # Repeat this step with 3 different learning rates a = {0.01, 0.1, 1}
    learning_rates = [0.01, 0.1, 1]

    # regularization parameters
    regularization_parameters = [0, 1, 10, 100, 1000]

    for alpha in learning_rates:
        print("For alpha=" + str(alpha) + ": ")

        for num_epochs in epoch_parameters:
            print("   -> For nEpochs=" + str(num_epochs) + ": ")
            if not regularize:
                h, w, training_set_performance = logistic_regression_model(
                    training_data=[X_train_provided, y_train_provided],
                    num_epochs=num_epochs,
                    learning_rate=alpha,
                    normalize=min_max_scaling or regular_scaling,
                    min_max_standardized=min_max_scaling
                )

                avg_training_set_performance = CrossValidationMetricsHolder(
                    accuracy=np.mean(training_set_performance.accuracy, axis=0),
                    precision=np.mean(training_set_performance.precision, axis=0),
                    recall=np.mean(training_set_performance.recall, axis=0),
                    f1_score=np.mean(training_set_performance.f1_score, axis=0)
                )

                y_pred = h(X_test_provided, w)

                try:
                    testing_metrics = cv_metrics(y_test_provided, y_pred, average=None)
                except ValueError:
                    testing_metrics = cv_metrics(y_test_provided, y_pred.round(), average=None)
                    pass

                try:
                    accuracy = accuracy_score(y_test_provided, y_pred)
                except ValueError:
                    accuracy = accuracy_score(y_test_provided, y_pred.round())
                    pass

                testing_set_performance = CrossValidationMetricsHolder(
                    accuracy=accuracy,
                    precision=testing_metrics[0][0],
                    recall=testing_metrics[1][0],
                    f1_score=testing_metrics[2][0]
                )

                for j in range(0, 10, 1):
                    print("      -> CV Step " + str(j + 1) + ": accuracy=" + str(training_set_performance.accuracy[j]),
                          end="")
                    print(" | precision=" + str(training_set_performance.precision[j]), end="")
                    print(" | recall=" + str(training_set_performance.recall[j]), end="")
                    print(" | F1-score=" + str(training_set_performance.f1_score[j]))

                print("      -> Average of CV metrics: accuracy=" + str(avg_training_set_performance.accuracy), end="")
                print(" | precision=" + str(avg_training_set_performance.precision), end="")
                print(" | recall=" + str(avg_training_set_performance.recall), end="")
                print(" | F1-score=" + str(avg_training_set_performance.f1_score))

                print("      -> Test Performance: accuracy=" + str(testing_set_performance.accuracy), end="")
                print(" | precision=" + str(testing_set_performance.precision), end="")
                print(" | recall=" + str(testing_set_performance.recall), end="")
                print(" | F1-score=" + str(testing_set_performance.f1_score))

                plt.plot(training_set_performance.accuracy)
                plt.plot(training_set_performance.precision)
                plt.plot(training_set_performance.recall)
                plt.plot(training_set_performance.f1_score)
                plt.legend(['accuracy', 'precision', 'recall', 'F1-Score'], loc='upper left')
                plt.xlabel('Cross Validation Step')
                plt.ylabel('Metric Value')
                plt.title('Cross Validation Metrics over 10 folds (nEpochs = ' + str(num_epochs) + ", alpha = " + str(
                    alpha) + ")")
                plt.savefig("1.png")
                plt.close()
                plt.clf()
            else:
                for regularization_parameter in regularization_parameters:
                    print("      -> Lambda: " + str(regularization_parameter) + ": ")

                    h, w, training_set_performance = logistic_regression_model(
                        training_data=[X_train_provided, y_train_provided],
                        num_epochs=num_epochs,
                        learning_rate=alpha,
                        normalize=min_max_scaling or regular_scaling,
                        min_max_standardized=min_max_scaling,
                        regularize=regularize,
                        regularization_parameter=regularization_parameter
                    )

                    avg_training_set_performance = CrossValidationMetricsHolder(
                        accuracy=np.mean(training_set_performance.accuracy, axis=0),
                        precision=np.mean(training_set_performance.precision, axis=0),
                        recall=np.mean(training_set_performance.recall, axis=0),
                        f1_score=np.mean(training_set_performance.f1_score, axis=0)
                    )

                    y_pred = h(X_test_provided, w)

                    try:
                        testing_metrics = cv_metrics(y_test_provided, y_pred, average=None)
                    except ValueError:
                        testing_metrics = cv_metrics(y_test_provided, y_pred.round(), average=None)
                        pass

                    try:
                        accuracy = accuracy_score(y_test_provided, y_pred)
                    except ValueError:
                        accuracy = accuracy_score(y_test_provided, y_pred.round())
                        pass

                    testing_set_performance = CrossValidationMetricsHolder(
                        accuracy=accuracy,
                        precision=testing_metrics[0][0],
                        recall=testing_metrics[1][0],
                        f1_score=testing_metrics[2][0]
                    )

                    for j in range(0, 10, 1):
                        print("         -> CV Step " + str(j + 1) + ": accuracy=" + str(training_set_performance.accuracy[j]),
                              end="")
                        print(" | precision=" + str(training_set_performance.precision[j]), end="")
                        print(" | recall=" + str(training_set_performance.recall[j]), end="")
                        print(" | F1-score=" + str(training_set_performance.f1_score[j]))

                    print("         -> Average of CV metrics: accuracy=" + str(avg_training_set_performance.accuracy), end="")
                    print(" | precision=" + str(avg_training_set_performance.precision), end="")
                    print(" | recall=" + str(avg_training_set_performance.recall), end="")
                    print(" | F1-score=" + str(avg_training_set_performance.f1_score))

                    print("         -> Test Performance: accuracy=" + str(testing_set_performance.accuracy), end="")
                    print(" | precision=" + str(testing_set_performance.precision), end="")
                    print(" | recall=" + str(testing_set_performance.recall), end="")
                    print(" | F1-score=" + str(testing_set_performance.f1_score))

                    plt.plot(training_set_performance.accuracy)
                    plt.plot(training_set_performance.precision)
                    plt.plot(training_set_performance.recall)
                    plt.plot(training_set_performance.f1_score)
                    plt.legend(['accuracy', 'precision', 'recall', 'F1-Score'], loc='upper left')
                    plt.xlabel('Cross Validation Step')
                    plt.ylabel('Metric Value')
                    plt.title('Cross Validation Metrics over 10 folds (nEpochs = ' + str(num_epochs) + ", alpha = " + str(
                        alpha) + ", " + "lambda = " + str(regularization_parameter) + ")")
                    plt.savefig("1.png")
                    plt.close()
                    plt.clf()


##################
# Problems 1 - 8 #
##################

# 1. Load the data into memory. Then, convert each of the categorical variables into numerical
this_location = str(os.path.dirname(os.path.realpath(__file__)))

# locations for the datafiles used in this assignment
DATA_PATH = this_location + "\\Dataset\\bank.csv"
SMALL_DATA_PATH = this_location + "\\Dataset\\bank-small.csv"
SMALL_TRAIN_DATA_PATH = this_location + "\\Dataset\\Training\\bank-small-train.csv"
SMALL_TEST_DATA_PATH = this_location + "\\Dataset\\Testing\\bank-small-test.csv"

data, data_yes_targets, data_no_targets = encode_labels_if_needed(pd.read_csv(DATA_PATH, sep=';'))
small_data, small_data_yes_targets, small_data_no_targets = encode_labels_if_needed(pd.read_csv(SMALL_DATA_PATH, sep=';'))

X = data.drop('y', axis=1)
X_small = small_data.drop('y', axis=1)
y = data['y']
y_small = small_data['y']

m, n = data.shape
n -= 1  # last column is the target/label, not an attribute

m_small, n_small = small_data.shape
n_small -= 1  # last column is the target/label, not an attribute

# add the bias term (x_0) here
X['bias'] = np.ones(m)
X = X[[X.columns.tolist()[-1]] + X.columns.tolist()[:-1]]
X_small['bias'] = np.ones(m_small)
X_small = X_small[[X_small.columns.tolist()[-1]] + X_small.columns.tolist()[:-1]]

# The dimension of X should be m x n and dimension of y vector would be m x 1.
assert(X.shape == (m, n + 1) and X_small.shape == (m_small, n_small + 1))
assert(y.shape == (m,) and y_small.shape == (m_small,))

# Note: Shape (m, 1) corresponds to a numpy shape of (m,).
#       This simplifies things for now, but when we want to multiply with it, we need to cast it with y[:, np.newaxis].

# we expect the number of samples in the bank-small dataset to be 8334
# we also expect there to be 1000 "yes" (1) targets and 7334 "no" (0) targets
assert(m == 45211 and data_yes_targets == 5289 and data_no_targets == 39922)

# we expect the number of samples in the bank-small dataset to be 8334
# we also expect there to be 1000 "yes" (1) targets and 7334 "no" (0) targets
assert(m_small == 8334 and small_data_yes_targets == 1000 and small_data_no_targets == 7334)

print("Problem #1: \n---------------")
print("The bank dataset has " + str(m) + " samples with " + str(n) + " attributes per sample")
print("    -> There are " + str(data_yes_targets) + " \"yes\" (1) targets and " + str(data_no_targets) + " \"no\" (0) targets")
print("The bank-small dataset has " + str(m_small) + " samples with " + str(n_small) + " attributes per sample")
print("    -> There are " + str(small_data_yes_targets) + " \"yes\" (1) targets and " + str(small_data_no_targets) + " \"no\" (0) targets\n")

# 2. Now implement logistic regression with the cost function from class 2/5/2018.
#    You need to solve it using the batch "Gradient Descent" algorithm
#    For the convergence test, choose option 1 with different nEpoch parameters from {100, 500, 1000)

print("Problem #2: \n---------------")
print("There is nothing to output here.")
print("See Problem 3 for the output of the Logistic Regression function built in this step\n")

# 3. Split the data randomly into two equal parts, containing 50% of the samples which will be used for training, and
#    a test set containing the remaining 50% of the samples. Alternatively, you can use the training & test splits
#    provided that preserves density of "yes" and "no" targets

rng_seed = 123456

# split up training and test data from the bank.csv dataset
X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X, y, train_size=0.5, test_size=0.5, random_state=rng_seed)
assert(X_train_A.shape[0] == int(m * 0.5) and y_train_A.shape[0] == int(m * 0.5))

# split up training and test data from the bank-small.csv dataset
X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X_small, y_small, train_size=0.5, test_size=0.5, random_state=rng_seed)
assert(X_train_B.shape[0] == int(m_small * 0.5) and y_train_B.shape[0] == int(m_small * 0.5))

# also bring the provided training and test splits and split them up
provided_training_set, _, _ = encode_labels_if_needed(pd.read_csv(SMALL_TRAIN_DATA_PATH, sep=';'))
provided_testing_set, _, _ = encode_labels_if_needed(pd.read_csv(SMALL_TEST_DATA_PATH, sep=';'))

m_provided_train, n_provided_train = provided_training_set.shape
n_provided_train -= 1
m_provided_test, n_provided_test = provided_testing_set.shape
n_provided_test -= 1

X_train_provided = provided_training_set.drop('y', axis=1)
y_train_provided = provided_training_set['y']
X_test_provided = provided_testing_set.drop('y', axis=1)
y_test_provided = provided_testing_set['y']

# add the bias term (x_0) here to the provided data sets
X_train_provided['bias'] = np.ones(m_provided_train)
X_train_provided = X_train_provided[[X_train_provided.columns.tolist()[-1]] + X_train_provided.columns.tolist()[:-1]]
X_test_provided['bias'] = np.ones(m_provided_test)
X_test_provided = X_test_provided[[X_test_provided.columns.tolist()[-1]] + X_test_provided.columns.tolist()[:-1]]

print("Problem #3: \n---------------")
print("Set from bank.csv has " + str(X_train_A.shape[0]) + " (~" + str(int(np.ceil((X_train_A.shape[0] / m) * 100))) + "%) training samples and " + str(y_test_A.shape[0]) + " (~" + str(int(np.floor((y_test_A.shape[0] / m) * 100))) + "%) testing samples")
print("Set from bank-small.csv has " + str(X_train_B.shape[0]) + " (~" + str(int(np.ceil((X_train_B.shape[0] / m_small) * 100))) + "%) training samples and " + str(y_test_B.shape[0]) + " (~" + str(int(np.floor((y_test_B.shape[0] / m_small) * 100))) + "%) testing samples")
print("Set from provided CSVs have " + str(X_train_provided.shape[0]) + " (~" + str(int(np.ceil((X_train_provided.shape[0] / m_provided_train) * 100))) + "%) training samples and " + str(y_test_provided.shape[0]) + " (~" + str(int(np.floor((y_test_provided.shape[0] / m_provided_test) * 100))) + "%) testing samples\n")

print("For now on, using only the Provided test and train datasets (bank-small-train.csv and bank-small-test.csv)\n")

run_logistic_regression()  # runs logistic regression without normalization or regularization

# 4. Scale the features of the dataset using Min-Max scaling to [0,1] range, and repeat step 3.
print("Problem #4: \n---------------")
#run_logistic_regression(min_max_scaling=True)  # runs logistic regression with min/max scaling

# 5. Scale the features of the dataset using standardization, and repeat step 3
print("Problem #5: \n---------------")
run_logistic_regression(regular_scaling=True)  # runs logistic regression with standardization

# 6. Implement regularized logistic regression with the cost function from class 2/5/2018.
#    You need to solve it using the batch "Gradient Descent" algorithm
#    For the convergence test, choose option 1 with different nEpoch parameters from {100, 500, 1000
print("Problem #6: \n---------------")
print("There is nothing to output here.")
print("See Problem 7 for the output of the Logistic Regression with regularization implemented in this step\n")

# 7. On the standardized or the scaled dataset, repeat step 3 except using the regularized logistic regression you
#    developed in step 6, by varying the parameter, lambda={0, 1, 10, 100, 1000}
print("Problem #7: \n---------------")
run_logistic_regression(regular_scaling=True, regularize=True)  # runs logistic regression with standardization and regularization

# 8. Summarize (using a plot, or a table) the classification performance metrics (i.e., accuracy, recall, precision,
#   F1-score) you would obtain in each of the experiments above.
print("Problem #8: \n---------------")
print("Graphs and detailed output have been output as the previous problems executed")