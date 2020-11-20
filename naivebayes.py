from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import heapq

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# preprocessing

x_train_prep = train_images / 255
x_test_prep = test_images /255

x_train_prep_1d = x_train_prep.reshape(-1, 28*28)
x_test_prep_1d = x_test_prep.reshape(-1, 28*28)

x_test_prep_3d = x_test_prep.reshape(-1, 28, 28, 1)
x_train_prep_3d = x_train_prep.reshape(-1, 28, 28, 1)

logistic = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=100)
logistic_param_grid = {'C': [0.001, 0.01, 0.1, 1., 10., 100.]}
lr_gridsearch = GridSearchCV(cv=4, estimator=logistic, param_grid=logistic_param_grid, scoring='accuracy')
lr_gridsearch.fit(x_train_prep_1d, train_labels)

print(lr_gridsearch.best_params_)
print(lr_gridsearch.best_score_)

def find_example(model, x, y, true_class, predicted_class):
    y_true = y
    y_pred = model.predict(x)
    found_index = None
    for index, (current_y_true, current_y_pred) in enumerate(zip(y_true, y_pred)):
        if current_y_true == true_class and current_y_pred == predicted_class:
            found_index = index
            break
    return found_index


def plot_example(model, x, y, true_class, predicted_class, value=None):
    index = find_example(model, x, y, true_class, predicted_class)
    print('True class:', class_names[true_class])
    print('Predicted class:', class_names[predicted_class])
    if value is not None:
        print('Misclassified', value, 'times')
    if index is not None:
        plt.imshow(x_test_prep[index])
        plt.show()
    print('')

def analyze_model(model, x, y, inspect_n=10):
    y_pred = model.predict(x)
    conf_matrix = confusion_matrix(y, y_pred)
    print('Confusion matrix:')
    print(conf_matrix)
    print('')
    for _ in range(10):
        conf_matrix[_][_] = 0
    conf_matrix_flat = conf_matrix.reshape(-1, 1)
    biggest_indices = heapq.nlargest(inspect_n, range(len(conf_matrix_flat)), conf_matrix_flat.take)
    biggest_indices = np.unravel_index(biggest_indices, conf_matrix.shape)
    highest_values = conf_matrix[biggest_indices]
    for x_index, y_index, value in zip(biggest_indices[0], biggest_indices[1], highest_values):
        plot_example(model, x, y, x_index, y_index, value)

# plot
def plot_image(X, y=None):
    if y is None:
        y = 'unknown'
    else:
        y = class_names[y]
    plt.title('Label is {label}'.format(label=y))
    plt.imshow(X, cmap='gray')
    plt.show()
