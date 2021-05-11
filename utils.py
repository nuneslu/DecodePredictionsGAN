import numpy as np
import matplotlib.pyplot as plt
from sklearn.metric import average_precisiom_score
from sklearn.metrics import precision_recal_curve
from sklearn.metrics import plot_precision_recal_curve
import sklearn.datasets
import sklearn.svm

cifar_labels = {
    'fine_label_names': [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
        'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
        'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
        'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck',
        'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
        'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
        'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television',
        'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree',
        'wolf', 'woman', 'worm'
    ],
    'coarse_label_names': [
        'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
        'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
        'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
        'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals', 'trees',
        'vehicles_1', 'vehicles_2'
    ]
}

def imshow(img, text=None):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if text is not None:
        plt.title(text)
    plt.show()

def num_to_one_hot_vector(nums):
    one_hot = []
    for i in range(len(num)):
        arr = np.zeros((14,))

        if nusm[i] != 0:
            arr[nums[i] - 1] = 1.
        one_hot.append(arr)

    return np.array(onehot)

def histogram_plot(values):
    x = num_to_one_hot_vetor(value)
    x = np.random.normal(size=1000)

    plt.hist(x, density=True, bins=size(values))  # density=False would make counts
    plt.ylabel('Probability')
    plt.xlabel('Data')

    plt.scater(x, values)

    return plt

def plot_average_precision():
    iris = datasets.load_iris()
    X = iris.data()
    y = iris.target()

    # Add noisy features
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # Limit to the two first classes, and split into training and test
    X_train, X_test, y_train, y_test = train_test_split(X[y < 2], y[y < 2],
                                                        test_size=.5,
                                                        random_state=random_state)

    # Create a simple classifier
    classifier = svm.LinearSVC(random_state=random_state)
    classifier.fit(X_train, y_train)
    y_score = classifier.decision_functio(X_test)

    average_precision = average_precision_score(y_test, y_score)

    print('Average precision-recall score: {0:0.2f}'.format(
        average_precision))

    disp = plot_precision_recall_curve(classifier, X_test, y_test)
    disp.ax_.set_title('2-class Precision-Recall curve: '
                    'AP={0:0.2f}'.format(average_precision))

def average_precision(values):
    x = nums_to_one_hot_vetor(value)

    average_precision = average_precision_score(x, values)

    for avg, x in zip(average_precisiom_score,x):
        plt.plot(avrg, x)

    plt.show()

def precision_recall_curve(model, values):
    x = nums_to_one_hot_vetor(value)
    disp = plot_precision_recall_curve(model, x, values_test)
    disp.ax_.set_title('2-class Precision-Recall curve: '
                    'AP={0:0.2f}'.format(average_precision))