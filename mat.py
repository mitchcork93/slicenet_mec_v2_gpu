import sys
sys.path.append('libsvm\python')
from svmutil import *
from scipy.spatial import distance

maximum = float(-10000.0)     # For normalizing data
minimum = float(10000.0)     # For normalizing data
model = svm_load_model('models/pain.model')


def label_to_emotion(label):

    """
        Method takes in a predicted label (int) and converts to an corresponding emotion (string)
        :param label:
        :return emotion:
    """

    emotions = {
        0: 'None',
        1: 'Mild',
        2: 'Moderate',  # Not used at the moment
        3: 'Severe',
    }

    return emotions[label]


def process_params(data):

    """
        Method takes in raw data from images and converts to required format
        (*see https://github.com/cjlin1/libsvm/tree/master/python) for predictions.
        :param data:
        :return processed data:
    """

    params = []
    distances = []

    global maximum
    global minimum

    for x in range(0, len(data)):

        point_one = data[str(x)]

        for y in range((x+1), len(data)):

            point_two = data[str(y)]

            dist = find_distance(point_one, point_two)

            if dist > maximum:
                maximum = dist
            if dist < minimum:
                minimum = dist

            distances = distances + [dist]

    index = 1
    for z in distances:

        val = normalize(z)
        params = params + [val]
        index = index + 1

    reset_min_max()
    return params


def find_distance(point_one, point_two):

    """
        Method takes in two points {x,y} and returns the euclidean distance between them.
        :param point_one:
        :param point_two:
        :return distance:
    """

    a = (point_one["x"], point_one["y"])
    b = (point_two["x"], point_two["y"])
    return (distance.euclidean(a, b)).item()


def normalize(param):

    """
        Method takes in a parameter and rescales it to a real number between [-1,1]
        :param param:
        :return normalized(param):
    """

    global maximum
    global minimum

    return (param - minimum) / (maximum - minimum)


def reset_min_max():

    """
        Method resets param min / max value used for scaling
    """

    global minimum
    global maximum

    maximum = float(-10000.0)  # For normalizing data
    minimum = float(10000.0)   # For normalizing data


def predict(landmarks):

    """
        Method takes in parameters, preforms a prediction and returns the result
        :param landmarks:
        :return emotion:
    """

    y = []
    x = []

    y = y + [0]
    x = x + [process_params(landmarks)]

    global model
    p_label, p_acc, p_val = svm_predict(y, x, model)

    return label_to_emotion(p_label[0])
