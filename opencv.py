import cv2
import dlib
import json
import requests
import mat
from PIL import Image

cnn_face_detector = dlib.cnn_face_detection_model_v1("cascades/mmod_human_face_detector.dat")
predictor = dlib.shape_predictor("cascades/shape_predictor_68_face_landmarks.dat")

votes_need = 2  # Can be edited, this means 3 classifications of a pain is needed before its shown.
votes = 0
current_pain = ""
vote_for = ""

colour_codes = {
    'None': (0, 255, 0),
    'Mild': (0, 255, 0),
    'Moderate': (0, 165, 255),
    'Severe': (255, 0, 0),
}


def get_rects(img, scale, all=True):

    global cnn_face_detector
    dets = cnn_face_detector(img, scale)

    return dets


def detect_face(img, scale):

    """
        Method takes in an image and returns the image with a box drawn around every face in the image.
        commented code below is for use on non CUDA pcs, the other code is for CUDA ONLY
        :param img: image to detect faces in
        :param scale: how large to scale image
        :return img with overlay:
    """

    dets = get_rects(img, scale)

    for i, d in enumerate(dets):
        cv2.rectangle(img, (d.rect.left(), d.rect.top()), (d.rect.right(), d.rect.bottom()), (255, 0, 255), 2)

    return img


def get_prediction_acceptance(pain):

    global votes
    global current_pain
    global vote_for
    global votes_need

    if votes_need == 0:
        return pain

    if current_pain == "":    # first prediction
        current_pain = pain

        return pain

    elif current_pain == pain:  # If the current predicted is the same as whats showing

        votes = 0
        return pain

    else:

        if vote_for == pain:

            votes = votes + 1

            if votes >= votes_need:
                current_pain = pain
                votes = 0

                return current_pain

            else:
                return current_pain

        else:                   # Change current polling category
            vote_for = pain
            votes = 0

            return current_pain


def detect_pain(img):

    """
    :param img:
    :return:
    """

    global predictor

    dets = get_rects(img, 0)

    for k, d in enumerate(dets):
        shape = predictor(img, d.rect)

        landmarks = dict()
        points = dict()

        index = 0
        for x in range(17, 68):

            landmark = shape.part(x)
            points[str(index)] = {'x': landmark.x, 'y': landmark.y}
            index = index + 1

        landmarks['landmarks'] = points

        classification = mat.predict(points)
        pain = get_prediction_acceptance(classification)

        font = cv2.FONT_HERSHEY_SIMPLEX

        pain_text = "Pain: {}".format(str(pain))

        for i, d in enumerate(dets):
            cv2.rectangle(img, (d.rect.left(), d.rect.top()), (d.rect.right(), d.rect.bottom()), colour_codes[pain], 2)

        cv2.putText(img, pain_text, (10, 450), font, 2, colour_codes[pain], 2, cv2.LINE_AA)

    return img
