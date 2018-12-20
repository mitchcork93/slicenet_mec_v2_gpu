import sys
import os
import dlib
import glob
from scipy.spatial import distance

detector = dlib.cnn_face_detection_model_v1("cascades/mmod_human_face_detector.dat")
sp = dlib.shape_predictor("cascades/shape_predictor_5_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("cascades/dlib_face_recognition_resnet_model_v1.dat")

"""
	change the image in this folder to a image of the person who is testing the application
"""
base_image = dlib.load_rgb_image("images/me.jpeg")

"""
    Calculate recognition data that will be used to compare against person
    who will be seen in the camera. Can be changed by switching the base
    image file listed above. 
"""

dets = detector(base_image, 1)
for k, d in enumerate(dets):

    shape = sp(base_image, d.rect)
    base_face_descriptor = facerec.compute_face_descriptor(base_image, shape)


"""
    Compare the likeness of the base image which is calculated initially
    to the image given by the camera
"""


def find_patient(image):

    dets = detector(image, 1)

    if len(dets) != 1:
        return {"ERROR": "FACES"}

    else:

        for k, d in enumerate(dets):

            shape = sp(image, d.rect)
            external_face_descriptor = facerec.compute_face_descriptor(image, shape)
            is_match = compare_faces(external_face_descriptor)

            if is_match:
                return {"mode": "identify", "type": "patient", "name": "Person 1", "age": 25, "medications": ["paracetamol", "ibuprofen"]}
            else:
                return {"ERROR": "PERSON"}


def compare_faces(external_face_descriptor):

    dist = distance.euclidean(base_face_descriptor, external_face_descriptor)

    if dist < .6:
        return True
    else:
        return False
