import sys
import os
import dlib
import glob
from scipy.spatial import distance

detector = dlib.cnn_face_detection_model_v1("cascades/mmod_human_face_detector.dat")
sp = dlib.shape_predictor("cascades/shape_predictor_5_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("cascades/dlib_face_recognition_resnet_model_v1.dat")


"""
    Make and append an entry in the dictionary for each person there is an image for. Be sure to keep correct syntax 
    when entering. Image used should be cropped to remove other people and background. See obama.jpg as an example.
"""

people = list()
people.append({"details": {"name": "Barack Obama", "age": "57", "sex": "Male", "height": "183 CM", "weight": "75 KG", "bmi": "24", "image_path": "images/obama.jpg"}})
#people.append({"details": {"name": "Example2", "age": "27", "sex": "Male", "height": "170 CM", "weight": "83 KG", "bmi": "23", "image_path": "images/x.png"}})

"""
    Calculate recognition data that will be used to compare against person
    who will be seen in the camera. A descriptor is then added to each person.
"""

for person in people:

    try:
        base_image = dlib.load_rgb_image(person["details"]["image_path"])

        dets = detector(base_image, 1)
        for k, d in enumerate(dets):

            shape = sp(base_image, d.rect)
            base_face_descriptor = facerec.compute_face_descriptor(base_image, shape)
            person["descriptor"] = base_face_descriptor

    except Exception as e:
        print "Couldn't add user. ERROR - {}".format(e)
        people.remove(person)


"""
    Compare the likeness of the base image which is calculated initially
    to the image given by the camera
"""


def find_patient(image):

    dets = detector(image, 1)
    external_face_descriptor = None

    if len(dets) != 1:
        return {"ERROR": "FACES"}

    else:

        for k, d in enumerate(dets):

            shape = sp(image, d.rect)
            external_face_descriptor = facerec.compute_face_descriptor(image, shape)

        if external_face_descriptor is not None:

            for person in people:

                is_match = compare_faces(external_face_descriptor,person["descriptor"])

                if is_match:
                    return person["details"]

            return {"ERROR": "PERSON"}


def compare_faces(external_face_descriptor,persons_descriptor):

    dist = distance.euclidean(persons_descriptor, external_face_descriptor)

    if dist < .6:
        return True
    else:
        return False
