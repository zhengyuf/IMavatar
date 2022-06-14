# Code borrowed and adapted from Neural Head Avatar
# https://github.com/philgras/neural-head-avatars/blob/473457eef83c9ee26f316451e31c2aa01a74603c/python_scripts/video_to_dataset.py#L26
from fdlite import (
    FaceDetection,
    FaceLandmark,
    face_detection_to_roi,
    IrisLandmark,
    iris_roi_from_face_landmarks,
)
from PIL import Image
import numpy as np
import os
import argparse
import re


def annotate_iris_landmarks(image_path, savefolder):
    """
    Annotates each frame with 2 iris landmarks
    :return: dict mapping frame number to landmarks numpy array
    """

    # iris detector
    detect_faces = FaceDetection()
    detect_face_landmarks = FaceLandmark()
    detect_iris_landmarks = IrisLandmark()
    frames = os.listdir(image_path)
    frames.sort(key=lambda f: int(re.sub('\D', '', f)))

    # frames = self._get_frame_list()
    landmarks = {}

    for frame in frames:
        img = Image.open(os.path.join(image_path, frame))

        width, height = img.size
        img_size = (width, height)

        face_detections = detect_faces(img)
        landmarks[frame] = None
        if len(face_detections) != 1:
            print("Empty iris landmarks")
        else:
            for face_detection in face_detections:
                try:
                    face_roi = face_detection_to_roi(face_detection, img_size)
                except ValueError:
                    print("Empty iris landmarks")
                    break

                face_landmarks = detect_face_landmarks(img, face_roi)
                if len(face_landmarks) == 0:
                    print("Empty iris landmarks")
                    break

                iris_rois = iris_roi_from_face_landmarks(face_landmarks, img_size)

                if len(iris_rois) != 2:
                    print("Empty iris landmarks")
                    break

                lmks = []
                for iris_roi in iris_rois[::-1]:
                    try:
                        iris_landmarks = detect_iris_landmarks(img, iris_roi).iris[
                                         0:1
                                         ]
                    except np.linalg.LinAlgError:
                        print("Failed to get iris landmarks")
                        break

                    for landmark in iris_landmarks:
                        lmks.append(landmark.x * width)
                        lmks.append(landmark.y * height)

        landmarks[frame] = lmks # if empty iris landmarks, then just use previous frame...but if the first frame is empty, then we have a problem.
    import json
    json.dump(landmarks, open(os.path.join(savefolder, 'iris.json'), 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', type=str, help='Path to images and deca and landmark jsons')
    args = parser.parse_args()
    image_path = os.path.join(args.path, 'image')
    annotate_iris_landmarks(image_path=image_path, savefolder=args.path)