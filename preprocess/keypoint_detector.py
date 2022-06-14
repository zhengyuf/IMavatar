import face_alignment
from skimage import io
import argparse
import os

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', type=str,  help='.')


args = parser.parse_args()

image_path = args.path + '/image/'
print(image_path)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

preds = fa.get_landmarks_from_directory(image_path)
import json
path = args.path
print(path)
save = {}
for k, code in preds.items():
	k = k.split('/')[-1]
	save[k] = code[0].tolist()
	
json.dump(save, open(os.path.join(path, 'keypoint.json'), 'w'))
