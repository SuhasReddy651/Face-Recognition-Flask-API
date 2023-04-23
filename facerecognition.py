import face_recognition
import cv2
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template
import os

app = Flask(__name__)

# Load data from CSV file
data = pd.read_csv('data.csv')

# Extract face encodings and labels from data
known_face_encodings = []
known_face_labels = []
for i, row in data.iterrows():
    for img_path in row['images'].split():
        img = cv2.imread(img_path)
        face_encodings = face_recognition.face_encodings(img)[0]
        known_face_encodings.append(face_encodings)
        known_face_labels.append(row['name'])

# Define a function to recognize faces in an image
@app.route('/face_recognition', methods=['POST',"GET"])
def recognize_faces():
    if request.method == 'POST':
        input_image = request.files.get('image')
        img_path = f'static/{input_image.filename}'
        input_image.save(img_path)
        img = cv2.imread(img_path)
        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)
        os.remove(img_path)
        # Loop through each face in the image
        for face_encoding in face_encodings:
            # Compare face encoding with known face encodings
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                # Recognized face
                return known_face_labels[best_match_index]

        # Unknown face
        return 'Unknown'
    else:
        return render_template('index.html')
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
# Test the model on a test image
# test_img_path = 'colegejpg.jpg'
# predicted_name = recognize_faces(test_img_path)
# print(predicted_name)