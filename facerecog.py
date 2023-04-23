import csv
import urllib.request
import io
import numpy as np
import face_recognition
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

# Load CSV file
with open('data.csv') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header row
    names = []
    image_urls = []
    for row in reader:
        names.append(row[0])
        image_urls.append(row[1])

# Load images from URLs
images = []
for url in image_urls:
    with open(url,'rb') as resp:
    #resp = urllib.request.urlopen(url)
        image_data = resp.read()
        image_file = io.BytesIO(image_data)
        image = face_recognition.load_image_file(image_file)
        images.append(image)

# Extract face encodings
encodings = []
for image in images:
    encoding = face_recognition.face_encodings(image)[0]
    encodings.append(encoding)

@app.route('/face_recognition', methods=['POST',"GET"])
def face_recognition_endpoint():
    if request.method =="POST":
        # Get input image from request
        input_image = request.files['image'].read()
        input_image = io.BytesIO(input_image)
        input_image = face_recognition.load_image_file(input_image)
        # Extract face encoding from input image
        face_encodings =  face_recognition.face_encodings(input_image)
        matching_names = []
        if len(face_encodings) >0:
            input_encoding = face_recognition.face_encodings(input_image)[0]
            # Perform face recognition on input image
            results = face_recognition.compare_faces(encodings, input_encoding)

            # Find matching names
            for i, match in enumerate(results):
                if match:
                    matching_names.append(names[i])
            print(matching_names)
            return jsonify(matching_names)
    else:
        return render_template('index.html')
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)

