import os
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import pickle
import numpy as np
import mediapipe as mp

# Set up Flask app
app = Flask(__name__)

# Define paths for file uploads and processed files
UPLOAD_FOLDER = r"F:\Huawei\Sign_project_1\host\uploads"
PROCESSED_FOLDER = r"F:\Huawei\Sign_project_1\host\processed"

# Ensure the directories exist, if not, create them

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions for file uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the trained model
try:
    model_dict = pickle.load(open(r"F:\Huawei\Sign_project_1\model.p", 'rb'))
    model = model_dict['model']
except Exception as e:
    print(f"Error loading model: {e}")

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7)

# Labels dictionary
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
    22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2',
    29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9',
    36: 'I love You', 37: 'yes', 38: 'No', 39: 'Hello', 40: 'Thanks',
    41: 'Sorry', 43: 'space'
} 

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to process image and return prediction
def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, "Error loading image."

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    data_aux = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = []
            y_ = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                data_aux.append(hand_landmarks.landmark[i].y - min(y_))

            # Make a prediction using the model
            prediction = model.predict([np.asarray(data_aux)])
            predicted_index = int(prediction[0])
            predicted_label = labels_dict.get(predicted_index, "Unknown label")

            # Draw the label on the image
            cv2.putText(img, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)

            # Save the processed image
            processed_image_path = os.path.join(PROCESSED_FOLDER, 'processed_' + os.path.basename(image_path))
            cv2.imwrite(processed_image_path, img)

            return processed_image_path, predicted_label

    return None, "No hands detected."

# Route to upload an image
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process the image and return the label
            processed_image_path, label = process_image(file_path)
            if processed_image_path:
                return render_template('result.html', label=label, image_path='/processed/' + os.path.basename(processed_image_path))

    return render_template('upload.html')

# Route to serve processed images
@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
