from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the skin disease detection model
model = load_model("skin_cancer.h5")

# Function to preprocess the image
def preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    # Resize the image to match the input size of the model
    image = cv2.resize(image, (28, 28))
    # Normalize pixel values to the range [0, 1]
    image = image / 255.0
    # Expand dimensions to create a batch of 1 image
    image = np.expand_dims(image, axis=0)
    return image

# Function to make predictions
def predict_skin_disease(image_path):
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    # Make prediction using the model
    prediction = model.predict(processed_image)
    # Get the predicted class index
    class_index = np.argmax(prediction)
    # Return the predicted class label
    return class_index

# Route to render the index.html template
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission
@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return "No file uploaded"
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return "No selected file"
        # If the file is valid, make prediction
        if file:
            # Save the uploaded image
            file_path = "uploaded_image.jpg"
            file.save(Skin-Disease)
            # Make prediction
            predicted_class_index = predict_skin_disease(file_path)
            # Render the result template with prediction
            return render_template('result.html', prediction=predicted_class_index)

if __name__ == '__main__':
    app.run(debug=True)
