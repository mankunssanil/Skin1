from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import os

app = Flask(__name__)

# Load the skin disease detection model
model = load_model("skin_cancer.h5")

# Define classes and other relevant information
classes = {
    4: ('nv', 'Melanocytic nevi', ['Sensitivity to touch around the mole', 'Redness or inflammation around the mole'],
        ['Avoid tight Clothing', 'Limit exposure to direct sunlight']),
    6: ('mel', 'Melanoma', ['Multiple colors within a mole', 'Bleeding or oozing from a mole'],
        ['Eat a balanced diet rich in antioxidants and vitamins', 'Avoid smoking and limit alcohol consumption']),
    2: ('bkl', 'Benign keratosis-like lesions', ['Itching or irritation in affected areas, Round or oval shaped growths',
                                                 'Very small growths clustered around the eyes or elsewhere on the face'],
        ['Moisturize Regularly', 'Manage Stress by meditation or yoga']),
    1: ('bcc', 'Basal cell carcinoma', ['Surrounding skin becoming sunken or depressed',
                                         'Formation of a flesh-coloured, pearl like bump'],
        ['Avoid harmful chemicals', 'Wear Protective Clothing']),
    5: ('vasc', 'Pyogenic granulomas and hemorrhage', ['Prone to Ulceration', 'Moist or friable surface structure'],
        ['Use sunscreen with a high SPF', 'Keep the affected arear covered with a sterile dressing']),
    0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae',
        ['Swelling and burning in affected region', 'Thickening of the skin'],
        ['Avoid tanning beds and sunlamps', 'Avoid hot shower and opt for lukewarm water']),
    3: ('df', 'Dermatofibroma', ['Dimpled appearance when pressed', 'Growing in size over time'],
        ['Avoid using harsh chemicals or irritants', 'Drink plenty of water and maintain proper hydration'])
}

# Define route for home page
@app.route("/")
def home():
    return render_template("index.html")

# Define route for image upload and prediction
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Check if a file was uploaded
        if "file" not in request.files:
            return render_template("index.html", prediction="No file uploaded")

        file = request.files["file"]

        # Check if the file is empty
        if file.filename == "":
            return render_template("index.html", prediction="No file selected")

        # Check if the file is valid
        if file:
            # Read and preprocess the image
            image = Image.open(file)
            image = image.resize((28, 28))
            img = np.array(image).reshape(-1, 28, 28, 3)

            # Make prediction
            result = model.predict(img)
            max_prob = max(result[0])
            class_ind = result[0].tolist().index(max_prob)

            # Get disease information from the classes dictionary
            disease_code, disease_name, symptoms, precautions = classes[class_ind]

            # Calculate confidence as a percentage
            confidence_percentage = max_prob * 100

            # Render prediction template with results
            return render_template("index.html", prediction=disease_name,
                                   confidence=f"{confidence_percentage:.2f}%", symptoms=symptoms, precautions=precautions)


if __name__ == "__main__":
    app.run(debug=True)
