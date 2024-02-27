from flask import Flask, request, jsonify
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the models dictionary to hold loaded models
models = {}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to load the selected model
def load_selected_model(model_name):
    if model_name in models:
        return models[model_name]
    else:
        # Load the model based on the selected model name
        model_path = "/home/guest/Downloads/Final year project/t2/models/model1.h5"
        if os.path.exists(model_path):
            loaded_model = load_model(model_path)
            models[model_name] = loaded_model
            return loaded_model
        else:
            return None

# Function to preprocess the uploaded image
def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def process_prediction(prediction):
    # Placeholder for processing the prediction result
    # This example assumes prediction is a probability distribution
    # and returns the class with the highest probability

    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(prediction)

    # Define the classes or categories for your prediction
    classes = ['Class 1', 'Class 2', 'Class 3']  # Replace with your actual class labels

    # Get the predicted class label
    predicted_class = classes[predicted_class_index]

    return predicted_class


@app.route("/predict", methods=["POST"])  # Specify that this endpoint accepts POST requests
def predict():
    print("Received request:")
    print("Form data:", request.form)
    print("Files:", request.files)

    # Extract data from request
    selected_model = request.form.get("selectedModel")
    uploaded_file = request.files.get("file")

    print("Selected Model:", selected_model)
    print("Uploaded File:", uploaded_file)
    # Error handling (check for missing data, invalid model names, etc.)
    if not selected_model or not uploaded_file:
        return jsonify({"error": "Missing required data"}), 400

    # Check if the selected model is valid
    if selected_model not in models:
        return jsonify({"error": "Invalid model name"}), 400

    # Load the selected model
    model = load_selected_model(selected_model)
    if model is None:
        return jsonify({"error": "Failed to load the selected model"}), 500

    # Save the uploaded file
    filename = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
    uploaded_file.save(filename)

    # Preprocess the uploaded image
    preprocessed_image = preprocess_image(filename)

    # Perform prediction
    prediction = model.predict(preprocessed_image)

    # Process prediction result
    processed_result = process_prediction(prediction)

    # Response creation
    response = jsonify({"result": processed_result})
    return response

if __name__ == "__main__":
    app.run(debug=True)
