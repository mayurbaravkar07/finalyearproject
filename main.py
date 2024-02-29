

from flask import Flask, render_template, request
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
import numpy as np

threshold=0.10
app = Flask(__name__)
model = load_model('models/model1.h5')
 # Load your custom trained CVD risk prediction model

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    
    # Assuming your model returns probabilities for "Potential risk of the CVD" and "No Signs of the CVD"
    prediction = model.predict(image)
    
        
    # Given prediction array
    prediction_array = np.array(prediction)

    # Calculate the average of the prediction array
    average = np.mean(prediction_array)

    # Normalize each element by dividing by the average
    normalized_array = prediction_array / average

    # Scale the normalized array to the range 0 to 1
    normalized_array = (normalized_array - np.min(normalized_array)) / (np.max(normalized_array) - np.min(normalized_array))
    result=np.mean(normalized_array)
    print(result)
    if result>threshold:
        result = "Potential risk of the CVD"
    else:
        result = "No Signs of the CVD"

    return result  # Return only the prediction result

if __name__ == '__main__':
    app.run(port=3000, debug=True)



