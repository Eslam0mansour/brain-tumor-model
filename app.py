import time

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="path/model.tflite")
interpreter.allocate_tensors()

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # Get the input image from the request
    file = request.files['image']
    # Save the image to ./uploads
    file.save("./uploads/image.png")
    # Load the saved image using PIL
    img = Image.open("./uploads/image.png")
    # Preprocess the image and convert it to a numpy array
    img = img.resize((224, 224))
    img = np.asarray(img)
    # # Get the input and output tensors
    # img = img[:, :, ::-1].copy()
    # Add the batch dimension
    img = np.expand_dims(img, axis=0)
    # Normalize the image to [0,1] range from [0,255] range and convert it to a float32
    img = img / 255.0
    # make it flot32
    img = img.astype(np.float32)
    # Get the input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Set the value of the input tensor
    interpreter.set_tensor(input_details[0]['index'], img)
    # Run the inference
    interpreter.invoke()
    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # Get the predicted class
    predicted_class = np.argmax(output_data[0])
    print(predicted_class)
    # Return the predicted class
    return jsonify({'class': str(predicted_class)})


if __name__ == '__main__':
    app.run(debug=True)
