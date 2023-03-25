from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="path/model_50.tflite")
interpreter.allocate_tensors()

# Get the input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


@app.route('/classify', methods=['POST'])
def classify_image():
    # Get the image data from the request
    image_data = request.files['image'].read()

    # Preprocess the image
    image = tf.io.decode_image(image_data, channels=1) # Convert the image to a 1D array
    image = tf.image.resize(image, [48, 48]) # Resize the image to 48x48 pixels
    image = tf.cast(image, tf.float32) # Convert the image to float32
    image /= 255.0 # Normalize the image to the range 0-1
    image = np.expand_dims(image, axis=0) # Add a dimension to the image
    # as the model expects a batch of images
    # Run the model inference
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    # Return the highest probability class label as a string

    highest_probability_index = np.argmax(output.flatten())

    class_labels = ['happy', 'sad', 'surprise', 'neutral']

    return jsonify({'class': class_labels[highest_probability_index]})


if __name__ == '__main__':
    app.run(debug=True)




