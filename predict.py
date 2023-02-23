import os
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Set the input and output tensor names
INPUT_TENSOR_NAME = 'input_1:0'
OUTPUT_TENSOR_NAMES = ['Identity:0']

# Set the threshold for object detection
DETECTION_THRESHOLD = 0.5

# Load the saved model
model_path = os.environ['SM_MODEL_DIR']
model = tf.keras.models.load_model(model_path)

# Load the image preprocessing module from TensorFlow Hub
preprocess = hub.load('https://tfhub.dev/tensorflow/cropnet/classifier/cropnet_keras/1')

# Parse the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=True)
args, _ = parser.parse_known_args()

# Load the image and preprocess it
image = tf.keras.preprocessing.image.load_img(args.image, target_size=(416, 416))
image_array = tf.keras.preprocessing.image.img_to_array(image)
image_tensor = preprocess(image_array / 255.)

# Make a prediction with the model
outputs = model.predict(image_tensor[np.newaxis, ...])
boxes, scores, classes = tf.raw_ops.NonMaxSuppressionV5(
    inputs=outputs[0]['output_1'],
    score_threshold=tf.constant(DETECTION_THRESHOLD),
    iou_threshold=tf.constant(0.5),
    max_output_size=tf.constant(100),
)

# Convert the boxes to relative coordinates
height, width = image_array.shape[:2]
boxes_relative = boxes / np.array([width, height, width, height])

# Print the predictions
for i in range(len(boxes)):
    print(f'{classes[i]}: {scores[i]}')
    print(f'box: {boxes_relative[i]}')
