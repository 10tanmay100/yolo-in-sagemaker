from sagemaker.tensorflow import TensorFlowModel

# Set the name of the endpoint
ENDPOINT_NAME = 'yolov5-endpoint'

# Create a TensorFlowModel from the output directory of the training job
model = TensorFlowModel(
    model_data=f's3://{BUCKET_NAME}/{PREFIX}/output/model.tar.gz',
    role=sagemaker.get_execution_role(),
    framework_version='2.5',
    entry_point='predict.py',
    source_dir='.',
)

# Deploy the model to a SageMaker endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',
    endpoint_name=ENDPOINT_NAME,
)

import numpy as np
import PIL.Image as Image

# Load an image
image = Image.open('test.jpg')

# Convert the image to a NumPy array
image_array = np.array(image)

# Make a prediction with the endpoint
result = predictor.predict({'inputs': image_array})

# Print the prediction results
print(result)

