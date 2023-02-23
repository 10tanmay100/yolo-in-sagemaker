import sagemaker
from sagemaker.tensorflow import TensorFlow

# Set the S3 bucket and prefix where training data will be stored
BUCKET_NAME = '<your-bucket-name>'
PREFIX = 'yolov5'

# Set the path to the training data
DATA_PATH = f's3://{BUCKET_NAME}/{PREFIX}/data'

# Set the name of the training job
JOB_NAME = 'yolov5-training'

# Set the ECR image URI
IMAGE_URI = '<your-account-id>.dkr.ecr.us-west-2.amazonaws.com/my-ecr-repo:latest'

# Set the hyperparameters
HYPERPARAMETERS = {
    'epochs': 50,
    'batch-size': 16,
    'learning-rate': 0.001,
    'data': DATA_PATH,
    'output-dir': f's3://{BUCKET_NAME}/{PREFIX}/output',
}

# Set up the SageMaker TensorFlow estimator
estimator = TensorFlow(
    entry_point='train.py',
    source_dir='.',
    role=sagemaker.get_execution_role(),
    instance_count=1,
    instance_type='ml.p3.2xlarge',
    framework_version='2.5',
    py_version='py37',
    hyperparameters=HYPERPARAMETERS,
    output_path=f's3://{BUCKET_NAME}/{PREFIX}/output',
    image_uri=IMAGE_URI,
    sagemaker_session=sagemaker.Session(),
)

# Start the training job
estimator.fit(inputs={'training': DATA_PATH}, job_name=JOB_NAME)
