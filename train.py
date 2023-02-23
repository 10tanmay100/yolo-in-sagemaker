import os
import argparse
import subprocess
from sagemaker_tensorflow import TensorFlow

# Set the paths to the training and validation datasets
TRAIN_DATASET = '/opt/ml/input/data/train'
VAL_DATASET = '/opt/ml/input/data/validation'

# Set the output path for the trained model
MODEL_PATH = '/opt/ml/model'

# Parse the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--learning-rate', type=float, default=0.001)
args, _ = parser.parse_known_args()

# Start the training job
if __name__ == '__main__':
    tensorflow = TensorFlow(
        entry_point='yolov5/train.py',
        role=os.environ['SM_ROLE'],
        instance_count=os.environ['SM_NUM_GPUS'],
        instance_type=os.environ['SM_INSTANCE_TYPE'],
        framework_version='2.5',
        py_version='py37',
        distribution={
            'smdistributed': {
                'dataparallel': {
                    'enabled': True
                }
            }
        },
        hyperparameters={
            'batch-size': args.batch_size,
            'epochs': args.epochs,
            'learning-rate': args.learning_rate,
        },
        output_path=MODEL_PATH,
    )

    train_data = tensorflow.sagemaker_session.upload_data(
        path=TRAIN_DATASET,
        key_prefix='data/train',
    )

    val_data = tensorflow.sagemaker_session.upload_data(
        path=VAL_DATASET,
        key_prefix='data/validation',
    )

    tensorflow.fit({'train': train_data, 'validation': val_data})
