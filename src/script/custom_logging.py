import os
import logging


def set_logging():
    # set logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
    # Disable c10d logging
    logging.getLogger('c10d').setLevel(logging.ERROR)
