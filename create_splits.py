import argparse
import glob
import os
import random
import shutil
import numpy as np

from utils import get_module_logger


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    # TODO: Implement function
    train = os.path.join(destination, 'train')
    if os.path.exists(train) == False:
        os.makedirs(train)
        
    val = os.path.join(destination, 'val')
    if os.path.exists(val) == False:
        os.makedirs(val)

    test = os.path.join(destination, 'test')
    if os.path.exists(test) == False:
        os.makedirs(test)
        
    # split data
    files = [filename for filename in glob.glob(f'{source}/*.tfrecord')]
    np.random.shuffle(files)

    train_data, val_data, test_data = np.split(files, [int(.86*len(files)), int(.10*len(files))])
    
    # move data to created directories
    for data in train_data:
        shutil.move(data, train)
    
    for data in val_data:
        shutil.move(data, val)
    
    for data in test_data:
        shutil.move(data, test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)