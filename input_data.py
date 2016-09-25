import numpy as np
import os
import csv

class DataSet(object):
    def __init__(self, filename):
        # Open up the file with data
        trainDataFile = open(filename, 'r');
        # Read it as a csv
        dataReader = csv.reader(trainDataFile);
        # Turn the data into an array
        data = list(dataReader);
        # Delete the csv header row
        del data[0];
        # Skip the first two columns (elem index and date)
        data = [x[2:8] for x in data];
        # Randomize the array indexing
        np.random.shuffle(data);
        # Parse out the inputs (first five entries)
        self._features = np.array([x[0:5] for x in data], np.float32);
        # Normalize the features
        self._features = self._features / self._features.max(axis=0)
        # Parse out labels (last entry)
        self._labels = np.array([x[5:6] for x in data], np.uint8);
        # self._labels = [];
        # # Use two classifiers instead of boolean for labels
        # # Todo: figure out how to do it w/o this
        # for bl in [x[5:6] for x in data]:
        #     self._labels.append([0, 1] if int(bl[0]) == 1 else [1, 0]);
        # # Convert into numpy array
        # self._labels = np.array(self._labels, np.uint8);

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

def read_data_sets(datasetDir):
    class DataSets(object):
        pass

    data_sets = DataSets();

    data_sets.train = DataSet(os.path.join(datasetDir, 'data_training.csv'));
    data_sets.validate = DataSet(os.path.join(datasetDir, 'data_validate.csv'));
    data_sets.test = DataSet(os.path.join(datasetDir, 'data_test.csv'));

    return data_sets;
