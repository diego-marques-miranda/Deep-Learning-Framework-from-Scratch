import numpy as np

class DataLoader:
    """Iterates over dataset arrays in shuffled, fixed-size batches."""
    def __init__(self, data, target, batch_size, shuffle=True):
        self.data = data
        self.target = target
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current_index = 0
        self.num_samples = len(data)

    def __iter__(self):
        """Resets iteration state and shuffles data arrays in unison if enabled."""
        self.current_index = 0

        if self.shuffle:
            indexes = np.random.permutation(len(self.data))
            self.data = self.data[indexes]
            self.target = self.target[indexes]

        return self

    def __next__(self):
        """Fetches the next batch of data and targets, raising StopIteration when finished."""
        if self.current_index >= self.num_samples:
            raise StopIteration
        
        end_index = self.current_index + self.batch_size

        if end_index > self.num_samples:
            end_index = self.num_samples

        batch_x = self.data[self.current_index:end_index]
        batch_y = self.target[self.current_index:end_index]

        self.current_index = end_index

        return batch_x, batch_y