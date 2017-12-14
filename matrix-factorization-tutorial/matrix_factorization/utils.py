"""
Utils (service code) for matrix factorization

Author: Alexander Hentschel
        alex.hentschel@axiomzen.co
"""
import numpy as np


def random_matrix(shape):
    std = 1.0 / np.sqrt(np.prod(shape))
    m = np.random.normal(0, std, shape)
    return np.abs(m)

def compute_loss(tf_session, loss, batch_size, data, user_idcs_var, item_idcs_var, target_var):
    number_batches = len(data) // batch_size
    avg_loss = 0.0
    for i in range(number_batches):
        start_idx = batch_size * i
        batch = data.iloc[start_idx: start_idx + batch_size]
        batch_loss = tf_session.run(loss, feed_dict={ user_idcs_var: batch['user index'].as_matrix(),
                                                      item_idcs_var: batch['item index'].as_matrix(),
                                                      target_var: batch['score'].as_matrix()
                                                     })
        avg_loss += batch_loss / number_batches
    return avg_loss

class Batcher():
    '''
    Batcher provides randomized mini-batches over a given data set. The batches
    will cover the entire data set without repetition. Only after the entire dataset has been
    covered, a new iteration over the dataset will start.
    '''

    def __init__(self, data):
        self.data = data
        self._data_indices = np.array(data.index.tolist()).copy()
        self._data_size = len(self._data_indices)
        self._first_unused_index = 0
        self._shuffle_data_indices()

    def _shuffle_data_indices(self):
        np.random.shuffle(self._data_indices)

    def next_batch(self, batch_size):
        if batch_size < 1:
            raise ValueError("Batchsize must be positive")
        if batch_size > self._data_size:
            raise ValueError("Requested batch size (%d) larger than data size (%d)" % (batch_size, self._data_size))
        end_idx = self._first_unused_index + batch_size
        if end_idx < self._data_size:
            batch_indices = self._data_indices[self._first_unused_index : end_idx]
        else:
            end_idx %= self._data_size
            batch_part1 = self._data_indices[self._first_unused_index:].copy() # array slicing creates a view only, which
            self._shuffle_data_indices()                                       # would get shuffled without a copy()
            batch_part2 = self._data_indices[: end_idx]
            batch_indices = np.append(batch_part1, batch_part2)
        self._first_unused_index = end_idx
        # CAUTION: batch_indices can be a view on self._data_indices which is shuffled in-place.
        # Need to copy batch_indices to prevent in-place modification of content
        return self.data.loc[batch_indices.copy()]
