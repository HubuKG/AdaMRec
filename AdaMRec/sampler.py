import numpy
from multiprocessing import Process, Queue
from scipy.sparse import lil_matrix


def sample_function(user_item_matrix, batch_size, n_negative, result_queue, check_negative=True):
    """
    positive and negative training samples are generated from the user-item matrix for training.
    param:
        user_item_matrix: the user-item matrix for positive user-item pairs
        batch_size: number of samples to return
        n_negative: number of negative samples per user-positive-item pair
        result_queue: the output queue
    return: None
    """
    user_item_matrix = lil_matrix(user_item_matrix)
    user_item_pairs = numpy.asarray(user_item_matrix.nonzero()).T
    user_to_positive_set = {u: set(row) for u, row in enumerate(user_item_matrix.rows)}
    while True:
        numpy.random.shuffle(user_item_pairs)
        for i in range(int(len(user_item_pairs) / batch_size)):

            user_positive_items_pairs = user_item_pairs[i * batch_size: (i + 1) * batch_size, :]

            # negative samples
            negative_samples = numpy.random.randint(
                0,
                user_item_matrix.shape[1],
                size=(batch_size, n_negative))

            if check_negative:
                for user_positive, negatives, i in zip(user_positive_items_pairs,
                                                       negative_samples,
                                                       range(len(negative_samples))):
                    user = user_positive[0]
                    for j, neg in enumerate(negatives):
                        while neg in user_to_positive_set[user]:
                            negative_samples[i, j] = neg = numpy.random.randint(0, user_item_matrix.shape[1])
            result_queue.put((user_positive_items_pairs, negative_samples))



class WarpSampler(object):
    """
    A generator that generates user-positive item pairs and negative item pairs in parallel
    """
    def __init__(self, user_item_matrix, batch_size=10000, n_negative=10, n_workers=5, check_negative=True):
        self.result_queue = Queue(maxsize=n_workers*2)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(user_item_matrix,
                                                      batch_size,
                                                      n_negative,
                                                      self.result_queue,
                                                      check_negative)))
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
