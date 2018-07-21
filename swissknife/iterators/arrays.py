class BatchArrayIterator:
    """

    """

    def __init__(self,
                 array, *arrays,
                 batch_size: int=32,
                 infinite: bool=False,
                 same_size_batches: bool=False):

        arrays = [array] + list(arrays)