"""
Data transformers.
"""
from itertools import islice


class GeneratorPipeline:
    """Concatenates together a group of generators transforming original data
    into format ready for model's training.

    The pipeline makes two assumptions about generators passed into its
    constructor:

        1) A `source` generator shouldn't expect getting any values via
        `send()` method, i.e. it should only emit samples.

        2) Each generator in `steps` list should expect receiving value from
        previous step as well as produce some result for next steps. It case
        when step returns a None value as result of its processing, this value
        is treated as sentinel meaning that subsequent steps should be ignored
        and the pipeline should be advanced to next iteration.

    Attributes:
        source: Generator that yields input samples.
        steps: List of generators accepting data in sequence from each other.
        max_iters: Limits number of calls made to source generator.

    """
    def __init__(self, source, *steps, max_iters=None):
        self.max_iters = max_iters
        self.source = source
        self._source = self.source
        self._steps = list(steps)
        self._need_reset = []
        self._init()

    def configure(self, max_iters=None):
        """Sets additional parameters changing pipeline behaviour."""
        self.max_iters = max_iters
        self._init()

    def add(self, generator):
        """Adds new generator into pipeline."""
        self._steps.append(generator)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        """Pass next data batch through sequence of transformers."""
        while True:
            batch = next(self._source)
            transformed = self.send(batch)
            self.reset_if_needed()
            if transformed is not None:
                break
        return transformed

    def send(self, batch):
        """Sends data batch into sequence of transforming generators."""

        processed = batch
        for generator in self._steps:
            processed = generator.send(processed)
            self._need_reset.append(generator)
            if processed is None:
                break
        return processed

    def reset_if_needed(self):
        """Advances steps which yielded not None value on previous iteration
        to make them ready accept a new value.
        """
        generators = self._need_reset
        while generators:
            generator = generators.pop()
            generator.send(None)

    def _init(self):
        if self.max_iters is None:
            self._source = self.source
        else:
            self._source = islice(self.source, 0, self.max_iters)
        for generator in self._steps:
            generator.send(None)
