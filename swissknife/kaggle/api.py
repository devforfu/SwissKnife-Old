class ClassifierSubmission:
    """Utility to make submissions for Kaggle's classification competitions."""

    def __init__(self, floats_format='%1.17f'):
        self.floats_format = floats_format

    def create(self, classes, predictions, output):
        """Creates a submission file.

        Args:
            classes: List of classes.
            predictions: Dictionary with predictions probabilities for each
                testing sample.
            output: File-like object where submission results will be written,
                including header and sample IDs column.

        """
        header = ['id'] + list(classes)
        rows = [header]
        fmt = self.floats_format

        for identifier, probabilities in predictions.items():
            row = [identifier] + [fmt % p for p in probabilities]
            rows.append(row)

        strings = [','.join(row) + '\n' for row in rows]
        if isinstance(output, str):
            with open(output, 'w') as fp:
                for string in strings:
                    fp.write(string)
        elif hasattr(output, 'write'):
            for string in strings:
                output.write(string)
        else:
            raise ValueError(
                'unexpected output type: %s.'
                ' Only strings and file-like '
                'objects are supported', type(output))
