from io import StringIO
from swissknife.kaggle.api import ClassifierSubmission


def test_building_kaggle_submission_file():
    predictions = {
        'id1': [0.1, 0.6, 0.3],
        'id2': [0.05, 0.95, 0.0],
        'id3': [1.0, 0.0, 0.0]
    }
    classes = ['first', 'second', 'third']
    submit = ClassifierSubmission(floats_format='%1.17f')

    buffer = StringIO()
    submit.create(classes, predictions, output=buffer)
    buffer.seek(0)
    string = buffer.getvalue()

    assert string
    assert len(string.split()) == len(predictions) + 1
    assert all(uid in string for uid in predictions.keys())
