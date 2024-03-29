from swissknife.kaggle.api import ClassifierSubmission
from swissknife.tests import StringBuffer


def test_classifier_submission_generates_valid_file():
    """Tests generating submission CSV string in valid format from dictionary
    with predictions and list with classes names.
    """
    predictions = {
        'id1': [0.1, 0.6, 0.3],
        'id2': [0.05, 0.95, 0.0],
        'id3': [1.0, 0.0, 0.0]
    }
    classes = ['first', 'second', 'third']
    submission = ClassifierSubmission(floats_format='%1.17f')

    with StringBuffer() as buffer:
        submission.create(classes, predictions, output=buffer)

    string = buffer.captured
    assert string
    assert len(string.split()) == len(predictions) + 1
    assert all(uid in string for uid in predictions.keys())


def test_classifier_submission_users_specified_floats_formatting():
    """Tests formatting floating point values representing submission scores
    using provided formatting string.
    """
    predictions = {'id1': [0.0001, 0.0005, 0.0005]}
    classes = ['first', 'second', 'third']
    submission = ClassifierSubmission(floats_format='%1.3f')

    with StringBuffer() as buffer:
        submission.create(classes, predictions, output=buffer)

    string = buffer.lines[-1]
    assert [float(x) == 0 for x in string.split(',')[1:]]
