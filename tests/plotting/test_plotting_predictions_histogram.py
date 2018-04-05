import pytest
import matplotlib
matplotlib.use('Agg')

from swissknife.plotting import plot_predictions


@pytest.mark.parametrize('params', [
    {'best_color': 'red', 'other_colors': 'blue', 'alpha': 0.2},
    {'title': 'There Is A Title'},
    {'title': 'Sorted By Prediction Probability', 'sort_by_probability': True},
    {'title': 'Sorted Alphabetically By Labels', 'sort_by_probability': False}
])
def test_plotting_predictions_histogram(params, dog_image, request):
    predictions = {
        'dog': 0.90,
        'fox': 0.05,
        'cat': 0.04,
        'whale': 0.009,
        'elephant': 0.001
    }

    figure = plot_predictions(dog_image, predictions, **params)

    figure.savefig('%s.png' % request.node.name)
