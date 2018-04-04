from swissknife.files import SavingFolder


def test_model_saving_paths_relative_to_root_folder(tmpdir):
    root = tmpdir.mkdir('all_models')
    model = root.join('model').join('model.h5')
    history = root.join('model').join('model.csv')

    saver = SavingFolder('model', models_root=str(root))

    assert saver.model_name == 'model'
    assert saver.models_root == str(root)
    assert saver.model_path == str(model)
    assert saver.history_path == str(history)
