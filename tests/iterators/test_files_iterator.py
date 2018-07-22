from swissknife.iterators.files import FilesIterator


def test_files_iterator_has_required_properties():
    iterator = FilesIterator('/path', pattern='a|b|c')

    assert iterator.folder == '/path'
    assert iterator.pattern == 'a|b|c'
    assert iterator.batch_size == 32
    assert not iterator.infinite
    assert not iterator.same_size_batches
    assert iterator.extensions == ['a', 'b', 'c']


def test_files_iterator_yields_batches_with_files_paths(tmpdir):
    folder = tmpdir.mkdir('files')
    for filename in list('abc'):
        folder.join('%s.txt' % filename).write('content')
    iterator = FilesIterator(folder, pattern='txt', batch_size=1)

    [a], [b], [c] = list(iterator)

    assert all([path.endswith('txt') for path in (a, b, c)])
