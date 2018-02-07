from pathlib import Path

from swissknife.utils import gather_files


def test_gathering_files_with_extensions_from_tree(tmpdir):
    """Tests collecting files with provided extensions from the source folder
    without deleting source folder.
    """
    content = 'content'
    src = tmpdir.mkdir('src')
    dst = tmpdir.mkdir('dst')
    files = (
        src.join('results.log'),
        src.join('failure.out'),
        src.mkdir('sub').join('log.txt'))
    for file in files:
        file.write(content)

    gather_files(src=src, dst=dst, exts='log|txt|out')

    contents = [file.read_text() for file in Path(dst).glob('**/*.*')]
    assert len(contents) == len(files)
    assert all(content == 'content' for content in contents)


