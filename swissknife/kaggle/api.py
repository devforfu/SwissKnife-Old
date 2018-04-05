import time
from io import StringIO
from subprocess import Popen, PIPE

import pandas as pd

from ..config import console_logger


class ClassifierSubmission:
    """Utility to make submissions for Kaggle's classification competitions."""

    def __init__(self, floats_format='%1.17f', log=None):
        self.floats_format = floats_format
        self.log = log or console_logger()

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

    def submit(self, filename, competition, message=None, timeout=60):
        if message is None:
            message = filename

        self.log.info('Sending submission...')
        ok = self._send_submission(competition, message, filename)
        if not ok:
            self.log.error('Cannot submit results: submission command failed')
            return None

        self.log.info('Evaluating results...')
        result = self._wait_for_evaluation(competition, timeout)
        if not result:
            self.log.error('Cannot retrieve submission result')
            return None

        return result

    def _send_submission(self, competition, message, output):
        cmd = 'kaggle competitions submit -c {} -m "{}" -f {}'
        ok, _ = self._run_command(cmd.format(competition, message, output))
        return ok

    def _wait_for_evaluation(self, competition, timeout):
        cmd = 'kaggle competitions submissions -c {} --csv'.format(competition)
        start = time.time()
        while True:
            elapsed = time.time() - start
            if elapsed >= timeout:
                return None
            ok, output = self._run_command(cmd, suppress=True)
            if not ok:
                return None
            submission_csv = '\n'.join(output)
            submissions = pd.read_csv(StringIO(submission_csv))
            latest_submission = submissions.iloc[0]
            if latest_submission.status != 'pending':
                return latest_submission.to_dict()
            time.sleep(1.0)

    def _run_command(self, command, suppress=False):
        process = Popen(
            command, stdout=PIPE, stderr=PIPE,
            bufsize=1, close_fds=True, shell=True)

        command_output = []

        while True:
            if process.poll() is not None:
                break
            for pipe in (process.stdout, process.stderr):
                while True:
                    raw_output = pipe.readline()
                    if not raw_output:
                        break
                    decoded = raw_output.decode(encoding='utf-8').strip('\n')
                    if not decoded:
                        continue
                    command_output.append(decoded)
                    if not suppress:
                        self.log.info(decoded)

        ok = process.returncode == 0
        return ok, command_output

