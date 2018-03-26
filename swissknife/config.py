"""
A set of utilities helpful to configure package or script before running.
"""
import os
import string
import logging
import logging.config
from io import StringIO
from os.path import dirname, join


DEFAULT_LOGGER = join(dirname(__file__), 'configs', 'logger.yaml')


def debug_logger(output_file=None):
    """Returns the most verbose logger, saving all messages starting from
    DEBUG level both into stdout and file (if name provided).
    """
    if output_file is not None:
        logger = get_logger(
            name='main',
            output_file=output_file,
            console_level='debug',
            file_level='debug')
    else:
        logger = get_logger('console', console_level='debug')
    return logger


def main_logger(output_file='run.log'):
    """Returns logger which sends output to stdout and file."""
    return get_logger('main', output_file=output_file)


def console_logger():
    """Returns logger which sends output to stdout only."""
    return get_logger('console')


def notebook_logger():
    """Returns logger with simplified message format sending output into
    stdout.
    """
    return get_logger('notebook')


def get_logger(name='main',
               output_file='run.log',
               console_level='info',
               file_level='warning',
               config_file=DEFAULT_LOGGER):
    """Configures logger using YAML configuration file.

    Args:
        name: Logger name.
        output_file: File to save logging messages.
        console_level: Minimal severity level of messages printed into stdout.
        file_level: Minimal severity level of messages saved into log file.
        config_file: Path to YAML file with logger configuration.

    Returns:
        log: An instantiated logger object.

    """
    try:
        import yaml
    except ImportError:
        raise ValueError(
            'cannot initialize logger with YAML config - '
            'yaml package is not installed')
    else:
        with open(config_file) as fp:
            content = fp.read()

    template = string.Template(content)
    config_string = template.substitute(
        logfile=output_file,
        file_level=file_level.upper(),
        console_level=console_level.upper())
    config = yaml.load(StringIO(config_string))
    logging.config.dictConfig(config)
    logger = logging.getLogger(name)
    return logger


def get_env_variable(name: str, default=None):
    """Gets environment variable if available.

    Args:
        name: An environment variable name.
        default: A fallback value if variable is not defined.

    """
    value = os.environ.get(name, default)
    if not value:
        log = console_logger()
        log.warning('%s environment variable is not set and has no default '
                    'fallback value', name)
        return None
    return value
