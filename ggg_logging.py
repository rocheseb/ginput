from __future__ import print_function, division

import logging


###########
# LOGGING #
###########

# Set up the logger for this package
# Add a new level (https://stackoverflow.com/a/13638084)
_important_log_level = 29  # just below warning at 30
logging.addLevelName(_important_log_level, 'IMPORTANT')


def _log_important(self, message, *args, **kwargs):
    if self.isEnabledFor(_important_log_level):
        # it is correct - *args in is passes just as args
        self._log(_important_log_level, message, args, **kwargs)


logging.Logger.important = _log_important

_formatter = logging.Formatter('%(levelname)s (from %(funcName)s in %(filename)s): %(message)s')
_log_handler = logging.StreamHandler()
_log_handler.setFormatter(_formatter)
logger = logging.getLogger('val-logger')
logger.addHandler(_log_handler)

logger.setLevel('DEBUG')


def _log_to_file(filename=''):
    fh = logging.FileHandler(filename=filename, mode='w')
    fh.setFormatter(_formatter)
    logger.addHandler(fh)


def setup_logger(log_file=None, level=0):
    """
    Modify the package logger to add log file and change the verbosity level.

    :param log_file: a filename to write the log to.
    :type log_file: str

    :param level: the verbosity level. < 0 suppresses all messages less important than warnings, 0 prints important
     messages, 1 print informational messages, >= 2 prints everything.
    :type level: int

    :return: None
    """
    if log_file is not None:
        _log_to_file(log_file)

    if level < 0:
        logger.setLevel('WARNING')
    elif level == 0:
        logger.setLevel('IMPORTANT')
    elif level == 1:
        logger.setLevel('INFO')
    elif level >= 2:
        logger.setLevel("DEBUG")
