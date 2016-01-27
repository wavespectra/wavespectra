"""
Fine grain timing for checking different parts of script
Use it by wrapping blocks of code that you want to time
e.g.:
from pymsl.core.timer import Timer
with Timer(msg='Total elapsed time') as t:
    do stuff
    do more stuff
"""

class Timer(object):
    """
    Wrap blocks of code to be timed using this class
    """
    def __init__(self, msg='elapsed time'):
        self.msg = msg

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.secs = time.time() - self.start
        logging.info('%s: %f sec' % (self.msg, self.secs))
