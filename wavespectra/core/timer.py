"""
Fine grain timing for checking different parts of script
Use it by wrapping blocks of code that you want to time
"""

import time


class Timer(object):
    """Wrap blocks of code to be timed.

    with Timer(msg='Elapsed time to do stuff') as t:
        do stuff

    """

    def __init__(self, msg="elapsed time", verbose=True):
        """
        msg :: will be printed before the elapsed time value.
        verbose :: if True prints Elapsed time, if False only creates instance
            with attributes.

        """
        self.msg = msg

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.secs = time.time() - self.start
        print(f"{self.msg}: {self.secs:f} sec")
