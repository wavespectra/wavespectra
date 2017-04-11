"""
Miscellaneous
"""
import datetime

to_datetime = lambda t: datetime.datetime.fromtimestamp(t.astype('int')/1e9)