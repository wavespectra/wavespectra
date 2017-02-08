"""
SwanSpecFile class from pymo with a change to read up NODATA with same shape
"""
import copy
import datetime
from pymo.core.wavespec import Spectrum
from pymo.data.spectra import SwanSpecFile

class SwanSpecFile2(SwanSpecFile):
    """
    Change read method to define NODATA as a zeroed array with expected shape (as opposed to one-element zero array)
    """
    def __init__(self, filename, Stmpl=None, locations=None, time=False, id='Swan Spectrum', dirorder=False, append=False):
        SwanSpecFile.__init__(self, filename, Stmpl=Stmpl, locations=locations, time=time, id=id, dirorder=dirorder, append=append)

    def read(self):
        if not self.f:
            return None
        if isinstance(self.times, list):
            line = self.f.readline()
            if line:
                ttime = datetime.datetime.strptime(line[0:15], '%Y%m%d.%H%M%S')
                self.times.append(ttime)
            else:
                return None
        Sout = []
        for ip, pp in enumerate(self.locations):
            if self._readhdr('NODATA'):
                # Snew = Spectrum()
                Snew = copy.deepcopy(self.S)
                Snew.S[:] = 0.
            else:
                Snew = copy.deepcopy(self.S)
                if self._readhdr('ZERO'):
                    Snew.S[:] = 0.
                elif self._readhdr('FACTOR'):
                    fac = float(self.f.readline())
                    for i, f in enumerate(self.S.freqs):
                        line = self.f.readline()
                        lsplit = line.split()
                        try:
                            s1 = map(float,lsplit)
                        except:
                            s1 = numpy.nan * numpy.ones((1, len(lsplit)))
                        Snew.S[i,:] = s1
                    Snew.S = fac * Snew.S
                    if self.dirmap:
                        Snew.S = Snew.S[:, self.dirmap]
            Sout.append(Snew)
        return Sout[0] if len(Sout)==1 else Sout