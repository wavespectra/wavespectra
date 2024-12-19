"""Read and write swan spectra files"""

from pathlib import Path
import re
import gzip
import datetime
import pandas as pd
import numpy as np

from wavespectra.core.attributes import attrs
from wavespectra.core.utils import to_nautical

E2V = 1025 * 9.81


class SwanSpecFile(object):
    """Read spectra in SWAN ASCII format."""

    def __init__(
        self,
        filename,
        freqs=None,
        dirs=None,
        x=None,
        y=None,
        time=False,
        id="Swan Spectrum",
        dirorder=False,
        append=False,
        tabfile=None,
        compresslevel=6,
    ):
        self.times = False
        self.filename = Path(filename)
        self.tabfile = tabfile or Path(
            str(self.filename).replace(".gz", "")
        ).with_suffix(".tab")
        self.is_tab = False
        self.buf = None

        if self.filename.suffix == ".gz":
            fopen = gzip.open
            kwargs = {"compresslevel": compresslevel}
        else:
            fopen = open
            kwargs = {}
        if freqs is not None:  # Writable file
            self.freqs = np.array(freqs)
            self.dirs = np.array(dirs)
            self.x = np.array(x)
            self.y = np.array(y)
            if time:
                self.times = []
            self.fid = fopen(filename, "wt", **kwargs)
            self.write_header(time, id)
            self.fmt = len(self.dirs) * "{:5.0f}"
        else:
            self.fid = fopen(filename, "rt+" if append else "rt")
            self._read_header("SWAN")
            while True:
                if not self._read_header("$"):
                    break
            if self._read_header("TIME"):
                self._read_header("1")
                self.times = []
            self.x = []
            self.y = []
            locs = self._read_header("LONLAT", True) or self._read_header(
                "LOCATIONS", True
            )
            for ip in locs:
                xy = [float(val) for val in ip.split()]
                self.x.append(xy[0])
                self.y.append(xy[1])
            self.x = np.array(self.x)
            self.y = np.array(self.y)
            self.afreq = self._read_header("AFREQ", True)
            self.rfreq = self._read_header("RFREQ", True)
            self.ndir = self._read_header("NDIR", True)
            self.cdir = self._read_header("CDIR", True)
            if self.afreq:
                self.freqs = np.array([float(val) for val in self.afreq])
            else:
                self.freqs = np.array([float(val) for val in self.rfreq])
            if self.ndir:
                self.dirs = np.array([float(val) for val in self.ndir])
            else:
                self.dirs = to_nautical(np.array([float(val) for val in self.cdir]))
            self._read_header("QUANT", True)
            # Figure units out, if Energy density factor needs to be applied
            units = self.fid.readline().strip().split()[0]
            if units.upper().startswith("J"):
                self.units_factor = E2V
            else:
                self.units_factor = 1.0
            self.excval = int(float(self.fid.readline().split()[0]))

        if dirorder:
            self.dirmap = list(np.argsort(self.dirs % 360.0))
            self.dirs = self.dirs[self.dirmap] % 360.0
        else:
            self.dirmap = False
        lons = np.unique(self.x)
        lats = np.unique(self.y)
        self.is_grid = len(lons) * len(lats) == len(self.x)
        self.is_tab = self.tabfile.is_file() & (len(lons) * len(lats) == 1)

    def _read_header(self, keyword, numspec=False):
        if not self.buf:
            self.buf = self.fid.readline()
        if self.buf.find(keyword) >= 0:
            if numspec:
                line = self.fid.readline()
                n = int(re.findall(r"\b(\d+)\b", line)[0])
                self.buf = [self.fid.readline() for i in range(0, n)]
            rtn = self.buf
            self.buf = None
        else:
            rtn = False
        return rtn

    def read(self):
        """Read single timestep from current position in file."""
        if not self.fid:
            return None
        if isinstance(self.times, list):
            line = self.fid.readline()
            if line:
                ttime = datetime.datetime.strptime(line[0:15], "%Y%m%d.%H%M%S")
                self.times.append(ttime)
            else:
                return None
        Sout = []
        for ip, pp in enumerate(self.x):
            Snew = np.nan * np.zeros((len(self.freqs), len(self.dirs)))
            if self._read_header("NODATA"):
                pass
            else:
                if self._read_header("ZERO"):
                    Snew = np.zeros((len(self.freqs), len(self.dirs)))
                elif self._read_header("FACTOR"):
                    fac = float(self.fid.readline())
                    for i, f in enumerate(self.freqs):
                        line = self.fid.readline()
                        lsplit = line.split()
                        try:
                            Snew[i, :] = [float(val) for val in lsplit]
                        except Exception:
                            import warnings

                            warnings.warn("Check what this is supposed to be doing.")
                            pass
                    Snew *= fac
                    if self.dirmap:
                        Snew = Snew[:, self.dirmap]
                else:  # For files with no timestamp
                    return None
            Sout.append(Snew / self.units_factor)
        return Sout

    def readall(self):
        """Read the entire file."""
        while True:
            sset = self.read()
            if sset:
                yield sset
            else:
                break

    def write_header(self, time=False, str1="", str2="", timecode=1, excval=-99):
        """Write header to file."""
        # Description
        strout = "{:40}{}\n".format("SWAN   1", "Swan standard spectral file")
        strout += "{:4}{}\n".format("$", str1)
        strout += "{:4}{}\n".format("$", str2)
        # Time
        if time:
            strout += "{:40}{}\n".format("TIME", "time-dependent data")
            strout += "{:>6d}{:34}{}\n".format(timecode, "", "time coding option")
        # Location
        strout += "{:40}{}\n".format("LONLAT", "locations in spherical coordinates")
        strout += "{:>6d}{:34}{}\n".format(len(self.x), "", "number of locations")
        for x, y in zip(self.x, self.y):
            strout += "{:2}{:<0.6f}{:2}{:<0.6f}\n".format("", x, "", y)
        # Frequency
        strout += "{:40}{}\n".format("AFREQ", "absolute frequencies in Hz")
        strout += "{:6d}{:34}{}\n".format(len(self.freqs), "", "number of frequencies")
        for freq in self.freqs:
            strout += "{:>11.5f}\n".format(freq)
        # Direction
        strout += "{:40}{}\n".format("NDIR", "spectral nautical directions in degr")
        strout += "{:6d}{:34}{}\n".format(len(self.dirs), "", "number of directions")
        for wdir in self.dirs:
            strout += "{:>11.4f}\n".format(wdir)
        # Data
        strout += "QUANT\n{:>6d}{:34}{}\n".format(
            1, "", "number of quantities in table"
        )
        strout += "{:40}{}\n".format("VaDens", "variance densities in m2/Hz/degr")
        strout += "{:40}{}\n".format("m2/Hz/degr", "unit")
        strout += "{:3}{:<37g}{}\n".format("", excval, "exception value")
        # Dumping
        self.fid.write(strout)

    def write_spectra(self, arr, time=None):
        """Write spectra from single timestamp.

        Args:
            arr (3D ndarray): spectra to write S(site, freq, dim).
            time (yyymmdd.HHMMSS): time of spectra to write.

        """
        if time is not None:
            self.fid.write(f"{time:40}{'date and time'}\n")
        for spec in arr:
            fac = spec.max() / 9998.0
            if np.isnan(fac):
                self.fid.write("NODATA\n")
            elif fac <= 0:
                self.fid.write("ZERO\n")
            else:
                self.fid.write(f"FACTOR\n{'':4}{fac:0.8E}\n")
                np.savetxt(self.fid, spec / fac, fmt="%5.0f", delimiter="")

    def close(self):
        """Close file handle."""
        if self.fid:
            self.fid.close()
        self.fid = False


def read_tab(filename, toff=0):
    """Read swan table file.

    Args:
        filename (str): name of SWAN tab file to read
        toff (float): timezone offset in hours

    Returns:
        Pandas DataFrame object

    """
    df = pd.read_csv(
        filename,
        sep=r"\s+",
        skiprows=[0, 1, 2, 3, 5, 6],
        parse_dates=[0],
        date_format="%Y%m%d.%H%M%S",
        index_col=0,
    )
    df.index.name = attrs.TIMENAME
    df.index = df.index.shift(toff, freq="1h")
    for col1, col2 in zip(df.columns[-1:0:-1], df.columns[-2::-1]):
        df = df.rename(columns={col2: col1})
    return df.iloc[:, 0:-1]
