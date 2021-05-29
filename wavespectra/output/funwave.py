"""FUNWAVE output plugin."""
import os
from zipfile import ZipFile, ZIP_DEFLATED
from io import StringIO
import logging
import numpy as np
from wavespectra.core.attributes import attrs


logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


def to_funwave(
    self,
    filename,
    clip=True,
):
    """Write spectra in Funwave format.

    Args:
        - filename (str): Name for output Funwave file.
        - clip (bool): Clip directions outside [-90, 90] range in cartesian convention.

    Note:
        - Format description: https://fengyanshi.github.io/build/html/wavemaker_para.html.
        - Directions converted from  wavespectra (0N, CW, from) to Cartesian (0E, CCW, to).
        - Funwave only seems to deal with directions in the [-90, 90] range in
          cartesian convention, use clip=True to clip spectra outside that range.
        - Both 2D E(f,d) and 1d E(f) spectra are supported.
        - If the SpecArray has more than one spectrum, multiple files are created in a
          zip archive defined by replacing the extension of `filename` by ".zip".

    """
    darr = self.efth.copy(deep=True)

    # Convert directions to Cartesian, going-to convention
    if attrs.DIRNAME in self.efth.dims:
        dir = (270 - self.dir.values) % 360
        dir[dir > 180] = dir[dir > 180] - 360
        darr = darr.assign_coords({attrs.DIRNAME: dir}).sortby(attrs.DIRNAME)

        # Clip directions outside [-90, 90] range
        if clip:
            darr = darr.sel(**{attrs.DIRNAME: slice(-90, 90)})

    stack_dims = list(darr.spec._non_spec_dims)
    dimsizes = set([darr[d].size for d in darr.spec._non_spec_dims])

    if not stack_dims or dimsizes == {1}:
        # Single spectrum in object, write directly to txt file
        funwave_spectrum(darr, filename)

    else:
        # Multiple spectra in object, write each txt file in a zip archive
        darrs = darr.stack({"stacked": stack_dims})
        fpath, fext = os.path.splitext(filename)
        zipname = fpath + ".zip"
        logger.info(f"Multiple spectra, writing txt files in zip archive {zipname}")
        with ZipFile(zipname, "w", compression=ZIP_DEFLATED) as zstream:
            for ind in range(darrs.stacked.size):
                darr = darrs.isel(stacked=[ind]).unstack()

                # Prefix to compose file name
                prefix = "_".join([f"{d}-{darr[d].values[0]}" for d in stack_dims])

                # Squeeze out non-spectral dimensions which should have lenght 1
                darr = darr.squeeze(drop=True)

                # Write spectrum to zip archive
                fname = os.path.basename(fpath) + "-" + prefix + fext
                logger.debug(f"Write {fname} to {zipname}")
                spectrum = funwave_spectrum(darr, None).getvalue()
                zstream.writestr(fname, spectrum)


def funwave_spectrum(darr, filename):
    """Funwave spectrum memory buffer.

    Args:
        darr (SpecArray): Spectrum to write (only `freq`, `dir` dims are allowed).
        filename (str): Name of file to save spectrum to, choose `None` if you don't
            want to save it to file but only return the memory buffer.

    Returns:
        StringIO memory buffer with spectrum object.

    """
    # Ensure direction dim exists for 1D spectrum and has value 0
    if attrs.DIRNAME not in darr.dims:
        darr = darr.expand_dims({attrs.DIRNAME: [0]}, axis=-1)
    elif darr[attrs.DIRNAME].size == 1:
        darr = darr.assign_coords({attrs.DIRNAME: [0]})

    # Amplitudes and phases
    amp = np.sqrt(darr * darr.spec.dfarr * darr.spec.dd * 8) / 2
    amp = amp.transpose()
    nd, nf = amp.shape
    phi = np.random.uniform(0, 1, (nd, nf)) * 360.0

    # Peak wave period
    tp = float(darr.spec.tp())

    # Create spectrum in memory buffer
    s = StringIO()
    s.write(f"{nf:>5d}{nd:>5d}   - NumFreq NumDir\n")
    s.write(f"{tp:>10.3f}   - PeakPeriod\n")
    s.write(nf * "%10.5f   - Freq\n" % tuple(darr[attrs.FREQNAME].values))
    s.write(nd * "%10.3f   - Dire\n" % tuple(darr[attrs.DIRNAME].values))
    if nd == 1:
        np.savetxt(s, amp.squeeze(drop=True), fmt="%12.8f", delimiter="")
    else:
        np.savetxt(s, amp, fmt="%12.8f", delimiter="")
        np.savetxt(s, phi, fmt="%12.3f", delimiter="")

    # Save to file
    if filename is not None:
        with open(filename, "w") as fid:
            fid.write(s.getvalue())

    return s
