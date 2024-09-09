"""Reader for AWAC spectra from NMEA data

https://www.nortekgroup.com/

The spectra are reconstructed from the Fourier coefficients using the maximum entropy method to remove negative energy.
Normalization is applied per frequency to keep energy over directions identical to energy in non-directional spectrum
The 2D spectral data is scaled to the significant wave height (Hs) given in the wave parameters.

The time is returned as read from the timestamps. The time-zone depends on the configuration of the device and is not included in the output.

revision history:
-----------------
2024-08-15 : First version - Ruben de Bruin


Data description and example data:
-------------


Fourier coeffcients:

Field Description Form
0 Identifier “$PNORF”
1 Fourier coefficient flag (A1/B1/A2/B2) "CC"
2 Date MMDDYY
3 Time hhmmss.s
4 Spectrum basis type (0=pressure, 1=velocity,
3=AST) n
5 Start Frequency (Hz) d.dd
6 Step Frequency (Hz) d.dd
7 Number of Frequencies N nnn
8 Fourier Coefficient CC [frequency 1] d.dddd
9 Fourier Coefficient CC [frequency 2] d.dddd
N+7 Fourier Coefficient CC [frequency N] d.dddd
N+8 Checksum (hex) *hh

$PNORF,A1,050120,000101,3,0.02,0.01,48,-0.1370,-0.1413,-0.2548,-0.7437,-0.9243,-0.9156,-0.9092,-0.9027,-0.8771,-0.8651,-0.8662,-0.8520,-0.8440,-0.8249,-0.8166,-0.8455,-0.8518,-0.8404,-0.8077,-0.7297,-0.6421,-0.5878,-0.5803,-0.5477,-0.5559,-0.5445,-0.3028,-0.2844,-0.1277,-0.2942,-0.2004,-0.1690,-0.1242,-0.0984,-0.0428,-0.0352,-0.0103,-0.0138,0.0057,-9.0000,-9.0000,-9.0000,-9.0000,-9.0000,-9.0000,-9.0000,-9.0000,-9.0000*06
$PNORF,B1,050120,000101,3,0.02,0.01,48,0.2317,0.2715,0.2530,0.2815,0.3104,0.3728,0.3932,0.3831,0.3758,0.3861,0.3942,0.4068,0.3551,0.3375,0.2897,0.2033,0.1489,0.0765,0.1139,0.1978,0.2613,0.1794,0.2080,0.0510,0.0194,-0.0606,0.0258,0.1475,0.1210,0.0400,0.0183,-0.0247,0.0287,0.0026,-0.0161,-0.0041,0.0134,0.0183,0.0113,-9.0000,-9.0000,-9.0000,-9.0000,-9.0000,-9.0000,-9.0000,-9.0000,-9.0000*05
$PNORF,A2,050120,000101,3,0.02,0.01,48,0.0298,-0.0182,0.0292,0.5327,0.7465,0.6843,0.6601,0.6511,0.5770,0.5368,0.5383,0.5018,0.5014,0.4659,0.4671,0.5400,0.5509,0.5443,0.4952,0.3610,0.2616,0.2780,0.3188,0.2016,0.1471,0.2066,0.1387,0.0577,0.0479,0.0829,0.0135,0.0651,-0.0119,0.0106,-0.0576,-0.0748,-0.0762,-0.0359,-0.0048,-9.0000,-9.0000,-9.0000,-9.0000,-9.0000,-9.0000,-9.0000,-9.0000,-9.0000*2B
$PNORF,B2,050120,000101,3,0.02,0.01,48,-0.0619,-0.0377,-0.0514,-0.3674,-0.5502,-0.6668,-0.7015,-0.6633,-0.6189,-0.6227,-0.6373,-0.6422,-0.5306,-0.4705,-0.3596,-0.2469,-0.1640,-0.0557,-0.0666,-0.1313,-0.1363,-0.0540,-0.0295,0.0234,0.0481,0.0893,0.0843,-0.0011,0.0024,0.0494,0.1149,0.1554,0.0620,-0.0601,-0.0865,-0.0371,-0.0058,-0.0191,-0.0185,-9.0000,-9.0000,-9.0000,-9.0000,-9.0000,-9.0000,-9.0000,-9.0000,-9.0000*0C

Spectra:

Field Description Form
0 Identifier “$PNORE”
1 Date MMDDYY
2 Time hhmmss.s
3 Spectrum basis type (0=pressure, 1=velocity,3=AST) n
4 Start Frequency (Hz) d.dd
5 Step Frequency (Hz) d.dd
6 Number of Frequencies N nnn
7 Energy Density [frequency 1] (cm2/Hz) dddd.ddd
8 Energy Density [frequency 2] (cm2/Hz) dddd.ddd
N+6 Energy Density [frequency N] (cm2/Hz) dddd.ddd

$PNORE,050120,000101,3,0.02,0.01,98,0.063,0.029,0.020,0.182,2.671,5.169,6.311,4.587,2.876,1.802,1.817,1.
442,1.163,0.878,0.653,0.575,0.549,0.409,0.306,0.195,0.184,0.167,0.159,0.119,0.105,0.085,0.072,0.052,0.053,0.045,0.034,0.025,0.022,0.018,0.018,0.018,0.014,0.011,0.008,0.008,0.008,0.007,0.006,0.006,0.005,0.005,0.005,0.005,0.005,0.004,0.003,0.003,0.002,0.003,0.002,0.002,0.002,0.001,0.002,0.002,0.002,0.002,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.000,0.000,0.000,0.000,0.000,0.000*75

Wave Parameters:

Field Description Form
0 Identifier “$PNORW”
1 Date MMDDYY
2 Time hhmmss
3 Spectrum basis type (0=pressure, 1=velocity,3=AST)
4 Processing method (1=PUV, 2=SUV, 3=MLM,4=MLMST)
5 Hm0 (m) dd.dd
6 H3 (m) dd.dd
7 H10 (m) dd.dd
8 Hmax (m) dd.dd
9 Tm02 (s) dd.dd
10 Tp (s) dd.dd
11 Tz (s) dd.dd
12 DirTp (deg) ddd.dd
13 SprTp (deg) ddd.dd
14 Main Direction (deg) ddd.dd
15 Unidirectivity Index dd.dd
16 Mean pressure (dbar) dd.dd
17 Number of no detects n
18 Number of bad detects n
19 Near surface Current speed (m/s) dd.dd
20 Near surface Current direction (deg) ddd.dd
21 Error Code hhhh
22 Checksum (hex) *hh

Example:
$PNORW,073010,051001,3,4,0.55,0.51,0.63,0.82,2.76,3.33,2.97,55.06,78.91,337.62,0.48,22.35,0,1,0.
27,129.11,0000*4E

Reconstructing a spectrum from these coefficients
--------------------------------------------------

DirRes = 4;
freq = 0.02:0.01:(48*0.01 + 0.01);
dDir = 0:pi*DirRes/180:(2*pi-pi*DirRes/180);
Spec = S;

 % (conversion from Cart to Compass)
 A1temp = A1;
 B1temp= B1;
 A1 = -B1temp;
 B1= -A1temp;
 A2 = -A2;
 B2 = B2;

% Direction From
dr = pi;
A1new = cos(dr)*A1 + sin(dr)*B1;
B1new = -sin(dr)*A1 + cos(dr)*B1;
A2new = cos(2*dr)*A2 + sin(2*dr)*B2;
B2new = -sin(2*dr)*A2 + cos(2*dr)*B2;

 % Now apply maximum entropy method to remove negative energy
 for j=1:length(freq)

    % This little gem switches from cart2compass
    for dd = 1:length(dDir)
       dNewdir = pi/2 - dDir(dd);

       %Pure Fourier coef soln w/o MEM
       DirFieldOLD(j,dd) = (A1new(j)*cos(dNewdir) + B1new(j)*sin(dNewdir) + A2new(j)*cos(2*dNewdir) + B2new(j)*sin(2*dNewdir))*Spec(ii,j);
    end
 end

 --OR--


% Now apply maximum entropy method to remove negative energy
for j=1:length(freq)
  c1 = A1new(j) + 1i*B1new(j);
  c2 = A2new(j) + 1i*B2new(j);
  %c1 = A1new(ii,j) + i*B1new(ii,j);
  %c2 = A2new(ii,j) + i*B2new(ii,j);

  fi1 = (c1 - c2*conj(c1)) / (1 - abs(c1)*abs(c1));
  fi2 = c2 - c1*fi1;

  % This little gem switches from cart2compass
  for dd = 1:length(dDir)
     dNewdir = pi/2 - dDir(dd);
     %dNewdir = dDir(dd);

     numer = 1 - fi1*conj(c1) - fi2*conj(c2);
     %denom = 1 - fi1*exp(-i*dDir(dd)) - fi2*exp(-2.0*i*dDir(dd));
     denom = 1 - fi1*exp(-1i*dNewdir) - fi2*exp(-2.0*1i*dNewdir);
     DirField(j,dd) = real(numer/(abs(denom)*abs(denom)))/(2.0*pi)*Spec(ii,j)*pi*DirRes/180;

     %Pure Fourier coef soln w/o ME
     DirFieldOLD(j,dd) = (A1new(j)*cos(dNewdir) + B1new(j)*sin(dNewdir) + A2new(j)*cos(2*dNewdir) + B2new(j)*sin(2*dNewdir))*Spec(ii,j);
  end
end


"""

import datetime
import warnings

import numpy as np

import xarray as xr
from wavespectra.core.attributes import set_spec_attributes


def read_awac(filename):
    """Read Spectra from Nortec AWAC NMEA file.

    The spectra are reconstructed from the Fourier coefficients using the maximum
    entropy method to remove negative energy. Normalisation is applied per frequency to
    keep energy over directions identical to energy in non-directional spectrum. The 2D
    spectral data is scaled to the significant wave height (Hs) given in the wave
    parameters.

    Args:
        - filename (str): path to AWAC file.

    Returns:
        - dset (SpecDataset): spectra dataset object read from file.

    """
    with open(filename, "r") as f:
        lines = [line for line in f.read().split("\n")]
    return read_awac_strings(lines)


def parse_awac_nmnea_wave_parameters(lines):
    for line in lines:
        if line.startswith("$PNORW"):
            blocks = line.split(",")
            timestamp = blocks[1] + blocks[2]
            Hm0 = float(blocks[5])
            H3 = float(blocks[6])
            H10 = float(blocks[7])
            Hmax = float(blocks[8])
            Tm02 = float(blocks[9])
            Tp = float(blocks[10])
            Tz = float(blocks[11])
            DirTp = float(blocks[12])
            SprTp = float(blocks[13])
            MainDir = float(blocks[14])
            UnidirectivityIndex = float(blocks[15])
            MeanPressure = float(blocks[16])
            NearSurfaceCurrentSpeed = float(blocks[19])
            NearSurfaceCurrentDirection = float(blocks[20])

            # yield as dict
            yield {
                "timestamp": timestamp,
                "Hm0": Hm0,
                "H3": H3,
                "H10": H10,
                "Hmax": Hmax,
                "Tm02": Tm02,
                "Tp": Tp,
                "Tz": Tz,
                "DirTp": DirTp,
                "SprTp": SprTp,
                "MainDir": MainDir,
                "UnidirectivityIndex": UnidirectivityIndex,
                "MeanPressure": MeanPressure,
                "NearSurfaceCurrentSpeed": NearSurfaceCurrentSpeed,
                "NearSurfaceCurrentDirection": NearSurfaceCurrentDirection,
            }


def parse_awac_nmea(lines):

    # scan to find frequency grid
    freq = None

    # read spectral frequencies from PNORF
    for line in lines:
        if line.startswith("$PNORF"):
            blocks = line.split(",")
            fstart = float(blocks[5])
            fstep = float(blocks[6])
            fn = int(blocks[7])

            freq = fstart + np.linspace(start=0, stop=fn * fstep, num=fn, endpoint=True)

            break

    if freq is None:
        raise ValueError("Can not determine frequency grid, $PNORF field not found")

    # read spectral frequencies from PNORE
    ok = False
    for line in lines:
        if line.startswith("$PNORE"):
            blocks = line.split(",")
            fstart2 = float(blocks[4])
            fstep2 = float(blocks[5])

            # assert that the frequency grid is the same
            assert (
                fstart == fstart2
            ), "Frequency start does not match between $PNORF and $PNORE"
            assert (
                fstep == fstep2
            ), "Frequency step does not match between $PNORF and $PNORE"

            ok = True
            break
    if not ok:
        raise ValueError("Can not determine frequency grid, $PNORE field not found")

    A1 = None
    A2 = None
    B1 = None
    B2 = None
    S = None
    Hs = None
    timestamp = None

    for line in lines:
        line = line.split("*")[0]

        # new timestamp?
        TS = None
        if line.startswith("$PNORF"):
            TS = line[10:16] + line[17:23]
        elif line.startswith("$PNORE"):
            TS = line[7:13] + line[14:20]
        elif line.startswith("$PNORW"):
            blocks = line.split(",")
            TS = blocks[1] + blocks[2]

        if TS is not None:
            if TS != timestamp:
                if A1 is not None:
                    yield timestamp, freq, A1, A2, B1, B2, S, Hs

                timestamp = TS
                A1 = None
                A2 = None
                B1 = None
                B2 = None
                S = None
                Hs = None

        if line.startswith("$PNORF"):
            # remove the checksum
            blocks = line.split(",")
            if blocks[1] == "A1":
                if A1 is not None:
                    warnings.warn(
                        f"Duplicate A1 spectrum found for timestamp {timestamp}"
                    )
                A1 = np.array([float(x) for x in blocks[8:]])
            elif blocks[1] == "A2":
                if A2 is not None:
                    warnings.warn(
                        f"Duplicate A2 spectrum found for timestamp {timestamp}"
                    )
                A2 = np.array([float(x) for x in blocks[8:]])
            elif blocks[1] == "B1":
                if B1 is not None:
                    warnings.warn(
                        f"Duplicate B1 spectrum found for timestamp {timestamp}"
                    )
                B1 = np.array([float(x) for x in blocks[8:]])
            elif blocks[1] == "B2":
                if B2 is not None:
                    warnings.warn(
                        f"Duplicate B2 spectrum found for timestamp {timestamp}"
                    )
                B2 = np.array([float(x) for x in blocks[8:]])
        elif line.startswith("$PNORE"):
            blocks = line.split(",")
            n = int(blocks[6])
            S = np.array([float(x) for x in blocks[7:]])
            if len(S) != n:
                warnings.warn(
                    f"Number of frequencies in spectrum does not match number of frequencies expected {n} got {len(S)}"
                )
        elif line.startswith("$PNORW"):
            blocks = line.split(",")
            Hs = float(blocks[5])

    if A1 is not None:
        yield timestamp, freq, A1, A2, B1, B2, S, Hs


def read_awac_strings(
    lines,
    nDirs=90,
):
    """Parses AWAC NMEA data into directional spectra."""

    data = [R for R in parse_awac_nmea(lines)]

    # each line of data contains a timestamp, freq, A1, A2, B1, B2, S
    # the first and last line may contain Nones for A1, A2, B1, B2, S if the file was incomplete. In that case we remove them

    if any(x is None for x in data[0]):
        warnings.warn(
            f"First spectrum of AWAC data incomplete, removed spectrum with timestamp {data[0][0]}"
        )
        data = data[1:]
    if any(x is None for x in data[-1]):
        warnings.warn(
            f"Last spectrum of AWAC data incomplete, removed with timestamp {data[0][0]}"
        )
        data = data[:-1]

    # convert to directional spectra
    dDir = np.linspace(0, 2 * np.pi, num=nDirs, endpoint=False)
    dir_step = dDir[1] - dDir[0]

    efth = []
    times = []

    for timestamp, freq, aA1, aA2, aB1, aB2, S, Hs in data:
        # %(conversion from Cart to Compass)
        # A1temp = A1.copy()
        # B1temp = B1.copy()
        #
        # print('A1 = [' + ','.join([str(a) for a in aA1]) + ']')
        # print('A2 = [' + ','.join([str(a) for a in aA2]) + ']')
        # print('B1 = [' + ','.join([str(a) for a in aB1]) + ']')
        # print('B2 = [' + ','.join([str(a) for a in aB2]) + ']')
        # print('S = [' + ','.join([str(a) for a in S]) + ']')

        A1 = -aB1.copy()
        B1 = -aA1.copy()
        A2 = -aA2.copy()
        B2 = aB2.copy()

        # Direction From
        dr = np.pi
        A1new = np.cos(dr) * A1 + np.sin(dr) * B1
        B1new = -np.sin(dr) * A1 + np.cos(dr) * B1
        A2new = np.cos(2 * dr) * A2 + np.sin(2 * dr) * B2
        B2new = -np.sin(2 * dr) * A2 + np.cos(2 * dr) * B2

        # Now apply maximum entropy method to remove negative energy
        # Initialize the directional field
        DirField = np.zeros((len(freq), len(dDir)))

        # Apply maximum entropy method to remove negative energy
        for iFreq in range(len(freq)):

            c1 = A1new[iFreq] + 1j * B1new[iFreq]
            c2 = A2new[iFreq] + 1j * B2new[iFreq]

            fi1 = (c1 - c2 * np.conj(c1)) / (1 - np.abs(c1) * np.abs(c1))
            fi2 = c2 - c1 * fi1

            # print(f"c1 = {c1}")
            # print(f"c2 = {c2}")
            # print(f"fi1 = {fi1}")
            # print(f"fi2 = {fi2}")

            factors = []
            for dd in range(len(dDir)):

                dNewdir = np.pi / 2 - dDir[dd]
                numer = 1 - fi1 * np.conj(c1) - fi2 * np.conj(c2)
                denom = (
                    1 - fi1 * np.exp(-1j * dNewdir) - fi2 * np.exp(-2.0 * 1j * dNewdir)
                )

                dir_factor = np.real(numer / (np.abs(denom) * np.abs(denom)))

                scale1 = 1 / (2 * np.pi)
                scale2 = dir_step

                scale = scale1 * scale2

                factor = scale * dir_factor

                DirField[iFreq, dd] = factor * S[iFreq]
                #
                # print("======================")
                # print(f"dd = {dd + 1}")
                # print(f"dNewdir = {dNewdir}")
                # print(f"numer = {numer}")
                # print(f"denom = {denom}")
                # print(f"dir_factor = {dir_factor}")

                factors.append(factor)

            # all factors should add up to 1.0, scale accordingly
            # print(f"Sum of factors = {np.sum(factors)}")
            # if np.sum(factors) > 1.1:
            #     print('Warning: sum of factors is larger than 1.0')

            directional_scaling = 1.0 / np.sum(factors)
            DirField[iFreq] *= directional_scaling

        # Calculate m0 of the spectrum
        Sd = [np.sum(DirField[iFreq]) for iFreq in range(len(freq))]
        m0 = np.trapz(Sd, freq)
        Hs_spectrum = 4 * np.sqrt(m0)

        # scale the spectrum to the given Hs IF Hs_spectrum is not zero
        if Hs_spectrum == 0:
            if Hs > 0:
                warnings.warn(
                    f"Spectrum with timestamp {timestamp} has zero significant wave height, can not scale to Hs = {Hs}m"
                )
        else:
            DirField *= (Hs / Hs_spectrum) ** 2

        # in wavespecrta we work in:
        # m
        # Hz
        # degrees
        # m2/Hz so that is already ok
        dir_step_deg = np.degrees(dir_step)
        efth.append(DirField / dir_step_deg)

        ts = datetime.datetime.strptime(timestamp, "%m%d%y%H%M%S")
        assert ts.tzinfo is None
        times.append(ts)

    # from matplotlib import pyplot as plt
    # plt.plot(Hss, label = 'integrated directional spectra')

    dir_degrees = np.degrees(dDir)
    freq = data[0][1]

    ds = xr.DataArray(
        data=efth,
        coords={"time": times, "freq": freq, "dir": dir_degrees},
        dims=("time", "freq", "dir"),
        name="efth",
    ).to_dataset()

    # Set attributes
    set_spec_attributes(ds)

    # add site dimension
    ds["site"] = [0]

    return ds
