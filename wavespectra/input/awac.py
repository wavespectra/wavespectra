"""Reader for AWAC spectra from NMEA data

https://www.nortekgroup.com/

questions:
--------------
[1.] What is the timezone?

revision history:
-----------------
2024-07-23 : First version - Ruben de Bruin


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

"""
import warnings

import numpy as np

def parse_awac_nmea(lines):

    # scan to find frequency grid
    freq = None
    for line in lines:
        if line.startswith("$PNORE"):
            blocks = line.split(',')
            fstart = float(blocks[4])
            fstep = float(blocks[5])
            fn = int(blocks[6])

            freq = fstart + np.linspace(start=0, stop=fn*fstep, num=fn, endpoint=True)

            break

    if freq is None:
        raise ValueError("Can not determine frequency grid, $PNORE field not found")


    # hard-coded
    DirRes = 4

    dDir = np.arange(start = 0, stop = (2*np.pi - np.pi * DirRes / 180), step = np.pi * DirRes / 180)

    A1 = None
    A2 = None
    B1 = None
    B2 = None
    S = None
    timestamp = None

    for line in lines:

        # new timestamp?
        TS = None
        if line.startswith("$PNORF"):
            TS = line[10:16] + line[17:23]
        elif line.startswith("$PNORE"):
            TS = line[7:13] + line[14:20]


        if TS is not None:
            if TS != timestamp:
                if A1 is not None:
                    yield timestamp, freq, A1, A2, B1, B2, S

                timestamp = TS
                A1 = None
                A2 = None
                B1 = None
                B2 = None
                S = None

        if line.startswith("$PNORF"):
            blocks = line.split(',')
            if blocks[1] == 'A1':
                A1 = np.array([float(x) for x in blocks[7:-1]])
            elif blocks[1] == 'A2':
                A2 = np.array([float(x) for x in blocks[7:-1]])
            elif blocks[1] == 'B1':
                B1 = np.array([float(x) for x in blocks[7:-1]])
            elif blocks[1] == 'B2':
                B2 = np.array([float(x) for x in blocks[7:-1]])
        elif line.startswith("$PNORE"):
            blocks = line.split(',')
            S = np.array([float(x) for x in blocks[6:-1]])

    if A1 is not None:
        yield timestamp, freq, A1, A2, B1, B2, S


def read_awac(lines):
    data = [R for R in parse_awac_nmea(lines)]

    # each line of data contains a timestamp, freq, A1, A2, B1, B2, S
    # the first and last line may contain Nones for A1, A2, B1, B2, S if the file was incomplete. In that case we remove them

    if any(x is None for x in data[0]):
        warnings.warn(f"First spectrum of AWAC data incomplete, removed spectrum with timestamp {data[0][0]}")
        data = data[1:]
    if any(x is None for x in data[-1]):
        warnings.warn(f"Last spectrum of AWAC data incomplete, removed with timestamp {data[0][0]}")
        data = data[:-1]

    return data
