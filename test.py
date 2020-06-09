from wavespectra.input.ndbc import read_ndbc
    
df=read_ndbc([
    "./tests/sample_files/ndbc/41010.data_spec",
    "./tests/sample_files/ndbc/41010.swdir",
    "./tests/sample_files/ndbc/41010.swdir2",
    "./tests/sample_files/ndbc/41010.swr1",
    "./tests/sample_files/ndbc/41010.swr2",
])

df[0].spec.plot()