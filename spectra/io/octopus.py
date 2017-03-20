
def read_octopus(filename):
    raise NotImplementedError('No Octopus read function defined')

def to_octopus(filename):
    sf=SwanSpecFile(swanfile)
    try:
        st=SwanTableFile(swanfile.replace('.spec','.tab'))
    except:
        st=None
    f=open(outfile,'w')
    specs=[s for s in sf.readall()]
    f.write('Forecast valid for %s\n' % (sf.times[0].strftime('%d-%b-%Y %H:%M:%S')))
    
    fmt=','.join(len(s.freqs)*['%6.5f'])+','
    dt=(sf.times[1]-sf.times[0]).seconds/3600 if len(sf.times)>1 else 0
    for i,t in enumerate(sf.times):
        s=specs[i]
        ttime,uwnd,vwnd,dep=st.read() if st else (0,-999.,-999.,-999)
        if i==0:
            f.write('nfreqs,%d\nndir,%d\nnrecs,%d\nLatitude, %7.4f\nLongitude, %7.4f\nDepth,%f\n\n' % (len(s.freqs),len(s.dirs),len(sf.times),sf.locations[-1].y,sf.locations[-1].x,dep))
            sdirs=numpy.mod(s.dirs,360.)
            idirs=numpy.argsort(sdirs)
            ddir=abs(sdirs[1]-sdirs[0])
            dfreq=numpy.hstack((0,s.freqs[1:]-s.freqs[:-1],0))
            dfreq=0.5*(dfreq[:-1]+dfreq[1:])
        lp=id+t.strftime('_%Y%m%d_%Hz')
        wd=(270-180*numpy.arctan2(vwnd,uwnd)/numpy.pi) % 360.
        ws=1.94*(uwnd**2+vwnd**2)**0.5 if uwnd>-100 else -999.
        s_sw=s.split([0,0.125])
        s_sea=s.split([0.125,1.])
        f.write('CCYYMM,DDHHmm,LPoint,WD,WS,ETot,TZ,VMD,ETotSe,TZSe,VMDSe,ETotSw,TZSw,VMDSw,Mo1,Mo2,HSig,DomDr,AngSpr,Tau\n')
        f.write('%s,\'%s,%s,%d,%.2f,%.4f,%.2f,%.1f,%.4f,%.2f,%.1f,%.4f,%.2f,%.1f,%.5f,%.5f,%.4f,%d,%d,%d\n' %
                (t.strftime('%Y%m'),t.strftime('%d%H%M'),lp,wd,ws,
                 (0.25*s.hs())**2,s.tm01(),s.dm(),
                 (0.25*s_sea.hs())**2,s_sea.tm01(),s_sea.dm(),
                 (0.25*s_sw.hs())**2,s_sw.tm01(),s_sw.dm(),
                 s.momf(1).momd(0)[0].S[0,0],s.momf(2).momd(0)[0].S[0,0],s.hs(),s.dpm(),s.dspr(),i*dt))
        f.write(('freq,'+fmt+'anspec\n') % tuple(s.freqs))
        for idir in idirs:
            f.write('%d,' % (numpy.round(sdirs[idir])))
            row=ddir*dfreq*s.S[:,idir]
            f.write((fmt+'%6.5f,\n') % (tuple(row)+(row.sum(),)))
        f.write(('fSpec,'+fmt+'\n') % tuple(dfreq*s.momd(0)[0].S[:,0]))
        f.write(('den,'+fmt+'\n\n') % tuple(s.momd(0)[0].S[:,0]))