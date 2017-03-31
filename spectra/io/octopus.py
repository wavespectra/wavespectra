import numpy
import datetime

to_datetime=lambda t:datetime.datetime.fromtimestamp(t.astype('int')/1e9)

def read_octopus(filename):
    raise NotImplementedError('No Octopus read function defined')

def to_octopus(self,filename,site_id='spec'):
    if len(self.lon)>1 or len(self.lat)>1:
        raise NotImplementedError('No Octopus export defined for multiple sites')
    f=open(filename,'w')
    f.write('Forecast valid for %s\n' % (to_datetime(self.time[0]).strftime('%d-%b-%Y %H:%M:%S')))
    fmt=','.join(len(self.freq)*['%6.5f'])+','
    dt=(self.time[1].astype('int')-self.time[0].astype('int'))/3.6e12 if len(self.time)>1 else 0
    swell=self.split(fmin=0,fmax=0.125)
    sea=self.split(fmin=0.125,fmax=1.)
    for i,t in enumerate(self.time):
        s=self.efth
        if i==0:
            f.write('nfreqs,%d\nndir,%d\nnrecs,%d\nLatitude, %7.4f\nLongitude, %7.4f\nDepth,%f\n\n' %
                    (len(self.freq),len(self.dir),len(self.time),self.lat[0],self.lon[0],self.dpt[0]))
            sdirs=numpy.mod(self.dir.values,360.)
            idirs=numpy.argsort(sdirs)
            ddir=abs(sdirs[1]-sdirs[0])
            dfreq=numpy.hstack((0,self.freq[1:].values-self.freq[:-1].values,0))
            dfreq=0.5*(dfreq[:-1]+dfreq[1:])
        lp=site_id+to_datetime(t).strftime('_%Y%m%d_%Hz')
        wd=self.wnddir[i]
        ws=self.wnd[i]
        s_sea=sea[i].spec
        s_sw=swell[i].spec
        s=self.isel(time=i,lon=0,lat=0).efth.spec
        f.write('CCYYMM,DDHHmm,LPoint,WD,WS,ETot,TZ,VMD,ETotSe,TZSe,VMDSe,ETotSw,TZSw,VMDSw,Mo1,Mo2,HSig,DomDr,AngSpr,Tau\n')
        f.write('%s,\'%s,%s,%d,%.2f,%.4f,%.2f,%.1f,%.4f,%.2f,%.1f,%.4f,%.2f,%.1f,%.5f,%.5f,%.4f,%d,%d,%d\n' %
                (to_datetime(t).strftime('%Y%m'),to_datetime(t).strftime('%d%H%M'),lp,wd,ws,
                 (0.25*s.hs())**2,s.tm01(),s.dm(),
                 (0.25*s_sea.hs())**2,s_sea.tm01(),s_sea.dm(),
                 (0.25*s_sw.hs())**2,s_sw.tm01(),s_sw.dm(),
                 s.momf(1).sum(),s.momf(2).sum(),s.hs(),s.dpm(),s.dspr(),i*dt))
        f.write(('freq,'+fmt+'anspec\n') % tuple(self.freq))
        for idir in idirs:
            f.write('%d,' % (numpy.round(sdirs[idir])))
            row=ddir*dfreq*s._obj.isel(dir=idir)
            f.write(fmt %tuple(row.values))
            f.write('%6.5f,\n'% row.sum())
        f.write(('fSpec,'+fmt+'\n') % tuple(dfreq*s.momd(0)[0].values))
        f.write(('den,'+fmt+'\n\n') % tuple(s.momd(0)[0].values))