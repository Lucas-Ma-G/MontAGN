# -*- coding: utf-8 -*-
#from __future__ import division
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import csv
import os

import montagn_utils as mu
import montagn_classes as mc
import montagn_polar as mpol
#from montagn_utils import *
#from montagn_utils_Lu import *
from astropy.io import fits
import mgui_colorbar as g_cb 
try:
    import patiencebar as g_pgb
except:
    #print 'For a better use you can install patiencebar (work with pip)'
    import mgui_progbar as g_pgb
    
from shutil import rmtree
from matplotlib.colors import LogNorm
#import png as png

### Plot configuration
#rc_params = {
#   'axes.labelsize': 'medium',
#   'font.size': 20,
#   'legend.fontsize': 'medium',
#   'text.usetex': False,
#   'image.aspect': 'equal',
#   'figure.autolayout': True,
#   'savefig.dpi': 300
#   }
#matplotlib.rcParams.update(rc_params)
# matplotlib.rcdefaults() # restore default settings

def write_photon(f, ph,x,y,z,i,Dt,Dt_tot,cst): #writes a photon into a file object (the file has to be opened before executing and closed after executing)
    """file, photon, exit coordinates"""
    en=ph.E #/Dt
    s = "%d\t"%i
    s += "%f\t"%ph.theta
    s += "%f\t"%ph.phi
    s += "%f\t%f\t"%(ph.Ss[1][0],ph.Ss[2][0])
    s += "%f\t"%ph.Ss[3][0]
    s += "%f\t"%ph.phiqu
    s += "%f\t%f\t%f\t"%(x/cst.pc,y/cst.pc,z/cst.pc)
    s += "%d\t"%ph.interact
    s += "%d\t"%ph.reem
    s += "%e\t"%ph.wvl
    s += "%s\t"%ph.label
    s += "%f\t"%en
    s += "%f\t"%Dt
    s += "%f\n"%Dt_tot
    f.write(s)


def write_photon_fits(f, ph,x,y,z,i,Dt,Dt_tot,cst): #writes a photon into a file object (the file has to be opened before executing and closed after executing)
    """file, photon, exit coordinates"""
    en=ph.E #/Dt
    s = "%d\t"%i
    s += "%f\t"%ph.theta
    s += "%f\t"%ph.phi
    s += "%f\t%f\t"%(ph.Ss[1][0],ph.Ss[2][0])
    s += "%f\t"%ph.Ss[3][0]
    s += "%f\t"%ph.phiqu
    s += "%f\t%f\t%f\t"%(x/cst.pc,y/cst.pc,z/cst.pc)
    s += "%d\t"%ph.interact
    s += "%d\t"%ph.reem
    s += "%e\t"%ph.wvl
    s += "%s\t"%ph.label
    s += "%f\t"%en
    s += "%f\t"%Dt
    s += "%f\n"%Dt_tot
    f.write(s)


def write_photon_hdf5(f, ph,x,y,z,i,Dt,Dt_tot,cst): #writes a photon into a file object (the file has to be opened before executing and closed after executing)
    """file, photon, exit coordinates"""
    en=ph.E #/Dt
    s = "%d\t"%i
    s += "%f\t"%ph.theta
    s += "%f\t"%ph.phi
    s += "%f\t%f\t"%(ph.Ss[1][0],ph.Ss[2][0])
    s += "%f\t"%ph.Ss[3][0]
    s += "%f\t"%ph.phiqu
    s += "%f\t%f\t%f\t"%(x/cst.pc,y/cst.pc,z/cst.pc)
    s += "%d\t"%ph.interact
    s += "%d\t"%ph.reem
    s += "%e\t"%ph.wvl
    s += "%s\t"%ph.label
    s += "%f\t"%en
    s += "%f\t"%Dt
    s += "%f\n"%Dt_tot
    s.write(f,append=True)


def display_SED(filerootname,**kwargs):
    #display_SED(filename,diam_angle=[],phi=[],bins=100, minwvl=1e-6, maxwvl = 1000.e-6, logx=0,logy=1,color=[],label=[],interact=-1,reem=-1,ret=0):
    # display the total output SED 
    """root of all files with stored photon results,
    viewing polar angle,
    viewing azimuthal angle,
    angular diameter
    [source label (string or set of strings)]
    [photons that interacted (0 or 1)]
    [photons that were reemitted (0 or 1)]"""
    
    pargs=plot_args(kwargs)
    pargs['spectra']=1

    #print pargs
    #print pargs['minwvl']

    #if isinstance(filerootname,str)==True:
    #    filerootname=[filerootname]
    
    if(pargs['polar']==1):
        binning,SED,SEDI,SEDQ,SEDU,SEDV=extract_from_file(filerootname,pargs)
        SEDIp,SEDP,SEDtheta,SEDpolV=polar_reduction(SEDI,SEDQ,SEDU,V=SEDV)
    else:
        binning,SED,SEDI=extract_from_file(filerootname,pargs)
            
    #plt.figure()
    mpol.plot_1D((binning)*1e9,SED,logy=pargs['logy'],title=pargs['title'],xlabel='Wavelength (nm)',ylabel='Intensity - number of photons packets',logx=pargs['logx'])
    mpol.plot_1D((binning)*1e9,SEDI,logy=pargs['logy'],title=pargs['title']+' I',xlabel='Wavelength (nm)',ylabel='Intensity (J)',logx=pargs['logx'])
    if(pargs['polar']==1):
       mpol.plot_1D((binning)*1e9,SEDQ,logy=0,title=pargs['title']+' Q',xlabel='Wavelength (nm)',ylabel='Intensity Q (J)',logx=pargs['logx'])
       mpol.plot_1D((binning)*1e9,SEDU,logy=0,title=pargs['title']+' U',xlabel='Wavelength (nm)',ylabel='Intensity U (J)',logx=pargs['logx'])
       mpol.plot_1D((binning)*1e9,SEDV,logy=0,title=pargs['title']+' V',xlabel='Wavelength (nm)',ylabel='Intensity V (J)',logx=pargs['logx'])
       mpol.plot_1D((binning)*1e9,SEDIp,logy=pargs['logy'],title=pargs['title']+' Ip',xlabel='Wavelength (nm)',ylabel='Polarised intensity (J)',logx=pargs['logx'])
       mpol.plot_1D((binning)*1e9,SEDP*100,logy=0,title=pargs['title']+' P',xlabel='Wavelength (nm)',ylabel='Linear Polarisation degree (%)',logx=pargs['logx'])
       mpol.plot_1D((binning)*1e9,SEDtheta,logy=0,title=pargs['title']+' theta',xlabel='Wavelength (nm)',ylabel='Linear Polarisation angle (degrees)',logx=pargs['logx'])
       mpol.plot_1D((binning)*1e9,SEDpolV*100,logy=0,title=pargs['title']+' V',xlabel='Wavelength (nm)',ylabel='Circular Polarisation degree (%)',logx=pargs['logx'])
    #if(pargs['logx']==1):
    #    plt.xscale('log')
    #if(pargs['logy']==1):      
    #    plt.yscale('log')
    #plt.title('SED total')
    #plt.xlabel('Wavelength (um)')
    #plt.ylabel('Intensity - number of photons packets')
    if(pargs['label']!=[]):
        plt.legend(pargs['label'],loc=0)
    plt.show()
    if(pargs['rec']==1):
        if(pargs['polar']==1):
            write_spectra(binning,[SED,SEDI,SEDQ,SEDU,SEDV,SEDIp,SEDP,SEDtheta,SEDpolV],['Number of packets','Intensity','Q','U','V','Ip','P','theta','CircP'],filerootname,path=pargs['outdir'])
        else:
            write_spectra(binning,[SED,SEDI],['Number of packets','Intensity'],filerootname,path=pargs['outdir'])
    if(pargs['ret']==1):
        if(pargs['polar']==1):
            return binning,[SED,SEDI,SEDQ,SEDU,SEDV,SEDIp,SEDP,SEDtheta,SEDpolV]
        else:
            return binning,[SED,SEDI]

def polar_reduction(I,Q,U,V=[]):
    Ip=np.sqrt(Q*Q+U*U)
    linpol=Ip/I 
    theta=-0.5*np.arctan2(U,Q)*180./np.pi
    Pq=Q/I
    Pu=U/I        
    if(V==[]):
        return Ip,linpol,theta
    else:
        circpol=V/I
        return Ip,linpol,theta,circpol

def plot_args(kwargs):
    """
    function of kwargs management
    get the parameters passed through the kwargs and give as an output the required args as a list,
    with default if not specified
    """

    #Theta selection keyword management
    l=1
    if('thetaobs' in kwargs.keys() and 'dtheta' in kwargs.keys()):
       l+=1 
    if('thetamin' in kwargs.keys() and 'thetamax' in kwargs.keys()):
        l*=-1
    if(l==-2):
        print("Warning, redundant keywords used to constrain theta")
        print("(among thetaobs, dtheta, thetamin and thetamax)")
        print("Using thetamin and thetamax")
    elif(l==2):
        kwargs['thetamin']=kwargs['thetaobs']-kwargs['dtheta']
        kwargs['thetamax']=kwargs['thetaobs']+kwargs['dtheta']

    #Args translation
    pargs={
        #IO parameters
        'path':kwargs.get('path',''),
        'outdir':kwargs.get('outdir',''),
        'outname':kwargs.get('outname',''),
        'outsuffixe':kwargs.get('outsuffixe',''),
        'file_ext':kwargs.get('file_ext','_phot.dat'),
        'ret':kwargs.get('ret',0),
        'rec':kwargs.get('rec',0),
        'saveformat':kwargs.get('saveformat','pdf'),
        
        #Photon selection parameters
        'minwvl':kwargs.get('minwvl',0.5e-6),
        'maxwvl':kwargs.get('maxwvl',1.e-3),
        'diam_angle':kwargs.get('diam_angle',[]),
        'thetamin':kwargs.get('thetamin',0),
        'thetamax':kwargs.get('thetamax',180),
        #'thetaobs':kwargs.get('thetaobs',-100),
        #'dtheta':kwargs.get('dtheta',-1),
        'phi':kwargs.get('phi',-1),
        'interact':kwargs.get('interact',-1),
        'reem':kwargs.get('reem',-1),
        'nscatt':kwargs.get('nscatt',-1),
        'nscattmax':kwargs.get('nscattmax',5),
        
        'polar':kwargs.get('polar',0),
        
        #display parameters
        'image':kwargs.get('image',0),
        'spectra':kwargs.get('spectra',0),
        'plot_per_file':kwargs.get('plot_per_file',0),
        'bins':kwargs.get('bins',100),
        'logx':kwargs.get('logx',0),
        'logy':kwargs.get('logy',1),
        'title':kwargs.get('title',''),
        'color':kwargs.get('color',[]),
        'obj':kwargs.get('obj','AGN'),
        'resimage':kwargs.get('resimage',26),
        'resunit':kwargs.get('resunit','pixel'),
        'label':kwargs.get('label',[]),
        'cmap':kwargs.get('cmap',[]),
        'rebin':kwargs.get('rebin',0),
        'rebinsize':kwargs.get('rebinsize',3),
        'enpaq':kwargs.get('enpaq',3.86e26),
        'coupe':kwargs.get('coupe',0),
        'extract':kwargs.get('extract',[]),
        'vectormode':kwargs.get('vectormode',0),
        'sym':kwargs.get('sym',0),
        'gif':kwargs.get('gif',1)
    }
    for i in kwargs:
        if i not in pargs.keys():
            print("Warning, keyword not recognised : ",i)
    return pargs

class Pparams:
    def __init__(self):
        """
        Gives the output parameters stored in the photon files with their corresponding row
        """
        self.n = 0
        self.theta = 1
        self.phi = 2
        self.Q = 3
        self.U = 4
        self.V = 5
        self.phiqu = 6
        self.x = 7
        self.y = 8
        self.z = 9
        self.interact = 10
        self.reem = 11
        self.wvl = 12
        self.label = 13
        self.en = 14
        self.Dt = 15
        self.Dt_tot = 16

def extract_from_file(filerootname,pargs):
    """
    Function to read the files (with the root name 'filerootname') and extract the wanted quantities ('quants')
    """
    photonparams=Pparams()
    nfig=[]
    try:
        filelist=np.sort(os.listdir(pargs['path']))
        #print 'found files:',filelist
    except:
        print('No files found')
        return 0
    #table=[]
    #t=fits.getval(directory+filename,'EXPTIME')
    #for j in range(len(filerootname)):
    if(pargs['spectra']==1):
        binning = np.logspace(np.log10(pargs['minwvl']),np.log10(pargs['maxwvl']),pargs['bins']+1)
        SED = 0*binning
        SEDI = 0*binning
        if(pargs['polar']==1):
            SEDQ = 0*binning
            SEDU = 0*binning
            SEDV = 0*binning
        if(pargs['plot_per_file']==1):
            try:
                nfig=np.max(plt.get_fignums())+1
            except:
                nfig=1
    #if(pargs['image']==1):
    nfile=0
    for filename in filelist:
        if(filerootname in filename):
            if(float(filename[-3-len(pargs['file_ext']):-len(pargs['file_ext'])])>=pargs['thetamin'] and float(filename[-3-len(pargs['file_ext']):-len(pargs['file_ext'])])<=pargs['thetamax']):
                print('Reading file:',filename)
                f = file(pargs['path']+filename,'r')
                r = csv.reader(f, delimiter='\t')
                if(pargs['polar']==0 and pargs['spectra']==1): #classical intensity spectra
                    tmp = []
                    tmpI = []
                    for row in r: #read the file line per line
                        tmp.append(float(row[photonparams.wvl]))
                        tmpI.append(float(row[photonparams.en]))
                    tmp1,tmp2=spectral_reduction(tmp,tmpI,binning,pargs,nfig=nfig,filename=filename,nfile=nfile)
                    SED+=tmp1
                    SEDI+=tmp2
                if(pargs['polar']==1 and pargs['spectra']==1): #polarised spectra
                    tmp = []
                    tmpI = []
                    tmpQ = []
                    tmpU = []
                    tmpV = []
                    for row in r: #read the file line per line
                        tmp.append(float(row[photonparams.wvl]))
                        tmpI.append(float(row[photonparams.en]))
                        tmpQ.append(float(row[photonparams.Q]))
                        tmpU.append(float(row[photonparams.U]))
                        tmpV.append(float(row[photonparams.V]))
                    tmp1,tmp2,tmp3,tmp4,tmp5=spectral_polar_reduction(tmp,tmpI,tmpQ,tmpU,tmpV,binning,pargs,nfig=nfig,filename=filename,nfile=nfile)
                    SED+=tmp1
                    SEDI+=tmp2
                    SEDQ+=tmp3
                    SEDU+=tmp4
                    SEDV+=tmp5
                nfile+=1

    if(pargs['spectra']==1 and pargs['polar']==0):
        return binning,SED,SEDI
    if(pargs['spectra']==1 and pargs['polar']==1):
        return binning,SED,SEDI,SEDQ,SEDU,SEDV

def spectral_reduction(tmp,tmpI,binning,pargs,nfig=[],filename='',nfile=0):
    """
    function to reduce spectra according to different requirement within the extract_from_file routine
    """
    SEDtmp = 0*binning
    SEDtmpI = 0*binning
    j=0
    for i in tmp: #attribute the photons to the binned spectral bands
        k = 0
        #print i,binning[k]
        while (i>binning[k]) * (k < pargs['bins']):
            k +=1
        SEDtmp[k]+=1
        SEDtmpI[k]+=tmpI[j]
        j+=1
    if(pargs['plot_per_file']==1):
        if(pargs['color']==[]):
            #plt.plot((binning)*1e9,SEDtmp)
            mpol.plot_1D((binning)*1e9,SEDtmp,logy=pargs['logy'],title='SED per file',xlabel='Wavelength (nm)',ylabel='Intensity - number of photons packets',label=filename,logx=pargs['logx'],n=nfig)
            mpol.plot_1D((binning)*1e9,SEDtmpI,logy=pargs['logy'],title='SED per file',xlabel='Wavelength (nm)',ylabel='Intensity (J)',label=filename,logx=pargs['logx'],n=nfig+1)
        else:
            #plt.plot((binning)*1e9,SEDtmp,color=pargs[color][j])
            mpol.plot_1D((binning)*1e9,SEDtmp,logy=pargs['logy'],title='SED per file',xlabel='Wavelength (nm)',ylabel='Intensity - number of photons packets',label=filename,logx=pargs['logx'],n=nfig,color=pargs['color'][nfile])
            mpol.plot_1D((binning)*1e9,SEDtmpI,logy=pargs['logy'],title='SED per file',xlabel='Wavelength (nm)',ylabel='Intensity (J)',label=filename,logx=pargs['logx'],n=nfig,color=pargs['color'][nfile])
        #SED+=SEDtmp
    return [SEDtmp,SEDtmpI]


def spectral_polar_reduction(tmp,tmpI,tmpQ,tmpU,tmpV,binning,pargs,nfig=[],filename='',nfile=0):
    """
    function to reduce spectra according to different requirement within the extract_from_file routine,
    polarised version
    """
    SEDtmp = 0*binning
    SEDtmpI = 0*binning
    SEDtmpQ = 0*binning
    SEDtmpU = 0*binning
    SEDtmpV = 0*binning
    j=0
    for i in tmp: #attribute the photons to the binned spectral bands
        k = 0
        while (i>binning[k]) * (k < pargs['bins']):
            k +=1
        SEDtmp[k]+=1
        SEDtmpI[k]+=tmpI[j]
        SEDtmpQ[k]+=tmpQ[j]*tmpI[j]
        SEDtmpU[k]+=tmpU[j]*tmpI[j]
        SEDtmpV[k]+=tmpV[j]*tmpI[j]
        j+=1
    if(pargs['plot_per_file']==1):
        if(pargs['color']==[]):
            mpol.plot_1D((binning)*1e9,SEDtmp,logy=pargs['logy'],title='SED per file',xlabel='Wavelength (nm)',ylabel='Intensity - number of photons packets',label=filename,logx=pargs['logx'],n=nfig)
            mpol.plot_1D((binning)*1e9,SEDtmpI,logy=pargs['logy'],title='SED per file',xlabel='Wavelength (nm)',ylabel='Intensity (J)',label=filename,logx=pargs['logx'],n=nfig+1)
            mpol.plot_1D((binning)*1e9,SEDtmpQ,logy=0,title='SED QUV per file',xlabel='Wavelength (nm)',ylabel='Intensity (J)',label=filename+' Q',logx=pargs['logx'],n=nfig+2)
            mpol.plot_1D((binning)*1e9,SEDtmpU,logy=0,title='SED QUV per file',xlabel='Wavelength (nm)',ylabel='Intensity (J)',label=filename+' U',logx=pargs['logx'],n=nfig+2)
            mpol.plot_1D((binning)*1e9,SEDtmpV,logy=0,title='SED QUV per file',xlabel='Wavelength (nm)',ylabel='Intensity (J)',label=filename+' V',logx=pargs['logx'],n=nfig+2)
        else:
            mpol.plot_1D((binning)*1e9,SEDtmp,logy=pargs['logy'],title='SED per file',xlabel='Wavelength (nm)',ylabel='Intensity - number of photons packets',label=filename,logx=pargs['logx'],n=nfig,color=pargs['color'][nfile])
            mpol.plot_1D((binning)*1e9,SEDtmpI,logy=pargs['logy'],title='SED per file',xlabel='Wavelength (nm)',ylabel='Intensity (J)',label=filename,logx=pargs['logx'],n=nfig+1,color=pargs['color'][nfile])
            mpol.plot_1D((binning)*1e9,SEDtmpQ,logy=0,title='SED QUV per file',xlabel='Wavelength (nm)',ylabel='Intensity (J)',label=filename+' Q',logx=pargs['logx'],n=nfig+2,color=pargs['color'][nfile])
            mpol.plot_1D((binning)*1e9,SEDtmpU,logy=0,title='SED QUV per file',xlabel='Wavelength (nm)',ylabel='Intensity (J)',label=filename+' U',logx=pargs['logx'],n=nfig+2,color=pargs['color'][nfile])
            mpol.plot_1D((binning)*1e9,SEDtmpV,logy=0,title='SED QUV per file',xlabel='Wavelength (nm)',ylabel='Intensity (J)',label=filename+' V',logx=pargs['logx'],n=nfig+2,color=pargs['color'][nfile])
    return [SEDtmp,SEDtmpI,SEDtmpQ,SEDtmpU,SEDtmpV]

def write_spectra(bins,SEDs,SEDsnames,filename,path=''):
    '''
    Function of properly recording the generated spectra on fits format
    Based on a routine written by S. Borgniet
    '''
    ###########################################################
    #### Writing Observed CCF in .fits file
    #def write_ccf_obs(path, ST, spectro, target, wav_range, method, sel_lines, mask_file, info_array, ccf_array, wide_rvgrid=False):
    
    #HEADER Keywords
    hdr = fits.Header()
    hdr['INSTR'] = 'MontAGN'
    #hdr['ST'], hdr['SPECTRO'] = ST, spectro
    #hdr['OBJECT'], hdr['HRS_ID'], hdr['CAT_ID'] = transf_name(target), target, search_id(path, target)[0]
    #hdr['GDR2_ID'] = search_id(path, target)[1]
    #hdr['RANGE'], hdr['METHOD'], hdr['LINES'] = wav_range, method, sel_lines
    #hdr['RVVAL1'], hdr['RVDELT1'], hdr['N_RV'] = rv_first, rv_step, N_rv
    #hdr.comments['RVVAL1'], hdr.comments['RVDELT1'] = 'km/s', 'km/s'
    #hdr.comments['N_RV'] = 'vrad = np.arange(N_RV)*RVDELT1 + RVVAL1 (km/s)'
    ext0 = fits.PrimaryHDU(header=hdr)
    ext_i = [ext0]
    #
    ###
    #Writing spectra
    col = fits.ColDefs([fits.Column(name='Wavelength', format='D', array=np.array(bins))])
    ext_i.append(fits.BinTableHDU.from_columns(col, name='Wavelentgh'))
    for i in range(len(SEDs)):
        #print np.shape(SEDs[i])
        #print type(SEDs[i])
        #print SEDs[i]
        col = fits.ColDefs([fits.Column(name=SEDsnames[i], format='D', array=np.array(SEDs[i]))])
        ext_i.append(fits.BinTableHDU.from_columns(col, name=SEDsnames[i]))
    #save_spectra = fits.HDUList([ext0, extp, ext_a, ext_w, ext_m, ext_d]) 
    save_spectra = fits.HDUList(ext_i)                       

    #Writing in file
    if(filename[-5:]!='.fits'):
        filename+='.fits'
    save_spectra.writeto(path+filename,overwrite=True)

def read_spectra(filename,path=''):
    f=fits.open(path+filename)
    SED=[]
    binning=[]
    for i in range(len(f)-1):
        tmp=np.array(f[i+1].data)
        #print tmp
        tmp2=[]
        for j in range(len(tmp)):
            tmp2.append(tmp[j][0])
        if(i==0):
            binning=np.copy(tmp2)
        else:
            SED.append(np.array(tmp2))
    return binning,SED

def display_SED_V2(filename,diam_angle=[],phi=[],bins=100,label='',interact=-1,reem=-1,ret=0):
    """file with stored photon results,
    viewing polar angle,
    viewing azimuthal angle,
    angular diameter
    [source label (string or set of strings)]
    [photons that interacted (0 or 1)]
    [photons that were reemitted (0 or 1)]"""
    f = file(filename,'r')
    r = csv.reader(f, delimiter='\t')
    t = []
    for row in r: #read the file line per line
        #if abs(row[1]-theta)<u and abs(row[2]-phi)<u \
        if (row[13] in label or label == '') \
        and (row[10] == interact or interact == -1) \
        and (row[11] == reem or reem == -1): #check if the photon is in the line of sight and is of the requested type
            t.append(float(row[12]))
    #print t
    maxwvl = max(t)
    minwvl = min(t)
    bin_space = (maxwvl-minwvl)/bins
    #binning = np.arange(minwvl,maxwvl,bin_space)
    binning = np.linspace(minwvl,maxwvl,bins+1)
    SED = 0*binning
    for i in t: #attribute the photons to the binned spectral bands
        k = 0
        #print i,binning[k]
        while i>binning[k]:
            k +=1
        SED[k]+=1
    #plt.figure()
    #plt.plot(binning+bin_space/2,SED,'ro')
    mpol.plot_1D((binning+bin_space/2)*1e6,SED,symbol='ro',title='SED',xlabel='Wavelength (um)',ylabel='Intensity - number of photons packets')
    mpol.plot_1D((binning+bin_space/2)*1e6,SED,logy=1,title='SED',xlabel='Wavelength (um)',ylabel='Intensity - number of photons packets')
    if(ret==1):
       return binning,SED


def display_SED_range(filename,diam_angle=[],phi=[],bins=100, minwvl=1e-6, maxwvl = 1000.e-6,label='',interact=-1,reem=-1,ret=0,logx=0):
    """file with stored photon results,
    viewing polar angle,
    viewing azimuthal angle,
    angular diameter
    minwvl and maxwvl 
    [source label (string or set of strings)]
    [photons that interacted (0 or 1)]
    [photons that were reemitted (0 or 1)]"""
    f = file(filename,'r')
    r = csv.reader(f, delimiter='\t')
    t = []
    for row in r: #read the file line per line
        #if abs(row[1]-theta)<u and abs(row[2]-phi)<u \
        #if (row[13] in label or label == ''): #\
        #and (row[10] == interact or interact == -1) \
        #and (row[11] == reem or reem == -1): #check if the photon is in the line of sight and is of the requested type
            t.append(float(row[12]))
    #print t
    #maxwvl = max(t)
    #minwvl = min(t)
    #bin_space = (maxwvl-minwvl)/bins
    #binning = np.arange(minwvl,maxwvl,bin_space)
    binning = np.logspace(np.log10(minwvl),np.log10(maxwvl),bins+1)
    SED = 0*binning
    for i in t: #attribute the photons to the binned spectral bands
        k = 0
        #print i,binning[k]
        while (i>binning[k]) * (k < bins):
            k +=1
        SED[k]+=1
    #plt.figure()
    #plt.plot(binning+bin_space/2,SED,'ro')
    #mpol.plot_1D((binning+bin_space/2)*1e6,SED,symbol='ro',title='SED',xlabel='Wavelength (um)',ylabel='Intensity - number of photons packets')
    mpol.plot_1D((binning)*1e6,SED,logy=1,title='SED',xlabel='Wavelength (um)',ylabel='Intensity - number of photons packets',logx=logx)
    if(ret==1):
       return binning,SED
       

def plot_spectre(spectre, wvl=[],bins=100, minwvl=1e-6, maxwvl = 1000.e-6, logx=0,logy=1):
    t=spectre
    if(wvl==[]):
        binning = np.logspace(np.log10(minwvl),np.log10(maxwvl),bins+1)
    else:
        binning = wvl
    SED = 0*binning
    for i in t: #attribute the photons to the binned spectral bands
        k = 0
        #print i,binning[k]
        while (i>binning[k]) * (k < bins):
            k +=1
        SED[k]+=1

    #dlambda=np.zeros(len(binning))
    #dlambda[1:-1]=(binning[2:]-binning[:-2])*0.5
    #dlambda[0]=binning[1]-binning[0]
    #dlambda[-1]=binning[-1]-binning[-2]
    #SEDl=SED/dlambda
    #SEDldl=SEDl*binning
    mpol.plot_1D((binning)*1e6,SED,logx=logx,logy=logy,title='SED total',xlabel='Wavelength (um)',ylabel='Intensity - number of photons packets')
    #mpol.plot_1D((binning)*1e6,SEDl,logx=logx,logy=logy,title='SED lambda total',xlabel='Wavelength (um)',ylabel='Intensity - number of photons packets')
    #mpol.plot_1D((binning)*1e6,SEDldl,logx=logx,logy=logy,title='SED integ total',xlabel='Wavelength (um)',ylabel='Intensity - number of photons packets')


def display_SED_range_tot(filename,diam_angle=[],phi=[],bins=100, minwvl=1e-6, maxwvl = 1000.e-6, logx=0,logy=1, label='',interact=-1,reem=-1,ret=0):
    # display the total output SED 
    """root of all files with stored photon results,
    viewing polar angle,
    viewing azimuthal angle,
    angular diameter
    [source label (string or set of strings)]
    [photons that interacted (0 or 1)]
    [photons that were reemitted (0 or 1)]"""
    t = []
    for l in range(10) :  # from 0 to 90°
        f = file(filename+'0'+str(l)+'0_phot.dat','r')
        r = csv.reader(f, delimiter='\t')
        for row in r: #read the file line per line
            if (row[13] in label or label == ''): 
                t.append(float(row[12]))
    for l in range(9) :  # from 100 to 180°
        f = file(filename+'1'+str(l)+'0_phot.dat','r')
        r = csv.reader(f, delimiter='\t')
        for row in r: #read the file line per line
            if (row[13] in label or label == ''): 
                t.append(float(row[12]))
    binning = np.logspace(np.log10(minwvl),np.log10(maxwvl),bins+1)
    SED = 0*binning
    for i in t: #attribute the photons to the binned spectral bands
        k = 0
        #print i,binning[k]
        while (i>binning[k]) * (k < bins):
            k +=1
        SED[k]+=1
    #dlambda=np.zeros(len(binning))
    #dlambda[1:-1]=(binning[2:]-binning[:-2])*0.5
    #dlambda[0]=binning[1]-binning[0]
    #dlambda[-1]=binning[-1]-binning[-2]
    #SEDl=SED/dlambda
    mpol.plot_1D((binning)*1e6,SED,logx=logx,logy=logy,title='SED total',xlabel='Wavelength (um)',ylabel='Intensity - number of photons packets')
    #mpol.plot_1D((binning)*1e6,SEDl,logx=logx,logy=logy,title='SED total',xlabel='Wavelength (um)',ylabel='Intensity - number of photons packets')
    if(ret==1):
       return binning,SED

def display_multi_SED_range_tot(filename,diam_angle=[],phi=[],bins=100, minwvl=1e-6, maxwvl = 1000.e-6, logx=0,logy=1,color=[],label=[],interact=-1,reem=-1,ret=0):
    # display the total output SED 
    """root of all files with stored photon results,
    viewing polar angle,
    viewing azimuthal angle,
    angular diameter
    [source label (string or set of strings)]
    [photons that interacted (0 or 1)]
    [photons that were reemitted (0 or 1)]"""
    
    plt.figure()

    if isinstance(filename,str)==True:
        filename=[filename]
    
    for j in range(len(filename)):
        t = []
        for l in range(10) :  # from 0 to 90°
            f = file(filename[j]+'0'+str(l)+'0_phot.dat','r')
            r = csv.reader(f, delimiter='\t')
            for row in r: #read the file line per line
                t.append(float(row[12]))
        for l in range(9) :  # from 100 to 180°
            f = file(filename[j]+'1'+str(l)+'0_phot.dat','r')
            r = csv.reader(f, delimiter='\t')
            for row in r: #read the file line per line
                t.append(float(row[12]))
        binning = np.logspace(np.log10(minwvl),np.log10(maxwvl),bins+1)
        SED = 0*binning
        for i in t: #attribute the photons to the binned spectral bands
            k = 0
            #print i,binning[k]
            while (i>binning[k]) * (k < bins):
                k +=1
            SED[k]+=1
        if color==[]:
            plt.plot((binning)*1e6,SED)
        else:
            plt.plot((binning)*1e6,SED,color=color[j])
            
    if(logx==1):
        plt.xscale('log')
    if(logy==1):      
        plt.yscale('log')
    plt.title('SED total')
    plt.xlabel('Wavelength (um)')
    plt.ylabel('Intensity - number of photons packets')
    if(label!=[]):
        plt.legend(label,loc=0)
    plt.show()
    if(ret==1):
       return binning,SED

def display_SED_old(filename,theta,phi,u,bins=20,label='',interact=-1,reem=-1):
    """file with stored photon results,
    viewing polar angle,
    viewing azimuthal angle,
    angular diameter
    [source label (string or set of strings)]
    [photons that interacted (0 or 1)]
    [photons that were reemitted (0 or 1)]"""
    f = file(filename,'r')
    r = csv.reader(f, delimiter='\t')
    t = []
    for row in r: #read the file line per line
        if abs(row[1]-theta)<u and abs(row[2]-phi)<u \
        and (row[3] in label or label == '') \
        and (row[4] == interact or interact == -1) \
        and (row[5] == reem or reem == -1): #check if the photon is in the line of sight and is of the requested type
            t.append(row[0])
    maxwvl = max(t)
    minwvl = min(t)
    bin_space = (maxwvl-minwvl)/bins
    binning = np.arange(minwvl,maxwvl,bin_space)
    SED = 0*binning
    for i in t: #attribute the photons to the binned spectral bands
        k = 0
        while i<binning[k]:
            k +=1
        SED[k]+=1
    plt.plot(binning+bin_space/2,SED,'ro')
    return binning,SED

def display_image(filename,theta,phi,u,res=50,minwvl=0,maxwvl=0,label='',interact=-1,reem=-1):
    """file with stored photon results,
    viewing polar angle,
    viewing azimuthal angle,
    angular diameter
    [resolution (in pxl)]
    [minimum observed wavelength]
    [maximum observed wavelength]
    [source label (string or set of strings)]
    [photons that interacted (0 or 1)]
    [photons that were reemitted (0 or 1)]"""
    f = file(filename,'r')
    a = np.zeros((res,res)) #image array


def display_spectral_image(filename,theta,phi,u,res=50,minwvl=0,maxwvl=0,label='',interact=-1,reem=-1):
    """file with stored photon results,
    viewing polar angle,
    viewing azimuthal angle,
    angular diameter
    [resolution (in pxl)]
    [minimum observed wavelength]
    [maximum observed wavelength]
    [source label (string or set of strings)]
    [photons that interacted (0 or 1)]
    [photons that were reemitted (0 or 1)]"""

def plot_Tupdate(filename,path=[],dat=1,unity="pc",rec=0,size=100,saveformat='pdf'):
    """plot_Tupdate(filename,path=[],dat=1,unity="pc",rec=0,size=100):

    Read the file filename+"_T_update.dat" or filename if "dat" keyword is disabled

    Plot the dust temperature profile using just the last temperature update for a given r,z (image) and a given R (profile)

    INPUTS :
    filename with the temperature update data (read "filename_T_update.dat" if dat=1 and "filename" if dat=0)
    [path      = (string or set of string []) define the directory wher to look for the file to load
                 if set to [], the directory will be "Output/"]
    [dat       = whether the file has a _T_update.dat extension (0 or [1])]
    [unity     = unit of Rmax (['pc'] or 'AU')]
    [rec       = saving or not the T profile ([0] or 1)
                 with filename "profile_cocon.png" for a star
                 and "profile_torus.png" for an AGN]
    [size      = (integer [100]) set the size of images]
    """

    ## Define the work directory
    directory=path
    if(directory==[]):
        #f=open("Output/"+filename+"_phot.dat",'r')
        if(dat==1):
            f=open("Output/"+filename+"_T_update.dat",'r')
        else:
            f=open("Output/"+filename,'r')

    else:
        #f=open(hostname+"_Output/"+filename,'r')
        if(dat==1):
            f=open(directory+filename+"_T_update.dat",'r')
        else:
            f=open(directory+filename,'r')

    #f=open("Output/"+filename+"_T_update.dat",'r')

    ## Reading file
    if(dat==1):
        print("Reading file",filename+'_T_update.dat')
    else:
        print("Reading file",filename)
    tab=[]
    for data in f.readlines():
        tmp=[float(data.split()[0]),float(data.split()[1]),float(data.split()[2])]
        tab.append(tmp)
    f.close()
    print("File read")

    cst=mc.Constant()

    ##Computing limits
    image=np.zeros([size,size])
    imageavg=np.zeros([size,size])
    imagemax=np.zeros([size,size])
    imageN=np.zeros([size,size])
    if(tab==[]):
        print("Warning ! No T update")
    else:
        xmax=max(np.transpose(tab)[0])
        ymax=max(np.abs(np.transpose(tab)[1]))
        Rmax=max(xmax,ymax)
        #print "max :",xmax,ymax,Rmax
        unit=np.linspace(0.,1.,size)*Rmax
        if(unity=="AU" or unity=="UA" or unity=="au" or unity=="ua"):
            unit=unit*cst.pc/cst.AU
            


        ## Computing images
        progbar=g_pgb.Patiencebar(valmax=len(tab),up_every=1)
        for i in range(len(tab)):
            progbar.update()
            x=int(np.floor(tab[i][0]/Rmax*0.98*size))
            y=int(np.floor(abs(tab[i][1]/Rmax*0.98*size)))
            image[x][y]=tab[i][2]
            imageN[x][y]+=1
            imageavg[x][y]+=tab[i][2]
            imagemax[x][y]=max(tab[i][2],imagemax[x][y])
            

        ## Rotating images
        image=np.transpose(image)
        imageN=np.transpose(imageN)
        imageavg=np.transpose(imageavg)
        imagemax=np.transpose(imagemax)

        ## Avoid the /0 and log(0) pb
        liste=np.nonzero(imageN==0)
        imageN[liste]=1

        imageavg=imageavg/imageN

        imageN[liste]=0.1
        imageN=np.log10(imageN)
       
        ## Ploting maps

        #res=Rmax/resimagef
        if(unity=='au' or unity=='AU' or unity=='UA' or unity=='ua'):
            Rmax*=cst.pc/cst.AU
            #res*=pc/AU
        #fig,ax=plt.subplots(1)
        #cax=ax.matshow(P*100,cmap=cmap,extent=[-xmax,xmax,-ymax,ymax],origin='lower')
        #cb=fig.colorbar(cax)
        #axe=[str(int(10*i)/10.) for i in ax.get_xticks()*resimage]

        fig,ax=plt.subplots(1)
        cax=ax.matshow(image,extent=[0,Rmax,0,Rmax],origin='lower',cmap=cm.jet)
        cb=fig.colorbar(cax)
        plt.title('Last temperature update in K')
        plt.xlabel('Radius from centre (in '+unity+')')
        plt.ylabel('Radius from centre (in '+unity+')')
        if(rec==1):
            if(saveformat=='png'):
                fig.savefig(directory+filename+'_Tlast.png',dpi=600)
            else:
                fig.savefig(directory+filename+'_Tlast.pdf',rasterized=True,dpi=300)
     
        fig,ax=plt.subplots(1)
        cax=ax.matshow(imageN,extent=[0,Rmax,0,Rmax],origin='lower',cmap=cm.jet)
        cb=fig.colorbar(cax)
        plt.title('Number of T update in log10')
        plt.xlabel('Radius from centre (in '+unity+')')
        plt.ylabel('Radius from centre (in '+unity+')')
        if(rec==1):
            if(saveformat=='png'):
                fig.savefig(directory+filename+'_Nupdate.png',dpi=600)
            else:
                fig.savefig(directory+filename+'_Nupdate.pdf',rasterized=True,dpi=300)
        
        fig,ax=plt.subplots(1)
        cax=ax.matshow(imageavg,extent=[0,Rmax,0,Rmax],origin='lower',cmap=cm.jet)
        cb=fig.colorbar(cax)
        plt.title('Averaged temperature in K')
        plt.xlabel('Radius from centre (in '+unity+')')
        plt.ylabel('Radius from centre (in '+unity+')')
        if(rec==1):
            if(saveformat=='png'):
                fig.savefig(directory+filename+'_Tavg.png',dpi=600)
            else:
                fig.savefig(directory+filename+'_Tavg.pdf',rasterized=True,dpi=300)
    
        fig,ax=plt.subplots(1)
        cax=ax.matshow(imagemax,extent=[0,Rmax,0,Rmax],origin='lower',cmap=cm.jet)
        cb=fig.colorbar(cax)
        plt.title('Maximum T update in K')
        plt.xlabel('Radius from centre (in '+unity+')')
        plt.ylabel('Radius from centre (in '+unity+')')
        if(rec==1):
            if(saveformat=='png'):
                fig.savefig(directory+filename+'_Tmax.png',dpi=600)
            else:
                fig.savefig(directory+filename+'_Tmax.pdf',rasterized=True,dpi=300)
        

        ## Computing profile
        profil=np.zeros([size])
        profilN=np.zeros([size])
        for i in range(size):
            for j in range(size):
                if(np.sqrt(i*i+j*j)<size):
                    profil[int(np.floor(np.sqrt(i*i+j*j)))]+=image[i][j]
                    profilN[int(np.floor(np.sqrt(i*i+j*j)))]+=1
      
    
        for j in range(len(profilN)):
            if(profilN[j]==0):
                profilN[j]=1
  
        profil=profil/profilN

        ## PLoting profile
        fig,ax=plt.subplots(1)
        plt.plot(unit,profil)
        #cax=ax.matshow(nphotstat,cmap=cmap)
        #cb=fig.colorbar(cax)
        plt.title('T profile in the cocoon')
        plt.xlabel('Radius from centre (in '+unity+')')
        plt.xlim(xmin=0)
        plt.ylabel('Temperature (in K)')
        plt.ylim(ymin=0)
        if(rec==1):
            if(saveformat=='png'):
                fig.savefig(directory+filename+'_Tprofile.png',dpi=600)
            else:
                fig.savefig(directory+filename+'_Tprofile.pdf',rasterized=True,dpi=300)

    #partie du code Yorick de legende a traduire
    #xtitre="Distance ("+unity+")"
    #xytitles, xtitre, "Temperature (K)"
    #limits, 0, size, 0, 350
    #plt,"L = 1Lsol, Teff = 5700K, TauV = 2.,",0.25,0.75
    #plt,"Rcocon = .001 pc, silicates with r^0",0.25,0.72
    #if(rec==1 && obj=="star"):
    #    png,"profile_cocon_up.png"
    #if(rec==1 && obj=="AGN"):
    #    png,"profile_torus_up.png"
  
def plot_image(filename,outname='',suffixe='',thetaobs=[],dtheta=[],obj="AGN",path=[],dat=1,resimage=26,resunit="pixel",diffn=[],ndiffmax=[],cmap=[],rebin=0,rebinsize=30,enpaq=3.86e26,coupe=0,extract=[],vectormode=0,sym=0,rec=0,outdir=[],gif=1,saveformat='pdf',display='intensity', xmax_forced=-1, ymax_forced=-1):
    """ 
    plot_image(filename,suffixe='',thetaobs=[],dtheta=[],obj="AGN",path=[],dat=1,resimage=51,resunit="pixel",diffn=[],ndiffmax=[],cmap=[],rebin=0,rebinsize=30,enpaq=3.86e26,coupe=0,extract=[],vectormode=1,sym=0,rec=0,gif=1)

    Computing and printing function of polarimetric images generated by MontAGN

    INPUTS :
    filename with the photons data
    [outname   = (string or set of string []) name of output files if different from filename]
    [suffixe   = (string or set of string []) added at the end of recorded files]
    [thetaobs  = observation inclination angle (float between 0 and 180 [])]
    [dtheta    = bandwidth of the angle (positive float [])]
    [obj       = just a label, "star" or "AGN"]
    [path      = (string or set of string []) define the directory wher to look for the file to load
                 if set to [], the directory will be "Output/"]
    [dat       = whether the file has a .dat extension (0 or [1])]
    [resimage  = size of the final images (integer [51])]
    [resunit   = unit of the resimage parameter (['pixel'],'AU'or'pc')]
    [diffn     = only consider photons with this number of diffusion is specified (integer [])]
    [ndiffmax  = maximum number of diffusion of photons to consider if specified (integer [])]
    [cmap      = colormap to be used (string or set of string ['jet'])]
    [rebin     = [0] for not rebinning
                 (integer) for integer x integer rebinning in the centre]
    [rebinsize = half-size of the binned zone (integer [30])] 
    [enpaq     = energy of all initials photon paquets used to compute the probability (float [3.86e26]) 
    [coupe     = enabled or disabled a cut: plot the average polar angle and polar degree for each line ([0] or 1)]
    [extract   = extract informations inside the region Rex, Zex in unit of resunit ([Rex,Zex])]
    [vectormod = add the polarisation vectors to the polarized intensity map (0 or [1])]
    [sym       = use or not the left-right and top-bottom symmetries to compute the U and Q maps ([0] or 1)]
    [rec       = record all images created as fits and png ([0] or 1)]
    [gif       = record the specified number of images to be exported as a gif (integer [1])]
    """

    ### Initialization ###
    ## Colormap choice
    if (cmap==[]):
        cmap='jet'
    elif (cmap=='yorick' or cmap=='Yorick'):
        cmap=cm.gist_earth
    cst=mc.Constant()


    ## Define the work directory
    directory=path
    if(directory==[]):
        if(filename=="merged"):
            f=open("Output/phot_merged.dat",'r')
        else:
            #f=open("Output/"+filename+"_phot.dat",'r')
            if(dat==1):
                f=open("Output/"+filename+".dat",'r')
            else:
                f=open("Output/"+filename,'r')

    else:
        #f=open(hostname+"_Output/"+filename,'r')
        if(dat==1):
            f=open(directory+filename+".dat",'r')
        else:
            f=open(directory+filename,'r')


    ### Read file ###
    tab=[]
    label=[]
    print("Reading file")
    #i=1
    nocirc=0
    nodt=0
    circ=1
    i=1
    if(nocirc==1): #to read old data
        for data in f.readlines():
            #print i
            if(i==1):
                try:
                    tmp=[float(data.split()[0]),float(data.split()[1]),float(data.split()[2]),float(data.split()[3]),float(data.split()[4]),float(data.split()[5]),float(data.split()[6]),float(data.split()[7]),float(data.split()[8]),float(data.split()[9]),float(data.split()[13]),float(data.split()[14])]
                except:
                    tmp=[float(data.split()[0]),float(data.split()[1]),float(data.split()[2]),float(data.split()[3]),float(data.split()[4]),float(data.split()[5]),float(data.split()[6]),float(data.split()[7]),float(data.split()[8]),float(data.split()[9]),float(data.split()[13])]
                tab.append(tmp)
                if(len(tmp)<12):
                    circ=0
                i+=1
            else:
                if(circ==1):
                    tmp=[float(data.split()[0]),float(data.split()[1]),float(data.split()[2]),float(data.split()[3]),float(data.split()[4]),float(data.split()[5]),float(data.split()[6]),float(data.split()[7]),float(data.split()[8]),float(data.split()[9]),float(data.split()[13]),float(data.split()[14])]
                else:
                    tmp=[float(data.split()[0]),float(data.split()[1]),float(data.split()[2]),float(data.split()[3]),float(data.split()[4]),float(data.split()[5]),float(data.split()[6]),float(data.split()[7]),float(data.split()[8]),float(data.split()[9]),float(data.split()[13])]
                tab.append(tmp)

    else: #to read normal data
        #try:
        for data in f.readlines():
            #tmp=[float(data.split()[0]),float(data.split()[1]),float(data.split()[2]),float(data.split()[3]),float(data.split()[4]),float(data.split()[5]),float(data.split()[6]),float(data.split()[7]),float(data.split()[8]),float(data.split()[9]),float(data.split()[13]),float(data.split()[14])]
            #print len(data.split())
            if(len(data.split())>16):
                tmp=[float(data.split()[0]),float(data.split()[1]),float(data.split()[2]),float(data.split()[3]),float(data.split()[4]),float(data.split()[7]),float(data.split()[8]),float(data.split()[9]),float(data.split()[10]),float(data.split()[6]),float(data.split()[14]),float(data.split()[5]),float(data.split()[11]),float(data.split()[12]),float(data.split()[15]),float(data.split()[16])]
                #tab.append(tmp)
                #label.append(data.split()[13])
                #except:
                #print "no total time specified"
                #for data in f.readlines():
                #tmp=[float(data.split()[0]),float(data.split()[1]),float(data.split()[2]),float(data.split()[3]),float(data.split()[4]),float(data.split()[5]),float(data.split()[6]),float(data.split()[7]),float(data.split()[8]),float(data.split()[9]),float(data.split()[13]),float(data.split()[14])]
            else:
                #print data.split()
                tmp=[float(data.split()[0]),float(data.split()[1]),float(data.split()[2]),float(data.split()[3]),float(data.split()[4]),float(data.split()[7]),float(data.split()[8]),float(data.split()[9]),float(data.split()[10]),float(data.split()[6]),float(data.split()[14]),float(data.split()[5]),float(data.split()[11]),float(data.split()[12]),float(data.split()[15])]
                nodt=1
            tab.append(tmp)
            label.append(data.split()[13])
        
    f.close()
    if(dat==1):
        print("File",filename+'.dat',"read")
    else:
        print("File",filename,"read")
    if(tab==[]):
        print("File empty, no photon detected")
        return 


    ### Photon selection based on theta - dtheta values ###
    nphottot=len(tab);
    thetalist=np.transpose(tab)[1];
    liste=[]
    if(nodt==1):
        Dt_tot=1.0
    else:
        Dt_tot=sum(np.transpose(tab)[15]/np.transpose(tab)[14]) #Ttot/dt
    progbar=g_pgb.Patiencebar(valmax=nphottot,up_every=1)
    print("Reading photons")
    #if(dtheta!=[] and thetaobs!=[]):
    if(thetaobs!=[]):
        for i in range(nphottot):   
            #if(((i+1))%(max(1,nphottot/100))==0):
            #    progbar.update(int(round((i+1)*100./nphottot)))
            progbar.update()
            #if(int(thetalist[i]*180./np.pi/(2*dtheta))-int(thetaobs/(2*dtheta))==0):
            if(thetalist[i]*180/np.pi>thetaobs-dtheta and thetalist[i]*180/np.pi<thetaobs+dtheta):
                liste.append(i)                       
        nphottot=len(liste)
    else:
        thetaobs=90
        print('thetaobs not given, taking 90 degrees')
        liste=[i for i in range(nphottot)] #linspace(1,nphot,nphot)
        progbar.update(100)
    print(" ")
    print(nphottot,"photons read")


    ### Translate photons into information vectors ###
    ## gif loop to translate step by step for gif exportations
    print("Translating photons")
    for m in range(gif):
        nphot=int(nphottot/gif*(m+1))

        ## Initialization
        posx=np.zeros([nphot]) #tab(6,*);
        posy=np.zeros([nphot]) #tab(7,*);
        posz=np.zeros([nphot]) #tab(8,*);
        axex=np.zeros([nphot]) 
        axey=np.zeros([nphot]) 
        ndiff=np.zeros([nphot]) #tab(9,*);
        energy=np.zeros([nphot])
        dt=np.zeros([nphot])
        progbar=g_pgb.Patiencebar(valmax=nphot,up_every=1)

        ## Translating loop with phi reorientations
        for i in range(nphot):   
            #if(((i+1))%(max(1,nphot/100))==0):
            #    progbar.update(int(round((i+1)*100./nphot)))
            progbar.update()
            posx[i]=np.sqrt(tab[liste[i]][5]*tab[liste[i]][5]+tab[liste[i]][6]*tab[liste[i]][6])*np.cos(-tab[liste[i]][2]+np.arctan2(tab[liste[i]][6],tab[liste[i]][5]))
            posy[i]=np.sqrt(tab[liste[i]][5]*tab[liste[i]][5]+tab[liste[i]][6]*tab[liste[i]][6])*np.sin(-tab[liste[i]][2]+np.arctan2(tab[liste[i]][6],tab[liste[i]][5]))
            posz[i]=tab[liste[i]][7]
            ndiff[i]=tab[liste[i]][8]
            energy[i]=tab[liste[i]][10]
            dt[i]=tab[liste[i]][14]
            #axex[i]=-posx[i]*np.cos(tab[liste[i]][1])+posz[i]*np.sin(tab[liste[i]][1]) ## ancienne version
            axex[i]=-posx[i]*np.cos(thetaobs*np.pi/180.)+posz[i]*np.sin(thetaobs*np.pi/180.) ## mettre un control sur thetaobs!=[]
            #axex[i]=np.copy(posz[i])
            axey[i]=np.copy(posy[i])
        print(" ")

        ## Setting axes limits
        if xmax_forced == -1:
            xmax=max(np.abs(axex))
        else:
            xmax = xmax_forced
        if ymax_forced == -1:
            ymax=max(np.abs(axey))
        else:
            ymax = xmax_forced
        tmp=max(xmax,ymax)
        ymax=xmax=tmp*1.01
        #print energy, min(energy)
        #print ndiff
        
        ## Adapting the energy into power --> not dependant of the observation time
        if Dt_tot>0:
            energy=energy/(dt*Dt_tot)
        else:
            energy = energy/dt
        #Ttot=sum(dt)

        ## Security : if no photons, image dimension > 0
        #print xmax,resimage
        
        if(resunit!='pixel'):
            #if(sum(ndiff)<1 or xmax<resimage/2.):
            if(False): #non used anymore
                if(obj=="star" or resunit=='AU' or resunit=='au' or resunit=='UA' or resunit=='ua'):
                    #if(ymax<0.0011):
                    ymax=xmax=0.001;
                    #elif(resunit=='pixel'):
                    #ymax
                else:
                    #if(ymax<11.0):
                    ymax=xmax=10.0;
  
        ## Possible axis rearrangement
        posx=np.copy(axex)
        posy=np.copy(axey)

        ## Determining the resolution
        if(resunit=='AU' or resunit=='au' or resunit=='UA' or resunit=='ua'):
            #if(xmax<resimage*pc/AU):
            if(xmax<2*resimage/cst.pc*cst.AU):
                ymax=xmax=2*resimage*cst.AU/cst.pc
            resimagef=int(round(xmax/resimage*cst.pc/cst.AU))
        elif(resunit=='pc'):
            if(xmax<2*resimage):
                ymax=xmax=2*resimage
            resimagef=int(round(xmax/resimage))
        else:
            resimagef=resimage

        #print resimagef,xmax
        #if(xmax<resimagef/2.):
        #    ymax=xmax=2*resimagef

        ## printing axes informations
        if(gif==1):
            if(xmax>=0.1):
                #print "Size of axe y (taumax x-z) :",xmax,"pc - of axe x (taumax y) :",ymax,"pc"
                print("Scale of axe y (taumax x-z) :",xmax/resimagef,"pc/pixel - of axe x (taumax y) :",ymax/resimagef,"pc/pixel")
            else:
                print("Size of axe y (taumax x-z) :",xmax/resimagef*cst.pc/cst.AU,"AU/pixel - of axe x (taumax y) :",ymax/resimagef*cst.pc/cst.AU,"AU/pixel")
        else:
            if(xmax>=0.1):
                print("Im",m+1,": Size of axe y (taumax x-z) :",xmax/resimagef,"pc/pixel - of axe x (taumax y) :",ymax/resimagef,"pc/pixel")
            else:
                print("Im",m+1,": Size of axe y (taumax x-z) :",xmax/resimagef*cst.pc/cst.AU,"AU/pixel - of axe x (taumax y) :",ymax/resimagef*cst.pc/cst.AU,"AU/pixel")


        ### Compute into images ###
        #print max(posx),np.mean(posx),xmax
        #print nphot

        ## initialization
        nbin=resimagef
        if(sym==2):
            nbintot=2*nbin
        else:
            nbintot=2*nbin+1
        #nbin=50
        #print nbintot
        dx=xmax/nbin
        dy=ymax/nbin
        pos=np.zeros([nbintot,2])
        QU=np.zeros([nbintot,nbintot,2])
        V=np.zeros([nbintot,nbintot])
        I=np.zeros([nbintot,nbintot])
        theta=np.zeros([nbintot,nbintot])
        P=np.zeros([nbintot,nbintot])
        ndiffmap=np.zeros([nbintot,nbintot])
        nphotstat=np.zeros([nbintot,nbintot])
        nphotstateff=np.zeros([nbintot,nbintot])
    
        pos=np.transpose([(np.linspace(1,nbintot,nbintot)-(nbintot+1)/2.)*xmax/nbin,(np.linspace(1,nbintot,nbintot)-(nbintot+1)/2.)*ymax/nbin])
  
        ## Computing maps : QU, I, ndiff, nphot
        print("Computing images")
        progbar=g_pgb.Patiencebar(valmax=nphot,up_every=1)
        for i in range(nphot):
            #if(((i+1))%(max(1,nphot/100))==0):
            #    progbar.update(int(round((i+1)*100./nphot)))
            progbar.update()
            if(ndiffmax==[] or ndiffmax>=ndiff[i]):
                if(diffn==[] or ndiff[i]==diffn):
                    if(np.abs(posx[i])<xmax and np.abs(posy[i])<ymax and energy[i]>1.):
                        QU[int(round(posx[i]/xmax*(nbintot-1)/2.))+nbin][int(round(posy[i]/ymax*(nbintot-1)/2.))+nbin][0]+=(tab[liste[i]][3]*np.cos(2.*(tab[liste[i]][9]))-tab[liste[i]][4]*np.sin(2.*(tab[liste[i]][9])))*energy[i]
                        #QU[int(round(posx[i]/xmax*(nbintot-1)/2.))+nbin][int(round(posy[i]/ymax*(nbintot-1)/2.))+nbin][0]-=(tab[liste[i]][3]*np.cos(2.*(tab[liste[i]][9]))-tab[liste[i]][4]*np.sin(2.*(tab[liste[i]][9])))*energy[i]
                        QU[int(round(posx[i]/xmax*(nbintot-1)/2.))+nbin][int(round(posy[i]/ymax*(nbintot-1)/2.))+nbin][1]+=(tab[liste[i]][4]*np.cos(2.*(tab[liste[i]][9]))+tab[liste[i]][3]*np.sin(2.*(tab[liste[i]][9])))*energy[i]
                        #QU[int(round(posx[i]/xmax*(nbintot-1)/2.))+nbin][int(round(posy[i]/ymax*(nbintot-1)/2.))+nbin][1]-=(tab[liste[i]][4]*np.cos(2.*(tab[liste[i]][9]))+tab[liste[i]][3]*np.sin(2.*(tab[liste[i]][9])))*energy[i]
                        if(circ==1):
                            V[int(round(posx[i]/xmax*(nbintot-1)/2.))+nbin][int(round(posy[i]/ymax*(nbintot-1)/2.))+nbin]+=tab[liste[i]][11]*energy[i]
                        I[int(round(posx[i]/xmax*(nbintot-1)/2.))+nbin][int(round(posy[i]/ymax*(nbintot-1)/2.))+nbin]+=energy[i]#1
                        ndiffmap[int(round(posx[i]/xmax*(nbintot-1)/2.))+nbin][int(round(posy[i]/ymax*(nbintot-1)/2.))+nbin]+=ndiff[i]*energy[i]
                        nphotstat[int(round(posx[i]/xmax*(nbintot-1)/2.))+nbin][int(round(posy[i]/ymax*(nbintot-1)/2.))+nbin]+=1
                        if(nphotstateff[int(round(posx[i]/xmax*(nbintot-1)/2.))+nbin][int(round(posy[i]/ymax*(nbintot-1)/2.))+nbin]<energy[i]):
                            nphotstateff[int(round(posx[i]/xmax*(nbintot-1)/2.))+nbin][int(round(posy[i]/ymax*(nbintot-1)/2.))+nbin]=energy[i]
        print(" ")

        ### Post-treatment ###
        ## QU rearrangement : use of symmetries
        I_nosym=np.copy(I)
        if(sym>0):
            QU1=np.copy(QU)
            QU2=np.zeros([nbintot,nbintot,2])
            QU3=np.zeros([nbintot,nbintot,2])
            QU4=np.zeros([nbintot,nbintot,2])
            I2=np.zeros([nbintot,nbintot])
            ndiffmap2=np.zeros([nbintot,nbintot])
            nphotstat2=np.zeros([nbintot,nbintot])
            nphotstateff2=np.zeros([nbintot,nbintot])
            I3=np.zeros([nbintot,nbintot])
            ndiffmap3=np.zeros([nbintot,nbintot])
            nphotstat3=np.zeros([nbintot,nbintot])
            nphotstateff3=np.zeros([nbintot,nbintot])
            I4=np.zeros([nbintot,nbintot])
            ndiffmap4=np.zeros([nbintot,nbintot])
            nphotstat4=np.zeros([nbintot,nbintot])
            nphotstateff4=np.zeros([nbintot,nbintot])
            for i in range(nbintot):
                for j in range(nbintot):
                    if (j!=nbin or sym==2):
                        QU2[i][j][0]=QU[i][nbintot-1-j][0]
                        QU2[i][j][1]=-QU[i][nbintot-1-j][1]
                        I2[i][j]=I[i][nbintot-1-j]
                        ndiffmap2[i][j]=ndiffmap[i][nbintot-1-j]
                        nphotstat2[i][j]=nphotstat[i][nbintot-1-j]
                        nphotstateff2[i][j]=nphotstateff[i][nbintot-1-j]
            for i in range(nbintot):
                for j in range(nbintot):
                    #if(nbin!=i and nbin!=j):
                    if(i!=nbin or sym==2):
                        QU3[i][j][:]=QU[nbintot-1-i][nbintot-1-j][:]
                        I3[i][j]=I[nbintot-1-i][nbintot-1-j]
                        ndiffmap3[i][j]=ndiffmap[nbintot-1-i][nbintot-1-j]
                        nphotstat3[i][j]=nphotstat[nbintot-1-i][nbintot-1-j]
                        nphotstateff3[i][j]=nphotstateff[nbintot-1-i][nbintot-1-j]
                        QU4[i][j][:]=QU2[nbintot-1-i][nbintot-1-j][:]
                        I4[i][j]=I2[nbintot-1-i][nbintot-1-j]
                        ndiffmap4[i][j]=ndiffmap2[nbintot-1-i][nbintot-1-j]
                        nphotstat4[i][j]=nphotstat2[nbintot-1-i][nbintot-1-j]
                        nphotstateff4[i][j]=nphotstateff2[nbintot-1-i][nbintot-1-j]
            QU+=QU2+QU3+QU4
            I+=I2+I3+I4
            ndiffmap+=ndiffmap2+ndiffmap3+ndiffmap4
            nphotstat+=nphotstat2+nphotstat3+nphotstat4
            for i in range(nbintot):
                for j in range(nbintot):
                    nphotstateff[i,j]=max(nphotstateff[i,j],nphotstateff2[i,j],nphotstateff3[i,j],nphotstateff4[i,j])

            # To check :
            #plt.matshow(QU1[:,:,0])
            #plt.matshow(QU2[:,:,0])
            #plt.matshow(QU3[:,:,0])
            #plt.matshow(QU4[:,:,0])
            #plt.matshow(QU1[:,:,1])
            #plt.matshow(QU2[:,:,1])
            #plt.matshow(QU3[:,:,1])
            #plt.matshow(QU4[:,:,1])



        ## Corrections of 0 to apply a log
        nphotdet=int(sum(sum(nphotstat)))
        listI=np.nonzero(I==0)
        for i in range(np.shape(listI)[1]):
            I[listI[0][i]][listI[1][i]]=1
            nphotstat[listI[0][i]][listI[1][i]]=0.1
            nphotstateff[listI[0][i]][listI[1][i]]=1
        nphotstateff=I/nphotstateff

        listI_nosym=np.nonzero(I_nosym==0)
        for i in range(np.shape(listI_nosym)[1]):
            I_nosym[listI_nosym[0][i]][listI_nosym[1][i]]=1

        ## QU --> P theta and I corrections
        #plt.matshow(I_nosym)
        #plt.matshow(I)
        #return I_nosym,V
        Pc=np.copy(V)
        for i in range(nbintot):
            for j in range(nbintot):
                P[i][j]=np.sqrt((QU[i][j][0]*QU[i][j][0]+QU[i][j][1]*QU[i][j][1])/(I[i][j]*I[i][j]))
                Pc[i][j]=V[i][j]/I_nosym[i][j]
                #theta[i][j]=-0.5*np.arctan2(QU[i][j][1]/I[i][j],QU[i][j][0]/I[i][j])*180./np.pi
                theta[i][j]=0.5*np.arctan2(QU[i][j][1]/I[i][j],QU[i][j][0]/I[i][j])*180./np.pi
                ndiffmap[i][j]=ndiffmap[i][j]/I[i][j]

        
        ## I into log
        logI=np.log10(I)

        ## Resetting I to 0 where it should
        for i in range(np.shape(listI)[1]):
            I[listI[0][i]][listI[1][i]]=0

            
        ## Computing probability maps
        Prob=I/nphotstat/enpaq
        nphotstat=np.log10(nphotstat)
                
        ## Resetting nphot to 0.1 where it should be nul
        for i in range(np.shape(listI)[1]):
            #nphotstat[listI[0][i]][listI[1][i]]=-0.1
            Prob[listI[0][i]][listI[1][i]]=1e-20
            nphotstateff[listI[0][i]][listI[1][i]]=0
        #nphotstateff=np.log10(nphotstateff)


        ## rebinning part -- Old way, acting on the initial resolution is much better
        if(rebin>0):
            #rebin=2
            #rebinsize=30
            thetavect=[]
            Pvect=[]
            for i in range(int(rebinsize*2/rebin)):
                for j in range(int(rebinsize*2/rebin)):
                    sommtheta=0.
                    nbtheta=0
                    sommP=0.
                    nbP=0
                    for k in range(rebin):
                        for l in range(rebin):
                            if(theta[nbin+1-rebinsize+rebin*i+k][nbin+1-rebinsize+rebin*j+l]>-95):
                                sommtheta+=theta[nbin+1-rebinsize+rebin*i+k][nbin+1-rebinsize+rebin*j+l]
                                thetavect.append(theta[nbin+1-rebinsize+rebin*i+k][nbin+1-rebinsize+rebin*j+l])
                                nbtheta+=1
                            if(P[nbin+1-rebinsize+rebin*i+k][nbin+1-rebinsize+rebin*j+l]>-95):
                                sommP+=P[nbin+1-rebinsize+rebin*i+k][nbin+1-rebinsize+rebin*j+l]
                                Pvect.append(P[nbin+1-rebinsize+rebin*i+k][nbin+1-rebinsize+rebin*j+l])
                                nbP+=1
                    if(nb>=1):
                        sommtheta=sommtheta/nbtheta
                        sommP=sommP/nbP
                    for k in range(rebin):
                        for l in range(rebin):
                            theta[nbin+1-rebinsize+rebin*i+k][nbin+1-rebinsize+rebin*j+l]=sommtheta
                            P[nbin+1-rebinsize+rebin*i+k][nbin+1-rebinsize+rebin*j+l]=sommP
            print("mean polar angle :",np.mean(thetavect),"with standard deviation :",np.sqrt(np.var(thetavect)))
      

        ## Computing Ip and I2 maps
        Ip=P*I
        I2=np.copy(I)
        I2[nbin][nbin]=0
        if(sym==2):
            I2[nbin-1][nbin-1]=0
            I2[nbin-1][nbin]=0
            I2[nbin][nbin-1]=0
        #if(proba==1):
        #    I=np.copy(Prob)
        if(sym==0):
            print(nphotdet,"photons detected")
        else:
            print(nphotdet,"photons detected using symmetries")
            print("for",nphotdet/4,"photons detected")


        ### Analyse ###
        
        ## making horrizontal cuts
        vectangle=np.zeros(nbintot)
        vectdegre=np.zeros(nbintot)
        if(coupe==1):
            #nbinhisto=45
            histotheta2=[]
            histoP2=[]
            #plt.figure()
            for i in range(nbintot):
                histotheta=[]
                histoP=[]
                vect=np.copy(theta[i][:])
                #vect=np.copy(np.transpose(theta)[:][i])
                listv=np.nonzero(vect+100)[0]
                #print i,listv
                if(len(listv)>0.5):
                    for j in range(len(listv)):
                        vectangle[i]+=(theta[i][listv[j]])/(len(listv))
                        vectdegre[i]+=(P[i][listv[j]])/(len(listv))*100
                        histotheta.append(theta[i][listv[j]])
                        histoP.append(P[i][listv[j]]*100)
                        #vectangle[i]+=(theta[listv[j]][i])/(len(listv))
                        #vectdegre[i]+=(P[listv[j]][i])/(len(listv))*100
                    if(i>nbin/2-5 and i<nbin/2+5):
                        #plt.figure()
                        #plt.hist(histotheta)
                        histotheta2.append(histotheta)
                        histoP2.append(histoP)
            ## Ploting cuts
            plt.figure()
            plt.hist(histotheta2,stacked=1)
            plt.show()
            plt.figure()
            plt.plot(vectangle)
            plt.title('averaged polar angle per line')
            plt.show()
            plt.figure()
            plt.plot(vectdegre)
            plt.title('averaged polar degre per line')
            plt.show()

        ## Computing the map centrosymetric-substracted
        #if(centrosymsubstract==1):
        thetasym=np.copy(theta)
        for i in range(nbintot):
            for j in range(nbintot):
                #thetasym[i][j]=np.arctan2(j-(nbintot-1)/2.,i-(nbintot-1)/2.)*180./np.pi-90
                thetasym[i][j]=np.arctan2(i-(nbintot-1)/2.,j-(nbintot-1)/2.)*180./np.pi
                if(thetasym[i][j]<-90):
                    thetasym[i][j]+=180
                if(thetasym[i][j]>90):
                    thetasym[i][j]-=180
        #thetas=thetasym
        thetas=theta-thetasym
        for i in range(nbintot):
            for j in range(nbintot):
                if(I[i][j]==0.):
                    thetas[i][j]=-120
                else:
                    if(thetas[i][j]<-90):
                        thetas[i][j]+=180
                    elif(thetas[i][j]>90):
                        thetas[i][j]-=180

        #alternative display of centrosym
        thetas_article=np.abs(thetas)
        phi=np.copy(thetasym)
        #print np.shape(QU)
        ##Qphi=-np.transpose(np.transpose(QU)[0]*np.cos(2*phi*np.pi/180.)+np.transpose(QU)[1]*np.sin(2*phi*np.pi/180.))
        ##Uphi=-np.transpose(np.transpose(QU)[0]*np.sin(2*phi*np.pi/180.)-np.transpose(QU)[1]*np.cos(2*phi*np.pi/180.))
        #Qphi=np.transpose(np.transpose(QU)[0]*np.cos(2*phi*np.pi/180.)+np.transpose(QU)[1]*np.sin(2*phi*np.pi/180.))
        #Uphi=-np.transpose(np.transpose(QU)[0]*np.sin(2*phi*np.pi/180.)+np.transpose(QU)[1]*np.cos(2*phi*np.pi/180.))
        Qphi=QU[:,:,0]*np.cos(2*phi*np.pi/180.)+QU[:,:,1]*np.sin(2*phi*np.pi/180.)
        Uphi=-QU[:,:,0]*np.sin(2*phi*np.pi/180.)+QU[:,:,1]*np.cos(2*phi*np.pi/180.)


        ##setting outname
        if(outname!=''):
            filename=outname
        filenametmp=''
        for i in range(len(filename)):
            if(filename[i]=='.'):
                filenametmp+='-'
            else:
                filenametmp+=filename[i]
        filename=filenametmp




        ## Setting the no-photon detection to -100 in angular maps
        thetaplot=np.copy(theta)
        #theta+=90
        for i in range(nbintot):
            for j in range(nbintot):
                if(theta[i][j]>90):
                    theta[i][j]-=180
        for i in range(nbintot):
            for j in range(nbintot):
                if(I[i][j]==0.):
                    theta[i][j]=-120.

        ## Extracting informations :
        if(extract!=[]):
            if(resunit=='pc' or resunit=='PC'):
                Rex=int(round(extract[0]/xmax*(nbintot-1)/2.)) ##in pixel
                Zex=int(round(extract[1]/xmax*(nbintot-1)/2.)) ##in pixel
            elif(resunit=='au' or resunit=='AU' or resunit=='ua' or resunit=='UA'):
                Rex=int(round(extract[0]/xmax*cst.AU/cst.pc*(nbintot-1)/2.)) ##in pixel
                Zex=int(round(extract[1]/xmax*cst.AU/cst.pc*(nbintot-1)/2.)) ##in pixel
            else:
                Rex=int(round(extract[0])) ##in pixel
                Zex=int(round(extract[1])) ##in pixel
            if(sym!=2):
                print('Analysed zone :',2*Rex+1,'x',2*Zex+1,'pixels')
                #print (QU[nbin-Rex:nbin+Rex+1,nbin-Zex:nbin+Zex+1,0]),sum(sum(QU[nbin-Rex:nbin+Rex+1,nbin-Zex:nbin+Zex+1,0]))
                Pex=np.sqrt((sum(sum(QU[nbin-Zex:nbin+Zex+1,nbin-Rex:nbin+Rex+1,0]))*sum(sum(QU[nbin-Zex:nbin+Zex+1,nbin-Rex:nbin+Rex+1,0]))+sum(sum(QU[nbin-Zex:nbin+Zex+1,nbin-Rex:nbin+Rex+1,1]))*sum(sum(QU[nbin-Zex:nbin+Zex+1,nbin-Rex:nbin+Rex+1,1])))/(sum(sum(I[nbin-Zex:nbin+Zex+1,nbin-Rex:nbin+Rex+1]))*sum(sum(I[nbin-Zex:nbin+Zex+1,nbin-Rex:nbin+Rex+1]))))
                thetaex=-0.5*np.arctan2(sum(sum(QU[nbin-Zex:nbin+Zex+1,nbin-Rex:nbin+Rex+1,1]))/sum(sum(I[nbin-Zex:nbin+Zex+1,nbin-Rex:nbin+Rex+1])),sum(sum(QU[nbin-Zex:nbin+Zex+1,nbin-Rex:nbin+Rex+1,0]))/sum(sum(I[nbin-Zex:nbin+Zex+1,nbin-Rex:nbin+Rex+1])))*180./np.pi
                Rextot=2*Rex+1
                Zextot=2*Zex+1
            else:
                print('Analysed zone :',2*Rex,'x',2*Zex,'pixels')
                #print (QU[nbin-Rex:nbin+Rex+1,nbin-Zex:nbin+Zex+1,0]),sum(sum(QU[nbin-Rex:nbin+Rex+1,nbin-Zex:nbin+Zex+1,0]))
                Pex=np.sqrt((sum(sum(QU[nbin-Zex:nbin+Zex,nbin-Rex:nbin+Rex,0]))*sum(sum(QU[nbin-Zex:nbin+Zex,nbin-Rex:nbin+Rex,0]))+sum(sum(QU[nbin-Zex:nbin+Zex,nbin-Rex:nbin+Rex,1]))*sum(sum(QU[nbin-Zex:nbin+Zex,nbin-Rex:nbin+Rex,1])))/(sum(sum(I[nbin-Zex:nbin+Zex,nbin-Rex:nbin+Rex]))*sum(sum(I[nbin-Zex:nbin+Zex,nbin-Rex:nbin+Rex]))))
                thetaex=-0.5*np.arctan2(sum(sum(QU[nbin-Zex:nbin+Zex,nbin-Rex:nbin+Rex,1]))/sum(sum(I[nbin-Zex:nbin+Zex,nbin-Rex:nbin+Rex])),sum(sum(QU[nbin-Zex:nbin+Zex,nbin-Rex:nbin+Rex,0]))/sum(sum(I[nbin-Zex:nbin+Zex,nbin-Rex:nbin+Rex])))*180./np.pi     
                Rextot=2*Rex
                Zextot=2*Zex           
            Pcen=[]
            thetacen=[]
            ncen=[]
            for i in range(Zextot):
                for j in range(Rextot):
                    Pcen.append(P[nbin-Zex+i][nbin-Rex+j]*100.)
                    thetacen.append(theta[nbin-Zex+i][nbin-Rex+j])
                    ncen.append(nphotstateff[nbin-Zex+i][nbin-Rex+j])
            print('Degree of polarisation in the central region :',Pex*100.,'%')
            print('Angle of polarisation in the central region :',thetaex,'degrees')
            plt.figure()
            plt.hist(thetacen,bins=18) 
            plt.title('theta histogram in the central region (degree)')   
            plt.show()    
            fig,ax=plt.subplots(1)
            if(sym==2):
                cax=ax.matshow(100*P[nbin-Zex:nbin+Zex,nbin-Rex:nbin+Rex],cmap=cmap)
            else:
                cax=ax.matshow(100*P[nbin-Zex:nbin+Zex+1,nbin-Rex:nbin+Rex+1],cmap=cmap)
            cb=fig.colorbar(cax)
            plt.title('Degree of polarisation in the analysed region (%)')
            plt.figure()
            plt.plot(thetacen,Pcen,'+')
            plt.title('P vs theta')
            plt.xlabel('Polarisation angle (degree)')
            plt.ylabel('Polarisation degree (%)')
            plt.show()
            plt.figure()
            plt.plot(ncen,Pcen,'+')
            #plt.plot(ncen,abs(np.array(thetacen)),'r+')
            plt.plot(ncen,thetacen,'r+')
            plt.title('Evolution of P and theta')
            plt.xlabel('Number of effective photons')
            plt.ylabel('Polarisation degree in blue (%) and angle in red (degree)')
            plt.show()


        ### Ploting all results ###

        ## set the directory
        if(outdir==[]):
            outdir='Output/'

        ## Save fits files (m==gif-1 if to select only the last files in case of step by step)
        if(rec==1 and m==gif-1):
            try:
                hdu = fits.PrimaryHDU()
                hdu.data=P*100
                hdu.writeto(outdir+filename+suffixe+'_P.fits')
                hdu = fits.PrimaryHDU()
                hdu.data=ndiffmap
                hdu.writeto(outdir+filename+suffixe+'_ndiffmap.fits')
                hdu = fits.PrimaryHDU()
                hdu.data=theta
                hdu.writeto(outdir+filename+suffixe+'_theta.fits')
                hdu = fits.PrimaryHDU()
                hdu.data=Ip
                hdu.writeto(outdir+filename+suffixe+'_Ip.fits')
                hdu = fits.PrimaryHDU()
                hdu.data=nphotstat
                hdu.writeto(outdir+filename+suffixe+'_nphot.fits')
                hdu = fits.PrimaryHDU()
                hdu.data=I2
                hdu.writeto(outdir+filename+suffixe+'_I2.fits')
                hdu = fits.PrimaryHDU()
                hdu.data=logI
                hdu.writeto(outdir+filename+suffixe+'_logI.fits')
                hdu = fits.PrimaryHDU()
                hdu.data=thetas
                hdu.writeto(outdir+filename+suffixe+'_theta-diff.fits')
                hdu = fits.PrimaryHDU()
                hdu.data=np.log10(Prob)
                hdu.writeto(outdir+filename+suffixe+'_logProba.fits')
                hdu = fits.PrimaryHDU()
                hdu.data=V*100
                hdu.writeto(outdir+filename+suffixe+'_V.fits')
            except:
                yn=input('Already existing fits files, overwrite them ? (y/n)')
                if(yn=='y' or yn=='Y'):
                    hdu = fits.PrimaryHDU()
                    hdu.data=P*100
                    hdu.writeto(outdir+filename+suffixe+'_P.fits',overwrite=True)
                    hdu = fits.PrimaryHDU()
                    hdu.data=ndiffmap
                    hdu.writeto(outdir+filename+suffixe+'_ndiffmap.fits',overwrite=True)
                    hdu = fits.PrimaryHDU()
                    hdu.data=theta
                    hdu.writeto(outdir+filename+suffixe+'_theta.fits',overwrite=True)
                    hdu = fits.PrimaryHDU()
                    hdu.data=Ip
                    hdu.writeto(outdir+filename+suffixe+'_Ip.fits',overwrite=True)
                    hdu = fits.PrimaryHDU()
                    hdu.data=nphotstat
                    hdu.writeto(outdir+filename+suffixe+'_nphot.fits',overwrite=True)
                    hdu = fits.PrimaryHDU()
                    hdu.data=I2
                    hdu.writeto(outdir+filename+suffixe+'_I2.fits',overwrite=True)
                    hdu = fits.PrimaryHDU()
                    hdu.data=logI
                    hdu.writeto(outdir+filename+suffixe+'_logI.fits',overwrite=True)
                    hdu = fits.PrimaryHDU()
                    hdu.data=thetas
                    hdu.writeto(outdir+filename+suffixe+'_theta-diff.fits',overwrite=True)
                    hdu = fits.PrimaryHDU()
                    hdu.data=np.log10(Prob)
                    hdu.writeto(outdir+filename+suffixe+'_logProba.fits',overwrite=True)
                    hdu = fits.PrimaryHDU()
                    hdu.data=V*100
                    hdu.writeto(outdir+filename+suffixe+'_V.fits',overwrite=True)


        ## ploting
        if(cmap=='jet'):
            cmaploop,normloop,mappableloop=g_cb.colorbar(cmap='loopingmask',cm_min=-120,cm_max=90)
        else:
            cmaploop,normloop,mappableloop=g_cb.colorbar(cmap='loopingearth',cm_min=-100,cm_max=90)
        maxlog=np.max(logI)
        minlog=maxlog-10
        #cmaplog,normlog,mappablelog=g_cb.colorbar(cmap=cmap,cm_min=np.log10(enpaq)-5,cm_max=np.log10(enpaq)+5)
        cmaplog,normlog,mappablelog=g_cb.colorbar(cmap=cmap,cm_min=minlog,cm_max=maxlog)
        cmapeff,normeff,mappableeff=g_cb.colorbar(cmap=cmap,cm_min=0,cm_max=min(nphotstateff.max(),50))
        cmapdiff,normdiff,mappablediff=g_cb.colorbar(cmap=cmap,cm_min=0,cm_max=5)
        cmap_art,norm_art,mappable_art=g_cb.colorbar(cmap=cmap,cm_min=0,cm_max=90)
        cmapV,normV,mappableV=g_cb.colorbar(cmap=cmap,cm_min=-100,cm_max=100)

        if(resunit=='pixel'):
            res=1
        else:
            res=xmax/resimagef
            if(resunit=='au' or resunit=='AU' or resunit=='UA' or resunit=='ua'):
                xmax*=cst.pc/cst.AU
                ymax*=cst.pc/cst.AU
                res*=cst.pc/cst.AU

        if(nodt==1):
            unitI="J"
        else:
            unitI="W"

        #suffixe=suffixe+str(thetaobs)
        titletheta = ' for i='+str(thetaobs)+' deg'
        #if(thetaobs<10):
        #    titletheta = 'for inclination of _00'+str(model.thetarec[i])+"_phot.dat"
        #elif(thetaobs<100):
        #    titletheta = '_0'+str(model.thetarec[i])+"_phot.dat"
        #else:
        #    titletheta = '_'+str(model.thetarec[i])+"_phot.dat"

        if(display=='polar' or display=='full'):
            fig,ax=plt.subplots(1)
            cax=ax.matshow(P*100,cmap=cmap,extent=[-xmax,xmax,-ymax,ymax],origin='lower')
            cb=fig.colorbar(cax)
            #axe=[str(int(10*i)/10.) for i in ax.get_xticks()*resimage]
            #print ax.get_xticks()
            #for i in range(len(ax.get_xticklabels())):
            #    print ax.get_xticklabels()[i]
            #print len(ax.get_xticks())
            #print np.linspace(-2*xmax/(len(ax.get_xticks())-2),2*xmax,len(ax.get_xticks()))
            
            #axe=[str(int(10*i)/10.) for i in np.linspace(-2*xmax/(len(ax.get_xticks())-2),2*xmax,len(ax.get_xticks()))]
            #ax.set_xticklabels(axe)
            #ax.set_yticklabels(axe)
            plt.title('Linear degree of polarisation (%)'+titletheta)
            plt.xlabel('Offset x ('+resunit+')')
            plt.ylabel('Offset y ('+resunit+')')
            if(rec==1):
                if(gif==1):
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+suffixe+'_P.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+suffixe+'_P.pdf',rasterized=True,dpi=300)
                else:
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_P.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_P.pdf',rasterized=True,dpi=300)
        if(display=='polar' or display=='full'):
            fig,ax=plt.subplots(1)
            cax=ax.matshow(Ip,cmap=cmap,extent=[-xmax,xmax,-ymax,ymax],origin='lower')
            cb=fig.colorbar(cax)
            #ax.set_xticklabels(axe)
            #ax.set_yticklabels(axe)
            plt.title('Linear polarised intensity ('+unitI+')'+titletheta)
            plt.xlabel('Offset x ('+resunit+')')
            plt.ylabel('Offset y ('+resunit+')')
            if(rec==1):
                if(gif==1):
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+suffixe+'_Ip.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+suffixe+'_Ip.pdf',rasterized=True,dpi=300)
                else:
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_Ip.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_Ip.pdf',rasterized=True,dpi=300)
        if(display=='polar' or display=='full'):
            fig,ax=plt.subplots(1)
            cax=ax.matshow(theta,cmap=cmaploop,norm=normloop,extent=[-xmax,xmax,-ymax,ymax],origin='lower')
            cb=fig.colorbar(cax)
            #ax.set_xticklabels(axe)
            #ax.set_yticklabels(axe)
            plt.title('Polarisation angle (degree)'+titletheta)
            plt.xlabel('Offset x ('+resunit+')')
            plt.ylabel('Offset y ('+resunit+')')
            if(rec==1):
                if(gif==1):
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+suffixe+'_theta.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+suffixe+'_theta.pdf',rasterized=True,dpi=300)
                else:
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_theta.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_theta.pdf',rasterized=True,dpi=300)
        if(display=='full'):
            fig,ax=plt.subplots(1)
            cax=ax.matshow(phi,cmap=cmaploop,norm=normloop,extent=[-xmax,xmax,-ymax,ymax],origin='lower')
            cb=fig.colorbar(cax)
            #ax.set_xticklabels(axe)
            #ax.set_yticklabels(axe)
            plt.title('Centro-symmetric angle (degree)'+titletheta)
            plt.xlabel('Offset x ('+resunit+')')
            plt.ylabel('Offset y ('+resunit+')')
            if(rec==1):
                if(gif==1):
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+suffixe+'_phi.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+suffixe+'_phi.pdf',rasterized=True,dpi=300)
                else:
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_phi.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_phi.pdf',rasterized=True,dpi=300)
        if(display=='intensity' or display=='polar' or display=='full'):
            fig,ax=plt.subplots(1)
            #cax=ax.matshow(logI,cmap=cmaplog,norm=normlog)
            #cax=ax.matshow(logI,cmap=cmap,extent=[-xmax,xmax,-ymax,ymax],origin='lower')
            cax=ax.matshow(logI,cmap=cmaplog,norm=normlog,extent=[-xmax,xmax,-ymax,ymax],origin='lower')
            cb=fig.colorbar(cax)
            #ax.set_xticklabels(axe)
            #ax.set_yticklabels(axe)
            plt.title('Intensity (log10('+unitI+'))'+titletheta)
            plt.xlabel('Offset x ('+resunit+')')
            plt.ylabel('Offset y ('+resunit+')')
            if(rec==1):
                if(gif==1):
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+suffixe+'_I.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+suffixe+'_I.pdf',rasterized=True,dpi=300)
                else:
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_I.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_I.pdf',rasterized=True,dpi=300)
        if(display=='intensity' or display=='polar' or display=='full'):
            fig,ax=plt.subplots(1)
            cax=ax.matshow(I2,cmap=cmap,extent=[-xmax,xmax,-ymax,ymax],origin='lower')
            cb=fig.colorbar(cax)
            #ax.set_xticklabels(axe)
            #ax.set_yticklabels(axe)
            plt.title('Intensity without central object ('+unitI+')'+titletheta)
            plt.xlabel('Offset x ('+resunit+')')
            plt.ylabel('Offset y ('+resunit+')')
            if(rec==1):
                if(gif==1):
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+suffixe+'_I2.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+suffixe+'_I2.pdf',rasterized=True,dpi=300)
                else:
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_I2.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_I2.pdf',rasterized=True,dpi=300)
        if(display=='intensity' or display=='polar' or display=='full'):
            fig,ax=plt.subplots(1)
            cax=ax.matshow(ndiffmap,cmap=cmap,norm=normdiff,extent=[-xmax,xmax,-ymax,ymax],origin='lower')
            cb=fig.colorbar(cax)
            #ax.set_xticklabels(axe)
            #ax.set_yticklabels(axe)
            plt.title('Averaged number of scatterings'+titletheta)
            plt.xlabel('Offset x ('+resunit+')')
            plt.ylabel('Offset y ('+resunit+')')
            if(rec==1):
                if(gif==1):
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+suffixe+'_ndiff.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+suffixe+'_ndiff.pdf',rasterized=True,dpi=300)
                else:
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_ndiff.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_ndiff.pdf',rasterized=True,dpi=300)
        if(display=='intensity' or display=='polar' or display=='full'):
            fig,ax=plt.subplots(1)
            cax=ax.matshow(nphotstat,cmap=cmap,extent=[-xmax,xmax,-ymax,ymax],origin='lower')
            cb=fig.colorbar(cax)
            #ax.set_xticklabels(axe)
            #ax.set_yticklabels(axe)
            plt.title('Number of packets per pixel (log10(npacket))'+titletheta)
            plt.xlabel('Offset x ('+resunit+')')
            plt.ylabel('Offset y ('+resunit+')')
            if(rec==1):
                if(gif==1):
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+suffixe+'_nphot.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+suffixe+'_nphot.pdf',rasterized=True,dpi=300)
                else:
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_nphot.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_nphot.pdf',rasterized=True,dpi=300)
        if(display=='full'):
            fig,ax=plt.subplots(1)
            cax=ax.matshow(nphotstateff,cmap=cmapeff,norm=normeff,extent=[-xmax,xmax,-ymax,ymax],origin='lower')
            cb=fig.colorbar(cax)
            #ax.set_xticklabels(axe)
            #ax.set_yticklabels(axe)
            plt.title('Number of effective packets per pixel'+titletheta)
            plt.xlabel('Offset x ('+resunit+')')
            plt.ylabel('Offset y ('+resunit+')')
            if(rec==1):
                if(gif==1):
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+suffixe+'_nphoteff.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+suffixe+'_nphoteff.pdf',rasterized=True,dpi=300)
                else:
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_nphoteff.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_nphoteff.pdf',rasterized=True,dpi=300)
        if(display=='polar' or display=='full'):
            fig,ax=plt.subplots(1)
            cax=ax.matshow(thetas,cmap=cmaploop,norm=normloop,extent=[-xmax,xmax,-ymax,ymax],origin='lower')
            cb=fig.colorbar(cax)
            #ax.set_xticklabels(axe)
            #ax.set_yticklabels(axe)
            plt.title('Difference to centrosymmetric (degree)'+titletheta)
            plt.xlabel('Offset x ('+resunit+')')
            plt.ylabel('Offset y ('+resunit+')')
            if(rec==1):
                if(gif==1):
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+suffixe+'_theta-diff.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+suffixe+'_theta-diff.pdf',rasterized=True,dpi=300)
                else:
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_theta-diff.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_theta-diff.pdf',rasterized=True,dpi=300)
        if(display=='full'):
            fig,ax=plt.subplots(1)
            cax=ax.matshow(Qphi,cmap=cmap,extent=[-xmax,xmax,-ymax,ymax],origin='lower')
            cb=fig.colorbar(cax)
            #ax.set_xticklabels(axe)
            #ax.set_yticklabels(axe)
            plt.title('Qphi ('+unitI+')'+titletheta)
            plt.xlabel('Offset x ('+resunit+')')
            plt.ylabel('Offset y ('+resunit+')')
            if(rec==1):
                if(gif==1):
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+suffixe+'_Qphi.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+suffixe+'_Qphi.pdf',rasterized=True,dpi=300)
                else:
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_Qphi.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_Qphi.pdf',rasterized=True,dpi=300)
        if(display=='full'):
            fig,ax=plt.subplots(1)
            cax=ax.matshow(Uphi,cmap=cmap,extent=[-xmax,xmax,-ymax,ymax],origin='lower')
            cb=fig.colorbar(cax)
            #ax.set_xticklabels(axe)
            #ax.set_yticklabels(axe)
            plt.title('Uphi ('+unitI+')'+titletheta)
            plt.xlabel('Offset x ('+resunit+')')
            plt.ylabel('Offset y ('+resunit+')')
            if(rec==1):
                if(gif==1):
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+suffixe+'_Uphi.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+suffixe+'_Uphi.pdf',rasterized=True,dpi=300)
                else:
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_Uphi.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_Uphi.pdf',rasterized=True,dpi=300)
        if(display=='full'):
            fig,ax=plt.subplots(1)
            cax=ax.matshow(QU[:,:,0],cmap=cmap,extent=[-xmax,xmax,-ymax,ymax],origin='lower')
            cb=fig.colorbar(cax)
            #ax.set_xticklabels(axe)
            #ax.set_yticklabels(axe)
            plt.title('Q ('+unitI+')'+titletheta)
            plt.xlabel('Offset x ('+resunit+')')
            plt.ylabel('Offset y ('+resunit+')')
            if(rec==1):
                if(gif==1):
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+suffixe+'_Q.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+suffixe+'_Q.pdf',rasterized=True,dpi=300)
                else:
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_Q.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_Q.pdf',rasterized=True,dpi=300)
        if(display=='full'):
            fig,ax=plt.subplots(1)
            cax=ax.matshow(QU[:,:,1],cmap=cmap,extent=[-xmax,xmax,-ymax,ymax],origin='lower')
            cb=fig.colorbar(cax)
            #ax.set_xticklabels(axe)
            #ax.set_yticklabels(axe)
            plt.title('U ('+unitI+')'+titletheta)
            plt.xlabel('Offset x ('+resunit+')')
            plt.ylabel('Offset y ('+resunit+')')
            if(rec==1):
                if(gif==1):
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+suffixe+'_U.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+suffixe+'_U.pdf',rasterized=True,dpi=300)
                else:
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_U.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_U.pdf',rasterized=True,dpi=300)
        if(display=='full'):
            fig,ax=plt.subplots(1)
            cax=ax.matshow(V,cmap=cmap,extent=[-xmax,xmax,-ymax,ymax],origin='lower')
            cb=fig.colorbar(cax)
            #ax.set_xticklabels(axe)
            #ax.set_yticklabels(axe)
            plt.title('V ('+unitI+')'+titletheta)
            plt.xlabel('Offset x ('+resunit+')')
            plt.ylabel('Offset y ('+resunit+')')
            if(rec==1):
                if(gif==1):
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+suffixe+'_V.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+suffixe+'_V.pdf',rasterized=True,dpi=300)
                else:
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_V.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_V.pdf',rasterized=True,dpi=300)
        if(display=='full'):
            fig,ax=plt.subplots(1)
            cax=ax.matshow(thetas_article,cmap=cmap_art,norm=norm_art,extent=[-xmax,xmax,-ymax,ymax],origin='lower')
            cb=fig.colorbar(cax)
            #ax.set_xticklabels(axe)
            #ax.set_yticklabels(axe)
            plt.title('Difference to centrosymmetric (degree)'+titletheta)
            plt.xlabel('Offset x ('+resunit+')')
            plt.ylabel('Offset y ('+resunit+')')
            if(rec==1):
                if(gif==1):
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+suffixe+'_theta-art.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+suffixe+'_theta-art.pdf',rasterized=True,dpi=300)
                else:
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_theta-art.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_theta-art.pdf',rasterized=True,dpi=300)
        if(display=='full'):
            cmap,norm,mappable=g_cb.colorbar(cmap=cmap,cm_min=-20,cm_max=0)
            fig,ax=plt.subplots(1)
            cax=ax.matshow(np.log10(Prob),norm=norm,cmap=cmap,extent=[-xmax,xmax,-ymax,ymax],origin='lower')
            cb=fig.colorbar(cax)
            #ax.set_xticklabels(axe)
            #ax.set_yticklabels(axe)
            plt.title('Probability'+titletheta)
            plt.xlabel('Offset x ('+resunit+')')
            plt.ylabel('Offset y ('+resunit+')')
            if(rec==1):
                if(gif==1):
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+suffixe+'_logProba.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+suffixe+'_logProba.pdf',rasterized=True,dpi=300)
                else:
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_logProba.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_logProba.pdf',rasterized=True,dpi=300)
        if(display=='polar' or display=='full'):
            fig,ax=plt.subplots(1)
            cax=ax.matshow(Pc*100,cmap=cmapV,norm=normV,extent=[-xmax,xmax,-ymax,ymax],origin='lower')
            cb=fig.colorbar(cax)
            #ax.set_xticklabels(axe)
            #ax.set_yticklabels(axe)
            plt.title('Circular degree of polarisation (%)'+titletheta)
            plt.xlabel('Offset x ('+resunit+')')
            plt.ylabel('Offset y ('+resunit+')')
            if(rec==1):
                if(gif==1):
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+suffixe+'_Pc.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+suffixe+'_Pc.pdf',rasterized=True,dpi=300)
                else:
                    if(saveformat=='png'):
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_Pc.png',dpi=600)
                    else:
                        fig.savefig(outdir+filename+str(m+1)+suffixe+'_Pc.pdf',rasterized=True,dpi=300)
                
        ## Compute the Ip-theta-P map with vectors
        
        #if(resunit=='AU' or resunit=='au' or resunit=='UA' or resunit=='ua'):
        #    resimage
        #elif(resunit=='pc'):
        #    resimage
        #else:
        #    resimage        
        
        if(vectormode>0):
            if(outname==''):
                #plot_vector(Ip,P,theta*np.pi/180.,title='Polarized Intensity (J)',cmap=cmap,fact=vectormode,rec=rec,directory=directory,filename=filename)
                #plot_vector(P*100,P,theta*np.pi/180.,title='Degree of polarization (%)',cmap=cmap,fact=vectormode,rec=rec,directory=directory,filename=filename)
                plot_vector(logI,P,thetaplot*np.pi/180.,title='Intensity (log10('+unitI+'))'+titletheta,suffixe=suffixe+'_logI',cmap=cmaplog,norm=normlog,res=res,rec=rec,directory=outdir,filename=filename,resunit=resunit,saveformat=saveformat)
                plot_vector(ndiffmap,P,thetaplot*np.pi/180.,title='Averaged number of scatterings'+titletheta,suffixe=suffixe+'_ndiff',cmap=cmap,norm=normdiff,res=res,rec=rec,directory=outdir,filename=filename,resunit=resunit,saveformat=saveformat)
                plot_vector(nphotstateff,P,thetaplot*np.pi/180.,title='Number of effective packets per pixel'+titletheta,suffixe=suffixe+'_nphoteff',cmap=cmapeff,norm=normeff,res=res,rec=rec,directory=outdir,filename=filename,resunit=resunit,saveformat=saveformat)
                plot_vector(Ip,P,thetaplot*np.pi/180.,title='Polarised intensity ('+unitI+')'+titletheta,suffixe=suffixe+'_Ip',cmap=cmap,res=res,rec=rec,directory=outdir,filename=filename,resunit=resunit,saveformat=saveformat)
                plot_vector(logI,P,thetaplot*np.pi/180.,title='Intensity (log10('+unitI+'))'+titletheta,suffixe=suffixe+'_clogI',contour=1,curve_range=[minlog,maxlog],cmap=cmaplog,norm=normlog,res=res,rec=rec,directory=outdir,filename=filename,resunit=resunit,saveformat=saveformat)
                plot_vector(ndiffmap,P,thetaplot*np.pi/180.,title='Averaged number of scatterings'+titletheta,suffixe=suffixe+'_cndiff',contour=1,curve_pos=[0.5,1.5,2.5],cmap=cmap,norm=normdiff,res=res,rec=rec,directory=outdir,filename=filename,resunit=resunit,saveformat=saveformat)
            else:
                plot_vector(logI,P,thetaplot*np.pi/180.,title='Intensity (log10('+unitI+'))'+titletheta,suffixe=suffixe+'_logI',cmap=cmaplog,norm=normlog,res=res,rec=rec,directory=outdir,filename=outname,resunit=resunit,saveformat=saveformat)
                plot_vector(ndiffmap,P,thetaplot*np.pi/180.,title='Averaged number of scatterings'+titletheta,suffixe=suffixe+'_ndiff',cmap=cmap,norm=normdiff,res=res,rec=rec,directory=outdir,filename=outname,resunit=resunit,saveformat=saveformat)
                plot_vector(nphotstateff,P,thetaplot*np.pi/180.,title='Number of effective packets per pixel'+titletheta,suffixe=suffixe+'_nphoteff',cmap=cmapeff,norm=normeff,res=res,rec=rec,directory=outdir,filename=outname,resunit=resunit,saveformat=saveformat)
                plot_vector(Ip,P,thetaplot*np.pi/180.,title='Polarised intensity ('+unitI+')'+titletheta,suffixe=suffixe+'_Ip',cmap=cmap,res=res,rec=rec,directory=outdir,filename=outname,resunit=resunit,saveformat=saveformat)
                plot_vector(logI,P,thetaplot*np.pi/180.,title='Intensity (log10('+unitI+'))'+titletheta,suffixe=suffixe+'_clogI',contour=1,curve_range=[minlog,maxlog],cmap=cmaplog,norm=normlog,res=res,rec=rec,directory=outdir,filename=outname,resunit=resunit,saveformat=saveformat)
                plot_vector(ndiffmap,P,thetaplot*np.pi/180.,title='Averaged number of scatterings'+titletheta,suffixe=suffixe+'_cndiff',contour=1,curve_pos=[0.5,1.5,2.5],cmap=cmap,norm=normdiff,res=res,rec=rec,directory=outdir,filename=outname,resunit=resunit,saveformat=saveformat)
            #plot_vector(P*100,P,thetaplot*np.pi/180.,title='Degree of polarisation (%)',suffixe=suffixe+'_p',cmap='jet',res=res,rec=rec,directory=outdir,filename=filename,resunit=resunit,saveformat=saveformat)

def make_dir(newpath):
    if not os.path.exists(newpath):
        os.makedirs(newpath)

def open_phot(path):
    with open(path, 'rb') as f:
        reader = csv.reader(f, delimiter='\t')
        your_list = list(reader)
    return your_list
"""    
def open_phot(path):
    tab = np.genfromtxt(path)
    return tab
"""
def select_phots(tab, lam_0, lam_1):
    new_tab = []
    for e in tab:
        el = float(e[12])
        flag = (el > lam_0)*(el < lam_1)
        if flag:
            new_tab.append(e)
    return new_tab
    
def make_copy_select_phots(path_read, lam_0, lam_1, path_save='default'):
    new_tab = select_phots(open_phot(path_read), lam_0, lam_1)
    if path_save is 'default':
        path_save = path_read + str(lam_0) + '_' + str(lam_1)
    with open(path_save,'wb') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(new_tab)
        
def make_ifu(path_read, bins, res_im, im_size=3, dir_ghost="./dir_ghost/", resunit='pc', thetaobs=90, obj='silicates', dtheta=5):
    if os.path.exists(dir_ghost):
        print("Error : ghost directory already exists!!")
        rm = raw_input("Do you want to remove it? (y/[n]) ( /!\ Danger /!\)")
        if rm is 'y':
            rmtree(dir_ghost)
        else:
            return 'Error', 'Error'
    make_dir(dir_ghost)
    ifu = []
    lams = []
    i = 0
    plt.ioff()
    for lam_0, lam_1 in bins:
        make_copy_select_phots(path_read, lam_0, lam_1, path_save=dir_ghost+'/'+str(lam_0)+'-'+str(lam_1)+'.dat')   
        plot_image(str(lam_0)+'-'+str(lam_1), path=dir_ghost, outdir=dir_ghost+'/', resimage=res_im, resunit=resunit, thetaobs=thetaobs, obj=obj, dtheta=dtheta, xmax_forced=im_size, ymax_forced=im_size, rec=1, outname=str(i)) 
        ifu.append(10**data_fits(dir_ghost+str(i)+'_logI.fits'))
        lams.append((lam_0+lam_1)/2)
        i += 1
        plt.close('all')
    plt.ion()
    x = np.arange(-im_size, im_size+res_im, res_im)
    y = np.arange(-im_size, im_size+res_im, res_im)
    xylam = np.meshgrid(x, lams, y)
    rmtree(dir_ghost)
    ifu = np.array(ifu)
    return ifu, xylam

def data_fits(file_name):
    return fits.open(file_name)[0].data

def merge_phots(filerootname, **kwargs):
    pargs=plot_args(kwargs)
    try:
        filelist=np.sort(os.listdir(pargs['path']))
        
#        print 'found files:',filelist
    except:
        print('No files found')
        return 0
    tab = []
    for filename in filelist:
        if(filerootname in filename):
            if ('_T_' not in filename and 'rho' not in filename):
                if(float(filename[-3-len(pargs['file_ext']):-len(pargs['file_ext'])])>=pargs['thetamin'] and float(filename[-3-len(pargs['file_ext']):-len(pargs['file_ext'])])<=pargs['thetamax']):
                    f = file(pargs['path']+filename,'r')
                    r = csv.reader(f, delimiter='\t')
                    for row in r: #read the file line per line
                        tab.append(row)
    return tab
            
                
def display_IFU(file_root_name, path, bins, res_im, im_size=3, dir_ghost="./dir_ghost/", dir_ghost2="./dir_ghost2/", resunit='pc', thetaobs=90, obj='silicates', dtheta=5, norm='lin', min_max_ratio = 1e-5,file_ext='_phot.dat'):
    """ 
    Example of use: display_IFU('t0b-dimensioned_3000000_001_','/home/pvermot/Documents/2019/montagn_bidouille/OUTPUT_tycho/Output/', [[2e-6,2.1e-6],[2.1e-6,2.2e-6],[2.2e-6,2.3e-6],[2.3e-6,2.4e-6]], 0.125, norm='log', im_size=5, dtheta=5, min_max_ratio = 1e-4)
    """
    if os.path.exists(dir_ghost2):
        print("Error : ghost directory already exists!!")
        rm = raw_input("Do you want to remove it? (y/[n]) ( /!\ Danger /!\)")
        if rm is 'y':
            rmtree(dir_ghost2)
        else:
            return 'Error', 'Error'
        return 'Error', 'Error'
    make_dir(dir_ghost2)
    phots = merge_phots(file_root_name, path=path, thetaobs=thetaobs, dtheta=dtheta,file_ext=file_ext)    
    with open(dir_ghost2+'phots_tmp','wb') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(phots)
    ifu, xylam = make_ifu(dir_ghost2+'phots_tmp', bins, res_im, im_size = im_size, dir_ghost=dir_ghost, resunit=resunit, thetaobs=thetaobs, obj=obj, dtheta=dtheta)
    rmtree(dir_ghost2)
    xylam = np.array(xylam)
    for im, mes, bi in zip(ifu, xylam[::2].swapaxes(0,1), bins):
        maxi = np.max(im)
        if norm is 'lin':
            plt.matshow(im, extent = [np.min(mes), np.max(mes), np.min(mes), np.max(mes)], origin='lower', vmin = maxi*min_max_ratio)
        if norm is 'log':
            plt.matshow(im, extent = [np.min(mes), np.max(mes), np.min(mes), np.max(mes)], norm = LogNorm(), origin='lower', vmin = maxi*min_max_ratio)            
        plt.title("Spectral range : "+str(bi[0])+' m - '+str(bi[1])+' m')
        plt.colorbar()

def compute_tau(model,wvl,angle='x',fromto=[],cst=[]):
    """ 
    Compute the value of tau
    """
    
    if(cst==[]):
        cst=mc.Constant()
    #rmin=0.005*1e-6
    #rmax=0.25*1e-6
    tau=0
    tau_max=0
    tau_species=[]
    j=0
    for grain in model.map.dust:
        if(True): #if(grain.usegrain==1):
            #print grain.name
            Qt=np.transpose(grain.Qext)
            xt=np.transpose(grain.x_M)
            iwvl=np.argsort(np.argsort(np.append(grain.wvlvect,wvl)))[len(grain.wvlvect)]-1
            #print iwvl,grain.wvlvect[iwvl],grain.wvlvect[iwvl+1]
            rmin=grain.rmin
            rmax=grain.rmax
            alpha=grain.alpha
            #a=[]
            Q=[]
            #print np.shape(grain.x_M)
            #print np.shape(xt)
            #print len(xt[:,iwvl]),np.shape(xt)[0]
            for i in range(np.shape(xt)[0]):
                #print i
                #a.append(loginterp([grain.wvlvect[iwvl],grain.wvlvect[iwvl+1]],[xt[i,iwvl],xt[i,iwvl+1]],wvl)*wvl/(2.*np.pi))
                Q.append(mpol.loginterp([grain.wvlvect[iwvl],grain.wvlvect[iwvl+1]],[Qt[i,iwvl],Qt[i,iwvl+1]],wvl))
            a=grain.sizevect
            ng_max=0.
            ng=0.
            jj=grain.number
            #xsub=int(model.Rsub[j]/model.map.res)
            xsub=model.Rsub[j]/model.map.res
            #print xsub,model.Rsub[j],model.map.res
            
            #print a,Q
            if(fromto==[]):
                start=[model.map.N,model.map.N,model.map.N]
                length=model.map.N
            else:
                start=fromto[0]
                length=fromto[1]
            #lensub=0
            if(angle=='x'):
                for i in range(length):
                    #ng_max+=((model.map.grid[start[0]+i][start[1]][start[2]].rho[j])/(length)) #the -1 stand for the middle cell which is empty
                    ng_max+=model.map.grid[start[0]+i][start[1]][start[2]].rho[jj] #the -1 stand for the middle cell which is empty
                    #if(i>=xsub):
                    if((start[0]+i-model.map.N)*(start[0]+i-model.map.N)+(start[1]-model.map.N)*(start[1]-model.map.N)+(start[2]-model.map.N)*(start[2]-model.map.N)>=xsub*xsub):
                        #ng+=((model.map.grid[start[0]+i][start[1]][start[2]].rho[j])/(lensub))
                        ng+=model.map.grid[start[0]+i][start[1]][start[2]].rho[jj]
                        #lensub+=1
                        #print lensub
                    #print ng_max
            elif(angle=='y'):
                for i in range(length):
                    #ng_max+=((model.map.grid[start[0]][start[1]+i][start[2]].rho[j])/(length))
                    ng_max+=model.map.grid[start[0]][start[1]+i][start[2]].rho[jj]
                    #if(i>=xsub):
                    if((start[0]-model.map.N)*(start[0]-model.map.N)+(start[1]+i-model.map.N)*(start[1]+i-model.map.N)+(start[2]-model.map.N)*(start[2]-model.map.N)>=xsub*xsub):
                        ng+=model.map.grid[start[0]][start[1]+i][start[2]].rho[jj]
                        #lensub+=1
                        #ng+=((model.map.grid[start[0]][start[1]+i][start[2]].rho[j])/(lensub))
                    #print ng_max
            elif(angle=='z'):
                for i in range(length):
                    #ng_max+=((model.map.grid[start[0]][start[1]][start[2]+i].rho[j])/(length))
                    ng_max+=model.map.grid[start[0]][start[1]][start[2]+i].rho[jj]
                    #if(i>=xsub):
                    if((start[0]-model.map.N)*(start[0]-model.map.N)+(start[1]-model.map.N)*(start[1]-model.map.N)+(start[2]+i-model.map.N)*(start[2]+i-model.map.N)>=xsub*xsub):
                        ng+=model.map.grid[start[0]][start[1]][start[2]+i].rho[jj]
                        #lensub+=1
                        #ng+=((model.map.grid[start[0]][start[1]][start[2]+i].rho[j])/(lensub))
                    #print ng_max
            #print ng_max, length
            #print ng, lensub
            #ng=ng/max(1,lensub)
            #print ng

            #if(jj!=3):
            if(grain.typeg!='electrons' and 'pah' not in grain.typeg):
                Qaa=0
                for i in range(len(Q)):
                    if(i<len(Q)-1):
                        da=a[i+1]-a[i]
                    else:
                        da=a[i]-a[i-1]
                    if(a[i]<rmax and a[i]>rmin):
                        #print Q[i]
                        #Qaa+=Q[i]*da*2.5/(a[i]**1.5*(1./(rmin**2.5)-1./(rmax**2.5)))
                        if(alpha==-1):
                            Qaa+=Q[i]*da*a[i]/np.log(rmax/rmin)
                        else:
                            Qaa+=Q[i]*da*(alpha+1.)*a[i]**(2.+alpha)/(rmax**(1+alpha)-rmin**(1+alpha))   
                        #print Qaa
                        #Qaa+=Q[i]*da*da*2.5/(2.*a[i]*a[i]*a[i]*(1./np.sqrt(rmin)-1./np.sqrt(rmax))*(1./(rmin**2.5)-1./(rmax**2.5)))
                        #Qaa+=Q[i]*da*2.5/(2.*a[i]*a[i]*(1./np.sqrt(rmin)-1./np.sqrt(rmax))*(1./(rmin**2.5)-1./(rmax**2.5)))
            else:
                if(grain.typeg=='electrons'):
                    Qaa=cst.sigt/np.pi
                elif('pah' in grain.typeg):
                    Qaa=mu.KPAH(grain,wvl)/np.pi
            L=model.map.Rmax-model.Rsub[j]
            Lmax=model.map.Rmax
            #print L, Lmax
            #print ng,Qaa,L
            #tau+=ng*Qaa*L*np.pi
            #tau_max+=ng_max*Qaa*Lmax*np.pi
            tau+=ng*Qaa*np.pi*model.map.res
            tau_max+=ng_max*Qaa*np.pi*model.map.res
            tau_species.append([ng_max*Qaa*model.map.res*np.pi,ng*Qaa*model.map.res*np.pi])
            j+=1
    return tau_max,tau,tau_species


def compute_tau_old(model,Q,x_M,wvl,rmin,rmax,alpha):
    """ 
    Compute the value of tau
    """
    
    #rmin=0.005*1e-6
    #rmax=0.25*1e-6
    a=x_M*wvl/(2.*np.pi)
    ng_max=0.
    ng=0.
    xsub=int(model.Rsub[0]/model.map.res)
    
    #print a,Q
    for i in range(model.map.N):
        ng_max+=((model.map.grid[model.map.N+i][model.map.N][model.map.N].rho[0])/(model.map.N))
        if(i>=xsub):
            ng+=((model.map.grid[model.map.N+i][model.map.N][model.map.N].rho[0])/(model.map.N-xsub))
        #print ng_max
    
    Qaa=0
    for i in range(len(Q)):
        if(i<len(Q)-1):
            da=a[i+1]-a[i]
        else:
            da=a[i]-a[i-1]
        if(a[i]<rmax and a[i]>rmin):
            #print Q[i]
            Qaa+=Q[i]*da*2.5/(a[i]**1.5*(1./(rmin**2.5)-1./(rmax**2.5)))
            #print Qaa
            #Qaa+=Q[i]*da*da*2.5/(2.*a[i]*a[i]*a[i]*(1./np.sqrt(rmin)-1./np.sqrt(rmax))*(1./(rmin**2.5)-1./(rmax**2.5)))
            #Qaa+=Q[i]*da*2.5/(2.*a[i]*a[i]*(1./np.sqrt(rmin)-1./np.sqrt(rmax))*(1./(rmin**2.5)-1./(rmax**2.5)))
    L=model.map.Rmax-model.Rsub[0]
    Lmax=model.map.Rmax
    #print ng,Qaa,L
    tau=ng*Qaa*L*np.pi
    tau_max=ng_max*Qaa*Lmax*np.pi
    return tau_max,tau


def plot_vector(data,P,theta,title=' ',suffixe='',contour=0,cmap='jet',norm=[],curve_pos=[],curve_range=[],rec=0,directory=[],filename='test',res=[],resunit='',saveformat='pdf'):
    """
    Plot the x data overploted by vectors of angle theta and lentgh P
    x,P and theta must be square array of the same lentgh
    """

    print('Creating vectors')
    Nx=len(data[0])
    Ny=len(data[1])
    if(res==[]):
        res=1
        resunit='pixels'
    xmax=Nx*res
    ymax=Ny*res
    #fact=6
    #theta-=90
    vect=np.zeros([Nx*Ny,4])
    for i in range(Nx):
        for j in range(Ny):
            #vect[i*Ny+j][0]=(-Ny/2.+j+0.5-0.4*P[i][j]*np.sin(theta[i][j]))*res
            #vect[i*Ny+j][1]=(-Nx/2.+i+0.5-0.4*P[i][j]*np.cos(theta[i][j]))*res
            #vect[i*Ny+j][2]=(-Ny/2.+j+0.5+0.4*P[i][j]*np.sin(theta[i][j]))*res
            #vect[i*Ny+j][3]=(-Nx/2.+i+0.5+0.4*P[i][j]*np.cos(theta[i][j]))*res
            vect[i*Ny+j][0]=(-Ny/2.+j+0.5-0.4*P[i][j]*np.sin(theta[i][j]))*res
            vect[i*Ny+j][1]=(-Nx/2.+i+0.5+0.4*P[i][j]*np.cos(theta[i][j]))*res
            vect[i*Ny+j][2]=(-Ny/2.+j+0.5+0.4*P[i][j]*np.sin(theta[i][j]))*res
            vect[i*Ny+j][3]=(-Nx/2.+i+0.5-0.4*P[i][j]*np.cos(theta[i][j]))*res

    fig,ax=plt.subplots(1)
    if(contour==0):
        if(norm==[]):
            cax=ax.matshow(data,cmap=cmap,extent=[-xmax*0.5,xmax*0.5,-ymax*0.5,ymax*0.5],origin='lower') #,interpolation=interpolation,extent=extent)
        else:
            cax=ax.matshow(data,cmap=cmap,norm=norm,extent=[-xmax*0.5,xmax*0.5,-ymax*0.5,ymax*0.5],origin='lower') #,interpolation=interpolation,extent=extent)
        #plt.matshow(data,cmap=cmap)
        cb=fig.colorbar(cax)
    else:
        x=np.linspace(-xmax*0.5,xmax*0.5,Nx)
        y=np.linspace(-ymax*0.5,ymax*0.5,Ny)
        if(curve_range==[] and curve_pos==[]):
            cax=plt.contour(x,y,data,colors='b')
        elif(curve_pos!=[]):
            cax=plt.contour(x,y,data,colors='b',levels=curve_pos)
        else:
            levels=np.linspace(curve_range[0],curve_range[1],5)
            cax=plt.contour(x,y,data,colors='b',levels=levels)
        plt.clabel(cax,fontsize=9,inline=1)
        cb=fig.colorbar(cax)
    #plt.colorbar()
    plt.title(title)
    progbar=g_pgb.Patiencebar(valmax=Nx*Ny,up_every=1)
    for i in range(Nx*Ny):
        #if(((i+1))%(max(1,(N*N)/100))==0):
        #    progbar.update(int(round((i+1)*100./(N*N))))
        progbar.update()
        ax.plot((vect[i][0],vect[i][2]),(vect[i][1],vect[i][3]),color='black')
    #plt.plot((np.transpose(vect_test)[0],np.transpose(vect_test)[1]),(np.transpose(vect_test)[2],np.transpose(vect_test)[3]))
    #print (vect_test[0],vect_test[2]),(vect_test[1],vect_test[3])
    #print (np.transpose(vect_test)[0],np.transpose(vect_test)[1]),(np.transpose(vect_test)[2],np.transpose(vect_test)[3])    
    #plt.axis([0,N,0,N])
    plt.axis([-xmax*0.5,xmax*0.5,-ymax*0.5,ymax*0.5])
    #print ax.get_xticks()
    #print ax.get_xticklines()
    #print ax.get_xticks()/fact
    #print str(ax.get_xticks()/fact)
    #if(xmax!=[]):
    #    axe=[str(int(10*i)/10.) for i in np.linspace(-2*xmax/(len(ax.get_xticks())-2),2*xmax,len(ax.get_xticks()))]
    #    ax.set_xticklabels(axe)
    #    ax.set_yticklabels(axe)
    #else:
    #    ax.set_xticklabels([str(int(10*i)/10.) for i in ax.get_xticks()/fact])
    #    ax.set_yticklabels([str(int(10*i)/10.) for i in ax.get_xticks()/fact])
        
    #if(resunit==''):
    #    plt.xlabel('Offset x (pixels)')
    #    plt.ylabel('Offset y (pixels)')
    #else:    
    plt.xlabel('Offset x ('+resunit+')')
    plt.ylabel('Offset y ('+resunit+')')
    #ax.xticks=ax.get_xticks()/fact
    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.show()
    if(rec==1):
        if(saveformat=='png'):
            fig.savefig(directory+filename+suffixe+'_vect.png',dpi=600)
        else:
            fig.savefig(directory+filename+suffixe+'_vect.pdf',rasterized=True,dpi=300)
    #plt.matshow(data)
    #plt.colorbar()
    #plt.title(title)
    #X,Y,U,V = zip(*vect)
    #ax = plt.gca()
    #ax.quiver(X,Y,U,V,angles='xy',scale_units='xy',scale=1,pivot='middle')
    ##ax.set_xlim([-1,10])
    ##ax.set_ylim([-1,10])
    #plt.draw()
    #plt.show()


def plot_vector_old(x,P,theta,fact=1,title=' ',suffixe='',cmap='jet',norm=[],rec=0,directory=[],filename='test',xmax=[],resunit=''):
    """
    Plot the x data overploted by vectors of angle theta and lentgh P
    x,P and theta must be square array of the same lentgh
    """

    print('Creating vectors')
    N=len(x[0])
    fact=6
    #theta-=90
    data=np.zeros([fact*N,fact*N])
    vect=np.zeros([N*N,4])
    vect_test=np.zeros([N*N,4])
    for i in range(N):
        for j in range(N):
            vect_test[i*N+j][0]=fact*j+2.5-2.5*P[i][j]*np.sin(theta[i][j])
            vect_test[i*N+j][1]=fact*i+2.5-2.5*P[i][j]*np.cos(theta[i][j])
            vect_test[i*N+j][2]=fact*j+2.5+2.5*P[i][j]*np.sin(theta[i][j])
            vect_test[i*N+j][3]=fact*i+2.5+2.5*P[i][j]*np.cos(theta[i][j])
            vect[i*N+j][0]=fact*j+2.5
            vect[i*N+j][1]=fact*i+2.5
            vect[i*N+j][2]=5.*P[i][j]*np.sin(theta[i][j])
            vect[i*N+j][3]=5.*P[i][j]*np.cos(theta[i][j])
            for k in range(fact):
                for l in range(fact):
                    data[i*fact+k][j*fact+l]=x[i][j]

    fig,ax=plt.subplots(1)
    if(norm==[]):
        cax=ax.matshow(data,cmap=cmap) #,interpolation=interpolation,extent=extent)
    else:
        cax=ax.matshow(data,cmap=cmap,norm=norm) #,interpolation=interpolation,extent=extent)
    #plt.matshow(data,cmap=cmap)
    cb=fig.colorbar(cax)
    #plt.colorbar()
    plt.title(title)
    progbar=g_pgb.Patiencebar(valmax=N*N,up_every=1)
    for i in range(N*N):
        #if(((i+1))%(max(1,(N*N)/100))==0):
        #    progbar.update(int(round((i+1)*100./(N*N))))
        progbar.update()
        ax.plot((vect_test[i][0],vect_test[i][2]),(vect_test[i][1],vect_test[i][3]),color='black')
    #plt.plot((np.transpose(vect_test)[0],np.transpose(vect_test)[1]),(np.transpose(vect_test)[2],np.transpose(vect_test)[3]))
    #print (vect_test[0],vect_test[2]),(vect_test[1],vect_test[3])
    #print (np.transpose(vect_test)[0],np.transpose(vect_test)[1]),(np.transpose(vect_test)[2],np.transpose(vect_test)[3])    
    #plt.axis([0,N,0,N])
    plt.axis([0,fact*N,0,fact*N])
    #print ax.get_xticks()
    #print ax.get_xticklines()
    #print ax.get_xticks()/fact
    #print str(ax.get_xticks()/fact)
    if(xmax!=[]):
        axe=[str(int(10*i)/10.) for i in np.linspace(-2*xmax/(len(ax.get_xticks())-2),2*xmax,len(ax.get_xticks()))]
        ax.set_xticklabels(axe)
        ax.set_yticklabels(axe)
    else:
        ax.set_xticklabels([str(int(10*i)/10.) for i in ax.get_xticks()/fact])
        ax.set_yticklabels([str(int(10*i)/10.) for i in ax.get_xticks()/fact])
        
    if(resunit==''):
        plt.xlabel('Offset x (pixels)')
        plt.ylabel('Offset y (pixels)')
    else:    
        plt.xlabel('Offset x ('+resunit+')')
        plt.ylabel('Offset y ('+resunit+')')
    #ax.xticks=ax.get_xticks()/fact
    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.show()
    if(rec==1):
        fig.savefig(directory+filename+suffixe+'_vect.png',dpi=600)
    #plt.matshow(data)
    #plt.colorbar()
    #plt.title(title)
    #X,Y,U,V = zip(*vect)
    #ax = plt.gca()
    #ax.quiver(X,Y,U,V,angles='xy',scale_units='xy',scale=1,pivot='middle')
    ##ax.set_xlim([-1,10])
    ##ax.set_ylim([-1,10])
    #plt.draw()
    #plt.show()


def plotT(model,l,prefixe,unity='pc',rec=1,display=1):
    cst=mc.Constant()
    print("Computing T map")
    N=model.map.N
    grilleT=np.zeros((N,N))
    grilleN=np.zeros((N,N))
    for i in range(2*N):
        for j in range(2*N):
            for k in range(2*N):
                x=np.int(np.sqrt((i-N+0.5)*(i-N+0.5)+(j-N+0.5)*(j-N+0.5)))
                y=np.int(np.abs(k-N+0.5))
                if (x<N and y<N):
                    grilleT[y][x]+=model.map.grid[i][j][k].T
                    grilleN[y][x]+=1

    for i in range(N):
        for j in range(N):
            if(grilleN[i][j]==0):
                grilleN[i][j]=1
    grilleT=grilleT/grilleN
    if(l=='i'):
        filename="Output/"+prefixe+"_T_init"
    elif(l=='f'):
        filename="Output/"+prefixe+"_T_final"
    else:
        filename="Output/"+prefixe+"_T_"+str(l)
    if(rec==1):
        f = open(filename+'.dat','w')
        for i in range(N):
            for j in range(N):
                if(j<N-1):
                    f.write("%f\t"%grilleT[i][j])
                else:
                    f.write("%f\n"%grilleT[i][j])
        f.close()
    if(display==1):
        Rmax=model.map.Rmax/cst.pc
        if(unity=='au' or unity=='AU' or unity=='UA' or unity=='ua'):
            Rmax*=cst.pc/cst.AU

        fig,ax=plt.subplots(1)
        cax=ax.matshow(grilleT,extent=[0,Rmax,0,Rmax],origin='lower',cmap=cm.jet)
        plt.xlabel('Radius from centre (in '+unity+')')
        plt.ylabel('Radius from centre (in '+unity+')')
        plt.colorbar(cax)
        if(l=='i'):
            plt.title('Initial grain temperature in K')
        elif(l=='f'):
            plt.title('Final grain temperature in K')
        else:
            plt.title('Grain temperature in K')
        plt.show() 
        if(rec==1):
            fig.savefig(filename+'.png',dpi=600)

def plotT_from_dat(filename,res,l,unity='pc'):
    T=np.genfromtxt(filename)
    Rmax=res*np.size(T,1)

    fig,ax=plt.subplots(1)
    cax=ax.matshow(T,extent=[0,Rmax,0,Rmax],origin='lower',cmap=cm.jet)
    plt.xlabel('Radius from centre (in '+unity+')')
    plt.ylabel('Radius from centre (in '+unity+')')
    plt.colorbar(cax)
    if(l=='i'):
        plt.title('Initial grain temperature in K')
    elif(l=='f'):
        plt.title('Final grain temperature in K')
    else:
        plt.title('Grain temperature in K')
    plt.show() 

def plotrho(model,prefixe,unity='pc',rec=1,display=1,nsimu=1,tycho=0):
    cst=mc.Constant()
    if(nsimu==1 or tycho==1):
        print("Computing density map")
    N=model.map.N
    grillerho1=np.zeros((2*N,2*N))
    grillerho2=np.zeros((2*N,2*N))
    mass=0.
    rmax=0.25*1e-6
    rmin=0.005*1e-6
    for i in range(2*N):
        for j in range(2*N):
            for l in range(len(model.map.grid[i][j][N].rho)):
                grillerho1[j][i]+=model.map.grid[i][j][N].rho[l]
                grillerho2[j][i]+=model.map.grid[i][N][j].rho[l]
                for k in range(2*N):
                    mass+=model.map.grid[i][j][k].rho[l]

    mass=mass*model.map.res*model.map.res*model.map.res
    #mass=mass*model.map.Rmax*model.map.Rmax*model.map.Rmax/(N*N*N*8)
    if(nsimu==1 or tycho==1):
        print("Number of dust particles in the model :",mass)
    density=3.3*1e3 #kg/m3 # ou 0.29*1e-3 ? ##cf Vincent Guillet 2008 p~130
    particlemass=density*20./3.*np.pi*(np.sqrt(rmax)-np.sqrt(rmin))/(1./rmin**2.5-1./rmax**2.5)
    mass=mass*particlemass
    if(nsimu==1 or tycho==1):
        print("Total mass of dust :",mass,"kg")
    grillerho1=grillerho1*particlemass
    grillerho2=grillerho2*particlemass

    if(rec==1):
        #filename="Output/"+prefixe+"_density_xy.dat"
        filename="Output/"+prefixe+"_rho_xy.dat"
        f = open(filename,'w')
        for i in range(N):
            for j in range(N):
                if(j<N-1):
                    f.write("%f\t"%grillerho1[i][j])
                else:
                    f.write("%f\n"%grillerho1[i][j])
        f.close()
        #filename="Output/"+prefixe+"_density_xz.dat"
        filename="Output/"+prefixe+"_rho_xz.dat"
        f = open(filename,'w')
        for i in range(N):
            for j in range(N):
                if(j<N-1):
                    f.write("%f\t"%grillerho2[i][j])
                else:
                    f.write("%f\n"%grillerho2[i][j])
        f.close()
    #seuil=np.ones([2*N,2*N])*min(min(grillerho1.reshape(4*N*N)[np.where(grillerho1)[0]][np.where(grillerho1)[1]]),min(grillerho2.reshape(4*N*N)))
    seuil=np.ones([2*N,2*N])*1e-30
    #plt.matshow(seuil)
    #plt.colorbar()
    #plt.title('seuil')
    if(display==1):
        Rmax=model.map.Rmax/cst.pc
        if(unity=='au' or unity=='AU' or unity=='UA' or unity=='ua'):
            Rmax*=cst.pc/cst.AU
        if(model.usemodel==1 or model.usemodel==2 or model.usemodel==3):
            cmap,norm,mappable=g_cb.colorbar(cmap="jet",cm_min=-20,cm_max=-10)
        else:
            cmap,norm,mappable=g_cb.colorbar(cmap="jet",cm_min=-25,cm_max=-15)
        fig,ax=plt.subplots(1)
        cax=ax.matshow(np.log10(grillerho1+seuil),norm=norm,extent=[-Rmax,Rmax,-Rmax,Rmax],origin='lower')
        plt.xlabel('Distance from centre (in '+unity+')')
        plt.ylabel('Distance from centre (in '+unity+')')
        plt.colorbar(mappable)
        #plt.title('Grain density - xy plan (kg/m3)')
        plt.title('Grain density - xy plan (log10(kg/m3))')
        plt.show() 
        fig,ax=plt.subplots(1)
        cax=ax.matshow(np.log10(grillerho2+seuil),norm=norm,extent=[-Rmax,Rmax,-Rmax,Rmax],origin='lower')
        plt.xlabel('Distance from centre (in '+unity+')')
        plt.ylabel('Distance from centre (in '+unity+')')
        plt.colorbar(mappable)
        #plt.title('Grain density - xz plan (kg/m3)')
        plt.title('Grain density - xz plan (log10(kg/m3))')
        plt.show() 

def plotrhodustype(model,prefixe,unity='pc',rec=1,display=1,nsimu=1,cluster=0):
    cst=mc.Constant()
    if(nsimu==1 or cluster==1):
        print("Computing density map")
    #poussiere=['silicates','graphites_ortho','graphites_para','electrons','pah']
    cst=mc.Constant()
    for l in range(len(model.map.grid[0][0][0].rho)):
        N=model.map.N
        grillerho1=np.zeros((2*N,2*N))
        grillerho2=np.zeros((2*N,2*N))
        mass=0.
        rmax=model.map.dust[l].rmax #0.25*1e-6
        rmin=model.map.dust[l].rmin #0.005*1e-6
        density=model.map.dust[l].density
        for i in range(2*N):
            for j in range(2*N):
                #for l in range(len(model.map.grid[i][j][N].rho)):
                grillerho1[j][i]+=model.map.grid[i][j][N].rho[l]
                grillerho2[j][i]+=model.map.grid[i][N][j].rho[l]
                for k in range(2*N):
                    mass+=model.map.grid[i][j][k].rho[l]

        mass=mass*model.map.res*model.map.res*model.map.res
        #mass=mass*model.map.Rmax*model.map.Rmax*model.map.Rmax/(N*N*N) #8 ou pas 8 ?
        if(nsimu==1 or cluster==1):
            if(mass>0.0):
                #print "Number of dust particles of "+poussiere[l]+" in the model :",mass
                if(model.map.dust[l].typeg=='electrons'):
                    print("Number of "+str(model.map.dust[l])+" in the model :",mass)
                else:
                    print("Number of dust particles of "+str(model.map.dust[l])+" in the model :",mass)
        #density=3.3*1e3 #kg/m3 # ou 0.29*1e-3 ? ##cf Vincent Guillet 2008 p~130
        #density=[3.3*1e3,3.3*1e3,3.3*1e3,0] #kg/m3 # ou 0.29*1e-3 ? ##cf Vincent Guillet 2008 p~130
        masse=0
        if(model.map.dust[l].typeg!='electrons'):
            #if(model.map.dust[l].alpha==-3.5):
            #    particlemass=density*20./3.*np.pi*(np.sqrt(rmax)-np.sqrt(rmin))/(1./rmin**2.5-1./rmax**2.5)
            #else:
            #    print "WARNING : alpha coefficent not supported :",model.map.dust[l].alphagrain
            #    particlemass=0.
            particlemass=density*mu.particle_volume(rmin,rmax,model.map.dust[l].alpha)
            mass=mass*particlemass
            if(nsimu==1 or cluster==1):
                if(mass>0.0):
                    #print "Total mass of "+poussiere[l]+" :",mass,"kg"
                    print("Total mass of "+str(model.map.dust[l])+" :",mass,"kg")
        else:
            if(masse==1):
                particlemass=density
            else:
                particlemass=1
            mass=mass*particlemass
            if(nsimu==1 or cluster==1):
                if(masse==1):
                    print("Total mass of electron :",mass,"kg")
        grillerho1=grillerho1*particlemass
        grillerho2=grillerho2*particlemass

        if(rec==1):
            #filename="Output/"+prefixe+"_density_xy.dat"
            #filename="Output/"+prefixe+'_'+poussiere[l]+"_rho_xy.dat"
            filename="Output/"+prefixe+'_'+str(model.map.dust[l])+"_rho_xy.dat"

            f = open(filename,'w')
            for i in range(N):
                for j in range(N):
                    if(j<N-1):
                        f.write("%f\t"%grillerho1[i][j])
                    else:
                        f.write("%f\n"%grillerho1[i][j])
            f.close()
            #filename="Output/"+prefixe+"_density_xz.dat"
            #filename="Output/"+prefixe+'_'+poussiere[l]+"_rho_xz.dat"
            filename="Output/"+prefixe+'_'+str(model.map.dust[l])+"_rho_xz.dat"
            f = open(filename,'w')
            for i in range(N):
                for j in range(N):
                    if(j<N-1):
                        f.write("%f\t"%grillerho2[i][j])
                    else:
                        f.write("%f\n"%grillerho2[i][j])
            f.close()
        #seuil=np.ones([2*N,2*N])*min(min(grillerho1.reshape(4*N*N)[np.where(grillerho1)[0]][np.where(grillerho1)[1]]),min(grillerho2.reshape(4*N*N)))
        seuil=np.ones([2*N,2*N])*1e-30
        #plt.matshow(seuil)
        #plt.colorbar()
        #plt.title('seuil')
        if(display==1 and model.map.dust[l].usegrain==1):
            plt.show()
            if(l==3):
                seuil=np.ones([2*N,2*N])*1e-50
                #if(model.usemodel==1 or model.usemodel==2 or model.usemodel==3):
                #    cmap,norm,mappable=g_cb.colorbar(cmap="jet",cm_min=-40,cm_max=-20)
                #else:
                #    cmap,norm,mappable=g_cb.colorbar(cmap="jet",cm_min=-45,cm_max=-25)
            #else:

            grilluseless=[]
            if(np.max(grillerho1)!=0):
                useless=np.nonzero(grillerho1!=0)
                #print useless
                for i in range(np.shape(useless)[1]):
                    grilluseless.append(grillerho1[useless[0][i],useless[1][i]])
            if(np.max(grillerho2)!=0):
                useless=np.nonzero(grillerho2)
                #print useless
                for i in range(np.shape(useless)[1]):
                    grilluseless.append(grillerho2[useless[0][i],useless[1][i]])
            #Cbmin=min(np.min(grillerho1[np.nonzero(grillerho1!=0)]),np.min(grillerho2[np.nonzero(grillerho2!=0)]))
            Cbmin=max(np.min(grilluseless)*0.1,seuil[0,0])
            Cbmax=max(np.max(grillerho1),np.max(grillerho2))*10

            Rmax=model.map.Rmax/cst.pc
            if(unity=='au' or unity=='AU' or unity=='UA' or unity=='ua'):
                Rmax*=cst.pc/cst.AU
            #if(True):
            cmap,norm,mappable=g_cb.colorbar(cmap="jet",cm_min=np.log10(Cbmin),cm_max=np.log10(Cbmax))
            #cmap,norm,mappable=g_cb.colorbar(cmap="jet")
            #if(masse==1 or l!=3):
            #    if(model.usemodel==1 or model.usemodel==2 or model.usemodel==3):
            #        cmap,norm,mappable=g_cb.colorbar(cmap="jet",cm_min=-20,cm_max=-10)
            #    else:
            #        cmap,norm,mappable=g_cb.colorbar(cmap="jet",cm_min=-30,cm_max=-20)
            #else:
            #    cmap,norm,mappable=g_cb.colorbar(cmap="jet",cm_min=-5,cm_max=10)
            plt.matshow(np.log10(grillerho1+seuil),norm=norm,extent=[-Rmax,Rmax,-Rmax,Rmax],origin='lower')
            plt.xlabel('Distance from centre (in '+unity+')')
            plt.ylabel('Distance from centre (in '+unity+')')
            plt.colorbar(mappable)
            #plt.title('Grain density - xy plan (kg/m3)')
            if(l!=3 or masse==1):
                #plt.title(poussiere[l]+' density - xy plan (log10(kg/m3))')
                plt.title(str(model.map.dust[l])+' density - xy plan (log10(kg/m3))')
            else:
                plt.title('Electron density - xy plan (log10(e-/m3))')
            plt.show() 
            plt.matshow(np.log10(grillerho2+seuil),norm=norm,extent=[-Rmax,Rmax,-Rmax,Rmax],origin='lower')
            plt.xlabel('Distance from centre (in '+unity+')')
            plt.ylabel('Distance from centre (in '+unity+')')
            plt.colorbar(mappable)
            #plt.title('Grain density - xz plan (kg/m3)')
            if(l!=3 or masse==1):
                #plt.title(poussiere[l]+' density - xz plan (log10(kg/m3))')
                plt.title(str(model.map.dust[l])+' density - xz plan (log10(kg/m3))')
            else:
                plt.title('Electron density - xz plan (log10(e-/m3))')
            plt.show() 

def plot_time(time,message,star=0,hmins=0):
    if(time>60):
        time=time/60.
        unite="min"
        if(time>60):
            time=time/60.
            unite="h"
            if(time>24):
                time=time/24.
                unite="d"
                if(time>365.25):
                    time=time/365.25
                    unite="yr"
    else:
        unite="s"
    if(hmins==1):
        if(unite=="yr"):
            timeyr=int(np.floor(time))
            time=(time-timeyr)*365.25
        if(unite=="yr" or unite=="d"):
            timed=int(np.floor(time))
            time=(time-timed)*24
        if(unite=="yr" or unite=="d" or unite=="h"):
            timeh=int(np.floor(time))
            time=(time-timeh)*60
        if(unite=="yr" or unite=="d" or unite=="h" or unite=="min"):
            timemin=int(np.floor(time))
            time=(time-timemin)*60
        times=int(np.floor(time))
        if(star==1):
            #print "*** ",message,timemessage,' ***'
            if(unite=="yr"):
                print('*** ',message,'%.d' % timeyr,'yr','%.d' % timed,'d','%.d' % timeh,'h','%.d' % timemin,'min','%.2f' % times,'s',' ***')
            elif(unite=="d"):
                print('*** ',message,'%.d' % timed,'d','%.d' % timeh,'h','%.d' % timemin,'min','%.2f' % times,'s',' ***')
            elif(unite=="h"):
                print('*** ',message,'%.d' % timeh,'h','%.d' % timemin,'min','%.2f' % times,'s',' ***')
            elif(unite=="min"):
                print('*** ',message,'%.d' % timemin,'min','%.2f' % times,'s',' ***')
            else:
                print('*** ',message,'%.2f' % times,'s',' ***')
        else:
            #print message,timemessage
            if(unite=="yr"):
                print(message,'%.d' % timeyr,'yr','%.d' % timed,'d','%.d' % timeh,'h','%.d' % timemin,'min','%.2f' % times,'s')
            elif(unite=="d"):
                print(message,'%.d' % timed,'d','%.d' % timeh,'h','%.d' % timemin,'min','%.2f' % times,'s')
            elif(unite=="h"):
                print(message,'%.d' % timeh,'h','%.d' % timemin,'min','%.2f' % times,'s')
            elif(unite=="min"):
                print(message,'%.d' % timemin,'min','%.2f' % times,'s')
            else:
                print(message,'%.2f' % times,'s')
    else:
        if(star==1):
            print("*** ",message,'%.2f' % time,unite,' ***')
        else:
            print(message,'%.2f' % time,unite)


def nphot_detected(filename,path=[],dat=1):
    """
    
    """


    ## Define the work directory
    directory=path
    if(directory==[]):
        if(filename=="merged"):
            f=open("Output/phot_merged.dat",'r')
        else:
            #f=open("Output/"+filename+"_phot.dat",'r')
            if(dat==1):
                f=open("Output/"+filename+".dat",'r')
            else:
                f=open("Output/"+filename,'r')

    else:
        #f=open(hostname+"_Output/"+filename,'r')
        if(dat==1):
            f=open(directory+filename+".dat",'r')
        else:
            f=open(directory+filename,'r')


    ### Read file ###
    tab=[]
    label=[]
    #print "Reading file"
    #i=1
    nocirc=0
    nodt=0
    circ=1
    nphot=0
    for data in f.readlines():
        if(float(data.split()[10])==0):
            nphot+=1
    f.close()
    print(nphot,' packets received')
    return nphot


def plot_diff_angle(filename,normsin=0,suffixe='',path='',dat=1,nang=180):
    """
    
    """    
    ## Define the work directory
    color=['b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k']
    anglerange=[0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180]
    tab_th=[]
    tab_en=[]
    tab_Q=[]
    tab_U=[]
    nphot=0
    for theta in anglerange:
        if(theta<10):
            thetastr='_00'+str(theta)
        elif(theta<100):
            thetastr='_0'+str(theta)
        else:
            thetastr='_'+str(theta)
        if(dat==1):
            f=open(path+filename+thetastr+suffixe+".dat",'r')
        else:
            f=open(path+filename+thetastr+suffixe,'r')
        for data in f.readlines():
            if(float(data.split()[10])==1):
                nphot+=1
                tab_th.append(float(data.split()[1]))
                tab_en.append(float(data.split()[14]))
                tab_Q.append(float(data.split()[3]))
                tab_U.append(float(data.split()[4]))
        f.close()

    tab=np.zeros([2,nang])
    tabpol=np.zeros([4,nang]) #[Q,U,P,theta]
    tab[0,:]=np.arange(nang)*180./nang+0.5
    progbar=g_pgb.Patiencebar(valmax=nphot,up_every=1)
    for i in range(nphot):
        progbar.update()
        th=tab_th[i]/np.pi*nang
        tab[1,th]+=tab_en[i]
        tabpol[0,th]+=tab_Q[i]*tab_en[i]
        tabpol[1,th]+=tab_U[i]*tab_en[i]

    tabpol[2]=np.sqrt(tabpol[0]*tabpol[0]+tabpol[1]*tabpol[1])/tab[1]
    tabpol[3]=-0.5*np.arctan2(tabpol[1],tabpol[0])

    if(normsin==1):
        tab[1]=tab[1]/np.sin(tab[0]*np.pi/180)

    legendpos=0
    plt.figure()
    plt.plot(tab[0],tab[1])
    plt.title('Scattering angle')
    plt.ylabel('Number of photons (arbitrary unit)')
    plt.xlabel('Theta (degrees)')
    plt.show()
    plt.figure()
    plt.polar(tab[0]*np.pi/180,tab[1],color[0]) #,label='after 1 scattering')
    plt.polar(-tab[0]*np.pi/180,tab[1],color[0]+'--')
    #plt.legend(loc=legendpos)
    plt.title('Scattering angle')
    #plt.ylabel('Number of photons (arbitrary unit)')
    plt.xlabel('Theta (degrees)')
    plt.show()

    plt.figure()
    plt.plot(tab[0],tabpol[2]*100)
    plt.title('Polarisation degree')
    plt.ylabel('Polarisation degree (%)')
    plt.xlabel('Theta (degrees)')
    plt.show()
    plt.figure()
    plt.polar(tab[0]*np.pi/180,tabpol[2],color[0]) #,label='after 1 scattering')
    plt.polar(-tab[0]*np.pi/180,tabpol[2],color[0]+'--')
    #plt.legend(loc=legendpos)
    plt.title('Polarisation degree (%)')
    #plt.ylabel('Number of photons (arbitrary unit)')
    plt.xlabel('Theta (degrees)')
    plt.show()

def plotT3d(model, Tmi,Tma):
	print("Computing T map")
	N=model.map.N
	grilleT=np.zeros((N,N,N))
	grilleN=np.zeros((N,N,N))
	for i in range(2*N):
		for j in range(2*N):
			for k in range(2*N):
				x=np.int(i-N+0.5)
				y=np.int(j-N+0.5)
				z=np.int(k-N+0.5)
				grilleT[x][y][z] = model.map.grid[i][j][k].T
				grilleN[x][y][z]+=1
	Rmax=model.map.Rmax/3.e16
	#gr = np.array(model.map.grid.T)
	
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	
	#cax=ax.matshow(grilleT,extent=[0,Rmax,0,Rmax],origin='lower')
	for i in range(10):
	#for T1,T2, c in [(150.,180.,'b'),(180.,210.,'g'),(210.,240.,'y'),(240.,270.,'r')]:
		T1 = i*(Tma - Tmi)/10 + Tmi    
		T2 = (i+1)*(Tma - Tmi)/10 + Tmi     
		Tsel = np.where((grilleT > T1) & (grilleT < T2))
		#print(T1,T2,len(Tsel), Tsel)
		ax.scatter(Tsel[:][0], Tsel[:][1], Tsel[:][2], c='C'+str(i))
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	plt.show() 

def save3d_T(model,prefix=''):
    print('Computing 3D T map')
    N=model.map.N
    T=np.zeros((2*N,2*N,2*N))
    for i in range(2*N):
        for j in range(2*N):
            for k in range(2*N):
                T[i,j,k] = model.map.grid[i][j][k].T
    f = open('Output/'+prefix+'_T_map.dat','w')
    for i in range(2*N):
        for j in range(2*N):
            for k in range(2*N):
                f.write("%f\t"%T[i,j,k])
    f.close()
    
def save3d_rho(model,prefix='',nsimu=1,cluster=0):
    if(nsimu==1 or cluster==1):
        print('Computing 3D density map')
    N=model.map.N
    for l in range(len(model.map.grid[0][0][0].rho)):
        #rho_sil=np.zeros((2*N,2*N,2*N))
        #rho_grap_ortho=np.zeros((2*N,2*N,2*N))
        #rho_grap_para=np.zeros((2*N,2*N,2*N))
        #rho_el=np.zeros((2*N,2*N,2*N))
        rho=np.zeros((2*N,2*N,2*N))

        for i in range(2*N):
            for j in range(2*N):
                for k in range(2*N):
                    #rho_sil[i,j,k] = model.map.grid[i][j][k].rho[0]
                    #rho_grap_ortho[i,j,k] = model.map.grid[i][j][k].rho[1]
                    #rho_grap_para[i,j,k] = model.map.grid[i][j][k].rho[2]
                    #rho_el[i,j,k] = model.map.grid[i][j][k].rho[3]
                    rho[i,j,k] = model.map.grid[i][j][k].rho[l]

        f = open('Output/'+prefix+'_rho_'+model.map.dust[l].name+'_map.dat','w')
        for i in range(2*N):
            for j in range(2*N):
                for k in range(2*N):
                    #f.write("%f\t"%rho_sil[i,j,k])
                    f.write("%f\t"%rho[i,j,k])
        f.close()

def load3d_T(data):
    if isinstance(data,str)==True:
        T=np.genfromtxt(data)
        N=int(((np.size(T)+1)**(1/3))/2)
        T=np.reshape(T,(2*N,2*N,2*N))
    else:
        N=data.map.N
        T=np.zeros((2*N,2*N,2*N))
        for i in range(2*N):
            for j in range(2*N):
                for k in range(2*N):
                    T[i,j,k] = data.map.grid[i][j][k].T
    return T

def load3d_rho(data,grain=None):
    if isinstance(data,str)==True:
        rho=np.genfromtxt(data)
        N=int(((np.size(rho)+1)**(1/3))/2)
        rho=np.reshape(rho,(2*N,2*N,2*N))
    else:
        for l in range(len(data.map.dust)):
            if (data.map.dust[l].name == grain):
                N=data.map.N
                rho=np.zeros((2*N,2*N,2*N))
                for i in range(2*N):
                    for j in range(2*N):
                        for k in range(2*N):
                            rho[i,j,k] = data.map.grid[i][j][k].rho[l]            
    return rho
    
def plot3d(param,res,step=1,xmin=None,xmax=None,ymin=None,ymax=None,zmin=None,zmax=None,param_min=None,param_max=None,param_log=False,vmin=None,vmax=None,cmap=cm.jet,alpha=1,title='',unit='pc'): 
    N=param.shape[0]/2
    param = param[0::step,0::step,0::step]
    ijk=np.unravel_index(np.arange(np.size(param)),np.shape(param))
    x=(ijk[0]*step-N+0.5)*res
    y=(ijk[1]*step-N+0.5)*res
    z=(ijk[2]*step-N+0.5)*res

    if xmin==None: xmin=np.amin(x)
    if xmax==None: xmax=np.amax(x)
    if ymin==None: ymin=np.amin(y)
    if ymax==None: ymax=np.amax(y)
    if zmin==None: zmin=np.amin(z)
    if zmax==None: zmax=np.amax(z)
    if param_min==None: param_min=np.amin(param)
    if param_max==None: param_max=np.amax(param)
    if vmax==None: vmax=np.amax(param)
    
    fig = plt.figure()   
    ax = fig.add_subplot(111,projection='3d')
    rng=np.where((x>=xmin)&(x<=xmax)&(y>=ymin)&(y<=ymax)&(z>=zmin)&(z<=zmax)&(param.flatten()>=param_min)&(param.flatten()<=param_max))
    if param_log==False:
        if vmin==None: vmin=np.amin(param)
        norm=matplotlib.colors.Normalize(vmin,vmax)
    elif param_log==True:
        if vmin==None: vmin=np.amin(param[param>0])
        norm=matplotlib.colors.LogNorm(vmin,vmax)
    p = ax.scatter(x[rng],y[rng],z[rng],c=param.flatten()[rng],marker=',',norm=norm,cmap=cmap,alpha=alpha,edgecolors='none')
    ax.set_title(title)
    ax.set_xlabel('X ('+unit+')')
    ax.set_xlim(xmin,xmax)
    ax.set_ylabel('Y ('+unit+')')
    ax.set_ylim(ymin,ymax)    
    ax.set_zlabel('Z ('+unit+')')
    ax.set_zlim(zmin,zmax)
    cbar = fig.colorbar(p)
    cbar.solids.set(alpha=1)
    plt.show()
    
def tomo3d(param,res,step=1,xmin=None,xmax=None,xstep=None,ymin=None,ymax=None,ystep=None,zmin=None,zmax=None,zstep=None,param_min=None,param_max=None,param_step=None,param_log=False,cmap=cm.jet,alpha=1,title='',unit='pc'):
    N=param.shape[0]/2
    param = param[0::step,0::step,0::step]
    ijk=np.unravel_index(np.arange(np.size(param)),np.shape(param))
    x=(ijk[0]*step-N+0.5)*res
    y=(ijk[1]*step-N+0.5)*res
    z=(ijk[2]*step-N+0.5)*res

    if xmin==None: xmin=np.amin(x)
    if xmax==None: xmax=np.amax(x)
    if xstep==None: xstep=xmax-xmin
    if ymin==None: ymin=np.amin(y)
    if ymax==None: ymax=np.amax(y)
    if ystep==None: ystep=ymax-ymin
    if zmin==None: zmin=np.amin(z)
    if zmax==None: zmax=np.amax(z)
    if zstep==None: zstep=zmax-zmin
    if param_min==None: param_min=np.amin(param)
    if param_max==None: param_max=np.amax(param)
    if param_step==None: param_step=param_max-param_min
    
    for xmin_ in np.arange(xmin,xmax,xstep):
        for ymin_ in np.arange(ymin,ymax,ystep):
            for zmin_ in np.arange(zmin,zmax,zstep):
                for param_min_ in np.arange(param_min,param_max,param_step):
                    plot3d(param,res,xmin=xmin_,xmax=min(xmin_+xstep,xmax),ymin=ymin_,ymax=min(ymin_+ystep,ymax),zmin=zmin_,zmax=min(zmin_+zstep,zmax),param_min=param_min_,param_max=min(param_min_+param_step,param_max),param_log=param_log,vmin=param_min_,vmax=min(param_min_+param_step,param_max),cmap=cmap,alpha=alpha,title=title,unit=unit)
                    #plot3d(param,res,xmin=xmin_,xmax=min(xmin_+xstep,xmax),ymin=ymin_,ymax=min(ymin_+ystep,ymax),zmin=zmin_,zmax=min(zmin_+zstep,zmax),param_min=param_min_,param_max=min(param_min_+param_step,param_max),param_log=param_log,vmin=param_min,vmax=param_max,cmap=cmap,alpha=alpha,title=title,unit=unit)

def displaystats(info,model,tau_speciesV,tau_species,band):

    # POLARIMETRIC PHASE FUNCTIONS

    #print 'histo-alpha : ',polar.alpha
    mpol.plotphase_xfix(np.sum(info.alpha,axis=1),title='Alpha probability density function')
    mpol.plotphase_xfix(np.sum(info.alpha,axis=1),normsin=1,title='Alpha phase function')
    for i in range(5):
        mpol.plotphase_xfix(np.sum(info.beta[:,i,:],axis=1),twopi=1,title='Beta probability density function after '+str(i+1)+' scatterings')
    for k in range(len(model.map.dust)):
        #print model.map.dust[i].name
        #if(model.map.dust[i].name=="silicates"):
        #isil=i
        mpol.plotphase_xfix(info.alpha[:,k],title='Alpha probability density function for '+model.map.dust[k].name)
        mpol.plotphase_xfix(info.alpha[:,k],normsin=1,title='Alpha phase function for '+model.map.dust[k].name)
        for i in range(5):
            mpol.plotphase_xfix(info.beta[:,i,k],twopi=1,title='Beta probability density function after '+str(i+1)+' scatterings for '+model.map.dust[k].name)
        if(model.map.dust[k].name!='electrons'):
            mpol.plot_1D(model.map.dust[k].sizevect,info.xstat[:,k],logx=1,title='Random grain radius function for '+model.map.dust[k].name,xlabel='Grain size (m)',ylabel='Number of occurrences')
            mpol.plot_1D(10**(-np.linspace(0.0,1.0,len(model.map.dust[k].sizevect))*3.)*100.0,info.albedo[:,k],logx=1,title='Albedo distribution for '+model.map.dust[k].name,xlabel='Albedo (%)',ylabel='Number of occurrences')
    for i in range(np.shape(info.pstat)[0]):
        for j in range(np.shape(info.pstat)[1]):
            for k in range(len(model.map.dust)):
                info.pstat[i,j,1,k]=max(1.0,info.pstat[i,j,1,k])
    for k in range(len(model.map.dust)):
        for i in range(5):
            mpol.plotphase_xfix(info.pstat[:,i,0,k]/info.pstat[:,i,1,k],title='Polarisation degree after '+str(i+1)+' scatterings for '+model.map.dust[k].name)

    # OPTICAL DEPTHS FOR EACH DUST TYPE

    for i in range(len(model.map.dust)):
        print("Effective tau in V for "+model.map.dust[i].name+" :",tau_speciesV[i][0],"and tau RSublimation in V :",tau_speciesV[i][1])
        print("Effective tau max at "+band+" m for "+model.map.dust[i].name+" :",tau_species[i][0], "and tau RSublimation at "+band+" m :",tau_species[i][1])
    

    # AVERAGED ALBEDO COMPUTATION

    for i in range(len(model.map.dust)):
        if(model.map.dust[i].name!="electrons"): # and model.map.dust[i].name!="pah"):
            ir=np.argsort(np.argsort(np.append(model.map.dust[i].sizevect,0.01e-6)))[len(model.map.dust[i].sizevect)]-1
            if(ir<0):
                ir=0
            if(ir>=len(model.map.dust[i].sizevect)-1):
                ir=len(model.map.dust[i].sizevect)-2
            alb=(model.map.dust[i].albedo[:,ir]+model.map.dust[i].albedo[:,ir+1])*0.5 #lin
            albV2=mpol.loginterp(model.map.dust[i].wvlvect,alb,0.5e-6) #log

            if(info.force_wvl==[]):
                #print "Mean silicate albedo in H :",albH[ir]*100,"%" #polar.x_M,polar.albedo
                #print "Mean silicate albedo in V :",albV[ir]*100,"%"
                print("Mean ",model.map.dust[i].name," albedo in V :",albV2*100,"%")
            else:
                alb=mpol.loginterp(model.map.dust[i].wvlvect,alb,info.force_wvl) #log
                print("Mean ",model.map.dust[i].name," albedo in the used wvl :",alb*100,"%")



def openoutfile(path,filename,model,info):
    if(info.add==0):
        f2=[]
        for i in range(len(model.thetarec)):
            if(model.thetarec[i]<10):
                #if(info.outformat=='ascii'):
                fi = open(path+filename+'_00'+str(model.thetarec[i])+"_phot.dat",'w')
                #elif(info.outformat=='fits'):
                #    #fi = open(path+filename+'_00'+str(model.thetarec[i])+"_phot.dat",'w',format='fits')
                #else:
                #    #fi = open(path+filename+'_00'+str(model.thetarec[i])+"_phot.dat",'w',format='hdf5')
            elif(model.thetarec[i]<100):
                #if(info.outformat=='ascii'):
                fi = open(path+filename+'_0'+str(model.thetarec[i])+"_phot.dat",'w')
                #elif(info.outformat=='fits'):
                #    #fi = open(path+filename+'_0'+str(model.thetarec[i])+"_phot.dat",'w',format='fits')
                #else:
                #    #fi = open(path+filename+'_0'+str(model.thetarec[i])+"_phot.dat",'w',format='hdf5')
            else:
                #if(info.outformat=='ascii'):
                fi = open(path+filename+'_'+str(model.thetarec[i])+"_phot.dat",'w')
                #elif(info.outformat=='fits'):
                #    #fi = open(path+filename+'_'+str(model.thetarec[i])+"_phot.dat",'w',format='fits')
                #else:
                #    #fi = open(path+filename+'_'+str(model.thetarec[i])+"_phot.dat",'w',format='hdf5')
            f2.append(fi)
        info.ftheta=f2
    if(info.add==1):
        f2=[]
        for i in range(len(model.thetarec)):
            if(model.thetarec[i]<10):
                #if(info.outformat=='ascii'):
                fi = open(path+filename+'_00'+str(model.thetarec[i])+"_phot.dat",'a')
                #elif(info.outformat=='fits'):
                #    #fi = open(path+filename+'_00'+str(model.thetarec[i])+"_phot.dat",'a',format='fits')
                #else:
                #    #fi = open(path+filename+'_00'+str(model.thetarec[i])+"_phot.dat",'a',format='hdf5')
            elif(model.thetarec[i]<100):
                #if(info.outformat=='ascii'):
                fi = open(path+filename+'_0'+str(model.thetarec[i])+"_phot.dat",'a')
                #elif(info.outformat=='fits'):
                #    #fi = open(path+filename+'_0'+str(model.thetarec[i])+"_phot.dat",'a',format='fits')
                #else:
                #    #fi = open(path+filename+'_0'+str(model.thetarec[i])+"_phot.dat",'a',format='hdf5')
            else:
                #if(info.outformat=='ascii'):
                fi = open(path+filename+'_'+str(model.thetarec[i])+"_phot.dat",'a')
                #elif(info.outformat=='fits'):
                #    #fi = open(path+filename+'_'+str(model.thetarec[i])+"_phot.dat",'a',format='fits')
                #else:
                #    #fi = open(path+filename+'_'+str(model.thetarec[i])+"_phot.dat",'a',format='hdf5')
            f2.append(fi)
        info.ftheta=f2
    if(model.usethermal==1):
        #if(info.outformat=='ascii'):
        fT = open(path+filename+"_T_update.dat",'w')
        #elif(info.outformat=='fits'):
        #    #fT = open(path+filename+"_T_update.dat",'w',format='fits')
        #else:
        #    #fT = open(path+filename+"_T_update.dat",'w',format='hdf5')
        info.fT=fT
    else:
        info.fT = []

