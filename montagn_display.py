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
import montagn_output as mout
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
        kwargs['thetaobs']=(kwargs['thetamax']+kwargs['thetamin'])*0.5
        kwargs['dtheta']=(kwargs['thetamax']-kwargs['thetamin'])*0.5
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
        'minwvl':kwargs.get('minwvl',0.1e-6),
        'maxwvl':kwargs.get('maxwvl',1.e-3),
        'diam_angle':kwargs.get('diam_angle',[]),
        'xmax':kwargs.get('xmax',-1),
        'ymax':kwargs.get('ymax',-1),
        'thetamin':kwargs.get('thetamin',0.),
        'thetamax':kwargs.get('thetamax',np.pi),
        #'thetaobs':kwargs.get('thetaobs',-100),
        #'dtheta':kwargs.get('dtheta',-1),
        'phimin':kwargs.get('phimin',-np.pi),
        'phimax':kwargs.get('phimax',np.pi),
        'phisym':kwargs.get('phisym',1),
        'interact':kwargs.get('interact',-1),
        'reem':kwargs.get('reem',-1),
        'nscatt':kwargs.get('nscatt',-1),
        'nscattmax':kwargs.get('nscattmax',5),
        'label':kwargs.get('label',-1),
        
        'polar':kwargs.get('polar',0),
        'polarsym':kwargs.get('polarsym',0),

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
        'resimage':kwargs.get('resimage',-1),
        'resunit':kwargs.get('resunit','m'),
        'legend':kwargs.get('legend',[]),
        'cmap':kwargs.get('cmap',plt.rcParams['image.cmap']),
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

class Spectrum:
    def __init__(self,pargs):
        """
        Defining the initial caharacteristics of any spectra
        """
        self.datatype='spectrum'
        self.binning = np.logspace(np.log10(pargs['minwvl']),np.log10(pargs['maxwvl']),pargs['bins']+1)
        self.SED = 0*self.binning
        self.SEDI = 0*self.binning
        if(pargs['polar']==1):
            self.SEDQ = 0*self.binning
            self.SEDU = 0*self.binning
            self.SEDV = 0*self.binning
        self.fileopened=[]
        self.filenotopened=[]

class Image:
    def __init__(self,pargs):
        """
        Defining the initial caharacteristics of any image
        """
        self.datatype = 'image'
        self.resunit = pargs['resunit']
        self.resimage = pargs['resimage']
        if(self.resunit=='pc' or self.resunit=='PC'):
            self.resimage*=mc.Constant().pc
        elif(self.resunit=='au' or self.resunit=='AU' or self.resunit=='ua' or self.resunit=='UA'):
            self.resimage*=mc.Constant().AU
        self.influx = 0.
        self.outflux = 0.
        self.default_size = 10 # default size of not specified
        if(pargs['xmax']==-1 and pargs['ymax']==-1):
            self.xmax=-1
            self.ymax=-1
            self.nx = 2*self.default_size # half-size of the square (2*nx ranges from -xmax to +xmax)
            self.ny = 2*self.default_size
        else:
            if(pargs['xmax']==-1):
                self.xmax=pargs['ymax']
                self.ymax=pargs['ymax']
            elif(pargs['ymax']==-1):
                self.xmax=pargs['xmax']
                self.ymax=pargs['xmax']
            else:
                self.xmax=pargs['xmax']
                self.ymax=pargs['ymax']
            self.nx = max(1,int((2*self.xmax)//self.resimage))
            self.ny = max(1,int((2*self.ymax)//self.resimage))
        if(pargs['polarsym']==1 and (self.nx%2!=0 or self.ny%2!=0)):
            print('WARNING! Odd number of pixel while using polar symmetries')
            print('         Artifact might appear on the central pixel lines')
            print('         Please consider adapting the x and y max values to avoid this problem')
        self.posx = np.zeros(self.nx)
        self.posy = np.zeros(self.ny)
        self.I = np.zeros([self.nx,self.ny])
        self.I2 = np.zeros([self.nx,self.ny])
        self.logI = np.zeros([self.nx,self.ny])
        self.nscattmap = np.zeros([self.nx,self.ny])
        self.nphotstat = np.zeros([self.nx,self.ny])
        self.nphotstateff = np.zeros([self.nx,self.ny])
        self.proba = np.zeros([self.nx,self.ny])
        if(pargs['polar']==1):
            self.Q = np.zeros([self.nx,self.ny])
            self.U = np.zeros([self.nx,self.ny])
            self.V = np.zeros([self.nx,self.ny])
            self.theta = np.zeros([self.nx,self.ny])
            self.thetas = np.zeros([self.nx,self.ny])
            self.thetasym = np.zeros([self.nx,self.ny])
            self.thetas_article = np.zeros([self.nx,self.ny])
            self.P = np.zeros([self.nx,self.ny])
            self.Ip = np.zeros([self.nx,self.ny])
            self.polV = np.zeros([self.nx,self.ny])
            self.Qphi = np.zeros([self.nx,self.ny])
            self.Uphi = np.zeros([self.nx,self.ny])
        self.fileopened=[]
        self.filenotopened=[]

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
        #binning,SED,SEDI,SEDQ,SEDU,SEDV=extract_from_file_old(filerootname,pargs)
        data=extract_from_file(filerootname,pargs,spectral_reduction)
        data.SEDIp,data.SEDP,data.SEDtheta,data.SEDpolV=polar_reduction(data.SEDI,data.SEDQ,data.SEDU,V=data.SEDV)
    else:
        data=extract_from_file(filerootname,pargs,spectral_reduction)
            
    plot_spectra(data,pargs)

    if(pargs['rec']==1):
        if(pargs['polar']==1):
            write_spectra(data.binning,[data.SED,data.SEDI,data.SEDQ,data.SEDU,data.SEDV,data.SEDIp,data.SEDP,data.SEDtheta,data.SEDpolV],['Number of packets','Intensity','Q','U','V','Ip','P','theta','CircP'],filerootname,path=pargs['outdir'])
        else:
            write_spectra(data.binning,[data.SED,data.SEDI],['Number of packets','Intensity'],filerootname,path=pargs['outdir'])
    print('Opened files:')
    print(data.fileopened)
    if(pargs['ret']==1):
        return data


def plot_spectra(data,pargs):
    #plt.figure()
    mpol.plot_1D((data.binning)*1e9,data.SED,logy=pargs['logy'],title=pargs['title'],xlabel='Wavelength (nm)',ylabel='Intensity - number of photons packets',logx=pargs['logx'])
    mpol.plot_1D((data.binning)*1e9,data.SEDI,logy=pargs['logy'],title=pargs['title']+' I',xlabel='Wavelength (nm)',ylabel='Intensity (J)',logx=pargs['logx'])
    if(pargs['polar']==1):
       mpol.plot_1D((data.binning)*1e9,data.SEDQ,logy=0,title=pargs['title']+' Q',xlabel='Wavelength (nm)',ylabel='Intensity Q (J)',logx=pargs['logx'])
       mpol.plot_1D((data.binning)*1e9,data.SEDU,logy=0,title=pargs['title']+' U',xlabel='Wavelength (nm)',ylabel='Intensity U (J)',logx=pargs['logx'])
       mpol.plot_1D((data.binning)*1e9,data.SEDV,logy=0,title=pargs['title']+' V',xlabel='Wavelength (nm)',ylabel='Intensity V (J)',logx=pargs['logx'])
       mpol.plot_1D((data.binning)*1e9,data.SEDIp,logy=pargs['logy'],title=pargs['title']+' Ip',xlabel='Wavelength (nm)',ylabel='Polarised intensity (J)',logx=pargs['logx'])
       mpol.plot_1D((data.binning)*1e9,data.SEDP*100,logy=0,title=pargs['title']+' P',xlabel='Wavelength (nm)',ylabel='Linear Polarisation degree (%)',logx=pargs['logx'])
       mpol.plot_1D((data.binning)*1e9,data.SEDtheta,logy=0,title=pargs['title']+' theta',xlabel='Wavelength (nm)',ylabel='Linear Polarisation angle (degrees)',logx=pargs['logx'])
       mpol.plot_1D((data.binning)*1e9,data.SEDpolV*100,logy=0,title=pargs['title']+' V',xlabel='Wavelength (nm)',ylabel='Circular Polarisation degree (%)',logx=pargs['logx'])
    #if(pargs['logx']==1):
    #    plt.xscale('log')
    #if(pargs['logy']==1):      
    #    plt.yscale('log')
    #plt.title('SED total')
    #plt.xlabel('Wavelength (um)')
    #plt.ylabel('Intensity - number of photons packets')
    if(pargs['legend']!=[]):
        plt.legend(pargs['legend'],loc=0)
    plt.show()


def display_image(filerootname,**kwargs):
    """root of all files with stored photon results,
    viewing polar angle,
    viewing azimuthal angle,
    angular diameter
    [source label (string or set of strings)]
    [photons that interacted (0 or 1)]
    [photons that were reemitted (0 or 1)]"""
    
    pargs=plot_args(kwargs)
    pargs['image']=1

    if(pargs['polar']==1):
        #binning,SED,SEDI,SEDQ,SEDU,SEDV=extract_from_file_old(filerootname,pargs)
        data=extract_from_file(filerootname,pargs,image_reduction)
    else:
        data=extract_from_file(filerootname,pargs,image_reduction)
        

    if(data.outflux>=data.influx*0.1):
        print("WARNING! Presence of flux outside the image")
        print('         ',(100*data.outflux)//(max(1,data.outflux+data.influx)),' % of light not displayed')

    # using polar symmetries
    if(pargs['polarsym']==1):
        I=data.I #to be kept in non-symmetrised version for Pcirc 
        data=symmetry(data)

    # Setting limits for log scales
    nphotdet=int(sum(sum(data.nphotstat)))
    listI=np.nonzero(data.I==0)
    for i in range(np.shape(listI)[1]):
        data.I[listI[0][i]][listI[1][i]]=1
        data.nphotstat[listI[0][i]][listI[1][i]]=0.1
        data.nphotstateff[listI[0][i]][listI[1][i]]=1
    data.nphotstateff=data.I/data.nphotstateff

    if(pargs['polar']==1):
        imageIp,imageP,imagetheta=polar_reduction(data.I,data.Q,data.U)
        data.Ip=imageIp
        data.P=imageP
        data.theta=imagetheta
        data.polV=data.V/data.I

    #log images
    data.logI=np.log10(data.I)

    ## Resetting I to 0 where it should
    for i in range(np.shape(listI)[1]):
        data.I[listI[0][i]][listI[1][i]]=0

            
    ## Computing probability maps
    data.proba=data.I/data.nphotstat # /enpaq  #to be implemented
    data.nphotstat=np.log10(data.nphotstat)
              
                
    ## Resetting nphot to 0.1 where it should be nul
    for i in range(np.shape(listI)[1]):
        #nphotstat[listI[0][i]][listI[1][i]]=-0.1
        data.proba[listI[0][i]][listI[1][i]]=1e-20
        data.nphotstateff[listI[0][i]][listI[1][i]]=0
    #nphotstateff=np.log10(nphotstateff)


    ## Computing Ip and I2 maps
    if(pargs['polar']==1):
        data.Ip=data.P*data.I #re-computing to avoid the intensity offset
    data.I2=np.copy(data.I)
    data.I2[data.nx//2][data.ny//2]=0
    if(data.nx%2!=0):
        data.I2[data.nx//2+1][data.ny//2]=0
    if(data.ny%2!=0):
        data.I2[data.nx//2][data.ny//2+1]=0
    if(data.nx%2!=0 and data.ny%2!=0):
        data.I2[data.nx//2+1][data.ny//2+1]=0


    if(pargs['polarsym']==0):
        print(nphotdet,"photons detected")
    else:
        print(nphotdet,"photons detected using symmetries")
        print("for",nphotdet/4,"photons detected")

    if(pargs['polar']==1):
        data=centrosym(data)

    # plotting images
    plot_images(data,pargs)

    #print 'Opened files:'
    #print data.fileopened
    print(len(data.fileopened),' files opened')
    if(pargs['ret']==1):
        return data

def centrosym(data):
    ## Computing the map centrosymetric-substracted
    #if(centrosymsubstract==1):
    thetasym=np.copy(data.theta)
    for i in range(data.nx):
        for j in range(data.ny):
            #thetasym[i][j]=np.arctan2(j-(nbintot-1)/2.,i-(nbintot-1)/2.)*180./np.pi-90
            thetasym[i][j]=np.arctan2(i-(data.nx-1)/2.,j-(data.ny-1)/2.)*180./np.pi
            if(thetasym[i][j]<-90):
                thetasym[i][j]+=180
            if(thetasym[i][j]>90):
                thetasym[i][j]-=180
    #thetas=thetasym
    thetas=data.theta-thetasym
    for i in range(data.nx):
        for j in range(data.ny):
            if(data.I[i][j]==0.):
                thetas[i][j]=-120
            else:
                if(thetas[i][j]<-90):
                    thetas[i][j]+=180
                elif(thetas[i][j]>90):
                    thetas[i][j]-=180

    data.thetasym=thetasym
    data.thetas=thetas

    #alternative display of centrosym
    data.thetas_article=np.abs(thetas)
    phi=np.copy(thetasym)
    #print np.shape(QU)
    ##Qphi=-np.transpose(np.transpose(QU)[0]*np.cos(2*phi*np.pi/180.)+np.transpose(QU)[1]*np.sin(2*phi*np.pi/180.))
    ##Uphi=-np.transpose(np.transpose(QU)[0]*np.sin(2*phi*np.pi/180.)-np.transpose(QU)[1]*np.cos(2*phi*np.pi/180.))
    #Qphi=np.transpose(np.transpose(QU)[0]*np.cos(2*phi*np.pi/180.)+np.transpose(QU)[1]*np.sin(2*phi*np.pi/180.))
    #Uphi=-np.transpose(np.transpose(QU)[0]*np.sin(2*phi*np.pi/180.)+np.transpose(QU)[1]*np.cos(2*phi*np.pi/180.))
    data.Qphi=data.Q[:,:]*np.cos(2*phi*np.pi/180.)+data.U[:,:]*np.sin(2*phi*np.pi/180.)
    data.Uphi=-data.Q[:,:]*np.sin(2*phi*np.pi/180.)+data.U[:,:]*np.cos(2*phi*np.pi/180.)
    return data


def check_outname(pargs):
    filenametmp=''
    for i in range(len(pargs['outname'])):
        if(pargs['outname'][i]=='.'):
            filenametmp+='-'
        else:
            filenametmp+=pargs['outname'][i]
    pargs['outname']=filenametmp
    return pargs


def plot_images(data,pargs):
    #pargs=check_outname(pargs)

    if(pargs['polar']==1):
        ## Setting the no-photon detection to -100 in angular maps
        thetaplot=np.copy(data.theta)
        #theta+=90
        for i in range(data.nx):
            for j in range(data.ny):
                if(data.theta[i][j]>90):
                    data.theta[i][j]-=180
        for i in range(data.nx):
            for j in range(data.ny):
                if(data.I[i][j]==0.):
                    data.theta[i][j]=-120.


    ### Ploting all results ###
    ## Save fits files
    if(pargs['rec']==1):
        save_fits(data.nscattmap,pargs['outdir']+pargs['outname']+pargs['outsuffixe']+'_nscattmap.fits')
        save_fits(data.nphotstat,pargs['outdir']+pargs['outname']+pargs['outsuffixe']+'_nphot.fits')
        save_fits(data.I2,pargs['outdir']+pargs['outname']+pargs['outsuffixe']+'_I2.fits')
        save_fits(data.logI,pargs['outdir']+pargs['outname']+pargs['outsuffixe']+'_logI.fits')
        save_fits(np.log10(data.proba),pargs['outdir']+pargs['outname']+pargs['outsuffixe']+'_logProba.fits')
        if(pargs['polar']==1):
            save_fits(data.P*100,pargs['outdir']+pargs['outname']+pargs['outsuffixe']+'_P.fits')
            save_fits(data.theta,pargs['outdir']+pargs['outname']+pargs['outsuffixe']+'_theta.fits')
            save_fits(data.Ip,pargs['outdir']+pargs['outname']+pargs['outsuffixe']+'_Ip.fits')
            save_fits(data.thetas,pargs['outdir']+pargs['outname']+pargs['outsuffixe']+'_theta-diff.fits')
            save_fits(data.V*100,pargs['outdir']+pargs['outname']+pargs['outsuffixe']+'_V.fits')


    ## ploting
    if(pargs['cmap']=='jet'):
        cmaploop,normloop,mappableloop=g_cb.colorbar(cmap='loopingmask',cm_min=-120,cm_max=90)
    else:
        cmaploop,normloop,mappableloop=g_cb.colorbar(cmap='loopingearth',cm_min=-100,cm_max=90)
    maxlog=np.max(data.logI)
    minlog=maxlog-10
    #cmaplog,normlog,mappablelog=g_cb.colorbar(cmap=cmap,cm_min=np.log10(enpaq)-5,cm_max=np.log10(enpaq)+5)
    cmaplog,normlog,mappablelog=g_cb.colorbar(cmap=pargs['cmap'],cm_min=minlog,cm_max=maxlog)
    cmapeff,normeff,mappableeff=g_cb.colorbar(cmap=pargs['cmap'],cm_min=0,cm_max=min(data.nphotstateff.max(),50))
    cmapscatt,normscatt,mappablescatt=g_cb.colorbar(cmap=pargs['cmap'],cm_min=0,cm_max=5)
    cmap_art,norm_art,mappable_art=g_cb.colorbar(cmap=pargs['cmap'],cm_min=0,cm_max=90)
    cmapV,normV,mappableV=g_cb.colorbar(cmap=pargs['cmap'],cm_min=-100,cm_max=100)
            
    nodt=1 #pour l instant
    if(nodt==1):
        unitI="J"
    else:
        unitI="W"
    
    #suffixe=suffixe+str(thetaobs)
    titletheta = ' for i=['+str(180*pargs['thetamin']//np.pi)+','+str(180*pargs['thetamax']//np.pi)+'] deg'
    #if(thetaobs<10):
    #    titletheta = 'for inclination of _00'+str(model.thetarec[i])+"_phot.dat"
    #elif(thetaobs<100):
    #    titletheta = '_0'+str(model.thetarec[i])+"_phot.dat"
    #else:
    #    titletheta = '_'+str(model.thetarec[i])+"_phot.dat"
    
    plot_image(data.logI,title='Intensity (log10('+unitI+'))'+titletheta,cmap=cmaplog,norm=normlog,scale=[-data.xmax,data.xmax,-data.ymax,data.ymax],rec=pargs['rec'],outname=pargs['outdir']+pargs['outname']+pargs['outsuffixe']+'_logI',resunit=pargs['resunit'],saveformat=pargs['saveformat'])
    plot_image(data.I2,title='Intensity without central object ('+unitI+')'+titletheta,cmap=pargs['cmap'],scale=[-data.xmax,data.xmax,-data.ymax,data.ymax],rec=pargs['rec'],outname=pargs['outdir']+pargs['outname']+pargs['outsuffixe']+'_I2',resunit=pargs['resunit'],saveformat=pargs['saveformat'])
    plot_image(data.nscattmap,title='Averaged number of scatterings'+titletheta,cmap=pargs['cmap'],norm=normscatt,scale=[-data.xmax,data.xmax,-data.ymax,data.ymax],rec=pargs['rec'],outname=pargs['outdir']+pargs['outname']+pargs['outsuffixe']+'_nscatt',resunit=pargs['resunit'],saveformat=pargs['saveformat'])
    plot_image(data.nphotstat,title='Number of packets per pixel (log10(npacket))'+titletheta,cmap=pargs['cmap'],scale=[-data.xmax,data.xmax,-data.ymax,data.ymax],rec=pargs['rec'],outname=pargs['outdir']+pargs['outname']+pargs['outsuffixe']+'_nphot',resunit=pargs['resunit'],saveformat=pargs['saveformat'])
    plot_image(data.nphotstateff,title='Number of effective packets per pixel'+titletheta,cmap=cmapeff,norm=normeff,scale=[-data.xmax,data.xmax,-data.ymax,data.ymax],rec=pargs['rec'],outname=pargs['outdir']+pargs['outname']+pargs['outsuffixe']+'_nphoteff',resunit=pargs['resunit'],saveformat=pargs['saveformat'])
    plot_image(np.log10(data.proba),title='Probability'+titletheta,scale=[-data.xmax,data.xmax,-data.ymax,data.ymax],rec=pargs['rec'],outname=pargs['outdir']+pargs['outname']+pargs['outsuffixe']+'_logProba',resunit=pargs['resunit'],saveformat=pargs['saveformat'])
    if(pargs['polar']==1):
        plot_image(data.P*100,title='Linear degree of polarisation (%)'+titletheta,cmap=pargs['cmap'],scale=[-data.xmax,data.xmax,-data.ymax,data.ymax],rec=pargs['rec'],outname=pargs['outdir']+pargs['outname']+pargs['outsuffixe']+'_P',resunit=pargs['resunit'],saveformat=pargs['saveformat'])
        plot_image(data.Ip,title='Linear polarised intensity ('+unitI+')'+titletheta,cmap=pargs['cmap'],scale=[-data.xmax,data.xmax,-data.ymax,data.ymax],rec=pargs['rec'],outname=pargs['outdir']+pargs['outname']+pargs['outsuffixe']+'_Ip',resunit=pargs['resunit'],saveformat=pargs['saveformat'])
        plot_image(data.theta,title='Polarisation angle (degree)'+titletheta,cmap=pargs['cmap'],scale=[-data.xmax,data.xmax,-data.ymax,data.ymax],rec=pargs['rec'],outname=pargs['outdir']+pargs['outname']+pargs['outsuffixe']+'_theta',resunit=pargs['resunit'],saveformat=pargs['saveformat'])
        plot_image(data.thetasym,title='Centro-symmetric angle (degree)'+titletheta,cmap=pargs['cmap'],scale=[-data.xmax,data.xmax,-data.ymax,data.ymax],rec=pargs['rec'],outname=pargs['outdir']+pargs['outname']+pargs['outsuffixe']+'_thetasym',resunit=pargs['resunit'],saveformat=pargs['saveformat'])
        plot_image(data.thetas,title='Difference to centrosymmetric (degree)'+titletheta,cmap=pargs['cmap'],scale=[-data.xmax,data.xmax,-data.ymax,data.ymax],rec=pargs['rec'],outname=pargs['outdir']+pargs['outname']+pargs['outsuffixe']+'_theta_diff.',resunit=pargs['resunit'],saveformat=pargs['saveformat'])
        plot_image(data.Qphi,title='Qphi ('+unitI+')'+titletheta,scale=[-data.xmax,data.xmax,-data.ymax,data.ymax],rec=pargs['rec'],outname=pargs['outdir']+pargs['outname']+pargs['outsuffixe']+'_Qphi',resunit=pargs['resunit'],saveformat=pargs['saveformat'])
        plot_image(data.Uphi,title='Uphi ('+unitI+')'+titletheta,scale=[-data.xmax,data.xmax,-data.ymax,data.ymax],rec=pargs['rec'],outname=pargs['outdir']+pargs['outname']+pargs['outsuffixe']+'_Uphi',resunit=pargs['resunit'],saveformat=pargs['saveformat'])
        plot_image(data.Q,title='Q ('+unitI+')'+titletheta,scale=[-data.xmax,data.xmax,-data.ymax,data.ymax],rec=pargs['rec'],outname=pargs['outdir']+pargs['outname']+pargs['outsuffixe']+'_Q',resunit=pargs['resunit'],saveformat=pargs['saveformat'])
        plot_image(data.U,title='U ('+unitI+')'+titletheta,scale=[-data.xmax,data.xmax,-data.ymax,data.ymax],rec=pargs['rec'],outname=pargs['outdir']+pargs['outname']+pargs['outsuffixe']+'_U',resunit=pargs['resunit'],saveformat=pargs['saveformat'])
        plot_image(data.V,title='V ('+unitI+')'+titletheta,scale=[-data.xmax,data.xmax,-data.ymax,data.ymax],rec=pargs['rec'],outname=pargs['outdir']+pargs['outname']+pargs['outsuffixe']+'_V',resunit=pargs['resunit'],saveformat=pargs['saveformat'])
        plot_image(data.thetas_article,title='Difference to centrosymmetric (degree)'+titletheta,cmap=cmap_art,norm=norm_art,scale=[-data.xmax,data.xmax,-data.ymax,data.ymax],rec=pargs['rec'],outname=pargs['outdir']+pargs['outname']+pargs['outsuffixe']+'_theta_art',resunit=pargs['resunit'],saveformat=pargs['saveformat'])
        plot_image(data.polV,title='Circular degree of polarisation (%)'+titletheta,cmap=cmapV,norm=normV,scale=[-data.xmax,data.xmax,-data.ymax,data.ymax],rec=pargs['rec'],outname=pargs['outdir']+pargs['outname']+pargs['outsuffixe']+'_polV',resunit=pargs['resunit'],saveformat=pargs['saveformat'])


        if(pargs['vectormode']>0):
            mout.plot_vector(data.logI,data.P,thetaplot*np.pi/180.,title='Intensity (log10('+unitI+'))'+titletheta,suffixe=pargs['outsuffixe']+'_logI',cmap=cmaplog,norm=normlog,res=data.resimage,rec=pargs['rec'],directory=pargs['outdir'],filename=pargs['outname'],resunit=data.resunit,saveformat=pargs['saveformat'])
            mout.plot_vector(data.nscattmap,data.P,thetaplot*np.pi/180.,title='Averaged number of scatterings'+titletheta,suffixe=pargs['outsuffixe']+'_ndiff',cmap=pargs['cmap'],norm=normscatt,res=data.resimage,rec=pargs['rec'],directory=pargs['outdir'],filename=pargs['outname'],resunit=data.resunit,saveformat=pargs['saveformat'])
            mout.plot_vector(data.nphotstateff,data.P,thetaplot*np.pi/180.,title='Number of effective packets per pixel'+titletheta,suffixe=pargs['outsuffixe']+'_nphoteff',cmap=cmapeff,norm=normeff,res=data.resimage,rec=pargs['rec'],directory=pargs['outdir'],filename=pargs['outname'],resunit=data.resunit,saveformat=pargs['saveformat'])
            mout.plot_vector(data.Ip,data.P,thetaplot*np.pi/180.,title='Polarised intensity ('+unitI+')'+titletheta,suffixe=pargs['outsuffixe']+'_Ip',cmap=pargs['cmap'],res=data.resimage,rec=pargs['rec'],directory=pargs['outdir'],filename=pargs['outname'],resunit=data.resunit,saveformat=pargs['saveformat'])
            mout.plot_vector(data.logI,data.P,thetaplot*np.pi/180.,title='Intensity (log10('+unitI+'))'+titletheta,suffixe=pargs['outsuffixe']+'_clogI',contour=1,curve_range=[minlog,maxlog],cmap=cmaplog,norm=normlog,res=data.resimage,rec=pargs['rec'],directory=pargs['outdir'],filename=pargs['outname'],resunit=data.resunit,saveformat=pargs['saveformat'])
            mout.plot_vector(data.nscattmap,data.P,thetaplot*np.pi/180.,title='Averaged number of scatterings'+titletheta,suffixe=pargs['outsuffixe']+'_cndiff',contour=1,curve_pos=[0.5,1.5,2.5],cmap=pargs['cmap'],norm=normscatt,res=data.resimage,rec=pargs['rec'],directory=pargs['outdir'],filename=pargs['outname'],resunit=data.resunit,saveformat=pargs['saveformat'])



def plot_image(data,title='',cmap=[],norm=[],scale=[],rec=0,outname='test',resunit='pixel',saveformat='pdf'):
    ## ploting
    if(saveformat=='pdf'):
        outname+='.pdf'
    elif(saveformat=='png'):
        outname+='.png'
    if(cmap==[]):
        cmap='jet'
    elif(cmap=='yorick' or cmap=='Yorick'):
        cmap=cm.gist_earth
    fig,ax=plt.subplots(1)
    if(norm==[]):
        if(scale==[]):
            cax=ax.matshow(data,cmap=cmap,origin='lower')
        else:
            cax=ax.matshow(data,cmap=cmap,origin='lower',extent=scale)
    else:
        if(scale==[]):
            cax=ax.matshow(data,cmap=cmap,norm=norm,origin='lower')
        else:
            cax=ax.matshow(data,cmap=cmap,norm=norm,origin='lower',extent=scale)
    cb=fig.colorbar(cax)
    plt.title(title)
    plt.xlabel('Offset x ('+resunit+')')
    plt.ylabel('Offset y ('+resunit+')')
    if(rec==1):
        if(saveformat=='png'):
            fig.savefig(outname,dpi=600)
        else:
            fig.savefig(outname,rasterized=True,dpi=300)


def symmetry(data):
    ## QU rearrangement : use of symmetries
    I_nosym=np.copy(data.I)
    #if(sym>0):
    QU=np.array([data.Q,data.U])
    ndiffmap=data.nscattmap
    nphotstat=data.nphotstat
    nphotstateff=data.nphotstateff
    I=data.I
    QU1=[data.Q,data.U]
    QU2=np.zeros([2,data.nx,data.ny])
    QU3=np.zeros([2,data.nx,data.ny])
    QU4=np.zeros([2,data.nx,data.ny])
    I2=np.zeros([data.nx,data.ny])
    ndiffmap2=np.zeros([data.nx,data.ny])
    nphotstat2=np.zeros([data.nx,data.ny])
    nphotstateff2=np.zeros([data.nx,data.ny])
    I3=np.zeros([data.nx,data.ny])
    ndiffmap3=np.zeros([data.nx,data.ny])
    nphotstat3=np.zeros([data.nx,data.ny])
    nphotstateff3=np.zeros([data.nx,data.ny])
    I4=np.zeros([data.nx,data.ny])
    ndiffmap4=np.zeros([data.nx,data.ny])
    nphotstat4=np.zeros([data.nx,data.ny])
    nphotstateff4=np.zeros([data.nx,data.ny])
    for i in range(data.nx):
        for j in range(data.ny):
            QU2[0][i][j]=QU[0][i][data.ny-1-j]
            QU2[1][i][j]=-QU[1][i][data.ny-1-j]
            I2[i][j]=I[i][data.ny-1-j]
            ndiffmap2[i][j]=ndiffmap[i][data.ny-1-j]
            nphotstat2[i][j]=nphotstat[i][data.ny-1-j]
            nphotstateff2[i][j]=nphotstateff[i][data.ny-1-j]
    for i in range(data.nx):
        for j in range(data.ny):
            QU3[:,i,j]=QU[:,data.nx-1-i,data.ny-1-j]
            I3[i][j]=I[data.nx-1-i][data.ny-1-j]
            ndiffmap3[i][j]=ndiffmap[data.nx-1-i][data.ny-1-j]
            nphotstat3[i][j]=nphotstat[data.nx-1-i][data.ny-1-j]
            nphotstateff3[i][j]=nphotstateff[data.nx-1-i][data.ny-1-j]
            QU4[:,i,j]=QU2[:,data.nx-1-i,data.ny-1-j]
            I4[i][j]=I2[data.nx-1-i][data.ny-1-j]
            ndiffmap4[i][j]=ndiffmap2[data.nx-1-i][data.ny-1-j]
            nphotstat4[i][j]=nphotstat2[data.nx-1-i][data.ny-1-j]
            nphotstateff4[i][j]=nphotstateff2[data.nx-1-i][data.ny-1-j]
    QU+=QU2+QU3+QU4
    I+=I2+I3+I4
    ndiffmap+=ndiffmap2+ndiffmap3+ndiffmap4
    nphotstat+=nphotstat2+nphotstat3+nphotstat4
    for i in range(data.nx):
        for j in range(data.ny):
            nphotstateff[i,j]=max(nphotstateff[i,j],nphotstateff2[i,j],nphotstateff3[i,j],nphotstateff4[i,j])
    data.Q=QU[0,:,:]
    data.U=QU[1,:,:]
    data.I=I
    data.nscattmap=ndiffmap
    data.nphotstat=nphotstat
    data.nphotstateff=nphotstateff
    return data

def polar_reduction(I,Q,U,V=[]):
    Ip=np.sqrt(Q*Q+U*U)
    linpol=Ip/I 
    theta=0.5*np.arctan2(U,Q)*180./np.pi #to be checked
    Pq=Q/I
    Pu=U/I        
    if(V==[]):
        return Ip,linpol,theta
    else:
        circpol=V/I
        return Ip,linpol,theta,circpol


def extract_from_file(filerootname,pargs,fonction):
    """
    Function to read the files (with the root name 'filerootname') and extract the wanted quantities ('quants')
    """
    photonparams=Pparams()
    try:
        filelist=np.sort(os.listdir(pargs['path']))
        #print 'found files:',filelist
    except:
        print('No files found')
        return 0
    #if(pargs['plot_per_file']==1):
    try:
        nfig=np.max(plt.get_fignums())+1
    except:
        nfig=1
    nfile=0
    paramlist,data=select_params(pargs)
    nparamtokeep=len(paramlist)
    nligne=0
    fileopened=[]
    filenotopened=[]
    progbar=g_pgb.Patiencebar(valmax=len(filelist),up_every=1)
    for filename in filelist:
        if(filerootname in filename):
            if('_T_' not in filename and 'rho' not in filename):
                if(float(filename[-3-len(pargs['file_ext']):-len(pargs['file_ext'])])>=180/np.pi*pargs['thetamin'] and float(filename[-3-len(pargs['file_ext']):-len(pargs['file_ext'])])<=180/np.pi*pargs['thetamax']):
                    #print 'Reading file:',filename
                    nligne=len(open(pargs['path']+filename).readlines())
                    f = open(pargs['path']+filename,'r')
                    r = csv.reader(f, delimiter='\t')
                    n = 0
                    table=np.zeros([nparamtokeep,nligne])
                    for row in r: #read the file line per line
                        #conditions
                        if(conditions(pargs,row,photonparams)):
                            for i in range(nparamtokeep):
                                if(paramlist[i]!=13):
                                    table[i,n]=float(row[paramlist[i]])
                                else: # label and thus a str
                                    table[i,n]=row[paramlist[i]]
                            n+=1
                    data=fonction(table,n,data,pargs,nfig=nfig,filename=filename,nfile=nfile)
                    nfile+=1
                    fileopened.append(filename)
                else:
                    filenotopened.append(filename)
        progbar.update()
    data.fileopened=fileopened
    data.filenotopened=filenotopened
    return data

def conditions(pargs,row,photonparams):
    cond=True
    if(float(row[photonparams.wvl])<pargs['minwvl']):
        cond=False
    elif(float(row[photonparams.wvl])>pargs['maxwvl']):
        cond=False
    elif(float(row[photonparams.theta])<pargs['thetamin']):
        cond=False
    elif(float(row[photonparams.theta])>pargs['thetamax']):
        cond=False
    elif(float(row[photonparams.phi])<pargs['phimin']):
        cond=False
    elif(float(row[photonparams.phi])>pargs['phimax']):
        cond=False
    elif(pargs['label']!=-1 and row[photonparams.label] not in pargs['label']):
        cond=False
    elif(pargs['interact']!=-1 and int(row[photonparams.interact])!=pargs['interact']):
        cond=False
    elif(pargs['reem']!=-1 and int(row[photonparams.reem])!=pargs['reem']):
        cond=False
    elif(pargs['nscatt']!=-1 and int(row[photonparams.interact])-int(row[photonparams.reem])!=pargs['nscatt']):
        cond=False
    elif(int(row[photonparams.interact])-int(row[photonparams.reem])>pargs['nscattmax']):
        cond=False
    return cond

def select_params(pargs):
    """
    Function defining which parameters to read and what format to use for data
    """
    photonparams=Pparams()
    if(pargs['spectra']==1):
        if(pargs['polar']==0):
            paramlist=[photonparams.wvl,photonparams.en]
        else:
            #to include !!
            #paramlist=[photonparams.wvl,photonparams.en,photonparams.phi,photonparams.phiqu,photonparams.Q,photonparams.U,photonparams.V]
            paramlist=[photonparams.wvl,photonparams.en,photonparams.Q,photonparams.U,photonparams.V]
        data=Spectrum(pargs)
    elif(pargs['image']==1):
        if(pargs['polar']==0):
            paramlist=[photonparams.wvl,photonparams.en,photonparams.theta,photonparams.phi,photonparams.x,photonparams.y,photonparams.z,photonparams.interact,photonparams.reem]
        else:
            paramlist=[photonparams.wvl,photonparams.en,photonparams.theta,photonparams.phi,photonparams.x,photonparams.y,photonparams.z,photonparams.interact,photonparams.reem,photonparams.phiqu,photonparams.Q,photonparams.U,photonparams.V]
        data=Image(pargs)
    return paramlist,data

def spectral_reduction(table,n,data,pargs,nfig=[],filename='',nfile=0):
    """
    function to reduce spectra according to different requirement within the extract_from_file routine
    """
    tmp=table[0,:n]
    tmpI=table[1,:n]

    SEDtmp = 0*data.binning
    SEDtmpI = 0*data.binning

    if(pargs['polar']==1):
        tmpQ=table[2,:n]
        tmpU=table[3,:n]
        tmpV=table[4,:n]

    SEDtmpQ = 0*data.binning
    SEDtmpU = 0*data.binning
    SEDtmpV = 0*data.binning

    j=0
    for i in tmp: #attribute the photons to the binned spectral bands
        k = 0
        #print i,binning[k]
        while (i>data.binning[k]) * (k < pargs['bins']):
            k +=1
        SEDtmp[k]+=1
        SEDtmpI[k]+=tmpI[j]
        if(pargs['polar']==1):
           SEDtmpQ[k]+=tmpQ[j]*tmpI[j]
           SEDtmpU[k]+=tmpU[j]*tmpI[j]
           SEDtmpV[k]+=tmpV[j]*tmpI[j]
        j+=1
    if(pargs['plot_per_file']==1):
        plot_file(nfig,nfile,data.binning,SEDtmp,SEDtmpI,SEDtmpQ,SEDtmpU,SEDtmpV)

        #SED+=SEDtmp
    #return [SEDtmp,SEDtmpI]
    data.SED+=SEDtmp
    data.SEDI+=SEDtmpI
    if(pargs['polar']==1):
        data.SEDQ+=SEDtmpQ
        data.SEDU+=SEDtmpU
        data.SEDV+=SEDtmpV
    return data

def image_reduction(table,n,data,pargs,nfig=[],filename='',nfile=0):

    ### Translate photons into maps ###
    ## Initialisation
    cst=mc.Constant()
    influx=0.
    outflux=0.
    I=np.zeros([data.nx,data.ny])
    nscattmap=np.zeros([data.nx,data.ny])
    nphotstat=np.zeros([data.nx,data.ny])
    nphotstateff=np.zeros([data.nx,data.ny])
    if(pargs['polar']==1):
        Q=np.zeros([data.nx,data.ny])
        U=np.zeros([data.nx,data.ny])
        V=np.zeros([data.nx,data.ny])
    if(nfile==0):
        #checking dimensions for the first file
        data=check_dimension(data,table,pargs,n,cst)
    #dt #to implemant ?
    
    ## Translating phi reorientations
    if(pargs['phisym']==1):
        posx_t=np.sqrt(table[4,:n]*table[4,:n]+table[5,:n]*table[5,:n])*np.cos(-table[3,:n]+np.arctan2(table[5,:n],table[4,:n]))
        posy_t=np.sqrt(table[4,:n]*table[4,:n]+table[5,:n]*table[5,:n])*np.sin(-table[3,:n]+np.arctan2(table[5,:n],table[4,:n]))
    else:
        posx_t=table[4,:n]
        posy_t=table[5,:n]
    posz_t=table[6,:n]
    axex=-posx_t*np.cos(table[2,:n]*np.pi/180.)+posz_t*np.sin(table[2,:n]*np.pi/180.)
    axey=np.copy(posy_t)

    ## Adapting the energy into power --> not dependant of the observation time
    if(False): #to be implemented
        if Dt_tot>0:
            energy = energy/(dt*Dt_tot)
        else:
            energy = energy/dt
            #Ttot=sum(dt)

    #dx=xmax/nbin
    #dy=ymax/nbin
    
    if(nfile==0):
        data.posx=np.array([np.linspace(-data.xmax,data.xmax,data.nx)])+data.resimage*0.5
        data.posy=np.array([np.linspace(-data.ymax,data.ymax,data.ny)])+data.resimage*0.5
  
    ## Computing maps : QU, I, ndiff, nphot
    for i in range(n):
        wvlfact=1. # to implement filters profiles, based on table[0,i]
        if(np.abs(axex[i])<data.xmax and np.abs(axey[i])<data.ymax and table[1,i]>1.): # TBD: adapting the energy value to relative energy
            xx=int(round((axex[i]+data.xmax)/data.resimage))
            yy=int(round((axey[i]+data.ymax)/data.resimage))
            if(xx>=data.nx):
                print('Warning: x axis too far',xx)
            elif(yy>=data.ny):
                print('Warning: y axis too far',yy)
            else:
                if(pargs['polar']==1):
                    Q[xx,yy]+=(table[10,i]*np.cos(2.*(table[9,i]))-table[11,i]*np.sin(2.*(table[9,i])))*table[1,i]*wvlfact
                    U[xx,yy]+=(table[11,i]*np.cos(2.*(table[9,i]))+table[10,i]*np.sin(2.*(table[9,i])))*table[1,i]*wvlfact
                    V[xx,yy]+=table[12,i]*table[1,i]*wvlfact
                I[xx,yy]+=table[1,i]*wvlfact
                nscattmap[xx,yy]+=table[7,i]*table[1,i]*wvlfact
                nphotstat[xx,yy]+=1
                if(nphotstateff[xx,yy]<table[1,i]):
                    nphotstateff[xx,yy]=table[1,i]
                influx+=table[1,i]*wvlfact
        elif(np.abs(axex[i])>data.xmax or np.abs(axey[i])>data.ymax):
            outflux+=table[1,i]*wvlfact
    data.influx+=influx
    data.outflux+=outflux
    data.I+=I
    data.nscattmap+=nscattmap
    data.nphotstat+=nphotstat
    data.nphotstateff+=nphotstateff
    if(pargs['polar']==1):
        data.Q+=Q
        data.U+=U
        data.V+=V
    return data


def check_dimension(data,table,pargs,n,cst):
    if(data.xmax==-1 or data.ymax==-1):
        try:
            xmax=np.max(table[3,:n])*cst.pc
            ymax=np.max(table[4,:n])*cst.pc
            zmax=np.max(table[5,:n])*cst.pc
            data.xmax=np.max(np.array([xmax,ymax,zmax]))*2. # the factor 2 is present here as a margin as we are based here only on the first file
            data.ymax=data.xmax
            if(data.xmax==0):
                ## Security : if no photons, image dimension > 0
                data.xmax=10.
                data.ymax=10.
        except:
            data.xmax=10.
            data.ymax=10.
    if(data.resimage==-1):
        res=data.xmax/data.default_size #in m
        ## Determining the resolution
        if(data.resunit=='AU' or data.resunit=='au' or data.resunit=='UA' or data.resunit=='ua'):
            data.resimage=res/cst.AU
        elif(data.resunit=='pc'):
            data.resimage=res/cst.pc
        else:
            data.resimage=res
    return data


def plot_file(nfig,nfile,binning,SEDtmp,SEDtmpI,SEDtmpQ,SEDtmpU,SEDtmpV):
    if(pargs['polar']==0):
        if(pargs['color']==[]):
            #plt.plot((binning)*1e9,SEDtmp)
            mpol.plot_1D((binning)*1e9,SEDtmp,logy=pargs['logy'],title='SED per file',xlabel='Wavelength (nm)',ylabel='Intensity - number of photons packets',label=filename,logx=pargs['logx'],n=nfig)
            mpol.plot_1D((binning)*1e9,SEDtmpI,logy=pargs['logy'],title='SED per file',xlabel='Wavelength (nm)',ylabel='Intensity (J)',label=filename,logx=pargs['logx'],n=nfig+1)
        else:
            #plt.plot((binning)*1e9,SEDtmp,color=pargs[color][j])
            mpol.plot_1D((binning)*1e9,SEDtmp,logy=pargs['logy'],title='SED per file',xlabel='Wavelength (nm)',ylabel='Intensity - number of photons packets',label=filename,logx=pargs['logx'],n=nfig,color=pargs['color'][nfile])
            mpol.plot_1D((binning)*1e9,SEDtmpI,logy=pargs['logy'],title='SED per file',xlabel='Wavelength (nm)',ylabel='Intensity (J)',label=filename,logx=pargs['logx'],n=nfig,color=pargs['color'][nfile])
    else:
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
    save_spectra.writeto(path+filename,clobber=True)

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
    for i in xrange(2*N):
        for j in xrange(2*N):
            for k in xrange(2*N):
                T[i,j,k] = model.map.grid[i][j][k].T
    f = open('Output/'+prefix+'_T_map.dat','w')
    for i in xrange(2*N):
        for j in xrange(2*N):
            for k in xrange(2*N):
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

        for i in xrange(2*N):
            for j in xrange(2*N):
                for k in xrange(2*N):
                    #rho_sil[i,j,k] = model.map.grid[i][j][k].rho[0]
                    #rho_grap_ortho[i,j,k] = model.map.grid[i][j][k].rho[1]
                    #rho_grap_para[i,j,k] = model.map.grid[i][j][k].rho[2]
                    #rho_el[i,j,k] = model.map.grid[i][j][k].rho[3]
                    rho[i,j,k] = model.map.grid[i][j][k].rho[l]

        f = open('Output/'+prefix+'_rho_'+model.map.dust[l].name+'_map.dat','w')
        for i in xrange(2*N):
            for j in xrange(2*N):
                for k in xrange(2*N):
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
        for i in xrange(2*N):
            for j in xrange(2*N):
                for k in xrange(2*N):
                    T[i,j,k] = data.map.grid[i][j][k].T
    return T

def load3d_rho(data,grain=None):
    if isinstance(data,str)==True:
        rho=np.genfromtxt(data)
        N=int(((np.size(rho)+1)**(1/3))/2)
        rho=np.reshape(rho,(2*N,2*N,2*N))
    else:
        for l in xrange(len(data.map.dust)):
            if (data.map.dust[l].name == grain):
                N=data.map.N
                rho=np.zeros((2*N,2*N,2*N))
                for i in xrange(2*N):
                    for j in xrange(2*N):
                        for k in xrange(2*N):
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
                print("Mean ",model.map.dust[i].name," albedo in the used wvl :",alb*100,"%" )



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

