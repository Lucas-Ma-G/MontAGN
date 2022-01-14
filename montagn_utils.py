# -*- coding: utf-8 -*-
#from __future__ import division
import numpy as np
import scipy.integrate as sci
import scipy.optimize as opt
import matplotlib.pyplot as plt
import time
import tarfile
import os
import csv

import montagn_classes as mc
import montagn_polar as mpol

import mgui_colorbar as g_cb 
import mgui_random as g_rand
try:
    import patiencebar as g_pgb
except:
    import mgui_progbar as g_pgb

########    Constants (USI)    ########
b = 2.8977721e-3 #Wien's displacement cst
h = 6.62606957e-34
c = 299792458
kB = 1.3806488e-23
sigma = 5.670373e-8 #Stefan-Boltzmann constant, in USI
G= 6.67384e-11 #Gravitationnal constant, in USI

Msol = 1.9884e30 #kg/Msol
pc = 3.08567758e16 #m/pc
AU = 1.4960e11 #m/AU
nm = 1.0e-9
yr = 3.15576e+07 #s/yr

#############    function factories   ##############################

def spectrum_factory(wvl,spct_in,type_spct='F'):
    """
    Spectrum factory, build a wvl_proba distribution and method
    INPUT: n wavelengths, n-point spectrum in unit of Flux (W) !
                          --> lambda x Flambda or nu x Fnu W - or homogeneous (/m2)
                          if input spectrum in other unit --> use type_spct to specify Flam or Fnu
    """
    #Management of the wavelength array --> converting to log scale for interpolation
    #                                       using bins centered on the given values and not the values directly
    wvl=np.log10(wvl)
    wvl_bins=np.zeros(len(wvl)+1)
    wvl_bins[1:-1]=0.5*(wvl[1:]+wvl[:-1])
    wvl_bins[0]=wvl[0]-(wvl_bins[1]-wvl[0])
    wvl_bins[-1]=wvl[-1]+(-wvl_bins[-2]+wvl[-1])

    dwvl=wvl_bins[1:]-wvl_bins[:-1]

    #Management of the spectrum itself
    spct=spct_in*dwvl #Correct the input spectrum in case of non constant bin size
    #(does not affect the dimension because spct is divided by cumsum(spct) which recorrect bin size)
    if(type_spct=='Flam'):
        spct=spct*10**wvl #Convert the spectrum into W/m2 for interpolation purpose
    if(type_spct=='Fnu'):
        print('WARNING ! Using Fnu with input in wavelength ! Assuming that wvl is indeed in unit m.')
        spct=spct/10**wvl*mc.Constant().c #Convert the spectrum into W/m2 for interpolation purpose
    spct_N = np.zeros(len(spct)+1)
    spct_N[1:] = np.array(spct)*1./np.sum(np.array(spct)) #values for end of interval

    spct_NC = np.cumsum(spct_N) #values for end of interval --> 0 for 1st bin
    spct_NC = np.array(spct_NC)/np.max(np.array(spct_NC))

    #Factory
    def spectrum(wavelength):
        #return np.interp(wavelength,wvl,spct_N)
        return np.interp(np.log10(wavelength),wvl_bins,spct_N)
        #return mpol.loginterp(wvl,spct_N,wavelength)
    def wvl_proba(rand1):
        #return np.interp(rand1,spct_NC,wvl)
        return 10**np.interp(rand1,spct_NC,wvl_bins)
        #return mpol.loginterp(spct_NC,wvl,rand1)

    return spectrum, wvl_proba


def spectrum_factory_nolog(wvl,spct_in,type_spct='F'):
    """
    Spectrum factory, build a wvl_proba distribution and method
    INPUT: n wavelengths, n-point spectrum in unit of Flux (W) !
                          --> lambda x Flambda or nu x Fnu W - or homogeneous (/m2)
                          if input spectrum in other unit --> use type_spct to specify Flam or Fnu
    """
    #Management of the wavelength array --> converting to log scale for interpolation
    #                                       using bins centered on the given values and not the values directly
    wvl=np.log10(wvl)
    wvl_bins=np.zeros(len(wvl)+1)
    wvl_bins[1:-1]=0.5*(wvl[1:]+wvl[:-1])
    wvl_bins[0]=wvl[0]-(wvl_bins[1]-wvl[0])
    wvl_bins[-1]=wvl[-1]+(-wvl_bins[-2]+wvl[-1])

    dwvl=10**wvl_bins[1:]-10**wvl_bins[:-1]

    #Management of the spectrum itself
    spct=spct_in*dwvl #Correct the input spectrum in case of non constant bin size
    #(does not affect the dimension because spct is divided by cumsum(spct) which recorrect bin size)
    if(type_spct=='Flam'):
        spct=spct*10**wvl #Convert the spectrum into W/m2 for interpolation purpose
    if(type_spct=='Fnu'):
        print('WARNING ! Using Fnu with input in wavelength ! Assuming that wvl is indeed in unit m.')
        spct=spct/10**wvl*mc.Constant().c #Convert the spectrum into W/m2 for interpolation purpose
    spct_N = np.zeros(len(spct)+1)
    spct_N[1:] = np.array(spct)*1./np.sum(np.array(spct)) #values for end of interval

    spct_NC = np.cumsum(spct_N) #values for end of interval --> 0 for 1st bin
    spct_NC = np.array(spct_NC)/np.max(np.array(spct_NC))

    #Factory
    def spectrum(wavelength):
        #return np.interp(wavelength,wvl,spct_N)
        return np.interp(np.log10(wavelength),wvl_bins,spct_N)
        #return mpol.loginterp(wvl,spct_N,wavelength)
    def wvl_proba(rand1):
        #return np.interp(rand1,spct_NC,wvl)
        return 10**np.interp(rand1,spct_NC,wvl_bins)
        #return mpol.loginterp(spct_NC,wvl,rand1)

    return spectrum, wvl_proba

def spectrum_factory_old(wvl,spct):
    """n wavelengths, n-point spectrum"""
    wvl=np.log10(wvl)
    spct_N = np.array(spct)*1./np.sum(np.array(spct))
    #normalization of the spectrum to unit sum
    spct_NC = np.cumsum(spct_N)
    #cumulative spectrum, used as probability function
    #spct_N = np.array(spct_N)*1./max(spct_N)
    spct_NC = (np.array(spct_NC)*1.-np.min(spct_NC))/np.max(np.array(spct_NC)-np.min(spct_NC)) ###### DR
    #spct_NC = np.array(spct_NC)*wvl*1./max(spct_NC*wvl) ###### DR
    #renormalization to unit maximum
    #mpol.plot_1D(10**wvl,spct_N,n=1,logx=1)
    #mpol.plot_1D(10**wvl,spct_NC,n=1,symbol='r')
    #print spct_NC
    def spectrum(wavelength):
        #return np.interp(wavelength,wvl,spct_N)
        return np.interp(np.log10(wavelength),wvl,spct_N)
        #return mpol.loginterp(wvl,spct_N,wavelength)
    def wvl_proba(rand1):
        #return np.interp(rand1,spct_NC,wvl)
        return 10**np.interp(rand1,spct_NC,wvl)
        #return mpol.loginterp(spct_NC,wvl,rand1)
    return spectrum, wvl_proba


def spectrum_factory_old2(wvl,spct):
    """n wavelengths, n-point spectrum"""
    wvln=np.zeros(len(wvl)+2)
    wvln[1:-1]=np.log10(np.array(wvl))
    wvln[0]=np.log10(wvl[0]*0.9)
    wvln[-1]=np.log10(wvl[-1]*1.1)
    wvlmid=(wvln[1:]+wvln[:-1])*0.5
    spctn=np.zeros(len(spct)+2)
    spctn[1:-1]=np.array(spct)
    spctn[0]=0.
    spctn[-1]=0.
    spct_N = np.array(spctn)*1./np.sum(np.array(spctn))
    #normalization of the spectrum to unit sum
    spct_NC = np.cumsum(spct_N)[:-1]
    #cumulative spectrum, used as probability function
    #spct_N = np.array(spct_N)*1./max(spct_N)
    spct_NC = np.array(spct_NC)*1./max(spct_NC) ###### DR
    #spct_NC = np.array(spct_NC)*wvl*1./max(spct_NC*wvl) ###### DR
    #renormalization to unit maximum
    mpol.plot_1D(10**wvln,spct_N,n=1,logx=1)
    mpol.plot_1D(10**wvlmid,spct_NC,n=1,symbol='r')
    #print spct_NC
    def spectrum(wavelength):
        #return np.interp(wavelength,wvl,spct_N)
        return np.interp(np.log10(wavelength),wvln,spct_N)
        #return mpol.loginterp(wvl,spct_N,wavelength)
    def wvl_proba(rand1):
        #return np.interp(rand1,spct_NC,wvl)
        return 10**np.interp(rand1,spct_NC,wvlmid)
        #return mpol.loginterp(spct_NC,wvl,rand1)
    return spectrum, wvl_proba

#def spectrum_factory(wvl,T,spct):
#    """n wavelengths, m temperatures, n*m spectrum"""
#    spct_N = []
#    for i in range(len(T)):
#        spct_N.append(np.array(spct[i])*1./np.sum(spct[i]))
#    #normalization of the spectra to unit integral
#    spct_NC = []
#    for i in range(len(T)):
#        spct_NC.append(np.cumsum(spct_N[i]))
#    #cumulative spectra, used as probability function
#    def spectrum(wavelength, temp):
#        s = [np.interp(wavelength,wvl,spct_N[i]) for i in range(len(T))]
#        return np.interp(temp,T,s)
#    def wvl_proba(rand1, temp):
#        s = [np.interp(rand1,spct_NC[i],wvl) for i in range(len(T))]
#        return np.interp(temp,T,s)
#    return spectrum, wvl_proba

def BB_old(T,wvl):
    return 2*h*c**2/(wvl**5*(np.exp(h*c/(wvl*kB*T))-1))

def BB(T,wvl):
    """
    Return the value of emission of a black body at temperature T
    and wavelength wvl.
    Given in W.sr-1.m-3
    """
    return 2*h*c*c/(wvl*wvl*wvl*wvl*wvl*(np.exp(h*c/(wvl*kB*T))-1))

def BBs(T, Npoints=1000):
    """returns wavelength and black-body spectrum lists for a given temperature"""
    wbb = b/T #black body peak wavelength
    wvl = np.linspace(0.175,75,Npoints)*wbb #wvl of the black body peak at 10-6 relative flux
    sp = BB(T,wvl)
    sp[0] = 0
    sp[-1] = 0
    return wvl,sp


def grain_size(rmin,rmax,z):
    """minimum radius, maximum radius, powerlaw exponent (z =< 0).
    Return a size generator and the average section, needed in the Grain class"""
    if z == -1:
        def size(genseed):
            x = genseed.uniform()
            return (rmax/rmin)**x*rmin
        avsec = np.pi*(rmax**2-rmin**2)/(2*np.log(rmax/rmin))
    elif z == -3:
        def size(genseed):
            x = genseed.uniform()
            return ((rmax**(z+1)-rmin**(z+1))*x+rmin**(z+1))**(1./(z+1))
        avsec = 2*np.pi/(rmin**(-2)-rmax**(-2))*np.log(rmax/rmin)
    else:
        def size(genseed):
            x = genseed.uniform()
            return ((rmax**(z+1)-rmin**(z+1))*x+rmin**(z+1))**(1./(z+1))
        avsec = np.pi*(z+1)/(z+3)*(rmax**(z+3)-rmin**(z+3))/(rmax**(z+1)-rmin**(z+1))
    return size, avsec


def grain_size_surf(rmin,rmax,z):
    """minimum radius, maximum radius, powerlaw exponent (z =< 0).
    Return a size generator and the average section, needed in the Grain class
    Take into consideration the surface of the grain"""

    if z == -1:
        def size(genseed):
            x = genseed.uniform()
            return ((rmax**(z+3)-rmin**(z+3))*x+rmin**(z+3))**(1./(z+3))
        avsec = np.pi*(rmax**2-rmin**2)/(2*np.log(rmax/rmin))
    elif z == -3:
        def size(genseed):
            x = genseed.uniform()
            return (rmax/rmin)**x*rmin
        avsec = 2*np.pi/(rmin**(-2)-rmax**(-2))*np.log(rmax/rmin)
    else:
        def size(genseed):
            x = genseed.uniform()
            return ((rmax**(z+3)-rmin**(z+3))*x+rmin**(z+3))**(1./(z+3))
        avsec = np.pi*(z+1)/(z+3)*(rmax**(z+3)-rmin**(z+3))/(rmax**(z+1)-rmin**(z+1))
    return size, avsec

def Kp_tab(Cab):
    """Cab(wvl) : absorption coefficient
    Returns a tabulated version of precalculated Kp"""
    Temp = np.linspace(10,2000,100)
    kpt = []
    def KpBB(wvl,i):
        return BB(Temp[i],wvl)*Cab(wvl)
    for i in range(100):
        kpt.append(sci.quad(KpBB, 0.175*b/Temp[i], 75*b/Temp[i], args = i)[0]/(sigma*Temp[i]**4/np.pi))
    def Kp(T):
        return np.interp(T,Temp,kpt)
    return Kp

def Kp_expl(Cab):
    """Cab(wvl) : absorption coefficient
    Returns an explicit calculation of Kp"""
    def Kp(T):
        return sci.quad(lambda x:BB(T,x)*Cab(x), 0.175*b/T, 75*b/T)[0]/(sigma*T**4/np.pi)
    return Kp

def particle_mass():
    """
    Return the masses per particle
    """
    
    #Computing the mass per particle
    alpha=-3.5

    # silicate
    rmax_sil=0.25e-6
    rmin_sil=0.005e-6
    density_sil=3.3e3 #kg/m3 # ou 0.29*1e-3 ? ##cf Vincent Guillet 2008
    #particlemass_sil=density_sil*20./3.*np.pi*(np.sqrt(rmax_sil)-np.sqrt(rmin_sil))/(1./rmin_sil**2.5-1./rmax_sil**2.5) #kg/particle pour -3.5
    particlemass_sil=density_sil*particle_volume(rmin_sil,rmax_sil,alpha)

    # graphite ortho
    rmax_grap_ortho=0.25e-6
    rmin_grap_ortho=0.005e-6
    density_grap_ortho=2.2e3 #kg/m3
    #particlemass_grap_ortho=density_grap_ortho*20./3.*np.pi*(np.sqrt(rmax_grap_ortho)-np.sqrt(rmin_grap_ortho))/(1./rmin_grap_ortho**2.5-1./rmax_grap_ortho**2.5) #kg/particle pour -3.5
    particlemass_grap_ortho=density_grap_ortho*particle_volume(rmin_grap_ortho,rmax_grap_ortho,alpha)

    # graphite para
    rmax_grap_para=0.25e-6
    rmin_grap_para=0.005e-6
    density_grap_para=2.2e3 #kg/m3
    #particlemass_grap_para=density_grap_para*20./3.*np.pi*(np.sqrt(rmax_grap_para)-np.sqrt(rmin_grap_para))/(1./rmin_grap_para**2.5-1./rmax_grap_para**2.5) #kg/particle pour -3.5
    particlemass_grap_para=density_grap_para*particle_volume(rmin_grap_para,rmax_grap_para,alpha)

    # PAH neutral
    rmax_pah_neu=0.25e-6
    rmin_pah_neu=0.005e-6
    density_pah_neu=2.2e3 #kg/m3ah
    particlemass_pah_neu=density_pah_neu*particle_volume(rmin_pah_neu,rmax_pah_neu,alpha)

    # PAH ionised
    rmax_pah_ion=0.25e-6
    rmin_pah_ion=0.005e-6
    density_pah_ion=2.2e3 #kg/m3ah
    particlemass_pah_ion=density_pah_ion*particle_volume(rmin_pah_ion,rmax_pah_ion,alpha)

    # electron
    particlemass_el=9.109e-31 #kg
    
    return [particlemass_sil,particlemass_grap_ortho,particlemass_grap_para,particlemass_pah_neu,particlemass_pah_ion,particlemass_el]


def particle_mass_new(rmin,rmax,alpha):
    """
    Return the masses per particle
    """
    
    #Computing the mass per particle

    # silicate
    rmax_sil=rmax
    rmin_sil=rmin
    density_sil=3.3e3 #kg/m3 # ou 0.29*1e-3 ? ##cf Vincent Guillet 2008
    particlemass_sil=density_sil*particle_volume(rmin_sil,rmax_sil,alpha)

    # graphite ortho
    rmax_grap_ortho=rmax
    rmin_grap_ortho=rmin
    density_grap_ortho=2.2e3 #kg/m3
    particlemass_grap_ortho=density_grap_ortho*particle_volume(rmin_grap_ortho,rmax_grap_ortho,alpha)

    # graphite para
    rmax_grap_para=rmax
    rmin_grap_para=rmin
    density_grap_para=2.2e3 #kg/m3
    particlemass_grap_para=density_grap_para*particle_volume(rmin_grap_para,rmax_grap_para,alpha)

    # graphite para
    rmax_grap_para=rmax
    rmin_grap_para=rmin
    density_grap_para=2.2e3 #kg/m3
    particlemass_grap_para=density_grap_para*particle_volume(rmin_grap_para,rmax_grap_para,alpha)

    # PAH neutral
    rmax_pah_neu=0.25e-6
    rmin_pah_neu=0.005e-6
    density_pah_neu=2.2e3 #kg/m3ah
    particlemass_pah_neu=density_pah_neu*particle_volume(rmin_pah_neu,rmax_pah_neu,alpha)

    # PAH ionised
    rmax_pah_ion=0.25e-6
    rmin_pah_ion=0.005e-6
    density_pah_ion=2.2e3 #kg/m3ah
    particlemass_pah_ion=density_pah_ion*particle_volume(rmin_pah_ion,rmax_pah_ion,alpha)

    # electron
    particlemass_el=9.109e-31 #kg

    return [particlemass_sil,particlemass_grap_ortho,particlemass_grap_para,particlemass_pah_neu,particlemass_pah_ion,particlemass_el]

def particle_volume(rmin,rmax,alpha):
    if(alpha==-1):
        V=4./3.*np.pi*(rmax**(4.+alpha)-rmin**(4.+alpha))/np.log(rmax/rmin)/(4+alpha)
    elif(alpha==-4):
        V=4./3.*np.pi*np.log(rmax/rmin)/(rmax**(1.+alpha)-rmin**(1.+alpha))*(1+alpha)
    else:
        V=4./3.*np.pi*(rmax**(4.+alpha)-rmin**(4.+alpha))/(rmax**(1.+alpha)-rmin**(1.+alpha))*(1+alpha)/(4+alpha)
    return V


##############    Scattering phase functions    ################

def scat_uniform(r,wvl):
    r1 = np.random.random()
    r2 = np.random.random()
    dth = np.arccos(2*r1-1) #polar angle
    ph = r2*2*np.pi #azimuthal angle
    return dth, ph

def scat_HG_factory(g):
    """returns a Henyey-Greenstein phase function, given a g(r,wvl) coefficient"""
    def scat_HG(r,wvl):
        r1 = np.random.random()
        r2 = np.random.random()
        if abs(g(r,wvl)) < 1e-6 :
            dth = np.arccos(2*r1-1)
        else:
            dth = np.arccos((1+g(r,wvl)**2-((1-g(r,wvl)**2)/(1+g(r,wvl)*(2*r1-1)))**2)/(2*g(r,wvl)))
        ph = r2*2*np.pi #azimuthal angle
        return dth, ph
    return scat_HG

def scat_HGD_factory(g,alpha):
    """Henyey-Greenstein phase function, modified by Draine (ApJ 2003), given g(r,wvl) and alpha(r,wvl) coefficients"""
    #copy of scat_HG for now...
    def scat_HGD(r,wvl):
        r1 = np.random.random()
        r2 = np.random.random()
        if abs(g(r,wvl)) < 1e-6 :
            dth = np.arccos(2*r1-1)
        else:
            dth = np.arccos((1+g(r,wvl)**2-((1-g(r,wvl)**2)/(1+g(r,wvl)*(2*r1-1)))**2)/(2*g(r,wvl)))
        ph = r2*2*np.pi #azimuthal angle
        return dth, ph
    return scat_HGD

def scat_BHMie(r,wvl):
    """Interface to BHMie code"""
    pass

##############    Utility functions    ####################

def sort_grains_old(grain_list):
    """sorts a list of Grain instances by decreasing sublimation temperature"""
    #to be used before inputing the Grain list to the MC code
    tsub={}
    for i in range(len(grain_list)):
        tsub[grain_list[i].Tsub] = i
    t = sorted(tsub.keys(),reverse=True)
    s_list = []
    for i in t:
        s_list.append(grain_list[tsub[i]])
    return s_list,t


def sort_grains(grain_list):
    """sorts a list of Grain instances by decreasing sublimation temperature"""
    #to be used before inputing the Grain list to the MC code
    tsub=np.zeros(len(grain_list))
    for i in range(len(grain_list)):
        tsub[i]=grain_list[i].Tsub
    t = sorted(tsub,reverse=True)
    j = len(grain_list)-np.argsort(tsub)-1
    s_list = []
    for i in range(len(t)):
        s_list.append(grain_list[j[i]])
    return s_list,t

def temp_solver(ini,Tmax,func):
    return opt.fminbound(func, ini, Tmax)


########    CSV interface    ########

def load_spectrum(filename):
    """loads a spectrum from a file, returns Spectrum type object
    No header in the file. Format :
    wavelength (in m)    intensity (arbitrary)
    wavelength (in m)    intensity (arbitrary)
    ..."""
    f = open(filename,'r')
    r = csv.reader(f,delimiter='\t')
    wvl = []
    sp = []
    cst=mc.Constant()
    #Other lines are :  WVL then spectrum(wvl,T)
    for row in r:
        wvl.append(readvalue(row[0],cst))
        sp.append(readvalue(row[1],cst))
        #wvl.append(eval(row[0]))
        #sp.append(eval(row[1]))
    f.close()
    spectrum, wvl_proba = spectrum_factory(wvl,sp)
    S = mc.Spectrum(spectrum,wvl_proba)
    return S

def save_map(m):
    filename = raw_input('Output file name ? ')
    f = file(filename,'w')
    s = str(m.Rmax)+'\t'+str(m.res)+'\n' #write the resolution and size first
    for i in m.dust:
        s += str(i)+'\t' #write what kind of grains the dust contained
    s = s[:-1]+'\n'
    f.write(s)
    for i in m.grid:
        for j in i:
            for k in j:
                s = ''
                for t in k.rho:
                    s += str(t)+'\t'
                s = s[:-1]+'\n'
                f.write(s)
    f.close()
    return filename

def load_map(filename):
    """Warning !!
    The returned map has a list of grain names instead of real Grain object as a .dust attribute.
    Create the corresponding Grain instances and replace the list before running simulations !"""
    f = file(filename,'r')
    r = csv.reader(f,delimiter='\t')
    l1 = r.next()
    l2 = r.next()
    m = mc.Map(eval(l1[0]),eval(l1[1]),l2)
    for i in range(2*m.N):
        for j in range(2*m.N):
            for k in range(2*m.N):
                a = r.next()
                b = []
                for ii in a:
                    b.append(eval(ii))
                m.grid[i][j][k].rho = b
    return m

def save_model(mod):
    filemod = raw_input('Output file name ? ')
    f = file(filemod,'w')
    s = str(mod.AF)+'\n'
    for i in mod.Rsub:
        s += str(i)+'\t' #list of Rsub
    s += '\n'
    s += str(mod.energy)+'\n'
    s += '\nLight sources :\n'
    for i in mod.sources.list_sources:
        s += "Type : %s \n" %i.type
        s += "Luminosity = %.4e W \n \n" %i.lum
    print("Now writing the map :")
    mf = save_map(mod.map)
    s += mf
    f.write(s)
    f.close()

def load_model(filename):
    """loads and returns the extra parameters of a Model object
    Import / create the corresponding sources and map (described at the end of the file)
    to create the object"""
    f = file(filename,'r')
    r = csv.reader(f,delimiter='\t')
    AF = eval(r.next()[0])
    Rs = r.next()
    Rsub = []
    for i in Rs:
        if len(i)>0:
            j = eval(i)
            Rsub.append(j)
    en = eval(r.next()[0])
    return AF,Rsub,en


def read_Jnu_Draine():
    fm1=open('Input/jpah_draine.dat','r')
    jnu_pah = []
    for li in fm1:
        ln = li.split()
        lnum = [float(x) for x in ln]
        jnu_pah.append(lnum)     
    fm1.close()
    jnupah = np.array(jnu_pah)
    
    return jnupah
	

def read_CC_PAH_V1():
    table=[]
    wvl=[]
    table2=[]
    wvl2=[]
    i=0
    f=open("Input/CC_PAH_V1.dat",'r')
    for data in f.readlines():
        if(i not in [0,1,82,83]):
            if(i<83):
                wvl2.append(float(data.split()[0]))
                table2.append(float(data.split()[1]))
            else:
                wvl.append(float(data.split()[0]))
                table.append(float(data.split()[1]))

        i+=1
    f.close()
    #wvl=np.array(wvl)
    wvl=(np.array(wvl)+0.3)*0.75+1.
    table=np.array(table)
    wvl2=np.array(wvl2)
    table2=np.array(table2)
    indices=np.argsort(wvl2)
    wvl2=np.sort(wvl2)
    table4=table2[indices]
    #plt.figure()
    #plt.plot(wvl,table)
    #plt.show()
    #plt.figure()
    #plt.plot(np.log10(wvl2),np.log10(table4))
    #plt.show()
    wvl=1./np.array(wvl)*1e-6
    wvl2=wvl2*1e-6
    indices=np.argsort(wvl)
    wvl=np.sort(wvl)
    table3=table[indices]
    j=len(wvl)
    k=len(wvl2)
    wvlvect=np.zeros(j+k)
    tablevect=np.zeros(j+k)
    for i in range(j+k):
        if(i<j):
            wvlvect[i]=wvl[i]
            tablevect[i]=table3[i]
        else:
            wvlvect[i]=wvl2[i-j]
            tablevect[i]=table4[i-j]
    #plt.figure()
    #plt.plot(np.log10(wvlvect),np.log10(tablevect))
    #plt.show()
    return wvlvect, tablevect

def read_CC_PAH_V2():
    table=[]
    wvl=[]
    table2=[]
    wvl2=[]
    i=0
    f=open("Input/CC_PAH_V2.dat",'r')
    for data in f.readlines():
        if(i not in [0,1]):
            wvl.append(float(data.split()[0]))
            table.append(float(data.split()[1]))
        i+=1
    f.close()
    #wvl=np.array(wvl)
    wvl=np.array(wvl)*1e-6
    table=np.array(table)
    return wvl, table


########    Geometry    ########

#functions used by volume_interesect for numerical integration. circle_intersectN has N angles of the square inside the circle
def circle_intersect0(x,y,R,res):
    return 0

def circle_intersect1(x,y,R,res):
    xi = np.sqrt(max(R**2-y**2,0))-x #the "max" avoids NaN in case the precision leads to negative nbr.
    yi = np.sqrt(max(R**2-x**2,0))-y
    #circle-square intersections at (x,y+yi) and (x+xi,y)
    alpha = 2*np.arcsin(np.sqrt(xi**2+yi**2)/(2*R)) #angle of the arc
    return xi*yi*.5 + R**2*(alpha - np.sin(alpha))*.5 #triangle + circular segment

def circle_intersect2(x,y,R,res):
    if x >= y:
        x0 = x
        y0 = y
    else:
        x0 = y
        y0 = x #use the symmetry to simplify further calculations

    xi = np.sqrt(R**2-(y0+res)**2)-x0
    Xi = np.sqrt(R**2-y0**2)-x0
    #circle-square intersections at (x+xi,y+res) and (x+Xi,y)
    alpha = 2*np.arcsin(np.sqrt((Xi-xi)**2+res**2)/(2*R)) #angle of the arc
    return res*xi + res*(Xi-xi)*.5 + R**2*(alpha - np.sin(alpha))*.5 #rectangle + triangle + circular segment

def circle_intersect3(x,y,R,res):
    xi = np.sqrt(R**2-(y+res)**2)-x
    yi = np.sqrt(R**2-(x+res)**2)-y
    #circle-square intersections at (x+res,y+yi) and (x+xi,y+res)
    alpha = 2*np.arcsin(np.sqrt((res-xi)**2+(res-yi)**2)/(2*R)) #angle of the arc
    return xi*res+(res-xi)*yi + (res-xi)*(res-yi)*.5 + R**2*(alpha - np.sin(alpha))*.5 #2 rectangles + triangle + circular segment

def circle_intersect4(x,y,R,res):
    return res**2

def angles_check(x,y,R,res):
    #how many angles of the square inside the circle ? (works in the positive 4th of plane)
    N = 0
    if x**2+y**2 < R**2:
        N += 1
    if x**2+(y+res)**2 < R**2:
        N += 1
    if (x+res)**2+y**2 < R**2:
        N += 1
    if (x+res)**2+(y+res)**2 < R**2:
        N += 1
    return N

c_i_dict = {0:circle_intersect0, 1:circle_intersect1, 2:circle_intersect2, 3:circle_intersect3, 4:circle_intersect4}
#dictionnary to call the right function depending on the number of angles inside the circle

def circle_intersect(x,y,R,res):
    #general purpose circle_intersect for any number of angles inside the circle
    N = angles_check(x,y,R,res)
    return c_i_dict[N](x,y,R,res)

def volume_intersect(xc,yc,zc,res,AF,Rsub):
    """lower boundaries of the integrated cell, resolution, aperture of the funnel (in °), sublimation radius
    Returns the volume of the cubic cell that is inside the funnel or inside the sublimation sphere"""
    if abs(AF)>= 90:
        print("Aperture angle > 90° is impossible")
        return 0

    Z = max(abs(zc), abs(zc+res))
    z = min(abs(zc), abs(zc+res))
    y = min(abs(yc), abs(yc+res))
    x = min(abs(xc), abs(xc+res))
    #reorganize the geometry to return to the positive 8th of volume

    def R(zz):
        return max(np.abs(zz*np.tan(2*np.pi*AF/360)),np.sqrt(max(0,Rsub**2-zz**2)))
    def c_i_R(zz):
        return circle_intersect(x,y,R(zz),res)
    return sci.quad(c_i_R,z,Z)[0] #integrate the circle/square interection area from z to Z

def funnel_intersect(xc,yc,zc,res,AF): #not used (volume_intersect does the job better)
    """lower boundaries of the integrated cell, resolution, aperture of the funnel (in °)
    Returns the volume of the cubic cell that is inside the funnel"""
    if abs(AF)>= 90:
        print("Aperture angle > 90° is impossible")
        return 0
    Z = max(abs(zc), abs(zc+res))
    Y = max(abs(yc), abs(yc+res))
    X = max(abs(xc), abs(xc+res))
    z = min(abs(zc), abs(zc+res))
    y = min(abs(yc), abs(yc+res))
    x = min(abs(xc), abs(xc+res))
    t = np.tan(2*np.pi*AF/360)
    #reorganize the geometry to return to the positive 8th of volume
    #the volume will be integrated from z to Z, we have to find the slice area
    N_low = 0 #nbr of summits inside the funnel at z
    N_high = 0 #nbr of summits inside the funnel at Z
    if x**2+y**2 < z**2*t**2:
        N_low +=1
    if x**2+Y**2 < z**2*t**2:
        N_low +=1
    if X**2+y**2 < z**2*t**2:
        N_low +=1
    if X**2+Y**2 < z**2*t**2:
        N_low +=1
    if x**2+y**2 < Z**2*t**2:
        N_high +=1
    if x**2+Y**2 < Z**2*t**2:
        N_high +=1
    if X**2+y**2 < Z**2*t**2:
        N_high +=1
    if X**2+Y**2 < Z**2*t**2:
        N_high +=1

    if N_low == 0:
        if N_high == 0:
            return 0
        if N_high == 1:
            zt1 = np.sqrt(x**2+y**2)/t #transition
            return sci.quad(circle_intersect1,zt1,Z,args=(x,y,t))[0]
        if N_high == 2:
            zt1 = np.sqrt(x**2+y**2)/t #1st transition
            zt2 = min(np.sqrt(x**2+Y**2)/t, np.sqrt(X**2+y**2)/t) #2nd transition
            return sci.quad(circle_intersect1,zt1,zt2,args=(x,y,t))[0] + sci.quad(circle_intersect2,zt2,Z,args=(x,y,t,res))[0]
        if N_high == 3:
            zt1 = np.sqrt(x**2+y**2)/t #1st transition
            zt2 = min(np.sqrt(x**2+Y**2)/t, np.sqrt(X**2+y**2)/t) #2nd transition
            zt3 = max(np.sqrt(x**2+Y**2)/t, np.sqrt(X**2+y**2)/t) #3nd transition
            return sci.quad(circle_intersect1,zt1,zt2,args=(x,y,t))[0] + sci.quad(circle_intersect2,zt2,zt3,args=(x,y,t,res))[0] \
            + sci.quad(circle_intersect3,zt3,Z,args=(x,y,t,res))[0]
        if N_high == 4:
            zt1 = np.sqrt(x**2+y**2)/t #1st transition
            zt2 = min(np.sqrt(x**2+Y**2)/t, np.sqrt(X**2+y**2)/t) #2nd transition
            zt3 = max(np.sqrt(x**2+Y**2)/t, np.sqrt(X**2+y**2)/t) #3nd transition
            zt4 = np.sqrt(X**2+Y**2)/t #4th transition
            print(zt1,zt2,zt3,zt4)
            return sci.quad(circle_intersect1,zt1,zt2,args=(x,y,t))[0]  + sci.quad(circle_intersect2,zt2,zt3,args=(x,y,t,res))[0] \
            + sci.quad(circle_intersect3,zt3,zt4,args=(x,y,t,res))[0] + (Z-zt4)*res**2
    if N_low ==1:
        if N_high == 1:
            return sci.quad(circle_intersect1,z,Z,args=(x,y,t))
        if N_high == 2:
            zt2 = min(np.sqrt(x**2+Y**2)/t, np.sqrt(X**2+y**2)/t) #2nd transition
            return sci.quad(circle_intersect1,z,zt2,args=(x,y,t))[0] + sci.quad(circle_intersect2,zt2,Z,args=(x,y,t,res))[0]
        if N_high == 3:
            zt2 = min(np.sqrt(x**2+Y**2)/t, np.sqrt(X**2+y**2)/t) #2nd transition
            zt3 = max(np.sqrt(x**2+Y**2)/t, np.sqrt(X**2+y**2)/t) #3nd transition
            return sci.quad(circle_intersect1,z,zt2,args=(x,y,t))[0] + sci.quad(circle_intersect2,zt2,zt3,args=(x,y,t,res))[0] \
            + sci.quad(circle_intersect3,zt3,Z,args=(x,y,t,res))[0]
        if N_high == 4:
            zt2 = min(np.sqrt(x**2+Y**2)/t, np.sqrt(X**2+y**2)/t) #2nd transition
            zt3 = max(np.sqrt(x**2+Y**2)/t, np.sqrt(X**2+y**2)/t) #3nd transition
            zt4 = np.sqrt(X**2+Y**2)/t #4th transition
            return sci.quad(circle_intersect1,z,zt2,args=(x,y,t))[0] + sci.quad(circle_intersect2,zt2,zt3,args=(x,y,t,res))[0] \
            + sci.quad(circle_intersect3,zt3,zt4,args=(x,y,t,res))[0] + (Z-zt4)*res**2
    if N_low ==2:
        if N_high == 2:
            return sci.quad(circle_intersect2,z,Z,args=(x,y,t,res))[0]
        if N_high == 3:
            zt3 = max(np.sqrt(x**2+Y**2)/t, np.sqrt(X**2+y**2)/t) #3nd transition
            return sci.quad(circle_intersect2,z,zt3,args=(x,y,t,res))[0] + sci.quad(circle_intersect3,zt3,Z,args=(x,y,t,res))[0]
        if N_high == 4:
            zt3 = max(np.sqrt(x**2+Y**2)/t, np.sqrt(X**2+y**2)/t) #3nd transition
            zt4 = np.sqrt(X**2+Y**2)/t #4th transition
            return sci.quad(circle_intersect2,z,zt3,args=(x,y,t,res))[0] + sci.quad(circle_intersect3,zt3,zt4,args=(x,y,t,res))[0] \
            + (Z-zt4)*res**2
    if N_low == 3:
        if N_high == 3:
            return sci.quad(circle_intersect3,z,Z,args=(x,y,t,res))[0]
        if N_high == 4:
            zt4 = np.sqrt(X**2+Y**2)/t #4th transition
            return sci.quad(circle_intersect3,z,zt4,args=(x,y,t,res))[0] + (Z-zt4)*res**2
    if N_low == 4:
        print("WARNING ! Dust interaction occured in an empty cell !")
        return res**3


##############################################################################
#Lutils

#############    function factories   ##############################

def whereT(size,model2):
    for k in range(size*2):
        for j in range(size*2):
            for i in range(size*2):
                if(model2.map.grid[101+k-size][101+j-size][101+i-size].T>3.1):
                    print("T de",model2.map.grid[101+k-size][101+j-size][101+i-size].T,"en case [",k-10,",",j-10,",",i-10,"]")


def star_spectrum(T,n,lambdamin=1.0e-9,lambdamax=1.0,rec=1):
    """Fonction de création d'un spectre de n point d'une
    étoile de température T (en K) entre 1nm et 1m"""
    #lambdamin=1.0e-9
    #lambdamax=1.0
    wvl=np.exp(np.linspace(np.log(lambdamin),np.log(lambdamax),n))
    spectrum=BB(T,wvl)
    #spectrum=Qabs(T,wvl)
    filename="s"+str(T)+"_spectrum.dat"
    if(rec==1):
        f = open(filename,'w')
        for i in range(n):
            s = "%e\t"%wvl[i]
            s += "%e\n"%spectrum[i]
            f.write(s)
        f.close()
    else:
        return spectrum


def spdist_centre():
    #Fonction qui donne en sortie la position 0,0,0 
    #(pour une source par exemple)
    res=0,0,0
    return res

def FCabs(wvl_in):
    #Fonction qui renvoie le coefficient d'absorption Qabs
    #en fonction de la(/les) longueur d'onde en entrée

    #Entrée des correspondances wvl-coefs Q (à améliorer)
    wvl_tab=[1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]
    Qabs_sil=[1.0e-05,4.33e-05,4.56e-03,4.09e-01,2.67e-01,9.56e-01,9.67e-01,3.92e-01]
    Qsca_sil=[1.0e-11,2.53e-11,2.55e-07,1.80e-03,2.55e+00,1.32e+00,1.02e+00,1.05e+00]
    cabs_sil=np.log(Qabs_sil) #Qext_sil-Qsca_sil
    cs_sil=np.log(Qsca_sil+Qabs_sil) #Qext_sil
    wvl_in=np.log(wvl_in)
    wvl_tab=np.log(wvl_tab)
    
    #print type(wvl_in),np.ndim(wvl_in)
    if np.ndim(wvl_in)>0 : #type(wvl_in)!=float : #Cas d'un tableau de wvl
        cabs=np.linspace(0.,0.,np.size(wvl_in))
        for j in range(np.size(wvl_in)): #Boucle pour interpoler chaque valeur de lambda en entrée
            i=0
            wvl=wvl_in[j]
            #print j,wvl
            #while#
            if wvl>wvl_tab[i]: #wvl plus grande que l'intervalle défini
                #C2=wvl_tab[i]
                #C1=wvl_tab[i+1]
                cabs[j]=cabs_sil[i]#*(1-(wvl-C2)*(cabs_sil[i]-cabs_sil[i+1])/(C2-C1))
            else:
                if wvl<=wvl_tab[7]: #wvl plus petite que l'intervalle défini
                    #C2=wvl_tab[6]
                    #C1=wvl_tab[7]
                    cabs[j]=cabs_sil[7]#*(1-(wvl-C1)*(cabs_sil[6]-cabs_sil[7])/(C2-C1))
                else :
                    while wvl<=wvl_tab[i]: #wvl compris dans l'intervalle défini
                        if wvl>wvl_tab[i+1]:
                            C2=wvl_tab[i]
                            C1=wvl_tab[i+1]
                        i=i+1
                    #Interpolation linéaire (à améliorer)
                    cabs[j]=cabs_sil[i]+(wvl-C1)*(cabs_sil[i-1]-cabs_sil[i])/(C2-C1)
    else: #Cas d'un réel en entrée
        i=0
        wvl=wvl_in#[j]
        if wvl>wvl_tab[i]: #wvl plus grande que l'intervalle défini
            #C2=wvl_tab[i]
            #C1=wvl_tab[i+1]
            cabs=cabs_sil[i]#*(1-(wvl-C2)*(cabs_sil[i]-cabs_sil[i+1])/(C2-C1))
        else:
            if wvl<=wvl_tab[7]: #wvl plus petite que l'intervalle défini
                #C2=wvl_tab[6]
                #C1=wvl_tab[7]
                cabs=cabs_sil[7]#*(1-(wvl-C1)*(cabs_sil[6]-cabs_sil[7])/(C2-C1))
            else :
                while wvl<=wvl_tab[i]: #wvl compris dans l'intervalle défini
                    if wvl>wvl_tab[i+1]:
                        C2=wvl_tab[i]
                        C1=wvl_tab[i+1]
                    i=i+1
                #Interpolation linéaire (à améliorer)
                cabs=cabs_sil[i]+(wvl-C1)*(cabs_sil[i-1]-cabs_sil[i])/(C2-C1)
    return np.exp(cabs)

def FCsca(wvl):
    #Fonction qui renvoie le coefficient de diffusion Qsca
    #en fonction de la longueur d'onde en entrée

    #Entrée des correspondances wvl-coefs Q (à améliorer)
    wvl_tab=[1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]
    Qabs_sil=[1.0e-05,4.33e-05,4.56e-03,4.09e-01,2.67e-01,9.56e-01,9.67e-01,3.92e-01]
    Qsca_sil=[1.0e-11,2.53e-11,2.55e-07,1.80e-03,2.55e+00,1.32e+00,1.02e+00,1.05e+00]
    cabs_sil=np.log(Qabs_sil) #Qext_sil-Qsca_sil
    csca_sil=np.log(Qsca_sil+Qabs_sil) #Qext_sil
    wvl=np.log(wvl)
    wvl_tab=np.log(wvl_tab)
    i=0
    if wvl>wvl_tab[i]: #wvl plus grande que l'intervalle défini
        #C2=wvl_tab[i]
        #C1=wvl_tab[i+1]
        csca=csca_sil[i]#*(1-(wvl-C2)*(csca_sil[i]-csca_sil[i+1])/(C2-C1))
    else:
        if wvl<=wvl_tab[7]: #wvl plus petite que l'intervalle défini
            #C2=wvl_tab[6]
            #C1=wvl_tab[7]
            csca=csca_sil[7]#*(1-(wvl-C1)*(csca_sil[6]-csca_sil[7])/(C2-C1))
        else :
            #test=1
            #while #test==1:
            while wvl<=wvl_tab[i]: #wvl compris dans l'intervalle défini
                if wvl>wvl_tab[i+1]:
                    C2=wvl_tab[i]
                    C1=wvl_tab[i+1]
                    #test=0
                    #print C1,wvl,C2,csca_sil[i],csca_sil[i+1],i
                i=i+1
            #Interpolation linéaire (à améliorer)
            #print i
            csca=csca_sil[i]+(wvl-C1)*(csca_sil[i-1]-csca_sil[i])/(C2-C1)
    return np.exp(csca)

def Qcompute(x,x_m,Q):
    #
    Qx=loginterp(x_m,Q,x)
    return Qx

def kcompute(T,x_m,Q,r):
    #Fonction qui retourne l'intégrale Kv Bv(T) en fonction de T en entrée
    #Actuellement non valable
    #print "Q de ",loginterp(x_m,Q,2*np.pi*r/(0.175*b/T)),"pour x =",2*np.pi*r/(0.175*b/T)
    #print "Q de ",loginterp(x_m,Q,2*np.pi*r/(75*b/T)),"pour x =",2*np.pi*r/(75*b/T)
    #k=sci.quad(lambda x:BB(T,x)*loginterp(x_m,Q,2*np.pi*r/x), 0.175*b/T, 75*b/T)[0]/(sigma*T**4/np.pi)
    k=sci.quad(lambda x:BB(T,x)*np.interp(2*np.pi*r/x,x_m,Q), 0.175*b/T, 75*b/T)[0]/(sigma*T**4/np.pi)
    #Remplacer Q par C
    return k


    
def Qabs_a2_MRN_compute(a,Q):
    #Fonction qui retourne l'intégrale de Qabs a^2  a^-3.5 pour chaque lambda 
    k = sci.simps(Q*np.power(a,-1.5),a)

    return k

def Qabs_a2_MRN_B_compute(wvl, qa2, Tdust):
    #Fonction qui retourne l'intégrale de <Qabs a^2>B(lambda) pour chaque température 
    k = sci.simps(BB(Tdust,wvl)*qa2, wvl)

    return k



def scat_rayleigh(r,wvl):
    #Fonction qui renvoie en fonction du paramètre de forme (longueur
    #d'onde et rayon du grain) les angles d'une diffusion Rayleigh
    r1 = np.random.random()
    r2 = np.random.random()
    dth = 0.375*(1.0+(2*r1-1)*(2*r1-1)) #polar angle
    ph = r2*2*np.pi #azimuthal angle
    return dth, ph


def loginterp_old(x,y,x0):
    #Fonction qui interpole logarithmiquement un point y0 selon deux vecteurs
    #de meme taille se correspondant, x et y, à l'abscisse x0 parmi x.
    #Attention !! x et y doivent etre de meme taille, positives
    # ET x doit etre croissant

    #eps1=-min(min(x),min(y))+2.0 #1e-20+min(min(x),min(y))*1e-2
    #print eps1
    #eps=np.zeros(np.size(x))+eps1
    x=np.log(x)#+eps)
    y=np.log(y)#+eps)
    x0=np.log(x0)#+eps1)
    #print eps, eps1, x0
    nx=np.size(x)
    #print 'nx : ',nx,', x : ',x

    if np.ndim(x0)>0 : #type(x0)!=float : #Cas d'un tableau de x0
        n=np.size(x0)
        y0=np.linspace(0.,0.,n)
        for j in range(n): #Boucle pour interpoler chaque valeur de x en entrée
            i=0
            xtmp=x0[j]
            if xtmp>=x[nx-1]: #x0 plus grand que l'intervalle défini
                y0[j]=y[nx-1]
            else:
                if xtmp<=x[0]: #x0 plus petit que l'intervalle défini
                    y0[j]=y[0]
                else :
                    while xtmp>=x[i]: #x0 compris dans l'intervalle défini
                        if xtmp<x[i+1]:
                            C1=x[i]
                            C2=x[i+1]
                        i=i+1
                    #Interpolation linéaire
                    i=i-1
                    y0[j]=y[i]+(xtmp-C1)*(y[i+1]-y[i])/(C2-C1)
        y0=np.exp(y0)#-eps
    else: #Cas d'un réel en entrée
        i=0
        if x0>=x[nx-1]: #x0 plus grand que l'intervalle défini
            y0=y[nx-1]
        else:
            if x0<=x[0]: #x0 plus petit que l'intervalle défini
                y0=y[0]
            else :
                while x0>=x[i]: #x0 compris dans l'intervalle défini
                    #print i, x[i],x0,x[i+1]
                    if x0<x[i+1]:
                        C1=x[i]
                        C2=x[i+1]
                    i=i+1
                #Interpolation linéaire
                i=i-1
                y0=y[i]+(x0-C1)*(y[i+1]-y[i])/(C2-C1)
        y0=np.exp(y0)#-eps1
    return y0

def Qaapi(grain,wvl,cst):
    """ 
    Compute the value of Q*a*a*pi
    """
    if(grain.typeg=='electrons'):
        tau=cst.sigt
    elif('pah' in grain.typeg):
        tau=KPAH(grain,wvl)
    else:
        Qt=np.transpose(grain.Qext)
        xt=np.transpose(grain.x_M)
        rmin=grain.rmin
        rmax=grain.rmax
        alpha=grain.alpha
        wvlvect=grain.wvlvect
        #rmin=0.005*1e-6
        #rmax=0.25*1e-6
        #iwvl=int(np.floor(wvl))
        iwvl=np.argsort(np.argsort(np.append(wvlvect,wvl)))[len(wvlvect)]-1
        if(iwvl==len(wvlvect)-1): #case where wvl is greater than the wvl vect
            iwvl-=1
            #print "xxxxxxxxx"
        #a1=x_M[:,iwvl-1]*wvl/(2.*np.pi)
        #a2=x_M[:,iwvl]*wvl/(2.*np.pi)
        #a=x_M[:,iwvl]*wvl/(2.*np.pi)
        #a=(a1+a2)*0.5
        a=grain.sizevect
        Q=[]
        #print iwvl,wvl,wvlvect[iwvl],wvlvect[0],wvlvect[105]
        for i in range(len(xt[:,iwvl])):
            #a.append(loginterp([grain.wvlvect[iwvl],grain.wvlvect[iwvl+1]],[xt[i,iwvl],xt[i,iwvl+1]],wvl)*wvl/(2.*np.pi))
            #a.append(grain.rmin+i*grain.rstep)
            Q.append(mpol.loginterp([grain.wvlvect[iwvl],grain.wvlvect[iwvl+1]],[Qt[i,iwvl],Qt[i,iwvl+1]],wvl))
        #Q=(Qt[:,iwvl-1]+Qt[:,iwvl])*0.5
        Qaa=0
        compt=0
        for i in range(np.shape(Q)[0]):
            if(i<np.shape(Q)[0]-1):
                da=a[i+1]-a[i]
            else:
                da=a[i]-a[i-1]
            if(a[i]<rmax and a[i]>rmin):
                #Qaa+=Q[i]*da*2.5/(a[i]**1.5*(1./(rmin**2.5)-1./(rmax**2.5)))
                if(alpha==-1):
                    Qaa+=Q[i]*da*a[i]/np.log(rmax/rmin)
                else:
                    Qaa+=Q[i]*da*(alpha+1.)*a[i]**(2.+alpha)/(rmax**(1+alpha)-rmin**(1+alpha))                    
                compt+=1
                #print('Q :', Qaa)
        if(compt==0):
            print('WARNING : Resolution too low in x')
        tau=Qaa*np.pi
    #print 'a :',a,'tau :',tau
    return tau

def KPAH(pah,wvl):
    #k=np.interp(wvl,pah.CC_wvl,pah.CC)
    k=mpol.loginterp(pah.CC_wvl,pah.CC,wvl) #in cm2.C-1
    k*=1e-22*pah.Nc #in m2
    return k

def av_Nc(pah,niter=100):
    Nc=[25,50,100,200,500,1e3,1e4,1e5] #Number of C atoms
    g_rad=np.array([4.e-4,5.e-4,6.e-4,8.e-4,1.e-3,1.1e-3,3.e-3,6.e-3])*1e-6 #Corresponding radius (in m)
    dist=np.logspace(np.log10(pah.rmin),np.log10(pah.rmax),niter)
    cc=0 #averaged number of carbons atoms per pah
    for i in range(niter):
        if(i==0):
            dr=dist[i+1]-dist[i]
        elif(i==niter-1):
            dr=dist[i]-dist[i-1]
        else:
            dr=(dist[i+1]-dist[i-1])*0.5
        cc+=mpol.loginterp(g_rad,Nc,dist[i])*dist[i]**pah.alpha*dr
    if(pah.alpha==-1):
        integdist=np.log(rmax/rmin)
    else:
        integdist=(pah.alpha+1)/(pah.rmax**(pah.alpha+1)-pah.rmin**(pah.alpha+1))
    Nc=cc*integdist
    #Nc=10. #tmp
    pah.Nc=Nc
    
def get_tau(Q,x_M,wvl,rmin,rmax,alpha):
    """ 
    Compute the value of Q*a*a*pi
    """

    #rmin=0.005*1e-6
    #rmax=0.25*1e-6
    a=x_M*wvl/(2.*np.pi)
    Qaa=0
    compt=0
    for i in range(len(Q)):
        if(i<len(Q)-1):
            da=a[i+1]-a[i]
        else:
            da=a[i]-a[i-1]
        if(a[i]<rmax and a[i]>rmin):
            #Qaa+=Q[i]*da*2.5/(a[i]**1.5*(1./(rmin**2.5)-1./(rmax**2.5)))
            if(alpha==-1):
                Qaa+=Q[i]*da*a[i]/np.log(rmax/rmin)
            else:
                Qaa+=Q[i]*da*(alpha+1.)*a[i]**(2.+alpha)/(rmax**(1+alpha)-rmin**(1+alpha))  
            compt+=1
            #print('Q :', Qaa)
    if(compt==0):
        print('WARNING : Resolution too low in x')
    tau=Qaa*np.pi
    #print 'a :',a,'tau :',tau
    return tau

def get_tauel(rmin,rmax,alpha):
    """ 
    Compute the value of Q*a*a*pi
    """

    #rmin=0.005*1e-6
    #rmax=0.25*1e-6
    #a=x_M*wvl/(2.*np.pi)
    #Qaa=0
    #compt=0
    #for i in range(len(Q)):
    #    if(i<len(Q)-1):
    #        da=a[i+1]-a[i]
    #    else:
    #        da=a[i]-a[i-1]
    #    if(a[i]<rmax and a[i]>rmin):
    #        Qaa+=Q[i]*da*2.5/(a[i]**1.5*(1./(rmin**2.5)-1./(rmax**2.5)))
    #        compt+=1
    #        #print('Q :', Qaa)
    #if(compt==0):
    #    print('WARNING : Resolution too low in x')
    #tau=Qaa*np.pi
    #print 'a :',a,'tau :',tau
    return 0 #tau

def jnu_pah_emission(u,jnupah):
    '''
    return the spectrum of emission by PAH and the pertinent wavelength range
    for a given value of the density of radiation u (ratio to the standard ISRF)
    jnupah is a numpy array containing the wavelength and Draine data for 4.6% and .47% PAH  
    '''
	
    lgu = np.log10(u)
    #extrem values
    if(lgu>4):
        lgu=4.
    elif(lgu<-1):
        lgu=-1.
    nlgu = int(np.floor(lgu))
    n0 = nlgu + 1
    v = lgu - nlgu 
    
    #print np.shape(jnupah)
    #print lgu, nlgu
    #print n0
    y1 = jnupah[:,2*n0+1]
    z1 = jnupah[:,2*n0+2]
    y2 = jnupah[:,2*n0+3]
    z2 = jnupah[:,2*n0+4]
    y = y1*(1.-v)+y2*v
    z = z1*(1.-v)+z2*v
    w = 10**z - 10**y
    domain = np.where(w > 0.)
    #j_pah = (w[domain])
    j_pah = w
    #wvl = 10.**(jnupah[domain,0])
    wvl = 10.**(jnupah[:,0])
    #wvl = wvl[0,:] * 1.e-6
    wvl = wvl * 1.e-6
    return wvl[::-1], j_pah[::-1]  
    


def plot_Miephases(x,grains='silicates',nang=999,rrange=[],savename='test',rec=0):

    if(grains=='silicates'):
        typeg=0
        tsub=1400 #K Temp de sublimation, inutilisée
        rsub=0.0
        rmin=0.01e-6 #m Rayon min des grains 0.005 0.2
        rmax=0.1e-6 #m Rayon max des grains
        alpha=-3.5 #exposant de la loi de puissance des grains (<=0)
        rho=3.3 #kg/m3 # ou 0.29*1e-3 ? ##cf Vincent Guillet 2008 p120 (Jones et al., 1994
    elif(grains=='electrons'):
        typeg=3
        tsub=1e10 #K Temp de sublimation, inutilisée
        rsub=0.0
        rmin=4.600*1.0e-15 #m Rayon min des e- get from sqrt(sigma_T/pi) with Sigma_T=6.65e-29 m2
        rmax=4.602*1.0e-15 #m Rayon max des e-
        alpha=0 #exposant de la loi de puissance des grains (<=0)
        rho=mass_el
    elif(grains=='graphites_o'):
        typeg=1
        tsub=1700 #K Temp de sublimation 1997-Thatte
        #ortho
        rmin=0.01e-6 #0.005*1.0e-6 #m Rayon min des grains 0.005 0.2
        rmax=0.1e-6 #0.25*1.0e-6 #m Rayon max des grains
        alpha=-3.5 #-3.5 #exposant de la loi de puissance des grains (<=0)
        rho=2.2 #kg/m3
    elif(grains=='graphites_p'):
        #para :
        typeg=2
        tsub=1700 #K Temp de sublimation 1997-Thatte
        rmin=0.01e-6 #0.005*1.0e-6 #m Rayon min des grains 0.005 0.2
        rmax=0.1e-6 #0.25*1.0e-6 #m Rayon max des grains
        alpha=-3.5 #-3.5 #exposant de la loi de puissance des grains (<=0)
        rho=2.2 #kg/m3

    size,avsec=grain_size_surf(rmin,rmax,alpha)
    grain=mc.Grain(grains,tsub,grain_size_surf(rmin,rmax,alpha)[0],avsec,rmin/0.9,rmax,alpha,rho,typeg,1)

    plotinfo=[]
    nsize=2
    sizes_str=['10 nm','100 nm']
    sizes=[0.01e-6,0.1e-6]
    genseed=g_rand.gen_generator()
    force_wvl=[]
    
    phase=mpol.fill_mueller(nang,nsize,genseed,grain,plotinfo,force_wvl)
    phaseNorm,gw,phaseEnv=mpol.init_phase(nang,nsize,phase,genseed,grain,plotinfo,ret=1)

    x_M=np.zeros(np.shape(grain.x_M))
    phasex=np.zeros(np.shape(phase))
    phaseN=np.zeros(np.shape(phase))
    phaseE=np.zeros(np.shape(phase))
    l=len(grain.wvlvect)
    for i in range(l):
        x_M[i,:]=grain.x_M[l-i-1,:]
        phasex[:,i,:]=phase[:,l-i-1,:]
        phaseN[:,i,:]=phaseNorm[:,l-i-1,:]
        phaseE[:,i,:]=phaseEnv[:,l-i-1,:]

    #return x_M,phasex
    k=0
    fig=[]
    ax=[]
    for i in range(3+len(x)):
        fig_t,ax_t=plt.subplots(1,subplot_kw=dict(polar=True))
        fig.append(fig_t)
        ax.append(ax_t)
    for xx in x:
        wvl1=2.0*np.pi*0.01e-6/xx
        wvl2=2.0*np.pi*0.1e-6/xx
        iwvl=[np.argsort(np.argsort(np.append(grain.wvlvect,wvl1)))[len(grain.wvlvect)]-1,np.argsort(np.argsort(np.append(grain.wvlvect,wvl2)))[len(grain.wvlvect)]-1]
        #print wvl1, wvl2
        for i in range(2):
            #print grain.wvlvect[iwvl[i]]
            #print grain.x_M[iwvl[i]-1,i]
            #print grain.x_M[iwvl[i],i]
            #print grain.x_M[iwvl[i]+1,i]
            phasexx=np.zeros(nang)
            phaseNN=np.zeros(nang)
            phaseEE=np.zeros(nang)
            for j in range(nang):
                phasexx[j]=np.interp(xx,x_M[:,i],phasex[j,:,i])
                phaseNN[j]=np.interp(xx,x_M[:,i],phaseN[j,:,i])
                phaseEE[j]=np.interp(xx,x_M[:,i],phaseE[j,:,i])

            #plotphase_xfix(phase[:,iwvl[i],i],title='Phase function for a = '+sizes_str[i]+' ('+grain.name+')',label='x = '+str(grain.x_M[iwvl[i],i]),n=i,legendpos=3)
            mpol.plotphase_xfix(phasexx,title='Phase function for a = '+sizes_str[i]+' ('+grain.name+')',label='x = '+str(xx),n=i,icolor=k,legendpos=3,rrange=rrange,ax=ax[i])
            if(i==1):
                if(k==0):
                    mpol.plotphase_xfix(phasexx,title='x = '+str(xx)+' and a = '+sizes_str[i]+' ('+grain.name+')',label='Phase function',n=i,icolor=0,legendpos=3,rrange=rrange,normsin=0,ax=ax[2])
                    mpol.plotphase_xfix(phasexx,title='x = '+str(xx)+' and a = '+sizes_str[i]+' ('+grain.name+')',label='Probability density',n=i,icolor=1,legendpos=3,rrange=rrange,normsin=-1,ax=ax[2])
                mpol.plotphase_xfix(phasexx,title='x = '+str(xx)+' and a = '+sizes_str[i]+' ('+grain.name+')',label='Phase function',n=k+2,icolor=0,legendpos=3,rrange=rrange,ax=ax[k+3])
                mpol.plotphase_xfix(phaseEE,title='x = '+str(xx)+' and a = '+sizes_str[i]+' ('+grain.name+')',label='Envelope phase function',n=k+2,icolor=1,legendpos=3,rrange=rrange,ax=ax[k+3])
                mpol.plotphase_xfix(phaseNN,title='x = '+str(xx)+' and a = '+sizes_str[i]+' ('+grain.name+')',label='Normalised phase function',n=k+2,icolor=2,legendpos=3,rrange=rrange,ax=ax[k+3])
        k+=1
    if(rec==1):
        for i in range(2+len(x)):
            fig[i].savefig(savename+'_'+str(i)+'.pdf',rasterized=True,dpi=300)
    return grain



def readvalue(truc,cst,array=0):
    if(array==0):
        if('pc' in truc):
            value=np.float(truc[:-2])*cst.pc
        elif('AU' in truc or 'au' in truc):
            value=np.float(truc[:-2])*cst.AU
        else:
            value=np.float(truc)
        return value
    else:
        valuetab=np.zeros(len(truc))
        #print truc
        for i in range(len(truc)):
            trucval=truc[i]
            #print trucval
            if('pc' in trucval):
                value=np.float(trucval[:-2])*cst.pc
            elif('AU' in trucval or 'au' in trucval):
                value=np.float(trucval[:-2])*cst.AU
            else:
                value=np.float(trucval)
            valuetab[i]=value
        return valuetab


def read_paramfile(filenameparam,info):
    cst = mc.Constant()
    dust_pop = []
    sources = []
    densturb = []
    spherepower = []
    gaussianprofile = []
    cloud = []
    clump = []
    torus = []
    cone = []
    AGN_simple = []
    torus_Murakawa = []    
    cylinder = []
    denspower = []
    shell = []
    fractal = []
    fp = open(filenameparam,'r')
    for line in fp:
        #print(repr(line))
        line = line.strip()
        col = line.split()
        #print(col)
        if(col[0] == 'dust'):
            dust_pop.append([col[1],col[2],readvalue(col[3],cst),readvalue(col[4],cst),readvalue(col[5],cst),readvalue(col[6],cst)]) 
            if(info.verbose==1):
                print(dust)
        if(col[0] == 'denspower'):
            denspower.append([col[1][1:-1].split(','),readvalue(col[2][1:-1].split(','),cst,array=1),readvalue(col[3],cst),readvalue(col[4],cst),readvalue(col[5],cst)]) 
            if(info.verbose==1):
                print(denspower)
        if(col[0] == 'spherepower'):
            spherepower.append([col[1][1:-1].split(','), readvalue(col[2][1:-1].split(','),cst,array=1), readvalue(col[3],cst),readvalue(col[4],cst)]) 
            if(info.verbose==1):
                print(spherepower)
        if(col[0] == 'gaussianprofile'):
            gaussianprofile.append([col[1][1:-1].split(','), readvalue(col[2][1:-1].split(','),cst,array=1), readvalue(col[3],cst),readvalue(col[4],cst),readvalue(col[5],cst),readvalue(col[6],cst),readvalue(col[7],cst),readvalue(col[8],cst)]) 
            if(info.verbose==1):
                print(gausianprofile)
        if(col[0] == 'torus_Murakawa'):
            torus_Murakawa.append([col[1][1:-1].split(','),readvalue(col[2][1:-1].split(','),cst,array=1),readvalue(col[3],cst),readvalue(col[4],cst),readvalue(col[5],cst),readvalue(col[6],cst)])
            if(info.verbose==1):
                print(torus_Murakawa)
        if(col[0] == 'torus'):
            torus.append([col[1][1:-1].split(','),readvalue(col[2][1:-1].split(','),cst,array=1),readvalue(col[3],cst),readvalue(col[4],cst)])
            if(info.verbose==1):
                print(torus)
        if(col[0] == 'cone'):
            cone.append([col[1][1:-1].split(','),readvalue(col[2][1:-1].split(','),cst,array=1),readvalue(col[3],cst),readvalue(col[4],cst)])
            if(info.verbose==1):
                print(cone)
        if(col[0] == 'AGN_simple'):
            #torus_const.append([col[1][1:-1].split(','),readvalue(col[2],cst),readvalue(col[3],cst),readvalue(col[4],cst),readvalue(col[5],cst)])
            AGN_simple.append([col[1][1:-1].split(','),readvalue(col[2][1:-1].split(','),cst,array=1),readvalue(col[3][1:-1].split(','),cst,array=1),readvalue(col[4][1:-1].split(','),cst,array=1),readvalue(col[5],cst),readvalue(col[6],cst),readvalue(col[7],cst),readvalue(col[8],cst)])
            if(info.verbose==1):
                print(AGN_simple)
        if(col[0] == 'cylinder'):
            cylinder.append([col[1][1:-1].split(','),readvalue(col[2][1:-1].split(','),cst,array=1),readvalue(col[3],cst),readvalue(col[4],cst)]) 
            if(info.verbose==1):
                print(cylinder)
        #if(col[0] == 'clump'):
        #    clump.append([col[1][1:-1].split(','),readvalue(col[2],cst),readvalue(col[3],cst),readvalue(col[4],cst)]) 
        #    if(info.verbose==1):
        #        print(clump)
        if(col[0] == 'cloud'):
            cloud.append([col[1][1:-1].split(','),readvalue(col[2][1:-1].split(','),cst,array=1),readvalue(col[3],cst),readvalue(col[4],cst),readvalue(col[5],cst),readvalue(col[6],cst),readvalue(col[7],cst),readvalue(col[8],cst),readvalue(col[9],cst),readvalue(col[10],cst)]) 
            if(info.verbose==1):
                print(cloud)
        if(col[0] == 'shell'):
            shell.append([col[1][1:-1].split(','),readvalue(col[2][1:-1].split(','),cst,array=1),readvalue(col[3],cst),readvalue(col[4],cst)])
            if(info.verbose==1):
                print(shell)
        if(col[0] == 'fractal'):
            fractal.append([col[1][1:-1].split(','),int(col[2]),readvalue(col[3],cst),readvalue(col[4],cst),readvalue(col[5],cst),readvalue(col[6],cst),readvalue(col[7],cst),map(float,col[8][1:-1].split(','))])
            print(fractal)
            if(info.verbose==1):
                print(fractal)
        
        if col[0] == 'res_map':
            res_map = readvalue(col[1],cst)
            if(info.verbose==1):
                print(res_map)
        if col[0] == 'rmax_map':
            rmax_map = readvalue(col[1],cst) 
            if(info.verbose==1):
                print(rmax_map)
        if col[0] == 'af':
            af = readvalue(col[1],cst) 
            if(info.verbose==1):
                print(af)
        if col[0] == 'enpaq':
            enpaq = readvalue(col[1],cst)
            if(info.verbose==1):
                print(enpaq)
        if col[0] == 'source':
            try:
                sources.append([col[1],col[2],readvalue(col[3],cst),[[[readvalue(col[4],cst)],[readvalue(col[5],cst)],[readvalue(col[6],cst)]],[[readvalue(col[7],cst)],[readvalue(col[8],cst)],[readvalue(col[9],cst)]]]])
            except:
                sources.append([col[1],col[2],readvalue(col[3],cst),[[[0],[0],[0]],[[0],[0],[0]]]])
            if(info.verbose==1):
                print(sources) 

    param_fill_map2=dict(
        densturb=densturb,
        spherepower=spherepower, 
        gaussianprofile=gaussianprofile,
        cloud=cloud,
        #clump=clump,
        torus=torus,
        cone=cone,
        torus_Murakawa=torus_Murakawa,
        AGN_simple=AGN_simple,
        cylinder=cylinder,
        denspower=denspower,
        shell=shell,
        fractal=fractal 
    )
    return dust_pop,param_fill_map2,[enpaq,af,res_map,rmax_map],sources

def impup(question,cst,expect='float',array=0):
    truc=raw_input(question)
    if(expect!='str'):
        truc=readvalue(truc,cst,array=array)
    return truc

def read_paramuser():
    cst = mc.Constant()
    dust_pop = []
    sources = []

    ### grid parameters ###    
    res_map=impup('Grid resolution (m) ?',cst,expect='float')
    rmax_map=impup('Grid size (m) ?',cst,expect='float')
    
    ### model parameters ### unused for model #2 ###
    adds=impup('Add a source (y/n) ?',cst,expect='str')
    while(adds=='y'):
        centrobject=impup('What central object is it (star, AGN...) ?',cst,expect='str')
        fichierspectre=impup('What spectrum file to load ?',cst,expect='str')
        l=impup('Source luminosity (W) ?',cst,expect='float')
        emdir=list(input('Direction of emission ? ([[[0],[0],[0]],[[0],[0],[0]]] for default)'))
        source=[centrobject,fichierspectre,l,emdir]
        sources.append(source)
        adds=impup('Add a new source (y/n) ?',cst,expect='str')

    af=impup('Funnel aperture (°) ?',cst,expect='float')
    enpaq=impup('Energy in each "photon" object (J) ?',cst,expect='float')
    
    ### Rsub for dust ###
    adds=impup('Add a dust population (y/n) ?',cst,expect='str')
    while(adds=='y'):
        name=impup('Dust population name ?',cst,expect='str')
        print('Dust types available : silicates, graphites_ortho, graphites_para, electrons, pah_neutral or pah_ionised.')
        dusttype=impup('Dust type ?',cst,expect='str')
        rgmin=impup('Minimal grain radius (m) ?',cst,expect='float')
        rgmax=impup('Maximal grain radius (m) ?',cst,expect='float')
        alpha=impup('Slope of the grain size distribution ?',cst,expect='float')
        rsub=impup('Sublimation radius (m) ?',cst,expect='float')
        dust_pop.append([name,dusttype,rgmin,rgmax,alpha,rsub])
        adds=impup('Add a new dust population (y/n) ?',cst,expect='str')

    param_fill_map2=dict(
        densturb=[],
        spherepower=[], 
        gaussianprofile=[],
        cloud=[],
        torus=[],
        torus_Murakawa=[],    
        cylinder=[],
        denspower=[],
        shell=[]
    )
    return dust_pop,param_fill_map2,[enpaq,af,res_map,rmax_map],sources


def read_defined_models(usemodel,info):
    cst = mc.Constant()
    dust_pop = []
    sources = []
    densturb = []
    spherepower = []
    gaussianprofile = []
    cloud = []
    torus = []
    cone = []
    AGN_simple = []
    torus_Murakawa = []    
    cylinder = []
    denspower = []
    shell = []
    if(info.nsimu==1 or info.tycho==1):
        print("Using model ",usemodel)
    if(int(usemodel)==0): ## tunable model ##
        if(info.nsimu==1 or info.tycho==1):
            print("Tunable model (0)")
        # Should be containing all informations ! One line per parameter combination
        # For multiple configuration use further lines
        # put "[]" for not using one of these density models            
        param_fill_map2=dict(           
            denspower=[], # [[Radial power index, Radial typical profile size (in m), Vertical decay size (in m), [Density of grains (in particles / m3) at radial typical profile size (one for each grain type)]]]
            spherepower=[], # [[Radial power index, Radial typical profile size (in m), [Density of grains (in particles / m3) at radial typical profile size (one for each grain type)]]]
            gaussianprofile=[],
            densturb=[], # not usable
            cloud=[], # too many imputs --> not usable
            torus=[], #[[Disc outer radius (in pc),[density coefficent of grains (in kg / m3 ?)],ratio of the disk height at the disk boundary to the disk outer radius,Envelope mass infall (in Msol/yr),Mass of the star (in Msol)]]
            AGN_simple=[], #[[Disc outer radius (in m),[density coefficent of grains in the torus (in particles / m3 )],[density coefficent of grains in the envelope (in particles / m3 )],[density coefficent of grains in the cone (in particles / m3 )]]]
            cylinder=[] #[[cylinder radius (in m),cylinder height (in m),[density of grains (in particles / m3)]]]            
        )                        

        ### grid parameters ###
        res_map=0.1*cst.pc # m Résolution de la grille
        rmax_map=10*cst.pc # m Taille de la grille
        
        ### model parameters ###
        l_agn=1e36 #W Luminosité de l'AGN (ou objet central - etoile) AGN : 1e36 ; etoile : 3.846*1e26
        af=20 #° Ouverture du cone d'ionisation 20
        enpaq=3.846*1e26  #J Energy in each "photon" object  #1.0 #1e10
        centrobject='AGN' #(star or AGN)
        fichierspectre="s5700_spectrum.dat" #("spectre_picH.dat") #spectre_agn.dat
        
        ### Rsub for dust ###
        rsubsilicate=0.5*cst.pc # m Rayon de sublimation approximatif pour convergence
        rsubgraphite=0.5*cst.pc
        emission_dir=[[[0],[0],[0]],[[0],[0],[0]]] #direction of emission
        dust_pop=[
            #['silicates','silicates',0.005e-6,0.25e-6,-3.5,rsubsilicate],
            #['graphites_ortho','graphites_ortho',0.005e-6,0.25e-6,-3.5,rsubgraphite],
            #['graphites_para','graphites_para',0.005e-6,0.25e-6,-3.5,rsubgraphite],
            #['pah_neu','pah_neutral',0.005e-6,0.25e-6,-3.5,0.03],
            #['pah_ion','pah_ionised',0.005e-6,0.25e-6,-3.5,0.03],
            #['electrons','electrons',0,0,0,0]
        ]
        
    elif(int(usemodel)==-1): ## Optical Depth test model ##
        if(info.nsimu==1 or info.tycho==1):
            print("Optical Depth silicates test model (-1)")
        #if(usemodel>-1.05):
        tauH=(-1.-usemodel)*10
        if(info.nsimu==1 or info.tycho==1):
            print("     tau =",tauH,"in H")
        #for tau=2 in V
        #for tau=2 in K : [21000.0,0,0,0]
        param_fill_map2=dict(
            denspower=[],
            spherepower=[],  #old [[0,[0.6]]]
            gaussianprofile=[],
            densturb=[],
            cloud=[],
            torus=[],
            AGN_simple=[],
            #cylinder=[[['silicates'],[5.43e4*tauH],60*cst.AU,20*cst.AU]]
            cylinder=[[['silicates'],[4826.67*tauH],60*cst.AU,20*cst.AU]]
        )
        emission_dir=[[[0.],[0.],[0.999]],[[0.999],[0.],[0.]]]
        dust_pop=[
            ['silicates','silicates',0.005e-6,0.25e-6,-3.5,0.]
        ]
        
        ### grid parameters ###
        res_map=4*cst.AU # m Résolution de la grille
        rmax_map=100*cst.AU # m Taille de la grille
        
        ### model parameters ### unused for model #2 ###
        l_agn=3.846*1e26 #W Luminosité de l'AGN (ou objet central - etoile) AGN : 1e36 ; etoile : 3.846*1e26
        af=0 #° Ouverture du cone d'ionisation
        enpaq=3.846*1e26  #J Energy in each "photon" object
        centrobject='star' #(star or AGN)
        fichierspectre="spectre_picH.dat"
        
        ### Rsub for dust ###
        rsubsilicate=0.000*cst.AU # m Rayon de sublimation approximatif pour convergence
        rsubgraphite=0.000*cst.AU
        
    elif(int(usemodel)==-2): ## Optical Depth test model ##
        if(info.nsimu==1 or info.tycho==1):
            print("Optical Depth graphites ortho test model (-2)")
        #if(usemodel>-1.05):
        tauH=(-2.-usemodel)*10
        if(info.nsimu==1 or info.tycho==1):
            print("     tau =",tauH,"in H")
        #for tau=2 in V
        #for tau=2 in K : [21000.0,0,0,0]
        param_fill_map2=dict(
            denspower=[],
            spherepower=[],  #old [[0,[0.6]]]
            gaussianprofile=[],
            densturb=[],
            cloud=[],
            torus=[],
            AGN_simple=[],
            cylinder=[[['graphites_ortho'],[4.4e3*tauH],60*cst.AU,20*cst.AU]]
        )
        emission_dir=[[[0.],[0.],[0.999]],[[0.999],[0.],[0.]]]
        dust_pop=[
            ['graphites_ortho','graphites_ortho',0.005e-6,0.25e-6,-3.5,rsubgraphite]
        ]
    
        
        ### grid parameters ###
        res_map=4*cst.AU # m Résolution de la grille
        rmax_map=100*cst.AU # m Taille de la grille
        
        ### model parameters ### unused for model #2 ###
        l_agn=3.846*1e26 #W Luminosité de l'AGN (ou objet central - etoile) AGN : 1e36 ; etoile : 3.846*1e26
        af=0 #° Ouverture du cone d'ionisation
        enpaq=3.846*1e26  #J Energy in each "photon" object
        centrobject='star' #(star or AGN)
        fichierspectre="spectre_picH.dat"
        
        ### Rsub for dust ###
        rsubsilicate=0.000*cst.AU # m Rayon de sublimation approximatif pour convergence
        rsubgraphite=0.000*cst.AU
        
    elif(int(usemodel)==-3): ## Optical Depth test model ##
        if(info.nsimu==1 or info.tycho==1):
            print("Optical Depth graphites para test model (-3)")
        #if(usemodel>-1.05):
        tauH=(-3.-usemodel)*10
        if(info.nsimu==1 or info.tycho==1):
            print("     tau =",tauH,"in H")
        #for tau=2 in V
        #for tau=2 in K : [21000.0,0,0,0]
        param_fill_map2=dict(
            denspower=[],
            spherepower=[],  #old [[0,[0.6]]]
            gaussianprofile=[],
            densturb=[],
            cloud=[],
            torus=[],
            AGN_simple=[],
            cylinder=[[['graphites_para'],[4.4e3*tauH],60*cst.AU,20*cst.AU]]
        )            
        emission_dir=[[[0.],[0.],[0.999]],[[0.999],[0.],[0.]]]
        dust_pop=[
            ['graphites_para','graphites_para',0.005e-6,0.25e-6,-3.5,rsubgraphite]
        ]
        
        
        ### grid parameters ###
        res_map=4*cst.AU # m Résolution de la grille
        rmax_map=100*cst.AU # m Taille de la grille
        
        ### model parameters ### unused for model #2 ###
        l_agn=3.846*1e26 #W Luminosité de l'AGN (ou objet central - etoile) AGN : 1e36 ; etoile : 3.846*1e26
        af=0 #° Ouverture du cone d'ionisation
        enpaq=3.846*1e26  #J Energy in each "photon" object
        centrobject='star' #(star or AGN)
        fichierspectre="spectre_picH.dat"
        
        ### Rsub for dust ###
        rsubsilicate=0.000*cst.AU # m Rayon de sublimation approximatif pour convergence
        rsubgraphite=0.000*cst.AU
        
    elif(int(usemodel)==-4): ## Optical Depth test model ##
        if(info.nsimu==1 or info.tycho==1):
            print("Optical Depth electrons test model (-4)")
        #if(usemodel>-1.05):
        tauH=(-4.-usemodel)*10
        if(info.nsimu==1 or info.tycho==1):
            print("     tau =",tauH,"in H")
        #for tau=2 in V
        #for tau=2 in K : [21000.0,0,0,0]
        param_fill_map2=dict(
            denspower=[],
            spherepower=[],  #old [[0,[0.6]]]
            gaussianprofile=[],
            densturb=[],
            cloud=[],
            torus=[],
            AGN_simple=[],
            cylinder=[[['electrons'],[5.00e15*tauH],60*cst.AU,20*cst.AU]]
        )
        emission_dir=[[[0.],[0.],[0.999]],[[0.999],[0.],[0.]]]
        dust_pop=[
            ['electrons','electrons',0,0,0,0]
        ]
        
        
        ### grid parameters ###
        res_map=4*cst.AU # m Résolution de la grille
        rmax_map=100*cst.AU # m Taille de la grille
        
        ### model parameters ### unused for model #2 ###
        l_agn=3.846*1e26 #W Luminosité de l'AGN (ou objet central - etoile) AGN : 1e36 ; etoile : 3.846*1e26
        af=0 #° Ouverture du cone d'ionisation
        enpaq=3.846*1e26  #J Energy in each "photon" object
        centrobject='star' #(star or AGN)
        fichierspectre="spectre_picH.dat"
        
        ### Rsub for dust ###
        rsubsilicate=0.000*cst.AU # m Rayon de sublimation approximatif pour convergence
        rsubgraphite=0.000*cst.AU
        
    elif(int(usemodel)==-5): ## Optical Depth test model ##
        if(info.nsimu==1 or info.tycho==1):
            print("Optical Depth pah test model (-5)")
        #if(usemodel>-1.05):
        tauH=(-5.-usemodel)*10
        if(info.nsimu==1 or info.tycho==1):
            print("     tau =",tauH,"in H")
        #for tau=2 in V
        #for tau=2 in K : [21000.0,0,0,0]
        param_fill_map2=dict(
            denspower=[],
            spherepower=[],  #old [[0,[0.6]]]
            gaussianprofile=[],
            densturb=[],
            cloud=[],
            torus=[],
            AGN_simple=[],
            cylinder=[[['pah_neu'],[1.787544e9*tauH],60*cst.AU,20*cst.AU]]
        )
        emission_dir=[[[0.],[0.],[0.999]],[[0.999],[0.],[0.]]]
        dust_pop=[
            ['pah_neu','pah_neutral',0.005e-6,0.25e-6,-3.5,0.03]
        ]
        
        
        ### grid parameters ###
        res_map=4*cst.AU # m Résolution de la grille
        rmax_map=100*cst.AU # m Taille de la grille
        
        ### model parameters ### unused for model #2 ###
        l_agn=3.846*1e26 #W Luminosité de l'AGN (ou objet central - etoile) AGN : 1e36 ; etoile : 3.846*1e26
        af=0 #° Ouverture du cone d'ionisation
        enpaq=3.846*1e26  #J Energy in each "photon" object
        centrobject='star' #(star or AGN)
        fichierspectre="spectre_picH.dat"
        
        ### Rsub for dust ###
        rsubsilicate=0.000*cst.AU # m Rayon de sublimation approximatif pour convergence
        rsubgraphite=0.000*cst.AU
        
        
    elif(int(usemodel)==-6): ## Optical Depth test model ##
        if(info.nsimu==1 or info.tycho==1):
            print("Optical Depth full test model (-6)")
        #if(usemodel>-1.05):
        tauH=(-6.-usemodel)*10
        if(info.nsimu==1 or info.tycho==1):
            print("     tau =",tauH,"in H")
        #for tau=2 in V
        #for tau=2 in K : [21000.0,0,0,0]
        param_fill_map2=dict(
            denspower=[],
            spherepower=[],  #old [[0,[0.6]]]
            gaussianprofile=[],
            densturb=[],
            cloud=[],
            torus=[],
            AGN_simple=[],
            cylinder=[[['silicates','graphites_ortho','graphites_para','electrons','pah_neu'],[5.43e4*tauH/4.,4.4e3*tauH/4.,4.4e3*tauH/4.,5.00e15*tauH/4.],60*cst.AU,20*cst.AU]]
        )
        emission_dir=[[[0.],[0.],[0.999]],[[0.999],[0.],[0.]]]
        dust_pop=[
            ['silicates','silicates',0.005e-6,0.25e-6,-3.5,rsubsilicate],
            ['graphites_ortho','graphites_ortho',0.005e-6,0.25e-6,-3.5,rsubgraphite],
            ['graphites_para','graphites_para',0.005e-6,0.25e-6,-3.5,rsubgraphite],
            #['pah_neu','pah_neutral',0.005e-6,0.25e-6,-3.5,0.03],
            #['pah_ion','pah_ionised',0.005e-6,0.25e-6,-3.5,0.03],
            ['electrons','electrons',0,0,0,0]
        ]
        
        
        ### grid parameters ###
        res_map=4*cst.AU # m Résolution de la grille
        rmax_map=100*cst.AU # m Taille de la grille
        
        ### model parameters ### unused for model #2 ###
        l_agn=3.846*1e26 #W Luminosité de l'AGN (ou objet central - etoile) AGN : 1e36 ; etoile : 3.846*1e26
        af=0 #° Ouverture du cone d'ionisation
        enpaq=3.846*1e26  #J Energy in each "photon" object
        centrobject='star' #(star or AGN)
        fichierspectre="spectre_picH.dat"
        
        ### Rsub for dust ###
        rsubsilicate=0.000*cst.AU # m Rayon de sublimation approximatif pour convergence
        rsubgraphite=0.000*cst.AU
        
    elif(np.floor(usemodel)==1): ## Whelan SSC model with hierarchical clumpiness ## #whelan
        if(info.nsimu==1 or info.tycho==1):
            print("Whelan SSC model with hierarchical clumpiness (5)")

        SFE=0.05
        dust_to_gas=0.01
        #ratio=[0.625/particle_mass()[0],0.25/particle_mass_new(0.005e-6,0.25e-6,-3.5)[1],0.125/particle_mass()[2],0] # Draine & Malhotra 1993; Weingartner & Draine 2001
        mass_ratio = np.array([0.75,0.133,0.067,0.05]) # Joblin, C., Tielens, A., & Draine, B. 2011
        ratio = np.divide(mass_ratio,particle_mass_new(0.005e-6,0.25e-6,-3.5)[:4])
        
        Mdust0=1e6*2e30*(1-SFE)/SFE*dust_to_gas/(1+dust_to_gas)         
        
        if (usemodel==1.1):Rin=5*pc
        elif (usemodel==1.2):Rin=10*pc
        elif (usemodel==1.3):Rin=15*pc
        elif (usemodel==1.4):Rin=20*pc
        elif (usemodel==1.5):Rin=25*pc
        elif (usemodel==1.6):Rin=30*pc
        elif (usemodel==1.7):Rin=35*pc
        elif (usemodel==1.8):Rin=40*pc
        print('Rin = '+str(Rin/pc)+' pc')
        
        Rout=50*pc
        Mdust=Mdust0*(1-Rin**3/Rout**3)

        clumpiness=0.99
        N=32
        D=2.6
        H=5
     
        ### grid parameters ###
        res_map=1*pc # m Résolution de la grille 10 25
        rmax_map=50*pc # m Taille de la grille 500 1500
        
        Vshell = 4/3*np.pi*(Rout**3-Rin**3)                
        rho_shell = Mdust*(1-clumpiness)/Vshell
        
        Mfractal = Mdust*clumpiness        

        param_fill_map2=dict(                    
            #shell=[[Rin,Rout,ratio*rho_shell]], 
            shell=[[ratio*rho_shell,Rin,Rout]],
            fractal=[[N,D,H,Rin,Rout,Mfractal,ratio]]            
        )

        emission_dir=[[[0.],[0.],[0.]],[[0.],[0.],[0.]]]

        ### model parameters ### unused for model #2 ###
        l_agn=1.6e9*3.846e26 #W Luminosité de l'AGN (ou objet central - etoile) AGN : 1e36 ; etoile : 3.846*1e26
        af=0 #° Ouverture du cone d'ionisation 20
        enpaq=1.6*1e9*3.846e26  #1.0 #1e10 #J Energy in each "photon" object
        centrobject='star' #(star or AGN)
        fichierspectre="whelan_spectrum.dat"

        ### Rsub for dust ###
        rsubsilicate=0.03*pc # m Rayon de sublimation approximatif pour convergence
        rsubgraphite=0.03*pc
        rsubpah=0.03*pc 

        dust_pop=[
            ['silicates','silicates',0.005e-6,0.25e-6,-3.5,rsubsilicate],
            ['graphites_ortho','graphites_ortho',0.005e-6,0.25e-6,-3.5,rsubgraphite],
            ['graphites_para','graphites_para',0.005e-6,0.25e-6,-3.5,rsubgraphite],
            ['pah_neu','pah_neutral',0.005e-6,0.25e-6,-3.5,rsubpah],
        ]
    else: 
        print("ERROR : No ask and no model parameters to use.")
        print("Please enter a valid number for usemodel keyword or choose ask=1")
           
    sources=[[centrobject,fichierspectre,l_agn,emission_dir]]
    return dust_pop,param_fill_map2,[enpaq,af,res_map,rmax_map],sources

def translate_map_param(paramd,dust,indices):
    if isinstance(paramd[0], (str,list))==True: # parameters read from file
        param=paramd[1:]
        ndust=len(dust)
        for i in indices:
            dens=param[i]
            denstab=np.zeros(ndust)
            for j in range(ndust):
                if (isinstance(paramd[0],list)==True):
                    for k in range(len(paramd[0])):
                        if(dust[j].name == paramd[0][k]):
                            denstab[j]=dens[k]  #ratiotab                 
                else:
                    if(dust[j].name==paramd[0]):
                        denstab[j]=dens
            param[i]=denstab
    else:
        param=paramd
    return param

def translate_map_param2(param,dust):
    geom=['densturb',
          'spherepower',
          'gaussianprofile',
          'cloud',
          'torus',
          'torus_const',
          'cylinder',
          'denspower',
          'shell']
    ngeom=len(geom)
    densturb=[]
    spherepower=[]
    gaussianprofile=[]
    cloud=[]
    torus=[]
    torus_const=[]  
    cylinder=[]
    denspower=[]
    shell=[]
    for j in geom:
        for i in range(np.shape(param[j])[0]):
            if(j=='densturb'):
                densturb.append(translate_geom(param[j][i],dust))
            if(j=='spherepower'):
                spherepower.append(translate_geom(param[j][i],dust))
            if(j=='gaussianprofile'):
                gaussianprofile.append(translate_geom(param[j][i],dust))
            if(j=='cloud'):
                cloud.append(translate_geom(param[j][i],dust))
            if(j=='torus'):
                torus.append(translate_geom(param[j][i],dust))
            if(j=='torus_const'):
                torus_const.append(translate_geom(param[j][i],dust))
            if(j=='cylinder'):
                cylinder.append(translate_geom(param[j][i],dust))
            if(j=='denspower'):
                denspower.append(translate_geom(param[j][i],dust))
            if(j=='shell'):
                shell.append(translate_geom(param[j][i],dust))


    param2=dict(
        densturb=densturb,
        spherepower=spherepower, 
        gaussianprofile=gaussianprofile,
        cloud=cloud,
        torus=torus,
        torus_const=torus_const,    
        cylinder=cylinder,
        denspower=denspower,
        shell=shell
    )
    print(torus_const)
    return param2

def translate_geom(param,dust):
    ndust=len(dust)
    tabvalue=np.zeros(ndust)
    for i in range(ndust):
        if(dust[i].name==param[0]):
            tabvalue[i]=param[-1]
            param2=[param[1:-1],tabvalue]
    #print param2
    return param2
