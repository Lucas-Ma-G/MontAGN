# -*- coding: utf-8 -*-
import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt

#import montagn_utils as mu
#import montagn_utils_Lu as mul
import montagn_polar as mpol

def scat_uniform(r,wvl):
    r1 = np.random.random()
    r2 = np.random.random()
    dth = np.arccos(2*r1-1) #polar angle
    ph = r2*2*np.pi #azimuthal angle
    return dth, ph

###########################    CLASSES    ############################


class Constant:
    def __init__(self):
        """

        """
        self.b = 2.8977721e-3 #Wien's displacement cst
        self.h = 6.62606957e-34 #in USI
        self.c = 299792458 #m/s
        self.kB = 1.3806488e-23 #in USI
        self.sigma = 5.670373e-8 #Stefan-Boltzmann constant, in USI
        self.G= 6.67384e-11 #Gravitationnal constant, in USI
        self.sigt = 6.65246e-29 #Thomson electron cross section, in m2
        self.mass_el = 9.109e-31 #Electron mass, in kg
        self.r_el = 4.60167772e-15 #sqrt(sigt/pi)

        self.pc = 3.08567758e16 #m/pc
        self.AU = 1.49598e11 #m/AU
        self.nm = 1.0e-9 #m/nm
        self.Msol = 1.9884e30 #kg/Msol
        self.yr = 3.15576e+07 #s/yr

class Spectrum:
    def __init__(self,spct,wvlp):
        """spectrum function, wavelength probability function (complying with the spectrum_factory types)"""
        self.spectrum = spct
        self.wvl_proba = wvlp

class Emission_properties:
    def __init__(self,emdir='isotropic',polardir='random',polar='unpolarised'):
        """Emission properties function, wavelength probability function (complying with the spectrum_factory types)"""
        if(emdir==[] or emdir=='default'):
            self.emdir = 'isotropic'
        else:
            self.emdir = emdir
        if(polardir==[] or polardir=='default'):
            self.polardir = 'random'
        else:
            self.polardir = polardir
        if(polar==[] or polar=='default'):
            self.polar = 'unpolarised'
        else:
            self.polar = polar

class Source:
    """primary light source of any kind"""
    def __init__(self,lum,spct,spdist,t='generic',emission_prop=Emission_properties()): #emission_dir=[[[0],[0],[0]],[[0],[0],[0]]]):
        """total luminosity (in W),
        Spectrum class object,
        random position generator according to the source's geometry
        source type (string) : 'AGN', 'jet', 'pointlike'... """
        self.lum = lum
        self.spectrum = spct
        self.spacedist = spdist
        self.emission_prop=emission_prop
        self.type = t
    def __repr__(self):
        t = "Type : %s \n" %self.type
        l = "Luminosity = %.4e W" %self.lum
        return "LIGHT SOURCE\n"+t+l
    #spatial distribution generator, returns random emission position
    def emission(self,genseed,pos=None,wvl=None,tau=None):
        """generates the random parameters for a Photon instance"""
        if(pos==None):
            #random emission position according to spatial distribution
            x,y,z = self.spacedist()
        else:
            x=pos[0]
            y=pos[1]
            z=pos[2]
        if(wvl==None):
            #random wavelength
            r1 = genseed.uniform()
            wvl = self.spectrum.wvl_proba(r1)
        #wvl = 2.2*1e-6 #for K band photons
        #print self.emission_dir[0],self.emission_dir[0]==[[0],[0],[0]]
        #Direction of emission
        if(self.emission_prop.emdir=='isotropic'):
            r1 = genseed.uniform()
            r2 = genseed.uniform()
            theta = 2*r1-1 #polar angle
            phi = r2*2*np.pi #azimuthal angle
        else:
            #print "using given emission direction", self.emission_dir
            theta=np.cos(self.emission_prop.emdir[0]*np.pi/180.)
            phi=self.emission_prop.emdir[1]*np.pi/180.
        #Polarisation orientation
        if(self.emission_prop.polardir=='random'):
            r3 = genseed.uniform()
            phiu = r3*2*np.pi #azimuthal angle of u
        else:
            phiu=self.emission_prop.polardir*np.pi/180.
            #p=np.copy(self.emission_dir[0])
            #u=np.copy(self.emission_dir[1])

        #if(theta>0.999):
        #    theta=0.999
        #if(theta<-0.999):
        #    theta=-0.999

        #Computation of propagation vectors p and u
        u = [[-np.sin(phi)],[np.cos(phi)],[0.0]] #rotation vector
        p= [[np.cos(phi)*np.sqrt(1-theta*theta)],[np.sin(phi)*np.sqrt(1-theta*theta)],[theta]] 

        Rotu=np.zeros([3,3])
        Rotu[0,0]=p[0][0]*p[0][0]+(1-p[0][0]*p[0][0])*np.cos(phiu)
        Rotu[0,1]=p[0][0]*p[1][0]*(1-np.cos(phiu))-p[2][0]*np.sin(phiu)
        Rotu[0,2]=p[0][0]*p[2][0]*(1-np.cos(phiu))+p[1][0]*np.sin(phiu)
        Rotu[1,0]=p[0][0]*p[1][0]*(1-np.cos(phiu))+p[2][0]*np.sin(phiu)
        Rotu[1,1]=p[1][0]*p[1][0]+(1-p[1][0]*p[1][0])*np.cos(phiu)
        Rotu[1,2]=p[1][0]*p[2][0]*(1-np.cos(phiu))-p[0][0]*np.sin(phiu)
        Rotu[2,0]=p[0][0]*p[2][0]*(1-np.cos(phiu))-p[1][0]*np.sin(phiu)
        Rotu[2,1]=p[1][0]*p[2][0]*(1-np.cos(phiu))+p[0][0]*np.sin(phiu)
        Rotu[2,2]=p[2][0]*p[2][0]+(1-p[2][0]*p[2][0])*np.cos(phiu)
        
        Mrotu=np.matrix(Rotu)
        
        u2=Mrotu*u
        
        u[0]=[float(u2[0])]
        u[1]=[float(u2[1])]
        u[2]=[float(u2[2])]

        theta = np.arccos(theta)
        #p=np.copy(self.emission_dir[0])
        #u=np.copy(self.emission_dir[1])
        #theta=p[2][0]
        #phi=np.arccos(p[0][0]/np.sqrt(1-theta*theta))

        if(tau==None):
            #Optical dpeth that the photon will go through before next interaction
            X = genseed.uniform()
            tau = -np.log(X)

        #Polarisation at emission
        if(self.emission_prop.polar=='unpolarised'):
            Ss_part=[0,0,0]
        else:
            Ss_part=self.emission_prop.polar
        #ph=Photon(wvl,theta,phi,p,u,'',x,y,z,polar=Ss_part)
        return wvl,theta,phi,p,u,x,y,z,tau,Ss_part

class Source_total:
    """Super-structure containing all the individual light sources in the model"""
    def __init__(self, lst_src):
        self.n_sources = len(lst_src)
        self.list_sources = lst_src
        self.proba_lum = []
        for i in self.list_sources:
            self.proba_lum.append(i.lum*1.)
        self.proba_lum = np.array(self.proba_lum)
        self.total_lum = sum(self.proba_lum)
        self.proba_lum /= self.total_lum
        #the probability for a photon to be emitted by a given light source
        #is given by the relative luminosities
    def __repr__(self):
        l = "LIGHT SOURCES :\n"
        for i in range(len(self.list_sources)):
            l += "Type : %s \n"%self.list_sources[i].type
            l += "  Absolute luminosity = %.4e W\n"%self.list_sources[i].lum
            l += "    Relative luminosity = %.4f\n"%self.proba_lum[i]
        return l

    def emission_photon(self,genseed,enpaq=0.,wvl=None):
        """call the random parameter generator from one source to create a Photon instance"""
        #calls emission from random source
        r1 = genseed.uniform()
        s = 0
        l = self.proba_lum[s]
        while l < r1:
            s +=1
            l += self.proba_lum[s]
        #print l, r1, s, self.proba_lum, self.list_sources
        wvl,theta,phi,p,u,x,y,z,tau,Ss_part = self.list_sources[s].emission(genseed,wvl=wvl)
        #ph,x,y,z = self.list_sources[s].emission(genseed)
        #ph.label = self.list_sources[s].type
        #return Photon(wvl,theta,phi,self.list_sources[s].type,x,y,z,polar=Ss_part),x,y,z,p,u
        return Photon(wvl,theta,phi,p,u,self.list_sources[s].type,x,y,z,enpaq,tau=tau,polar=Ss_part),x,y,z

class Photon:
    """energy packet with given wavelength (in m) and direction of propagation"""
    def __init__(self, wvl, theta, phi, p, u, label, x, y, z, enpaq, tau=-1., polar=[0.0,0.0,0.0]):
        self.wvl = wvl
        self.theta = theta
        self.phi = phi
        self.label = label
        self.write = True
        self.interact = 0 #has interacted (diffusion of absorption)
        self.reem = 0 #is a reemitted photon (after absorption)
        #self.Ss = [[1.0],[0.0],[0.0],[0.0]] #Stokes vector
        self.Ss = [[1.0],[polar[0]],[polar[1]],[polar[2]]] #Stokes vector
        #self.p = [[1.0],[0.0],[0.0]] #propagation vector (redondant with theta and phi)
        #self.u = [[0.0],[1.0],[0.0]] #rotation vector, needed for p evolution
        self.p = p
        self.u = u
        self.phiqu = 0.0
        self.E = enpaq
        self.tau = tau
        self.x_inter = x
        self.y_inter = y
        self.z_inter = z

class Grain:
    """physical properties of a type of dust grain"""
    #def __init__(self, name, tsub, size, av, cs, ca, k,rmin,rmax,alpha,typeg,phase = 0):
    def __init__(self, name, tsub, size, av,rmin,rmax,alpha,density,typeg,usegrain,number,phase = 0):
        """name : 'silicate', 'PAH' ...
        Tsub : sublimation temperature (in K)
        size : random size generator (radius in m)
        av_sec : average cross-section of a grain (in m2) = integral(pi*size)dr
        Cs : interaction coefficient as a function of wvl : total cross-section(wvl) = Cs(wvl)*pi*r2
        Cab : absorption coefficient as a function of wvl : abs cross-section(wvl) = Cab(wvl)*pi*r2
        K : integral(kv Bv(T))
        phase : phase function generating random deviation (Dtheta, phi), given radius and wavelength : F(r,wvl). Defaults to isotropic scattering."""
        self.name = name
        self.Tsub = tsub
        self.size = size
        self.av_sec = av
        #self.Cs = cs
        #self.Cab = ca
        #self.K = k
        self.rmin = rmin
        self.rmax = rmax
        self.alpha = alpha
        self.density = density
        self.typeg = typeg
        self.number = number
        self.usegrain = usegrain
        self.Rsub_init = []
        # 0 for silicates
        # 1 for graphites ortho
        # 2 for graphites para
        # 3 for electrons
        # 4 for nanodiamants


        #x_M,S11,S12,S33,S34,phase,albedo,Qext,Qabs,S11e,S12e,S33e,S34e,phasee=pol.fill_mueller(nang,nforme,genseed,dust,plotinfo,ndust,force_wvl)
        #x_M,S11,S12,S33,S34,phase,albedo,Qext,Qabs=pol.fill_mueller(nang,nforme,genseed,dust,plotinfo,ndust,force_wvl)
        #phaseNorm,gw=pol.init_phase(nang,nforme,x_M,phase,ndust,genseed,plotinfo)
        self.x_M = [] #x_M
        self.S11 = [] #S11
        self.S12 = [] #S12
        self.S33 = [] #S33
        self.S34 = [] #S34
        self.phaseNorm = [] #phaseNorm
        self.w = [] #polyn
        self.g = [] #HG
        self.albedo = [] #albedo
        self.Qext = [] #Qext
        self.Qabs = [] #Qabs
        self.wvlvect = []
        self.sizevect = []

        self.Qabs_a2_MRN = []
        self.EemitT = []
        self.tempRange = []

        self.CC_wvl = []
        self.CC = []
        self.jnupah = []
        self.Nc = 0.

        #if phase == 0:
        #    self.phase = scat_uniform #mu.scat_uniform
        #else:
        #    self.phase = phase

    def albedo(self,wvl):
        return 1 - self.Cab(wvl)/self.Cs(wvl)

    def __repr__(self):
        return self.name

class Cell:
    """spatial cell, at temperature T and containing dust at density rho"""
    def __init__(self, t, r,pah=0):
        self.T = t #cell temperature (in K)
        self.rho = r #list of densities of different grain types (in particles/m3)
        self.N = 0 #running number of absorbed photon packets
        if(pah==1):
            self.Npah = 0
    def __repr__(self):
        return "%d K"%self.T

class Map:
    #organized 3D grid of Cell instances
    #Spatial origin is set at the intersection of cells. The shape is a (2 Rmax)^3 cube
    def __init__(self, Rmax, res, dust, pah=0):
        """Rmax : total size of the map (in m)
        res : resolution of the the grid (in m)
        dust : list of Grain type objects present in the dust (sorted by lowest sublimation temperature)
        """
        self.res = res
        self.Rmax = Rmax
        self.N = int(Rmax/res)+1
        self.grid = []
        self.dust = dust

        for i in range(2*self.N):
            x = []
            for j in range(2*self.N):
                y = []
                for k in range(2*self.N):
                    y.append(Cell(3,[0]*len(self.dust),pah=pah)) #Cell initialization : vacuum at 3K
                x.append(y)
            self.grid.append(x)
    def __repr__(self):
        s1 = "Radius = %s m\n"%self.Rmax
        s2 = "Resolution = %s m\n"%self.res
        return s1+s2
    def get(self,x,y,z):
        """(x,y,z) is an absolute position (in m)"""
        xr = x/self.res
        yr = y/self.res
        zr = z/self.res
        i = int(np.floor(xr+self.N))
        j = int(np.floor(yr+self.N))
        k = int(np.floor(zr+self.N))
        return self.grid[i][j][k] #returns the corresponding Cell

    def set(self,x,y,z,V):
        """(x,y,z) is an absolute position (in m), V a Cell object"""
        xr = x/self.res
        yr = y/self.res
        zr = z/self.res
        i = int(np.floor(xr+self.N))
        j = int(np.floor(yr+self.N))
        k = int(np.floor(zr+self.N))
        self.grid[i][j][k] = V #assigns value V to the corresponding Cell


class Model:
    #structure containing all the geometrical parameters of the AGN torus model
    def __init__(self,sources,mp,af,rsub,en,usemodel,ispah=0,tempmode=1):
        """sources : Source_total type object
        map : Map object
        AF : aperture angle of the torus' funnel (in °)
        Rsub : list of sublimation radii of the grains (should be increasing, just as Tsub)
        energy : energy carried by each photon packet
        """
        self.sources = sources
        self.map = mp
        self.AF = af
        self.Rsub = rsub
        self.energy = en
        self.Dt = 0
        self.nscattmax = 50
        self.usemodel = usemodel
        self.thetaobs = [90]
        self.thetarec = [0,45,90,135,180]
        self.dthetaobs = 5
        self.usethermal = 1
        self.Temp_mode=tempmode
        self.nRsub_update = 0
        self.isrf = 0
        self.N_ph_em = 0
        if(ispah==1):
            #loading isrf table
            self.isrf=read_isrf()
        grainused=[]
        for i in range(len(mp.dust)):
            grainused.append(mp.dust[i].typeg)
        #fonction sort non redondant
        self.grainused = grainused
    def __repr__(self):
        s = "Model parameters :\n%s"%self.map
        s += "Aperture angle = %s°\n"%self.AF
        s += "Dust composition :\n"
        for i in self.map.dust:
            s += " %s\n"%i
        s += "Sublimation radii :\n"
        for i in self.Rsub:
            s += " %s m\n"%i
        return s


def read_isrf(d=8):
    table=[]
    wvl=[]
    i=0
    f=open("Input/isrf.dat",'r')
    if(d==5):
        j=1
    elif(d==6):
        j=2
    elif(d==8):
        j=3
    elif(d==10):
        j=4
    elif(d==13):
        j=5
    for data in f.readlines():
        if(i not in [0,1,2,3,4]):
            wvl.append(float(data.split()[0])*1e-6)
            table.append(float(data.split()[j]))
        i+=1
    f.close()
    #integration
    isrf=0.
    for i in range(len(wvl)):
        if(i==0):
            lmin=wvl[i]
            lmax=(wvl[i+1]+wvl[i])*0.5
        elif(i==len(wvl)-1):
            lmax=wvl[i]
            lmin=(wvl[i-1]+wvl[i])*0.5
        else:
            lmin=(wvl[i-1]+wvl[i])*0.5
            lmax=(wvl[i+1]+wvl[i])*0.5
        isrf+=table[i]*(lmax-lmin)
    return isrf


       
class Info:
    def __init__(self,ask=1,add=0,force_wvl=[],wvl_max=1e-5,plotinfo=[],display=1,nsimu=1,nang=999,nsize=100,cluster=0,verbose=0,debug=[],unit=''):
        """Group all the flags in one class"""
        #self.usethermal = usethermal
        self.ask = ask
        self.add = add
        self.force_wvl = force_wvl
        self.wvl_max = wvl_max
        self.f = []
        self.ftheta = []
        self.fT = []
        self.plotinfo = plotinfo
        self.display = display
        self.checktau = 0
        self.nsimu = nsimu
        self.cluster = cluster
        self.alpha = np.linspace(0.,0.,nang)
        self.beta = np.zeros([nang,5])
        self.xstat = np.linspace(0.,0.,nsize)
        self.pstat = np.zeros([nang,5,2])
        self.albedo = np.linspace(0.,0.,nsize)
        self.Dt_tot = []
        self.verbose = verbose
        self.debug = debug
        self.unit = unit
        self.outformat = 'ascii'
        def nwarning():
            """Function initialising the count of encountered warnings"""
            Nwar={}
            Nwar['Tmax']=Nwarning('Tmax','WARNING: sublimation temperature reached in a cell \n Please consider adapting the sublimation radius to the source luminosity')
            return Nwar
        self.nwarning = nwarning()

class Nwarning:
    def __init__(self,name,warningtxt):
        self.name = name
        self.n = 0
        self.warningtxt = warningtxt

class Polar:
    def __init__(self,nang,nforme,genseed,dust,grainused,plotinfo,force_wvl):
        """Function including all informations required for polarization :
        x_M corresponding x=2*pi*r_grain/wvl used in other objets
        S11 polar matrix element
        S12 polar matrix element
        S33 polar matrix element
        S34 polar matrix element
        albedo for each x
        phasenorm phase function for each x
        gw parameters to use in phase functions (g or w in HG or polynomial)
        Qext Extinction coefficent for each x --> integral of Kv Bv(T)
        Qabs Absorption coefficent for each x
        """
        ndust=0 #part to select the grain to use
        #for i in set(grainused):
        #    if(i!=2):
        #        ndust+=1
        ndust=2
        #ndust=len(grainused)-1 #because of ortho and para graphites
        x_M,S11,S12,S33,S34,phase,albedo,Qext,Qabs,S11e,S12e,S33e,S34e,phasee=mpol.fill_mueller(nang,nforme,genseed,dust,plotinfo,ndust,force_wvl)
        phaseNorm,gw=mpol.init_phase(nang,nforme,x_M,phase,ndust,genseed,plotinfo)
        self.nang = nang
        self.nform = nforme
        #self.grainused = grainused
        self.x_M = x_M
        self.S11 = S11
        self.S12 = S12
        self.S33 = S33
        self.S34 = S34
        self.S11el = S11e
        self.S12el = S12e
        self.S33el = S33e
        self.S34el = S34e
        self.albedo = albedo
        self.phasenorm = phaseNorm
        self.gw = gw
        self.Qext = Qext
        self.Qabs = Qabs
        self.alpha = np.linspace(0.,0.,nang)
        self.beta = np.zeros([nang,5])
        self.xstat = np.linspace(0.,0.,nforme)
        self.pstat = np.zeros([nang,5,2])
