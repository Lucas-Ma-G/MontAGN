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

import montagn_utils as mut
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

#############    Geometry management functions    ##############################

def fill_map(mp):
    cst=mc.Constant()
    """adds any of the available structures to a Map object."""
    add=0
    while add not in ['y','n']:
        add = raw_input('Add a powerlaw dust density ? (y/n) ')
    while add == 'y':
        add_density_powerlaw(mp,cst)
        add = 0
        while add not in ['y','n']:
            add = raw_input('Add a powerlaw dust density ? (y/n) ')

    add=0
    while add not in ['y','n']:
        add = raw_input('Add a spherical powerlaw dust density ? (y/n) ')
    while add == 'y':
        add_spherical_powerlaw(mp,cst)
        add = 0
        while add not in ['y','n']:
            add = raw_input('Add a spherical powerlaw dust density ? (y/n) ')

    add=0
    while add not in ['y','n']:
        add = raw_input('Add a Gaussian profile dust density ? (y/n) ')
    while add == 'y':
        add_gaussian_profile(mp,cst)
        add = 0
        while add not in ['y','n']:
            add = raw_input('Add a Gaussian profile dust density ? (y/n) ')

    add=0
    while add not in ['y','n']:
        add = raw_input('Add a circumstellar disk dust density ? (y/n) ')
    while add == 'y':
        add_circumstellar_disk(mp,cst)
        add = 0
        while add not in ['y','n']:
            add = raw_input('Add a circumstellar disk dust density ? (y/n) ')

    add=0
    while add not in ['y','n']:
        add = raw_input('Add a turbulent dust density ? (y/n) ')
    while add == 'y':
        add_density_turbulent(mp,cst)
        add = 0
        while add not in ['y','n']:
            add = raw_input('Add a turbulent dust density ? (y/n) ')

    add=0
    while add not in ['y','n']:
        add = raw_input('Add a cloud ? (y/n) ')
    while add == 'y':
        add_cloud(mp,cst)
        add = 0
        while add not in ['y','n']:
            add = raw_input('Add a cloud ? (y/n) ')

    add=0
    while add not in ['y','n']:
        add = raw_input('Add a torus (Murakawa 2010) ? (y/n) ')
    while add == 'y':
        add_torus(mp,cst)
        add = 0
        while add not in ['y','n']:
            add = raw_input('Add a torus (Murakawa 2010) ? (y/n) ')

    add=0
    while add not in ['y','n']:
        add = raw_input('Add a constant density torus ? (y/n) ')
    while add == 'y':
        add_torus_const(mp,cst)
        add = 0
        while add not in ['y','n']:
            add = raw_input('Add a constant density torus ? (y/n) ')

    add=0
    while add not in ['y','n']:
        add = raw_input('Add a cylindrical box ? (y/n) ')
    while add == 'y':
        add_camembert(mp,cst)
        add = 0
        while add not in ['y','n']:
            add = raw_input('Add a cylindrical box ? (y/n) ')




def fill_map2(mp,cst,denspower=[],spherepower=[],gaussianprofile=[],circumstellardisk=[],densturb=[],cloud=[],clump=[],torus=[],cone=[],torus_Murakawa=[],torus_grad=[],AGN_simple=[],cylinder=[],shell=[],fractal=[],display=1):
    """adds any of the available structures to a Map object."""

    if(denspower!=[]):
        for i in range(len(denspower)):
            add_density_powerlaw2(mp,denspower[i],cst,display=display)

    if(spherepower!=[]):
        for i in range(len(spherepower)):
            add_spherical_powerlaw2(mp,spherepower[i],cst,display=display)

    if(gaussianprofile!=[]):
        for i in range(len(gaussianprofile)):
            add_gaussian_profile2(mp,gaussianprofile[i],cst,display=display)

    if(circumstellardisk!=[]):
        for i in range(len(circumstellardisk)):
            add_circumstellar_disk2(mp,circumstellardisk[i],cst,display=display)

    if(densturb!=[]):
        for i in range(len(densturb)):
            add_density_turbulent2(mp,densturb[i],cst,display=display)

    if(cloud!=[]):
        for i in range(len(cloud)):
            add_cloud2(mp,cloud[i],cst,display=display)

    #if(clump!=[]):
    #    for i in range(len(clump)):
    #        add_clump2(mp,clump[i],cst,display=display)

    if(torus!=[]):
        for i in range(len(torus)):
            add_torus2(mp,torus[i],cst,display=display)

    if(cone!=[]):
        for i in range(len(cone)):
            add_cone2(mp,cone[i],cst,display=display)

    if(AGN_simple!=[]):
        for i in range(len(AGN_simple)):
            add_AGN_simple2(mp,AGN_simple[i],cst,display=display)

    if(torus_Murakawa!=[]):
        for i in range(len(torus_Murakawa)):
            add_torus_Murakawa2(mp,torus_Murakawa[i],cst,display=display)

    if(torus_grad!=[]):
        for i in range(len(torus_grad)):
            add_torus_grad2(mp,torus_grad[i],cst,display=display)

    if(cylinder!=[]):
        for i in range(len(cylinder)):
            add_camembert2(mp,cylinder[i],cst,display=display)

    if(shell!=[]): #whelan
        for i in range(len(shell)):
            add_shell2(mp,shell[i],cst,display=display)

    if(fractal!=[]): #whelan
        for i in range(len(fractal)):
            add_fractal2(mp,fractal[i],cst,display=display)
    


#############    Geometry functions -- user call    ##############################



def add_density_powerlaw(mp,cst):
    """adds a powerlaw density of given grains to a Map object"""
    lr = mut.impup('Radial power index ? ',cst,expect='float')
    lrd = mut.impup('Radial typical profile size (in m) ? ',cst,expect='float')
    lz = mut.impup('Vertical decay size (in m) ? ',cst,expect='float')
    r = []
    for i in mp.dust:
        #ri = input('Density of '+str(i)+' grains (in particles / m3) at 1 pc ? ')
        ri = input('Density of '+str(i)+' grains (in particles / m3) at '+str(lrd)+' m (radial typical profile size) ? ')
        if(ri!=0):
            i.usegrain=1
        r.append(ri)
    r = np.array(r)
    for i in range(mp.N*2):
        for j in range(mp.N*2):
            for k in range(mp.N*2):
                r0 = np.array(mp.grid[i][j][k].rho) #initial densities
                #r1 = r*(mp.res/pc)**lr*((i-mp.N+.5)**2+(j-mp.N+.5)**2)**(lr*.5)*np.exp(-abs(k-mp.N+.5)*mp.res/lz) #added densities
                r1 = r*(mp.res/lrd)**lr*((i-mp.N+.5)**2+(j-mp.N+.5)**2)**(lr*.5)*np.exp(-abs(k-mp.N+.5)*mp.res/lz) #added densities
                mp.grid[i][j][k].rho = list(r0+r1)


def add_circumstellar_disk(mp,cst):
    """adds a circumstellar disk density of given grains to a Map object"""
    rad0 = mut.impup('Typical radius where the grain density is normalised (in m) ? ',cst,expect='float')
    rc = mut.impup('Critical radius for the power law slope change (in m) ? ',cst,expect='float')
    h0 = mut.impup('Typical height at the typical radius (in m) ? ',cst,expect='float')
    alpha_in = mut.impup('Inner radial power index (>0) ?',cst,expect='float')
    alpha_out = mut.impup('Outer radial power index (<0) ?',cst,expect='float')
    beta = mut.impup('Radial dependency of the vertical profile index (>0) ?',cst,expect='float')
    gamma = mut.impup('Vertical power index ? ',cst,expect='float')

    r = []
    for i in mp.dust:
        ri = input('Density of '+str(i)+' grains (in particles / m3) at the given typical radius and at z=0 ?')
        if(ri!=0):
            i.usegrain=1
        r.append(ri)
    r = np.array(r)

    for i in range(mp.N*2):
        for j in range(mp.N*2):
            for k in range(mp.N*2):
                r0 = np.array(mp.grid[i][j][k].rho) #initial densities
                rad = ((i-mp.N+.5)**2+(j-mp.N+.5)**2)**(0.5)*mp.res
                z = (k-mp.N+.5)*mp.res
                z0 = h0*(rad/rad0)**beta
                Z = np.exp(-(np.abs(z/z0))**gamma)
                R = ((rad/rc)**(-2*alpha_in)+(rad/rc)**(-2*alpha_out))**(-0.5)
                r1 = r*R*Z
                mp.grid[i][j][k].rho = list(r0+r1)


def add_spherical_powerlaw(mp,cst):
    """adds a powerlaw density of given grains to a Map object"""
    lr = mut.impup('Radial power index ? ',cst,expect='float')
    lrd = mut.impup('Radial profile size (in m) ? ',cst,expect='float')
    r = []
    for i in mp.dust:
        #ri = input('Density of '+str(i)+' grains (in particles / m3) at 1 pc ? ')
        ri = input('Density of '+str(i)+' grains (in particles / m3) at '+str(lrd)+' m (radial typical profile size) ? ')
        if(ri!=0):
            i.usegrain=1
        r.append(ri)
    r = np.array(r)
    for i in range(mp.N*2):
        for j in range(mp.N*2):
            for k in range(mp.N*2):
                r0 = np.array(mp.grid[i][j][k].rho) #initial densities
                #r1 = r*(mp.res/pc)**lr*((i-mp.N+.5)**2+(j-mp.N+.5)**2+(k-mp.N+.5)**2)**(lr*.5) #added densities
                r1 = r*(mp.res/lrd)**lr*((i-mp.N+.5)**2+(j-mp.N+.5)**2+(k-mp.N+.5)**2)**(lr*.5) #added densities
                mp.grid[i][j][k].rho = list(r0+r1)

def add_gaussian_profile(mp,cst):
    pass

def add_density_turbulent(mp,cst):
    """adds a turbulent density of given grains to a Map object"""
    pass


def add_cloud(mp,cst):
    """adds a cloud (bump function) to a Map object"""
    x0 = mut.impup('x0 ? ',cst,expect='float')
    y0 = mut.impup('y0 ? ',cst,expect='float')
    z0 = mut.impup('z0 ? ',cst,expect='float')
    a = mut.impup('largest semi-axis ? ',cst,expect='float')
    b = mut.impup('second semi-axis ? ',cst,expect='float')
    c = mut.impup('third semi-axis ? ',cst,expect='float')
    theta = input('first axis polar angle ? ')
    phi = input('first axis azimutal angle ? ')
    if np.sqrt(x0**2+y0**2+z0**2)+max(a,b,c) > mp.Rmax:
        print('Warning ! The cloud might reach beyond Rmax')
    else:
        rho = []
        def cloud(x,y,z):
            X = (np.cos(theta)*np.cos(phi)*(x-x0)+ np.sin(theta)*np.cos(phi)*(z-z0) - np.sin(phi)*(y-y0))/a
            Y = (np.cos(theta)*np.sin(phi)*(x-x0) + np.sin(theta)*np.sin(phi)*(z-z0) + np.cos(phi)*(y-y0))/b
            Z = (-np.sin(theta)*(x-x0) + np.cos(theta)*(z-z0))/c
        #dilatation, then polar rotation (around y axis) then azimuthal rotation (around z axis) then translation
        #bump function : reprojected gaussian function
            if X**2+Y**2+Z**2 > 1 - 1e-6: #1e-6 for precision safety
                return 0
            else:
                return np.exp(1-1/(1-(X**2+Y**2+Z**2))) #maximum at (0,0,0) normalized to 1.

        for i in range(len(mp.dust)):
            rhoi = input('Central density of %s grains ? '%mp.dust[i].name)
            if(rhoi!=0):
                mp.dust[i].usegrain=1
            rho.append(rhoi)


        for x in np.arange(x0-1.05*a,x0+1.05*a+mp.res,mp.res):
            for y in np.arange(y0-1.05*a,y0+1.05*a+mp.res,mp.res):
                for z in np.arange(z0-1.05*a,z0+1.05*a+mp.res,mp.res):
                    xm = np.floor(x/mp.res)*mp.res
                    xp = np.ceil(x/mp.res)*mp.res
                    ym = np.floor(y/mp.res)*mp.res
                    yp = np.ceil(y/mp.res)*mp.res
                    zm = np.floor(z/mp.res)*mp.res
                    zp = np.ceil(z/mp.res)*mp.res
                    m = (cloud(xm,ym,zm)+cloud(xm,ym,zp)+cloud(xm,yp,zm)+cloud(xm,yp,zp)+\
                    cloud(xp,ym,zm)+cloud(xp,ym,zp)+cloud(xp,yp,zm)+cloud(xp,yp,zp))*.125
                    #average of the 8 corners of the cell (faster and simpler than an integration)
                    #r0 = np.array(mp.get(x,y,z).rho) #initial densities
                    xr = x/mp.res
                    yr = y/mp.res
                    zr = z/mp.res
                    i = int(np.floor(xr+mp.N))
                    j = int(np.floor(yr+mp.N))
                    k = int(np.floor(zr+mp.N))
                    #print rho
                    r0 = np.array(mp.grid[i][j][k].rho) #initial densities
                    r1 = np.array(rho)*m #added densities
                    #T = mp.get(x,y,z).T
                    #mp.set(x,y,z,mc.Cell(T,list(r0+r1)))
                    #print r1, list(r0+r1)
                    mp.grid[i][j][k].rho = list(r0+r1)


def add_torus(mp,cst):
    """adds a torus powerlaw density of given grains to a Map object
    as described in Murakawa - 2010 - page 2"""

    Rdisk=mut.impup('Rdisk in m ?',cst,expect='float')
    Rdisk=Rdisk/pc
    #rhod=input('rhod in kg/m3 ?')    
    rhod = []
    for i in mp.dust:
        ri = input('Density of '+str(i)+' grains (in kg / m3) at (?) ? ')
        if(ri!=0):
            i.usegrain=1
        rhod.append(ri)
    rhod = np.array(rhod)
    H=input('H ?')
    Mpenv=input('Mpenv in Msol/yr ?')
    Ms=input('Ms in Msol ?')
    Gs=G/(pc*pc*pc)*Msol*yr*yr
    mu0=np.cos(80./2.*np.pi/180.)
    Rc=Rdisk

    #Computing the mass per particle
    # silicate
    rmax=0.25*1e-6
    rmin=0.005*1e-6
    density=3.3*1e3 #kg/m3 # ou 0.29*1e-3 ? ##cf Vincent Guillet 2008
    particlemass=density*20./3.*np.pi*(np.sqrt(rmax)-np.sqrt(rmin))/(1./rmin**2.5-1./rmax**2.5) #kg/particle


    for i in range(len(param[1])):
        ri = param[1][i]
        if(ri!=0):
            mp.dust[i].usegrain=1
        rhod.append(ri)
    rhod = np.array(rhod)
    for i in range(mp.N*2):
        for j in range(mp.N*2):
            for k in range(mp.N*2):
                x=(i-mp.N+.5)*(mp.res/pc)
                y=(j-mp.N+.5)*(mp.res/pc)
                z=(k-mp.N+.5)*(mp.res/pc)
                r=np.sqrt(x*x+y*y)
                R=np.sqrt(x*x+y*y+z*z)
                mu=np.abs(z/R)
                
                r0 = np.array(mp.grid[i][j][k].rho) #initial densities
                #r1 = rhod*(r/Rdisk)**(-15/8.)*np.exp(-np.pi/4.*(z/(Rdisk*H*(r/Rdisk)**9/8.))**2) #added disk densities
                #mu0 = 
                #r2 = Mpenv*Msol/(4.*np.pi*np.sqrt(Gs*Ms*R*R*R))*1./(np.sqrt(1+mu/mu0))*1./(mu/mu0+2.*mu0*mu0*Rc/R) #added envelope densities


                if(R<Rdisk):
                    if(mu<=mu0):
                        r1 = rhod*((r/Rdisk)**(-15./8.))*np.exp(-np.pi/4.*(z/(Rdisk*H*(r/Rdisk)**(9./8.)))**2) #added disk densities
                    else:
                        r1 = rhod*((r/Rdisk)**(-15./8.))*np.exp(-np.pi/4.*(z/(Rdisk*H*(r/Rdisk)**(9./8.)))**2)*0.01
                    r2 = [0]
                else:
                    r1 = [0]
                    if(mu<=mu0):
                        r2 = [Mpenv*Msol/(4.*np.pi*np.sqrt(Gs*Ms*R*R*R)*np.sqrt(1+mu/mu0)*(mu/mu0+2.*mu0*mu0*Rc/R))/(particlemass*pc*pc*pc)] #added envelope densities
                    else:
                        r2 = [Mpenv*Msol/(4.*np.pi*np.sqrt(Gs*Ms*R*R*R)*np.sqrt(1+mu/mu0)*(mu/mu0+2.*mu0*mu0*Rc/R))/(particlemass*pc*pc*pc)*0.01]

                mp.grid[i][j][k].rho = list(r0+r1+r2)




def add_torus_const(mp,cst):
    """adds a torus powerlaw density of given grains to a Map object
    as described in Murakawa - 2010 - page 2"""
    #Rdisk=param[0] #in m
    Rdisk=mut.impup('Rdisk in m ?',cst,expect='float')
    rhod=[]
    rhoe=[]
    rhoc=[]
    #H=param[2]
    #Mpenv=param[3] #in Msol/yr
    #Ms=param[4] #in Msol
    Gs=G/(pc*pc*pc)*Msol*yr*yr
    #mu0=np.cos(80./2.*np.pi/180.)
    mu0=np.cos(25*np.pi/180.)
    Rc=Rdisk/pc
    Rdisk=Rdisk/pc
    theta0=30*np.pi/180.
    mutheta=np.cos(np.pi/2.-theta0)
    Rout=25 #pc

    #Computing the mass per particle 
    rmax=0.25*1e-6
    rmin=0.005*1e-6
    density=3.3*1e3 #kg/m3 # ou 0.29*1e-3 ? ##cf Vincent Guillet 2008
    particlemass=density*20./3.*np.pi*(np.sqrt(rmax)-np.sqrt(rmin))/(1./rmin**2.5-1./rmax**2.5) #kg/particle

    
    for i in mp.dust:
        ri = input('Density of '+str(i)+' grains (in kg / m3) in the torus ?')
        rhod.append(ri)
        rie = input('Density of '+str(i)+' grains (in kg / m3) in the envelope ?')
        rhoe.append(rie)
        ric = input('Density of '+str(i)+' grains (in kg / m3) in the cone ?')
        rhoc.append(ric)
        if(ri!=0 or rie!=0 or ric!=0):
            i.usegrain=1
    rhod = np.array(rhod)
    rhoe = np.array(rhoe)
    rhoc = np.array(rhoc)

    for i in range(mp.N*2):
        for j in range(mp.N*2):
            for k in range(mp.N*2):
                x=(i-mp.N+.5)*(mp.res/pc)
                y=(j-mp.N+.5)*(mp.res/pc)
                z=(k-mp.N+.5)*(mp.res/pc)
                r=np.sqrt(x*x+y*y)
                R=np.sqrt(x*x+y*y+z*z)
                mu=np.abs(z/R)
                
                r0 = np.array(mp.grid[i][j][k].rho) #initial densities
                if(R<Rdisk):
                    if(mu<=mu0):
                        if(mu<=mutheta):
                            r1 = rhod #added disk densities
                        else:
                            r1= [0]
                    else:
                        r1= rhoc
                    r2 = [0]
                elif(R<Rout):
                    r1 = [0]
                    if(mu<=mu0):
                        r2 = rhoe #added envelope densities
                    else:
                        r2 = rhoc
                else:
                    r1=[0]
                    r2=[0]
                mp.grid[i][j][k].rho = list(r0+r1+r2)




def add_camembert(mp,cst):
    """adds a constant density cylindrical box of given grains to a Map object"""
    radius = mut.impup('Radius of the box (in m) ?',cst,expect='float')
    h = mut.impup('Half-height of the box (in m) ?',cst,expect='float')
    r = []
    for i in mp.dust:
        ri = input('Density of '+str(i)+' grains (in particles / m3) ? ')
        if(ri!=0):
            i.usegrain=1
        r.append(ri)
    r = np.array(r)
    for i in range(mp.N*2):
        for j in range(mp.N*2):
            for k in range(mp.N*2):
                R=np.sqrt((i-mp.N+0.5)*(i-mp.N+0.5)+(j-mp.N+0.5)*(j-mp.N+0.5))*mp.res
                z=(k-mp.N+0.5)*mp.res
                r0 = np.array(mp.grid[i][j][k].rho) #initial densities
                if(np.abs(z)<h and R<radius):
                    r1 = r #r*(mp.res/pc)**lr*((i-mp.N+.5)**2+(j-mp.N+.5)**2+(k-mp.N+.5)**2)**(lr*.5) #added densities
                else:
                    r1=[0]
                mp.grid[i][j][k].rho = list(r0+r1)



#############    Geometry functions -- predifined params    ##############################


def add_density_powerlaw2(mp,paramd,cst,display=1):
    """adds a powerlaw density of given grains to a Map object"""
    param=mut.translate_map_param(paramd,mp.dust,[0])

    lr = param[1]
    lrd= param[2]
    lz = param[3]
    r = []
    for i in range(len(param[0])):
        ri = param[0][i]
        if(ri!=0):
            mp.dust[i].usegrain=1
        r.append(ri)
    r = np.array(r)

    for i in range(mp.N*2):
        for j in range(mp.N*2):
            for k in range(mp.N*2):
                r0 = np.array(mp.grid[i][j][k].rho) #initial densities
                r1 = r*(mp.res/lrd)**lr*((i-mp.N+.5)**2+(j-mp.N+.5)**2)**(lr*.5)*np.exp(-abs(k-mp.N+.5)*mp.res/lz) #added densities
                mp.grid[i][j][k].rho = list(r0+r1)

def add_gaussian_profile2(mp,paramd,cst,display=1):
    """adds a powerlaw density of given grains to a Map object"""
    param=mut.translate_map_param(paramd,mp.dust,[0])

    powgau = param[1] #2
    powh = param[2] #1.125
    powrad = param[3] #2.5
    rad0 = param[4] #100*AU
    h0 = param[5] #10*AU
    Rout = param[6] #400*AU

    r = []
    for i in range(len(param[0])):
        ri = param[0][i]
        if(ri!=0):
            mp.dust[i].usegrain=1
        r.append(ri)
    r = np.array(r)

    for i in range(mp.N*2):
        for j in range(mp.N*2):
            for k in range(mp.N*2):
                r0 = np.array(mp.grid[i][j][k].rho) #initial densities
                rad = ((i-mp.N+.5)**2+(j-mp.N+.5)**2)**(0.5)*mp.res
                z = (k-mp.N+.5)*mp.res
                # rho = 3/2 Sigma/rad
                if(rad<Rout):
                    r1 = 1.5*r*(rad0/rad)**powrad*np.exp(-z**powgau/(2*(h0*(rad/rad0)**powh)**powgau))
                else:
                    r1 = 0.
                mp.grid[i][j][k].rho = list(r0+r1)


def add_circumstellar_disk2(mp,paramd,cst,display=1):
    """adds a circumstellar disk density of given grains to a Map object"""
    param=mut.translate_map_param(paramd,mp.dust,[0])

    rad0 = param[1]
    rc = param[2]
    h0 = param[3]
    alpha_in = param[4] #>0
    alpha_out = param[5] #<0
    beta = param[6] #>0
    gamma = param[7]

    r = []
    for i in range(len(param[0])):
        ri = param[0][i]
        if(ri!=0):
            mp.dust[i].usegrain=1
        r.append(ri)
    r = np.array(r)

    for i in range(mp.N*2):
        for j in range(mp.N*2):
            for k in range(mp.N*2):
                r0 = np.array(mp.grid[i][j][k].rho) #initial densities
                rad = ((i-mp.N+.5)**2+(j-mp.N+.5)**2)**(0.5)*mp.res
                z = (k-mp.N+.5)*mp.res
                z0 = h0*(rad/rad0)**beta
                Z = np.exp(-(np.abs(z/z0))**gamma)
                R = ((rad/rc)**(-2*alpha_in)+(rad/rc)**(-2*alpha_out))**(-0.5)
                r1 = r*R*Z
                mp.grid[i][j][k].rho = list(r0+r1)


def add_spherical_powerlaw2(mp,paramd,cst,display=1):
    """adds a powerlaw density of given grains to a Map object"""
    param=mut.translate_map_param(paramd,mp.dust,[0])

    lr = param[1]
    lrd= param[2]
    r = []
    for i in range(len(param[0])):
        ri = param[0][i]
        if(ri!=0):
            mp.dust[i].usegrain=1
        r.append(ri)
    r = np.array(r)

    for i in range(mp.N*2):
        for j in range(mp.N*2):
            for k in range(mp.N*2):
                r0 = np.array(mp.grid[i][j][k].rho) #initial densities
                r1 = r*(mp.res/lrd)**lr*((i-mp.N+.5)**2+(j-mp.N+.5)**2+(k-mp.N+.5)**2)**(lr*.5) #added densities
                mp.grid[i][j][k].rho = list(r0+r1)

def add_density_turbulent2(mp,param,cst,display=1):
    """adds a turbulent density of given grains to a Map object"""
    pass


def add_cloud2(mp,paramd,cst,display=1): #,checktau=0):
    """adds a cloud (bump function) to a Map object"""
    #x0 = input('x0 ? ')
    #y0 = input('y0 ? ')
    #z0 = input('z0 ? ')
    #a = input('largest semi-axis ? ')
    #b = input('second semi-axis ? ')
    #c = input('third semi-axis ? ')
    #theta = input('first axis polar angle ? ')
    #phi = input('first axis azimutal angle ? ')
    param=mut.translate_map_param(paramd,mp.dust,[0])

    x0 = param[1]
    y0 = param[2]
    z0 = param[3]
    a = param[4]
    b = param[5]
    c = param[6]
    theta = param[7]
    phi = param[8]
    if np.sqrt(x0**2+y0**2+z0**2)+max(a,b,c) > mp.Rmax:
        print('Warning ! The cloud might reach beyond Rmax')
    else:
        rho = []
        def cloud(x,y,z):
            X = (np.cos(theta)*np.cos(phi)*(x-x0)+ np.sin(theta)*np.cos(phi)*(z-z0) - np.sin(phi)*(y-y0))/a
            Y = (np.cos(theta)*np.sin(phi)*(x-x0) + np.sin(theta)*np.sin(phi)*(z-z0) + np.cos(phi)*(y-y0))/b
            Z = (-np.sin(theta)*(x-x0) + np.cos(theta)*(z-z0))/c
        #dilatation, then polar rotation (around y axis) then azimuthal rotation (around z axis) then translation
        #bump function : reprojected gaussian function
            if X**2+Y**2+Z**2 > 1 - 1e-6: #1e-6 for precision safety
                return 0
            else:
                return np.exp(1-1/(1-(X**2+Y**2+Z**2))) #maximum at (0,0,0) normalized to 1.
        # Number of particles inside a spherical cloud = 4*np.pi*scipy.integrate.quad(lambda r: r**2*np.exp(1-1/(1-r**2/Rcloud**2)),0,Rcloud)[0]

        for i in range(len(param[0])):
            rhoi = param[0][i]
            if(rhoi!=0):
                mp.dust[i].usegrain=1
            rho.append(rhoi)
            #r = np.array(r)


        for x in np.arange(x0-1.05*a,x0+1.05*a+mp.res,mp.res):
            for y in np.arange(y0-1.05*a,y0+1.05*a+mp.res,mp.res):
                for z in np.arange(z0-1.05*a,z0+1.05*a+mp.res,mp.res):
                    xm = np.floor(x/mp.res)*mp.res
                    xp = np.ceil(x/mp.res)*mp.res
                    ym = np.floor(y/mp.res)*mp.res
                    yp = np.ceil(y/mp.res)*mp.res
                    zm = np.floor(z/mp.res)*mp.res
                    zp = np.ceil(z/mp.res)*mp.res
                    m = (cloud(xm,ym,zm)+cloud(xm,ym,zp)+cloud(xm,yp,zm)+cloud(xm,yp,zp)+\
                    cloud(xp,ym,zm)+cloud(xp,ym,zp)+cloud(xp,yp,zm)+cloud(xp,yp,zp))*.125
                    #average of the 8 corners of the cell (faster and simpler than an integration)
                    #r0 = np.array(mp.get(x,y,z).rho) #initial densities
                    r1 = np.array(rho)*m #added densities
                    xr = x/mp.res
                    yr = y/mp.res
                    zr = z/mp.res
                    i = int(np.floor(xr+mp.N))
                    j = int(np.floor(yr+mp.N))
                    k = int(np.floor(zr+mp.N))
                    r0 = np.array(mp.grid[i][j][k].rho) #initial densities
                    #T = mp.get(x,y,z).T
                    #mp.set(x,y,z,mc.Cell(T,list(r0+r1)))
                    mp.grid[i][j][k].rho = list(r0+r1)


def clump(Rcloud,rho,n,Rin,Rout): #whelan
    """returns spherical clouds of given grains at random position in a shell"""
    cloud=[]
    rnd = g_rand.gen_generator()
    for i in range(n):
        R = rnd.uniform(Rin+Rcloud,Rout-Rcloud)
        theta = rnd.uniform(0,np.pi)
        phi = rnd.uniform(0,2*np.pi)
        x0 = R*np.sin(theta)*np.cos(phi)
        y0 = R*np.sin(theta)*np.sin(phi)
        z0 = R*np.cos(theta)
        cloud.append([rho,x0,y0,z0,Rcloud,Rcloud,Rcloud,0,0])
    return cloud


def add_torus_Murakawa2(mp,paramd,cst,display=1):
    """adds a torus powerlaw density of given grains to a Map object
    as described in murakawa - 2010 - page 2"""
    param=mut.translate_map_param(paramd,mp.dust,[0])
    Rdisk=param[1] #in m
    rhod=[]
    H=param[2]
    Mpenv=param[3] #in Msol/yr
    Ms=param[4] #in Msol
    Gs=G/(cst.pc*cst.pc*cst.pc)*cst.Msol*cst.yr*cst.yr
    mu0=np.cos(80./2.*np.pi/180.)
    Rc=Rdisk/cst.pc
    Rdisk=Rdisk/cst.pc


    rmax=[] #rgmax #0.25*1e-6
    rmin=[] #rgmin #0.005*1e-6
    #density=[3.3*1e3,0,0,0] #kg/m3 # ou 0.29*1e-3 ? ##cf Vincent Guillet 2008
    invparticlemass=[]
    for i in range(len(param[0])):
        rmax.append(mp.dust[i].rmax)
        rmin.append(mp.dust[i].rmin)
        ri = param[0][i]
        if(ri!=0 and i!=3): #not working for e-
            mp.dust[i].usegrain=1
            #Computing the mass per particle 
            density=mp.dust[i].density
            invparticlemass.append(1.0/(density*20./3.*np.pi*(np.sqrt(rmax[i])-np.sqrt(rmin[i]))/(1./rmin[i]**2.5-1./rmax[i]**2.5))) #particles/kg
            #particlemass.append(1.0/(density[i]*20./3.*np.pi*(np.sqrt(rmax[i])-np.sqrt(rmin[i]))/(1./rmin[i]**2.5-1./rmax[i]**2.5))) #kg/particle
        else:
            invparticlemass.append(0.0)
        rhod.append(ri)
    rhod = np.array(rhod)

    invparticlemass=np.array(invparticlemass)


    for i in range(mp.N*2):
        for j in range(mp.N*2):
            for k in range(mp.N*2):
                x=(i-mp.N+.5)*(mp.res/cst.pc)
                y=(j-mp.N+.5)*(mp.res/cst.pc)
                z=(k-mp.N+.5)*(mp.res/cst.pc)
                r=np.sqrt(x*x+y*y)
                R=np.sqrt(x*x+y*y+z*z)
                mu=np.abs(z/R)
                
                r0 = np.array(mp.grid[i][j][k].rho) #initial densities
                if(R<Rdisk):
                    #if(j==mp.N and k==mp.N):
                        #r3[i] = rhod*((r/Rdisk)**(-15./8.))*np.exp(-np.pi/4.*(z/(Rdisk*H*(r/Rdisk)**(9./8.)))**2)
                    if(mu<=mu0):
                        r1 = rhod*((r/Rdisk)**(-15./8.))*np.exp(-np.pi/4.*(z/(Rdisk*H*(r/Rdisk)**(9./8.)))**2) #added disk densities
                    else:
                        r1 = rhod*((r/Rdisk)**(-15./8.))*np.exp(-np.pi/4.*(z/(Rdisk*H*(r/Rdisk)**(9./8.)))**2)*0.01
                    r2 = np.zeros(len(param[1]))
                else:
                    r1 = np.zeros(len(param[1]))
                    #mu0 = 
                    #r2 = [Mpenv*Msol/(3.0e-15*pc*pc*pc*4.*np.pi*np.sqrt(Gs*Ms*R*R*R)*np.sqrt(1+mu/mu0)*(mu/mu0+2.*mu0*mu0*Rc/R))] #added envelope densities
                
                    if(mu<=mu0):
                        #r2 = [Mpenv*Msol/(4.*np.pi*np.sqrt(Gs*Ms*R*R*R)*np.sqrt(1+mu/mu0)*(mu/mu0+2.*mu0*mu0*Rc/R))/(particlemass*pc*pc*pc)] #added envelope densities
                        r2 = Mpenv*Msol/(4.*np.pi*np.sqrt(Gs*Ms*R*R*R)*np.sqrt(1+mu/mu0)*(mu/mu0+2.*mu0*mu0*Rc/R))/(cst.pc*cst.pc*cst.pc)*invparticlemass
                    else:
                        #r2 = [Mpenv*Msol/(4.*np.pi*np.sqrt(Gs*Ms*R*R*R)*np.sqrt(1+mu/mu0)*(mu/mu0+2.*mu0*mu0*Rc/R))/(particlemass*pc*pc*pc)*0.01]
                        r2 = Mpenv*Msol/(4.*np.pi*np.sqrt(Gs*Ms*R*R*R)*np.sqrt(1+mu/mu0)*(mu/mu0+2.*mu0*mu0*Rc/R))/(cst.pc*cst.pc*cst.pc)*0.01*invparticlemass
                #r2 = [Mpenv*Msol/(3.0e-15)/(pc*pc*pc)/(4.*np.pi*np.sqrt(Gs*Ms*R*R*R))/(mu/mu0+2.*mu0*mu0*Rc/R)] #added envelope densities
                #if(i==mp.N and j==mp.N and k==mp.N):
                    #print "z, R :",z,y,x,R

                    #print "rho :",r1,r2,list(r1+r2)
                #print np.shape(r0),np.shape(r1),np.shape(r2),np.shape(invparticlemass)
                #print r0
                #print r1
                #print r2,invparticlemass
                mp.grid[i][j][k].rho = list(r0+r1+r2)

    #print "top"
    #n=10000
    #r3 = np.zeros([mp.N*2*n])
    #for i in range(n*2*mp.N):
    #    x=(i-mp.N*n+.5)*(mp.res/pc/n)
    #    y=0.0
    #    z=0.0
    #    r=np.sqrt(x*x+y*y)
    #    R=np.sqrt(x*x+y*y+z*z)
    #    mu=np.abs(z/R)
    #    if(r>0.05*AU/pc):
    #        r3[i] = rhod*((r/Rdisk)**(-15./8.))*np.exp(-np.pi/4.*(z/(Rdisk*H*(r/Rdisk)**(9./8.)))**2)
    #plt.plot(r3)
    #print sum(r3)*particlemass
    #print "bot"

def add_torus2(mp,paramd,cst,display=1):
    param=mut.translate_map_param(paramd,mp.dust,[0])

    Rdisk=param[1] #in m
    thetadisk=param[2] #in degree (old value 30)
    #WARNING thetadisk is defined as the half-angle of the torus and NOT as its half-opening angle
    rhod=[]
    #H=param[2]
    #Mpenv=param[3] #in Msol/yr
    #Ms=param[4] #in Msol
    Gs=cst.G/(cst.pc*cst.pc*cst.pc)*cst.Msol*cst.yr*cst.yr
    Rdisk=Rdisk/cst.pc
    theta0=thetadisk*np.pi/180.
    mutheta=np.cos(np.pi/2.-theta0) #remove the pi/2- to switch to opening angle

    for i in range(len(param[0])):
        ri = param[0][i]
        if(ri!=0):
            mp.dust[i].usegrain=1
        rhod.append(ri)
    rhod = np.array(rhod)

    for i in range(mp.N*2):
        for j in range(mp.N*2):
            for k in range(mp.N*2):
                x=(i-mp.N+.5)*(mp.res/cst.pc)
                y=(j-mp.N+.5)*(mp.res/cst.pc)
                z=(k-mp.N+.5)*(mp.res/cst.pc)
                r=np.sqrt(x*x+y*y)
                R=np.sqrt(x*x+y*y+z*z)
                mu=np.abs(z/R)
                
                r0 = np.array(mp.grid[i][j][k].rho) #initial densities
                if(R<Rdisk and mu<=mutheta):
                    r1 = rhod #*(Rdisk-R) #added disk densities
                else:
                    r1= [0]
                mp.grid[i][j][k].rho = list(r0+r1)

def add_cone2(mp,paramd,cst,display=1):
    param=mut.translate_map_param(paramd,mp.dust,[0])

    Rout=param[1] #in m
    thetacone=param[2] #in degree (old value 25)
    rhoc=[]
    #H=param[2]
    #Mpenv=param[3] #in Msol/yr
    #Ms=param[4] #in Msol
    Gs=cst.G/(cst.pc*cst.pc*cst.pc)*cst.Msol*cst.yr*cst.yr
    mu0=np.cos(thetacone*np.pi/180.)
    Rout=Rout/cst.pc

    for i in range(len(param[0])):
        ric = param[0][i]
        if(ric!=0):
            mp.dust[i].usegrain=1
        rhoc.append(ric)
    rhoc = np.array(rhoc)


    for i in range(mp.N*2):
        for j in range(mp.N*2):
            for k in range(mp.N*2):
                x=(i-mp.N+.5)*(mp.res/cst.pc)
                y=(j-mp.N+.5)*(mp.res/cst.pc)
                z=(k-mp.N+.5)*(mp.res/cst.pc)
                r=np.sqrt(x*x+y*y)
                R=np.sqrt(x*x+y*y+z*z)
                mu=np.abs(z/R)
                
                r0 = np.array(mp.grid[i][j][k].rho) #initial densities
                if(mu<=mu0 or R>Rout):
                    r1= [0]
                else:
                    r1= rhoc
                mp.grid[i][j][k].rho = list(r0+r1)


def add_AGN_simple2(mp,paramd,cst,display=1):
    """adds"""
    param=mut.translate_map_param(paramd,mp.dust,[0,1,2])
    #Rdisk=param[3] #in m
    #Rout=param[4] #in m
    #thetadisk=param[5] #in degree
    #thetacone=param[6] #in degree
    add_torus2(mp,[paramd[0],paramd[1],paramd[4],paramd[6]],cst,display=display)
    add_shell2(mp,[paramd[0],paramd[2],paramd[4],paramd[5]],cst,display=display)
    add_cone2(mp,[paramd[0],paramd[3],paramd[5],paramd[7]],cst,display=display)

def add_torus_const2(mp,paramd,cst,display=1):
    """adds"""
    param=mut.translate_map_param(paramd,mp.dust,[0,1,2])
    #print param
    Rdisk=param[3] #in m
    rhod=[]
    rhoe=[]
    rhoc=[]
    #H=param[2]
    #Mpenv=param[3] #in Msol/yr
    #Ms=param[4] #in Msol
    Gs=cst.G/(cst.pc*cst.pc*cst.pc)*cst.Msol*cst.yr*cst.yr
    #mu0=np.cos(80./2.*np.pi/180.)
    mu0=np.cos(25*np.pi/180.)
    Rc=Rdisk/cst.pc
    Rdisk=Rdisk/cst.pc
    theta0=30*np.pi/180.
    mutheta=np.cos(np.pi/2.-theta0)
    Rout=25 #pc

    #Computing the mass per particle 
    #rmax=#rgmax #0.25*1e-6
    #rmin=#rgmin #0.005*1e-6
    #density=3.3*1e3 #kg/m3 # ou 0.29*1e-3 ? ##cf Vincent Guillet 2008
    #particlemass=density*20./3.*np.pi*(np.sqrt(rmax)-np.sqrt(rmin))/(1./rmin**2.5-1./rmax**2.5) #kg/particle


    for i in range(len(param[0])):
        ri = param[0][i]
        if(ri!=0):
            mp.dust[i].usegrain=1
        rhod.append(ri)
    rhod = np.array(rhod)
    for i in range(len(param[1])):
        rie = param[1][i]
        if(rie!=0):
            mp.dust[i].usegrain=1
        rhoe.append(rie)
    rhoe = np.array(rhoe)
    for i in range(len(param[2])):
        ric = param[2][i]
        if(ric!=0):
            mp.dust[i].usegrain=1
        rhoc.append(ric)
    rhoc = np.array(rhoc)


    for i in range(mp.N*2):
        for j in range(mp.N*2):
            for k in range(mp.N*2):
                x=(i-mp.N+.5)*(mp.res/cst.pc)
                y=(j-mp.N+.5)*(mp.res/cst.pc)
                z=(k-mp.N+.5)*(mp.res/cst.pc)
                r=np.sqrt(x*x+y*y)
                R=np.sqrt(x*x+y*y+z*z)
                mu=np.abs(z/R)
                
                r0 = np.array(mp.grid[i][j][k].rho) #initial densities
                if(R<Rdisk):
                    if(mu<=mu0):
                        if(mu<=mutheta):
                            r1 = rhod #added disk densities
                        else:
                            r1= [0]
                    else:
                        r1= rhoc
                    r2 = [0]
                elif(R<Rout):
                    r1 = [0]
                    if(mu<=mu0):
                        r2 = rhoe #added envelope densities
                    else:
                        r2 = rhoc
                else:
                    r1=[0]
                    r2=[0]
                mp.grid[i][j][k].rho = list(r0+r1+r2)

    #print "top"
    #n=10000
    #r3 = np.zeros([mp.N*2*n])
    #for i in range(n*2*mp.N):
    #    x=(i-mp.N*n+.5)*(mp.res/pc/n)
    #    y=0.0
    #    z=0.0
    #    r=np.sqrt(x*x+y*y)
    #    R=np.sqrt(x*x+y*y+z*z)
    #    mu=np.abs(z/R)
    #    if(r>0.05*AU/pc):
    #        r3[i] = rhod*((r/Rdisk)**(-15./8.))*np.exp(-np.pi/4.*(z/(Rdisk*H*(r/Rdisk)**(9./8.)))**2)
    #plt.plot(r3)
    #print sum(r3)*particlemass
    #print "bot"

def add_torus_grad2(mp,paramd,cst,display=1):
    """adds a torus with a gradient density of given grains to a Map object
    """
    param=mut.translate_map_param(paramd,mp.dust,[0,1,2])

    Rdisk=param[3] #in m
    rhod=[]
    rhoe=[]
    rhoc=[]
    #H=param[2]
    #Mpenv=param[3] #in Msol/yr
    #Ms=param[4] #in Msol
    Gs=cst.G/(cst.pc*cst.pc*cst.pc)*cst.Msol*cst.yr*cst.yr
    #mu0=np.cos(80./2.*np.pi/180.)
    mu0=np.cos(25*np.pi/180.)
    Rc=Rdisk/cst.pc
    Rdisk=Rdisk/cst.pc
    theta0=30*np.pi/180.
    mutheta=np.cos(np.pi/2.-theta0)
    Rout=25 #pc

    #Computing the mass per particle 
    #rmax=#rgmax #0.25*1e-6
    #rmin=#rgmin #0.005*1e-6
    #density=3.3*1e3 #kg/m3 # ou 0.29*1e-3 ? ##cf Vincent Guillet 2008
    #particlemass=density*20./3.*np.pi*(np.sqrt(rmax)-np.sqrt(rmin))/(1./rmin**2.5-1./rmax**2.5) #kg/particle


    for i in range(len(param[0])):
        ri = param[0][i]
        if(ri!=0):
            mp.dust[i].usegrain=1
        rhod.append(ri)
    rhod = np.array(rhod)
    for i in range(len(param[1])):
        rie = param[1][i]
        if(rie!=0):
            mp.dust[i].usegrain=1
        rhoe.append(rie)
    rhoe = np.array(rhoe)
    for i in range(len(param[2])):
        ric = param[2][i]
        if(ric!=0):
            mp.dust[i].usegrain=1
        rhoc.append(ric)
    rhoc = np.array(rhoc)

    for i in range(mp.N*2):
        for j in range(mp.N*2):
            for k in range(mp.N*2):
                x=(i-mp.N+.5)*(mp.res/cst.pc)
                y=(j-mp.N+.5)*(mp.res/cst.pc)
                z=(k-mp.N+.5)*(mp.res/cst.pc)
                r=np.sqrt(x*x+y*y)
                R=np.sqrt(x*x+y*y+z*z)
                mu=np.abs(z/R)
                
                r0 = np.array(mp.grid[i][j][k].rho) #initial densities
                if(R<Rdisk):
                    if(mu<=mu0):
                        if(mu<=mutheta):
                            r1 = rhod*(Rdisk-R) #added disk densities
                        else:
                            r1= [0]
                    else:
                        r1= rhoc
                    r2 = [0]
                elif(R<Rout):
                    r1 = [0]
                    if(mu<=mu0):
                        r2 = rhoe #added envelope densities
                    else:
                        r2 = rhoc
                else:
                    r1=[0]
                    r2=[0]
                mp.grid[i][j][k].rho = list(r0+r1+r2)

    #print "top"
    #n=10000
    #r3 = np.zeros([mp.N*2*n])
    #for i in range(n*2*mp.N):
    #    x=(i-mp.N*n+.5)*(mp.res/pc/n)
    #    y=0.0
    #    z=0.0
    #    r=np.sqrt(x*x+y*y)
    #    R=np.sqrt(x*x+y*y+z*z)
    #    mu=np.abs(z/R)
    #    if(r>0.05*AU/pc):
    #        r3[i] = rhod*((r/Rdisk)**(-15./8.))*np.exp(-np.pi/4.*(z/(Rdisk*H*(r/Rdisk)**(9./8.)))**2)
    #plt.plot(r3)
    #print sum(r3)*particlemass
    #print "bot"


def add_camembert2(mp,paramd,cst,display=1):
    """adds a constant density cylindrical box of given grains to a Map object"""
    param=mut.translate_map_param(paramd,mp.dust,[0])

    radius = param[1]
    h = param[2]
    r = []
    for i in range(len(param[0])):
        ri = param[0][i]
        if(ri!=0):
            mp.dust[i].usegrain=1
        r.append(ri)
    r = np.array(r)

    for i in range(mp.N*2):
        for j in range(mp.N*2):
            for k in range(mp.N*2):
                R=np.sqrt((i-mp.N+0.5)*(i-mp.N+0.5)+(j-mp.N+0.5)*(j-mp.N+0.5))*mp.res
                z=(k-mp.N+0.5)*mp.res
                r0 = np.array(mp.grid[i][j][k].rho) #initial densities
                if(np.abs(z)<h and R<radius):
                    r1 = r #r*(mp.res/pc)**lr*((i-mp.N+.5)**2+(j-mp.N+.5)**2+(k-mp.N+.5)**2)**(lr*.5) #added densities
                else:
                    r1=[0]
                mp.grid[i][j][k].rho = list(r0+r1)


def add_shell2(mp,paramd,cst,display=1): #whelan
    """adds a constant density shell of given grains to a Map object"""
    param=mut.translate_map_param(paramd,mp.dust,[0])
    Rin = param[1] #inner radius
    Rout= param[2] #outer radius
    r = []
    for i in range(len(param[0])):
        ri = param[0][i]
        if(ri!=0):
            mp.dust[i].usegrain=1
        r.append(ri)
    r = np.array(r)

    for i in range(mp.N*2):
        for j in range(mp.N*2):
            for k in range(mp.N*2):
                R=np.sqrt((i-mp.N+0.5)*(i-mp.N+0.5)+(j-mp.N+0.5)*(j-mp.N+0.5)+(k-mp.N+0.5)*(k-mp.N+0.5))*mp.res
                if (R>=Rin)&(R<=Rout):
                    r0 = np.array(mp.grid[i][j][k].rho) #initial densities
                    r1 = r #added densities
                    mp.grid[i][j][k].rho = list(r0+r1)
                    
def add_fractal2(mp,param,cst,display=1): #whelan
    """adds a hierarchically clumped shell (Elmegreen 1997) of given grains to a Map object based on the
algorithm described in Mathis et al. (2002)"""
    if len(param)==8:
        param=mut.translate_map_param(param,mp.dust,[0])

    ratio = param[0] #ratio of grains
    ratio = ratio/np.sum(ratio) #normalization
    N = param[1] #number of new positions that, at each level, are substituted for each position in the previous level (N^H positions after H levels)
    D = param[2] #volume fractal dimension
    L = N**(1/D) #fractal length (delta in Mathis et al. 2002)
    H = param[3] #number of hierarchical levels
    Rin = param[4] #inner radius
    Rout = param[5] #outer radius
    Mfractal = param[6] #total mass of the hierarchically clumped shell 
    
    rnd = g_rand.gen_generator()    
    
    if len(param)==7:
        fractal_shell = []
        for n in range(N):
            [i,j,k] = rnd.uniform(0,mp.N*2,3).astype(int)
            add_fractal2(mp,param+[1]+[[i,j,k]]+[fractal_shell],cst)

        particle_mass = np.zeros(len(mp.dust))
        dust_ind={
            'silicates':0,
            'graphites_ortho':1,
            'graphites_para':2,
            'pah_neu':3,
            'pah_ion':4,
            'electrons':5
        }
        for i in xrange(len(mp.dust)):
            particle_mass[i] = mut.particle_mass_new(mp.dust[i].rmin,mp.dust[i].rmax,mp.dust[i].alpha)[dust_ind[mp.dust[i].name]]
            
        rho_fractal = Mfractal/(np.sum(np.multiply(ratio,particle_mass))*len(fractal_shell)*mp.res**3)          
                
        r = []
        for i in range(len([x*rho_fractal for x in ratio])):
            ri = [x*rho_fractal for x in ratio][i]
            if(ri!=0):
                mp.dust[i].usegrain=1
            r.append(ri)
        r = np.array(r)          
        
        for [i,j,k] in fractal_shell:    
            r0 = np.array(mp.grid[i][j][k].rho) #initial densities
            r1 = r #added densities
            mp.grid[i][j][k].rho = list(r0+r1)
    else:
        h = param[7]        
        [i,j,k] = param[8]
        fractal_shell = param[9]
        
        if h==H:            
            [i,j,k]=[np.mod(np.fmod(i,mp.N*2)+mp.N*2,mp.N*2),np.mod(np.fmod(j,mp.N*2)+mp.N*2,mp.N*2),np.mod(np.fmod(k,mp.N*2)+mp.N*2,mp.N*2)]
            R=np.sqrt((i-mp.N+0.5)**2+(j-mp.N+0.5)**2+(k-mp.N+0.5)**2)*mp.res
            if (R>=Rin)&(R<=Rout): 
                fractal_shell.append([i,j,k])
        else:
            for n in range(N):
                dist = rnd.uniform(0,mp.N/L,3).astype(int)
                param_ = param
                param_[7] = h+1
                param_[8] = [i,j,k]+dist
                param_[9] = fractal_shell
                add_fractal2(mp,param_,cst)                
           

#############    Import parameter functions    ##############################


def read_paramfile(filenameparam,info):
    cst = mc.Constant()
    dust_pop = []
    emission = []
    emdir = []
    polardir = []
    polar= []
    sources = []
    densturb = []
    spherepower = []
    gaussianprofile = []
    circumstellardisk = []
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
            dust_pop.append([col[1],col[2],mut.readvalue(col[3],cst),mut.readvalue(col[4],cst),mut.readvalue(col[5],cst),mut.readvalue(col[6],cst)]) 
            if(info.verbose==1):
                print(dust_pop)
        if(col[0] == 'denspower'):
            denspower.append([col[1][1:-1].split(','),mut.readvalue(col[2][1:-1].split(','),cst,array=1),mut.readvalue(col[3],cst),mut.readvalue(col[4],cst),mut.readvalue(col[5],cst)]) 
            if(info.verbose==1):
                print(denspower)
        if(col[0] == 'spherepower'):
            spherepower.append([col[1][1:-1].split(','), mut.readvalue(col[2][1:-1].split(','),cst,array=1), mut.readvalue(col[3],cst),mut.readvalue(col[4],cst)]) 
            if(info.verbose==1):
                print(spherepower)
        if(col[0] == 'gaussianprofile'):
            gaussianprofile.append([col[1][1:-1].split(','), mut.readvalue(col[2][1:-1].split(','),cst,array=1), mut.readvalue(col[3],cst),mut.readvalue(col[4],cst),mut.readvalue(col[5],cst),mut.readvalue(col[6],cst),mut.readvalue(col[7],cst),mut.readvalue(col[8],cst)]) 
            if(info.verbose==1):
                print(gausianprofile)
        if(col[0] == 'circumstellardisk'):
            circumstellardisk.append([col[1][1:-1].split(','), mut.readvalue(col[2][1:-1].split(','),cst,array=1), mut.readvalue(col[3],cst),mut.readvalue(col[4],cst),mut.readvalue(col[5],cst),mut.readvalue(col[6],cst),mut.readvalue(col[7],cst),mut.readvalue(col[8],cst),mut.readvalue(col[9],cst)]) 
            if(info.verbose==1):
                print(circumstellardisk)
        if(col[0] == 'torus_Murakawa'):
            torus_Murakawa.append([col[1][1:-1].split(','),mut.readvalue(col[2][1:-1].split(','),cst,array=1),mut.readvalue(col[3],cst),mut.readvalue(col[4],cst),mut.readvalue(col[5],cst),mut.readvalue(col[6],cst)])
            if(info.verbose==1):
                print(torus_Murakawa)
        if(col[0] == 'torus'):
            torus.append([col[1][1:-1].split(','),mut.readvalue(col[2][1:-1].split(','),cst,array=1),mut.readvalue(col[3],cst),mut.readvalue(col[4],cst)])
            if(info.verbose==1):
                print(torus)
        if(col[0] == 'cone'):
            cone.append([col[1][1:-1].split(','),mut.readvalue(col[2][1:-1].split(','),cst,array=1),mut.readvalue(col[3],cst),mut.readvalue(col[4],cst)])
            if(info.verbose==1):
                print(cone)
        if(col[0] == 'AGN_simple'):
            #torus_const.append([col[1][1:-1].split(','),mut.readvalue(col[2],cst),mut.readvalue(col[3],cst),mut.readvalue(col[4],cst),mut.readvalue(col[5],cst)])
            AGN_simple.append([col[1][1:-1].split(','),mut.readvalue(col[2][1:-1].split(','),cst,array=1),mut.readvalue(col[3][1:-1].split(','),cst,array=1),mut.readvalue(col[4][1:-1].split(','),cst,array=1),mut.readvalue(col[5],cst),mut.readvalue(col[6],cst),mut.readvalue(col[7],cst),mut.readvalue(col[8],cst)])
            if(info.verbose==1):
                print(AGN_simple)
        if(col[0] == 'cylinder'):
            cylinder.append([col[1][1:-1].split(','),mut.readvalue(col[2][1:-1].split(','),cst,array=1),mut.readvalue(col[3],cst),mut.readvalue(col[4],cst)]) 
            if(info.verbose==1):
                print(cylinder)
        #if(col[0] == 'clump'):
        #    clump.append([col[1][1:-1].split(','),mut.readvalue(col[2],cst),mut.readvalue(col[3],cst),mut.readvalue(col[4],cst)]) 
        #    if(info.verbose==1):
        #        print(clump)
        if(col[0] == 'cloud'):
            cloud.append([col[1][1:-1].split(','),mut.readvalue(col[2][1:-1].split(','),cst,array=1),mut.readvalue(col[3],cst),mut.readvalue(col[4],cst),mut.readvalue(col[5],cst),mut.readvalue(col[6],cst),mut.readvalue(col[7],cst),mut.readvalue(col[8],cst),mut.readvalue(col[9],cst),mut.readvalue(col[10],cst)]) 
            if(info.verbose==1):
                print(cloud)
        if(col[0] == 'shell'):
            shell.append([col[1][1:-1].split(','),mut.readvalue(col[2][1:-1].split(','),cst,array=1),mut.readvalue(col[3],cst),mut.readvalue(col[4],cst)])
            if(info.verbose==1):
                print(shell)
        if(col[0] == 'fractal'):
            fractal.append([col[1][1:-1].split(','),map(float,col[2][1:-1].split(',')),int(col[3]),mut.readvalue(col[4],cst),mut.readvalue(col[5],cst),mut.readvalue(col[6],cst),mut.readvalue(col[7],cst),mut.readvalue(col[8],cst)])
            if(info.verbose==1):
                print(fractal)
        
        if col[0] == 'res_map':
            res_map = mut.readvalue(col[1],cst)
            if(info.verbose==1):
                print(res_map)
        if col[0] == 'rmax_map':
            rmax_map = mut.readvalue(col[1],cst) 
            if(info.verbose==1):
                print(rmax_map)
        if col[0] == 'af':
            af = mut.readvalue(col[1],cst) 
            if(info.verbose==1):
                print(af)
        if col[0] == 'enpaq':
            enpaq = mut.readvalue(col[1],cst)
            if(info.verbose==1):
                print(enpaq)
        if col[0] == 'emdir':
            emdir.append([col[1],mut.readvalue(col[2],cst),mut.readvalue(col[3],cst)])
        if col[0] == 'polardir':
            polardir.append([col[1],mut.readvalue(col[2],cst)])
        if col[0] == 'polar':
            polar.append([col[1],mut.readvalue(col[2],cst),mut.readvalue(col[3],cst),mut.readvalue(col[4],cst)])
        if col[0] == 'emission':
            emission.append([col[1],col[2],col[3],col[4]]) #[mut.readvalue(col[1],cst)]
            if(info.verbose==1):
                print(emission)
        if col[0] == 'source':
            #try:
            #    sources.append([col[1],col[2],mut.readvalue(col[3],cst),[[[mut.readvalue(col[4],cst)],[mut.readvalue(col[5],cst)],[mut.readvalue(col[6],cst)]],[[mut.readvalue(col[7],cst)],[mut.readvalue(col[8],cst)],[mut.readvalue(col[9],cst)]]]])
            #except:
            #    sources.append([col[1],col[2],mut.readvalue(col[3],cst),[[[0],[0],[0]],[[0],[0],[0]]]])
            sources.append([col[1],col[2],mut.readvalue(col[3],cst),col[4]])
            if(info.verbose==1):
                print(sources) 

    param_fill_map2=dict(
        densturb=densturb,
        spherepower=spherepower, 
        gaussianprofile=gaussianprofile,
        circumstellardisk=circumstellardisk,
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
    return dust_pop,param_fill_map2,[enpaq,af,res_map,rmax_map],sources,[emission,emdir,polardir,polar]


def read_paramuser():
    cst = mc.Constant()
    dust_pop = []
    sources = []
    emdir = []
    polardir = []
    polar = []
    emission = []
    name_def = []

    ### grid parameters ###    
    res_map=mut.impup('Grid resolution (m) ?',cst,expect='float')
    rmax_map=mut.impup('Grid size (m) ?',cst,expect='float')
    
    ### model parameters ### unused for model #2 ###
    adds=mut.impup('Add a source (y/n) ?',cst,expect='str')
    while(adds=='y'):
        centrobject=mut.impup('What central object is it (star, AGN...) ?',cst,expect='str')
        fichierspectre=mut.impup('What spectrum file to load ?',cst,expect='str')
        l=mut.impup('Source luminosity (W) ?',cst,expect='float')
        #emdir=list(input('Direction of emission ? ([[[0],[0],[0]],[[0],[0],[0]]] for default)'))
        #emission.append(mut.impup('Emission properties name (or "default")',cst,expect='str'))
        emission_name=mut.impup('Emission properties name (or "default")?',cst,expect='str')
        source=[centrobject,fichierspectre,l,emission_name]
        if(emission_name!='default'):
            emission.append(emission_name)
        sources.append(source)
        adds=mut.impup('Add a new source (y/n) ?',cst,expect='str')

    for i in emission:
        if(i!='default'):
            print('Defining properties of '+i)
            name_em=mut.impup('Emission direction name (or "default")',cst,expect='str')
            if(name_em not in name_def and name_em!='default'):
                theta_em=mut.impup('Emission direction angle theta (degrees) ?',cst,expect='float')
                phi_em=mut.impup('Emission direction angle phi (degrees) ?',cst,expect='float')
                emdir.append([name_em,theta_em,phi_em])
            name_em=mut.impup('Orientation of the polarisation at emission name (or "default")',cst,expect='str')
            if(name_em not in name_def and name_em!='default'):
                phi_em=mut.impup('Orientation angle of the polarisation at emission (degrees) ?',cst,expect='float')
                polardir.append([name_em,phi_em])
            name_em=mut.impup('Stokes parameters name (or "default")',cst,expect='str')
            if(name_em not in name_def and name_em!='default'):
                Q=mut.impup('Stokes parameter Q (<1) ?',cst,expect='float')
                U=mut.impup('Stokes parameter U (<1) ?',cst,expect='float')
                V=mut.impup('Stokes parameter V (<1) ?',cst,expect='float')
                polar.append([name_em,Q,U,V])
            name_def.append(name_em)

    af=mut.impup('Funnel aperture (°) ?',cst,expect='float')
    enpaq=mut.impup('Energy in each "photon" object (J) ?',cst,expect='float')
    
    ### Rsub for dust ###
    adds=mut.impup('Add a dust population (y/n) ?',cst,expect='str')
    while(adds=='y'):
        name=mut.impup('Dust population name ?',cst,expect='str')
        print('Dust types available : silicates, graphites_ortho, graphites_para, electrons, pah_neutral or pah_ionised.')
        dusttype=mut.impup('Dust type ?',cst,expect='str')
        rgmin=mut.impup('Minimal grain radius (m) ?',cst,expect='float')
        rgmax=mut.impup('Maximal grain radius (m) ?',cst,expect='float')
        alpha=mut.impup('Slope of the grain size distribution ?',cst,expect='float')
        rsub=mut.impup('Sublimation radius (m) ?',cst,expect='float')
        dust_pop.append([name,dusttype,rgmin,rgmax,alpha,rsub])
        adds=mut.impup('Add a new dust population (y/n) ?',cst,expect='str')

    param_fill_map2=dict(
        densturb=[],
        spherepower=[], 
        gaussianprofile=[],
        circumstellardisk=[],
        cloud=[],
        torus=[],
        torus_Murakawa=[],    
        cylinder=[],
        denspower=[],
        shell=[]
    )
    return dust_pop,param_fill_map2,[enpaq,af,res_map,rmax_map],sources,[emission,emdir,polardir,polar]


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
    emission = []
    emdir = []
    polardir = []
    polar = []
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
        emission_name='default' #direction of emission
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
        #emission_dir=[[[0.],[0.],[0.999]],[[0.999],[0.],[0.]]]
        emission_name='forward'
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
        #emission_dir=[[[0.],[0.],[0.999]],[[0.999],[0.],[0.]]]
        emission_name='forward'
        dust_pop=[
            ['graphites_ortho','graphites_ortho',0.005e-6,0.25e-6,-3.5,0.]
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
        #emission_dir=[[[0.],[0.],[0.999]],[[0.999],[0.],[0.]]]
        emission_name='forward'
        dust_pop=[
            ['graphites_para','graphites_para',0.005e-6,0.25e-6,-3.5,0.]
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
        #emission_dir=[[[0.],[0.],[0.999]],[[0.999],[0.],[0.]]]
        emission_name='forward'
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
        #emission_dir=[[[0.],[0.],[0.999]],[[0.999],[0.],[0.]]]
        emission_name='forward'
        dust_pop=[
            ['pah_neu','pah_neutral',0.005e-6,0.25e-6,-3.5,0.]
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
        #emission_dir=[[[0.],[0.],[0.999]],[[0.999],[0.],[0.]]]
        emission_name='forward'
        dust_pop=[
            ['silicates','silicates',0.005e-6,0.25e-6,-3.5,0.],
            ['graphites_ortho','graphites_ortho',0.005e-6,0.25e-6,-3.5,0.],
            ['graphites_para','graphites_para',0.005e-6,0.25e-6,-3.5,0.],
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
        ratio = np.divide(mass_ratio,mut.particle_mass_new(0.005e-6,0.25e-6,-3.5)[:4])
        
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
            fractal=[[ratio,N,D,H,Rin,Rout,Mfractal]]            
        )

        emission_name='default'

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
           
    sources=[[centrobject,fichierspectre,l_agn,emission_name]]

    #emission definition
    emission.append(['forward','forwardp','forwardu','default'])
    emission.append(['forwardQ','forwardp','forwardu','polarQ'])
    emdir.append(['forwardp',0.,0.])
    polardir.append(['forwardu',0.])
    polar.append(['polarQ',0.999,0.,0.])
    
    return dust_pop,param_fill_map2,[enpaq,af,res_map,rmax_map],sources,[emission,emdir,polardir,polar]
