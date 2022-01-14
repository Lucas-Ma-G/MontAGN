# -*- coding: utf-8 -*-
import numpy as np
import scipy.integrate as sci
import time
import tarfile
import os

import montagn_utils as mu
import montagn_classes as mc
import montagn_polar as mpol
import montagn_output as mout
import montagn_launch as ml
import matplotlib.pyplot as plt
#import random as rand
import mgui_random as g_rand
try:
    import patiencebar as g_pgb
except:
    print('For a better use you can install patiencebar (work with pip)')
    import mgui_progbar as g_pgb

def nver():
    nversion=3.01
    return nversion

#############    Monte Carlo tools    ####################

def funnel_propagation(photon, x,y,z, model):
    """
    Function managing the packet propagation inside the funnel

    Faster than main_propagation because it assumes empty cells
    """
    #print "funnel ",np.sqrt(x**2+y**2),z
    a = np.sin(photon.theta)*np.sin(photon.theta) - np.cos(photon.theta)*np.cos(photon.theta)*np.tan(model.AF*np.pi/180.)
    b = 2*(np.sin(photon.theta)*(x*np.cos(photon.phi)+y*np.sin(photon.phi)) - z*np.cos(photon.theta)*np.tan(model.AF*np.pi/180.))
    c = x*x+y*y-np.tan(model.AF*np.pi/180.)*z*z
    D = max(b*b - 4*a*c,0)
    t = (-b + np.sqrt(D))/(2*a)
    if t<0: #the intersection with the funnel was in the past : exiting the medium
        t = model.map.Rmax*1.01
    xinter = x + t*np.sin(photon.theta)*np.cos(photon.phi)
    yinter = y + t*np.sin(photon.theta)*np.sin(photon.phi)
    zinter = z + t*np.cos(photon.theta)
    return xinter,yinter,zinter

def main_propagation(photon, x,y,z, model,info,genseed,cst):
    """
    Main function managing the packet propagation

    Check the cell density, the intersection of the trajectory with the cell's walls
    Determine whether the packet interact within the cell or exit the cell
    Call the grain_interaction function if required
    """

    nx = photon.p[0][0]
    ny = photon.p[1][0]
    nz = photon.p[2][0]
    intersect = set()
    ### intersections with spheres ###
    r = 0 #identify in which sublimation shell the photon is (the last being the Rmax sphere : limit of medium)
    Rl = model.Rsub+[model.map.Rmax*1.01]
    while(Rl[r]*Rl[r] < x*x+y*y+z*z):
        r +=1
    #print "r :",r
    if r==0:
        b = 2*(nx*x+ny*y+nz*z)
        c = x**2+y**2+z**2-Rl[0]**2
        D = b**2-4*c
        t = (-b+np.sqrt(D))/2
        intersect.add(t)
    else:
        b = 2*(nx*x+ny*y+nz*z)
        c = x**2+y**2+z**2-Rl[r]**2
        D = b**2-4*c
        if D > 0:
            t = (-b-np.sqrt(D))/2
            if t > 0:
                intersect.add(t)
            else:
                t = (-b+np.sqrt(D))/2
                if t > 0:
                    intersect.add(t)
            #add to the set of possible intersections the soonest intersection with the outer sublimation sphere
        c = x**2+y**2+z**2-Rl[r-1]**2
        D = b**2-4*c
        if D > 0:
            t = (-b-np.sqrt(D))/2
            if t > 0:
                intersect.add(t)
            else:
                t = (-b+np.sqrt(D))/2
                if t > 0:
                    intersect.add(t)
            #add to the set of possible intersections the soonest intersection with the inner sublimation sphere

    ### intersection with the funnel ###
    a = np.sin(photon.theta)**2 - np.cos(photon.theta)**2*np.tan(model.AF*np.pi/180.)
    b = 2*(np.sin(photon.theta)*(x*np.cos(photon.phi)+y*np.sin(photon.phi)) - z*np.cos(photon.theta)*np.tan(model.AF*np.pi/180.))
    c = x**2+y**2-np.tan(model.AF*np.pi/180.)*z**2
    D = b**2 - 4*a*c
    if D > 0:
        t = (-b-np.sqrt(D))/(2*a)
        if t > 0:
            intersect.add(t)
        else:
            t = (-b+np.sqrt(D))/(2*a)
            if t > 0:
                intersect.add(t)
        #add to the set of possible intersections the soonest intersection with the funnel

    ### intersection with the cell's walls ###
    tcell = []
    if nx > 0:
        xc = np.ceil(x/model.map.res)*model.map.res
        tcell.append(1.*(xc-x)/nx)
    elif nx < 0:
        xc = np.floor(x/model.map.res)*model.map.res
        tcell.append(1.*(xc-x)/nx)
    if ny > 0:
        yc = np.ceil(y/model.map.res)*model.map.res
        tcell.append(1.*(yc-y)/ny)
    elif ny < 0:
        yc = np.floor(y/model.map.res)*model.map.res
        tcell.append(1.*(yc-y)/ny)
    if nz > 0:
        zc = np.ceil(z/model.map.res)*model.map.res
        tcell.append(1.*(zc-z)/nz)
    elif nz < 0:
        zc = np.floor(z/model.map.res)*model.map.res
        tcell.append(1.*(zc-z)/nz)
    # coordinates of cell walls and intersection time depending on direction of propagation
    t = min(tcell)
    #tcell is non-empty (at least one coordinate of the direction vector is not nil)
    intersect.add(t)
    intersect_min = min(intersect)

    ### random interactions with different types of dust ###
    #interact = {model.map.Rmax*1.01:'out'}
    Ktab = {}
    Ktot=0.0
    for i in range(len(model.map.dust)):
        if('Rsub' in info.debug):
            print('nRsub :',r,', ngrain :',i)
        if (r > i): #be further than the sublimation radius of a given dust type
            j=model.map.dust[i].number
            rho=model.map.get(x, y, z).rho[j]
            if('dust' in info.debug):
                print('graintype :',j,', density :',rho)
            if(rho!=0.0):
                #if('pah' in model.map.dust[i].typeg):
                #    K=mu.KPAH(model.map.dust[i],photon.wvl)
                #else:
                K=mu.Qaapi(model.map.dust[i],photon.wvl,cst)*rho
                #print('K :',K)
            else:
                K=0.0
            Ktot+=K
            Ktab[K] = model.map.dust[i]
            #print "K : ", K,loginterp(polar.x_M,polar.Qext,2*np.pi*np.sqrt(model.map.dust[i].av_sec/np.pi)/photon.wvl),model.map.get(x, y, z).rho[i]
            #print "av_sec : ",model.map.dust[i].av_sec,np.sqrt(model.map.dust[i].av_sec)/np.pi
            #print "Qext - Cs : ",loginterp(polar.x_M,polar.Qext,2*np.pi*photon.wvl/(np.sqrt(model.map.dust[i].av_sec)/np.pi)),model.map.dust[i].Cs(photon.wvl)
            #print "wvl : ",photon.wvl
            #print "interpol : ",polar.x_M,polar.Qext,2*np.pi*photon.wvl/(np.sqrt(model.map.dust[i].av_sec)/np.pi)
    #print('Ktot: ',Ktot)
    if(Ktot==0):
        interact_min = model.map.Rmax*2
    else:
        interact_min = photon.tau/Ktot

    ### what happens first ? ###
    if intersect_min < interact_min:
        #no interaction had the time to occur before reaching another region of space
        t = intersect_min
        photon.tau -= t*Ktot
        if t<model.map.res*1e-6:
            #avoid the risk of being exactly on the limit between regions
            t=model.map.res*1e-6
        X = x + nx*t
        Y = y + ny*t
        Z = z + nz*t
    else:
        #interaction occurs with the first encountered grain
        if('tau' in info.debug):
            print('old tau :',photon.tau)
        t = interact_min
        X = genseed.uniform()
        photon.tau = -np.log(X)
        X = x + nx*t
        Y = y + ny*t
        Z = z + nz*t

        lK=len(Ktab.keys())
        a = genseed.uniform()
        i=-1
        while(a>0):
            i+=1
            #print a, np.sort(Ktab.keys())[lK-1-i]/Ktot
            a -= np.sort(list(Ktab.keys()))[lK-1-i]/Ktot
        grain = Ktab[np.sort(list(Ktab.keys()))[lK-1-i]]
        #print "Interaction with ",grain.name

        photon.interact = photon.interact + 1
        photon.x_inter = X
        photon.y_inter = Y
        photon.z_inter = Z
        grain_interaction(photon, X, Y, Z, model, grain,info,genseed,cst) #,polar
        #print 'diff',X,Y,Z
    #print "t,nx,X,x : ",t,nx,X,x #Y,Z
    return X,Y,Z


def grain_interaction(photon, x,y,z, model, grain,info,genseed,cst):
    wvl = photon.wvl
    #Determination of the grain size
    if(grain.typeg=='electrons'):
        r = cst.r_el
    else:
        r = grain.size(genseed)
    #finding the ir indices of the given radius in the radii list
    #ir=np.argsort(np.argsort(np.append(grain.sizevect,r)))[len(grain.sizevect)]-1
    ir=mpol.get_indice(grain.sizevect,r)

    ### Interpolation of albedo on radius :
    # lin
    alb=(grain.albedo[:,ir]+grain.albedo[:,ir+1])*0.5 #lin

    #log
    #alb=[]
    #for i in range(len(grain.albedo[:,ir])):
    #    alb.append(mpol.loginterp(grain.sizevect,grain.albedo[i],r)) #log

    alb=mpol.loginterp(grain.wvlvect,alb,wvl) #log
    if('albedo' in info.debug):
        print('albedo :',alb)
    if(model.usethermal==0):
        scattering(r,ir,alb,photon, x,y,z, model, grain,info,genseed,cst)
    else:
        xx = genseed.uniform()
        if xx < alb: #grain.albedo(wvl): #diffusion
            #print "scattering"
            scattering(r,ir,alb,photon, x,y,z, model, grain,info,genseed,cst)
        else: 
            thermalisation(r,ir,alb,photon, x,y,z, model, grain,info,genseed,cst)
    return

def scattering(r,ir,alb,photon, x,y,z, model, grain,info,genseed,cst):
    '''
    Function to manage scattering events
    ''' 
    alpha, beta = mpol.diffpolar(r,photon.wvl,photon,grain,genseed)
    if(info.plotinfo!=[]): # and grain.typeg!=3):
        #iwvl=np.argsort(np.argsort(np.append(grain.wvlvect,wvl)))[len(grain.wvlvect)]-1
        #iwvl=mpol.get_indice(grain.wvlvect,wvl)
        info.alpha[int(round(alpha/np.pi*(len(grain.phaseNorm[:,0,0])-1))),model.map.dust.index(grain)]+=1
        info.beta[int(round(beta/np.pi*0.5*(len(grain.phaseNorm[:,0,0])-1))),min(photon.interact-1,4),model.map.dust.index(grain)]+=1
        info.xstat[ir,model.map.dust.index(grain)]+=1  #,iwvl
        info.pstat[int(round(alpha/np.pi*(len(grain.phaseNorm[:,0,0])-1))),min(photon.interact-1,4),0,model.map.dust.index(grain)]+=np.sqrt(photon.Ss[2][0]*photon.Ss[2][0]+photon.Ss[1][0]*photon.Ss[1][0])
        info.pstat[int(round(alpha/np.pi*(len(grain.phaseNorm[:,0,0])-1))),min(photon.interact-1,4),1,model.map.dust.index(grain)]+=1
        if('albedo' in info.debug):
            print(alb,-np.log10(alb)/(np.log10(len(grain.sizevect))+1.0),min(-np.log10(alb)/3.,0.999),int(np.floor(min(-np.log10(alb)/3.,0.999)*len(grain.sizevect))))
        info.albedo[int(np.floor(min(-np.log10(alb)/(np.log10(len(grain.sizevect))+1.0),0.999)*len(grain.sizevect))),model.map.dust.index(grain)]+=1



    #print (2*np.pi*r/wvl),np.where(polar.x_M>(2*np.pi*r/wvl))[0][np.argmin(polar.x_M[np.where(polar.x_M>2*np.pi*r/wvl)])]
    #print int(np.lround(alpha/np.pi*len(polar.phase))), alpha
    
    p=photon.p
    u=photon.u
    
    #new rotation vector u
    u0=np.copy(u)
    p0=np.copy(p)
    
    u[0]=[u0[0][0]*np.cos(beta)+(p0[1][0]*u0[2][0]-p0[2][0]*u0[1][0])*np.sin(beta)]
    u[1]=[u0[1][0]*np.cos(beta)+(p0[2][0]*u0[0][0]-p0[0][0]*u0[2][0])*np.sin(beta)]
    u[2]=[u0[2][0]*np.cos(beta)+(p0[0][0]*u0[1][0]-p0[1][0]*u0[0][0])*np.sin(beta)]
    
    u=u/(np.sqrt(u[0][0]*u[0][0]+u[1][0]*u[1][0]+u[2][0]*u[2][0]))             #Renormalisation
    
    #New vector p
    p[0]=[p0[0][0]*np.cos(alpha)+(u[1][0]*p0[2][0]-u[2][0]*p0[1][0])*np.sin(alpha)]
    p[1]=[p0[1][0]*np.cos(alpha)+(u[2][0]*p0[0][0]-u[0][0]*p0[2][0])*np.sin(alpha)]
    p[2]=[p0[2][0]*np.cos(alpha)+(u[0][0]*p0[1][0]-u[1][0]*p0[0][0])*np.sin(alpha)]
    
    p=p/(np.sqrt(p[0][0]*p[0][0]+p[1][0]*p[1][0]+p[2][0]*p[2][0]))
    
    photon.p=p
    photon.u=u
    if(model.usethermal==0):
        photon.E*=alb
        if(photon.interact>model.nscattmax):
            photon.wvl=1.0
            photon.write=False
    return

def thermalisation(r,ir,alb,photon, x,y,z, model, grain,info,genseed,cst):
    '''
    Function to manage absorption, thermalisation and reemission events
    ''' 
    T_old = model.map.get(x,y,z).T
    if('pah' in grain.typeg):
        #print 'Work in Progress'
        N_pahold = model.map.get(x,y,z).Npah #number of absorbed ph in the cell (+1 for the current one)
        N_pah = N_pahold+1
        #N_phtot=int(model.sources.total_lum*model.Dt/model.energy)
        N_ph_em=model.N_ph_em
        model.map.get(x,y,z).Npah += 1 #updated absorbed photons
        surf=emit_surf_pah(x,y,z,model)
        
        #Two ways of computing the isrf, one with the emitted photons at the time of the
        #simulation (approximation) - one with the total number of photons that will be emitted
        #(better but need a correction, somewhat not trivial)
        #uisrf=float(N_pah)/float(N_phtot)*model.sources.total_lum/surf/model.isrf #total photons
        uisrf=float(N_pah)/float(N_ph_em)*model.sources.total_lum/surf/model.isrf #only emitted ones
        
        #uisrfold=float(N_pahold)/float(N_phtot)*model.sources.total_lum/surf/model.isrf
        #print N_pah,N_phtot,model.sources.total_lum,surf,model.isrf
        #print uisrf
        #wvlold,spcold=mu.jnu_pah_emission(uisrfold,grain.jnupah)
        wvl,spc=mu.jnu_pah_emission(uisrf,grain.jnupah)
        #spc-=spcold #Correction of precedent emissions for the total photons case
        spc=np.array(spc)
        spc[np.where(spc<0)]=0
    else:
        if('T' in info.debug):
            T_new = temperature_update(photon,x,y,z,model,grain,ir,verbose=1)
        else:
            T_new = temperature_update(photon,x,y,z,model,grain,ir,verbose=info.verbose)
        info.fT.write("%f\t%f\t%f\n"%(np.sqrt(x*x+y*y)/cst.pc,z/cst.pc,T_new))
        wbb = cst.b/T_new #black body peak wavelength
        wvl = np.linspace(0.175,75,1000)*wbb #wvl of the black body peak at 10-6 relative flux
        Qabs=(grain.Qabs[:,ir]+grain.Qabs[:,ir+1])*0.5 #lin
        Qabs=mpol.loginterp(grain.wvlvect,Qabs,wvl) #log
        if(T_new!=T_old):
            spc = Qabs*(mu.BB(T_new,wvl)-mu.BB(T_old,wvl)) #emission spectrum : BW01 eq 7
        else:
            info.nwarning['Tmax'].n+=1
            spc = Qabs*mu.BB(T_new,wvl)
    spc[0] = 0
    spc[-1] = 0
    spectrum,wvl_proba = mu.spectrum_factory(wvl,spc)
    #photon.wvl = wvl_proba(T_new)
    wvl = wvl_proba(genseed.uniform())
    wvl,theta,phi,p,u,x,y,z,tau,Ss_part = model.sources.list_sources[0].emission(genseed,pos=[x,y,z],wvl=wvl,tau=photon.tau)
    photon.wvl = wvl
    photon.theta = theta
    photon.phi = phi
    photon.p = p
    photon.u = u
    #photon.tau = tau #unnecessary since ph.tau has already been upgraded
    photon.label = grain.name
    #print "new wvl : ",photon.wvl," from T : ",T_new,' K'
    photon.Ss=[[1.0],[Ss_part[0]],[Ss_part[1]],[Ss_part[2]]]
    #photon.Ss=[[1.0],[0.0],[0.0],[0.0]]
    photon.reem = photon.reem + 1
    return

def emit_surf_pah(x,y,z,model):
    #print 'Work in Progress, not working yet'
    res = model.map.res
    xc = np.floor(x/res)*res
    yc = np.floor(y/res)*res
    zc = np.floor(z/res)*res
    #V = [] #volume of the cell filled with dust, by type of grain.
    V = np.zeros(len(model.map.dust))
    for i in range(len(model.map.dust)):
        if('pah' in model.map.dust[i].typeg):
            #V.append(res**3 - mu.volume_intersect(xc,yc,zc,res,model.AF,model.Rsub[i]))
            V[i] = res**3 - mu.volume_intersect(xc,yc,zc,res,model.AF,model.Rsub[i])
    #em_surf = [] #total emitting surface by type of grain
    em_surf = 0 #total emitting surface by type of grain
    for i in range(len(model.map.dust)):
        if('pah' in model.map.dust[i].typeg):
            j = model.map.dust[i].number
            em_surf+=V[i]*model.map.get(x,y,z).rho[j]*model.map.dust[i].av_sec*4
    return em_surf

def temperature_update(ph,x,y,z,model,grain,ir,verbose=0): #,iwvl): #,polar):
    res = model.map.res
    xc = np.floor(x/res)*res
    yc = np.floor(y/res)*res
    zc = np.floor(z/res)*res
    #lower angle of the cell (for volume_intersect)
    T_old = model.map.get(x,y,z).T
    N_ph = model.map.get(x,y,z).N+1 #number of absorbed ph in the cell (+1 for the current one)
    if(verbose==1):
        print('N_ph : ',N_ph)
    N_phtot=int(model.sources.total_lum*model.Dt/model.energy)
    #print model.map.N
    V = [] #volume of the cell filled with dust, by type of grain.
    for i in range(len(model.map.dust)):
        V.append(res**3 - mu.volume_intersect(xc,yc,zc,res,model.AF,model.Rsub[i]))
    if(model.Temp_mode==2):
        em_surf = [] #total emitting surface by type of grain
        for i in range(len(model.map.dust)):
            j = model.map.dust[i].typeg
            em_surf.append(V[i]*model.map.get(x,y,z).rho[j]*model.map.dust[i].av_sec*4)
        if(verbose==1):
            print('em_surf : ',em_surf)
    def Tup(T):
        em_surf_Kp = 0
        for i in range(len(model.map.dust)):
            j = model.map.dust[i].number
            if(model.map.dust[i].typeg!='electrons'):
                #em_surf_Kp += kcompute(T,polar.x_M[:,j],polar.Qabs[:,j],np.sqrt(model.map.dust[i].av_sec/np.pi))*em_surf[i]
                Q=(grain.Qabs[:,ir]+grain.Qabs[:,ir+1])*0.5
                #Q=loginterp(grain.wvlvect,Qabs,wvl)
                x=(grain.x_M[:,ir]+grain.x_M[:,ir+1])*0.5
                em_surf_Kp += mu.kcompute(T,x,Q,np.sqrt(model.map.dust[i].av_sec/np.pi))*em_surf[i]
            #em_surf_Kp += model.map.dust[i].K(T)*em_surf[i]
            #print N_ph,model.energy
        return np.abs(4*sigma*T*T*T*T*model.Dt*em_surf_Kp - N_ph*model.energy)#BW01 eq 6 (radiative equilibrium)
        #return np.abs(4*sigma*T**4*model.Dt*em_surf_Kp - N_ph*model.energy)#BW01 eq 6 (radiative equilibrium)
    def Tup2(T):
        em_surf_Kp = 0
        for i in range(len(model.map.dust)):
            em_surf_Kp += model.map.dust[i].K(T)*em_surf[i]
        return np.abs(4*sigma*T**4*model.Dt*em_surf_Kp - N_ph*model.energy)#BW01 eq 6 (radiative equilibrium)
    def Tup_Dan(T):
        Eemit_tot = 0.
        #Tdust =  model.map.dust[0].tempRange
        for i in range(len(model.map.dust)):
            Tdust =  model.map.dust[i].tempRange
            j = model.map.dust[i].number
            if(model.map.dust[i].typeg!='electrons'):
                Eemit_tot += np.array(model.map.dust[i].EemitT)*V[i]*model.map.get(x,y,z).rho[j]
        #print('Eemit_tot :',Eemit_tot)
        #print('Vol: ',V[i], 'rho:', model.map.get(x,y,z).rho[j])
        #print('N_ph*model.energy :', N_ph*model.energy)

        #print np.shape(model.map.dust[i].EemitT), np.shape(model.map.dust[0].tempRange)
        #print np.shape(Eemit_tot), np.shape(Tdust)

        Tinterp = np.interp(np.log(N_ph*model.energy), np.log(Eemit_tot), Tdust) #BW01 eq 6 (radiative equilibrium)
        #Tinterp = loginterp(Eemit_tot, Tdust,N_ph*model.energy) #BW01 eq 6 (radiative equilibrium)
        #print Tinterp,Tinterp2
        # interpolation of temperature where absorbed energy = Sigma_type grains [total(ng pi < a2 Qabs > B(T))] 
        #print('Tinterp: ',Tinterp) #BW01 eq 6 (radiative equilibrium)
        return Tinterp #BW01 eq 6 (radiative equilibrium)
    def testTemp():
        Tvect=np.linspace(1,1000,1000)
        Tres=np.zeros(1000)
        Tres2=np.zeros(1000)
        kres=np.zeros(1000)
        kres2=np.zeros(1000)
        kpres=np.zeros(1000)
        kpres2=np.zeros(1000)
        for i in range(1000):
            Tres[i]=Tup(Tvect[i])
            Tres2[i]=Tup2(Tvect[i])
            kres[i]=kcompute(Tvect[i],polar.x_M,polar.Qabs,np.sqrt(model.map.dust[0].av_sec/np.pi))
            kres2[i]=model.map.dust[0].K(Tvect[i])
            kpres[i]=kres[i]/(sigma*Tvect[i]**4/np.pi)
            kpres2[i]=kres2[i]/(sigma*Tvect[i]**4/np.pi)
        plt.figure()
        plt.plot(Tvect,kres) 
        plt.plot(Tvect,kres2) 
        plt.title('k value depending on T(K)')
        plt.show()   
        plt.figure()
        plt.plot(Tvect,Tres)
        plt.plot(Tvect,Tres2) 
        plt.title('Tout (K) depending on Tin (K)')
        plt.show()    
        plt.figure()
        plt.plot(Tvect,kpres)
        plt.plot(Tvect,kpres2) 
        plt.title('kp value depending on T(K)')
        plt.show() 
        
    #testTemp()
    if(model.map.dust[0].typeg!='electrons'):
        T_max = model.map.dust[0].Tsub*1.1
    else:
        T_max = model.map.dust[1].Tsub*1.1
    if(model.Temp_mode==2):
        T_new = temp_solver(T_old*.9, T_max, Tup) #starts at lower T to allow cooling if some overheating (T>Tsub) occured earlier
    else:
        T_new = Tup_Dan(T_old)
    if(verbose==1):
        print("Told et Tnew : ",T_old, T_new)
    for i in range(len(model.map.dust)):
        #if T_new > model.map.dust[i].Tsub and np.sqrt(x**2+y**2+z**2) > model.Rsub[i]:
        if T_new > model.map.dust[i].Tsub and x*x+y*y+z*z > model.Rsub[i]*model.Rsub[i]:
            #increase the sublimation radius if the reached temperature is too high
            f = .5 #fraction of (r-Rsub) by which the sublimation radius is corrected. 1 : jump to r, 0 : no correction.
            #smaller f avoids overshooting but increases convergence time.
            rsub = model.Rsub[i]
            #model.Rsub[i] = rsub + (np.sqrt(x**2+y**2+z**2)-rsub)*f #updated Rsub
            model.Rsub[i] = rsub + (np.sqrt(x*x+y*y+z*z)-rsub)*f #updated Rsub
            ph.label += '.RSUB' #Rsub has converged (or overshot) if there are no more such flags for the exitting photons
            #print "T update : x,y,z:",x,y,z
            if(verbose==1):
                print('Rsub update with new T and Tsub:',T_new, model.map.dust[i].Tsub)
            model.nRsub_update+=1
    model.map.get(x,y,z).T = T_new #updated temperature
    #model.map.get(x,y,z).N = N_ph+1 #updated absorbed photons
    model.map.get(x,y,z).N += 1 #updated absorbed photons
    #print "x,y,z :",x,y,z
    #print "T :",model.map.get(x,y,z).T
    return T_new


###########    MONTE CARLO LOOP    ##############

def MC_loop(ph,x,y,z,model,info,genseed,cst):
    out = False
    while out == False:
        if x*x+y*y+z*z>=model.map.Rmax*model.map.Rmax: #exiting the medium
            out = True
        else:
            nx = ph.p[0][0]
            ny = ph.p[1][0]
            nz = ph.p[2][0]
            X = x + nx*model.map.res*1e-6
            Y = y + ny*model.map.res*1e-6
            Z = z + nz*model.map.res*1e-6
            #avoid the risk of being exactly on the limit between regions
            if np.sqrt(X*X+Y*Y) < abs(Z)*np.tan(model.AF*np.pi/180.):
                x,y,z = funnel_propagation(ph,X,Y,Z,model)
            else:
                x,y,z = main_propagation(ph,X,Y,Z,model,info,genseed,cst)
            if ph.wvl > info.wvl_max:
                out = True
                #assume the medium is transparent in the far IR
    return ph,x,y,z

def run_simulation(model, info = [], add = 0, nphot = [], Dt = [], filename = [], nang = 999, nsize = 100, force_wvl = [], wvl_max = 1e5, nsimu = 1, plotinfo = [], display = 1, ret = 0, cluster = 0, cst = [], path = ""):

    #Initialization of the seed for multithreading
    genseed=g_rand.gen_generator()
    if(cst==[]):
        cst=mc.Constant()

    if(info==[]):
        info=mc.Info(0,add,force_wvl,wvl_max,plotinfo,display,0,nsimu,nang,nsize,cluster)
    if(nphot==[] and Dt==[]):
        nphot = input('Number of photons packets to launch ? ')
        Dt = nphot*model.energy/model.sources.total_lum
        #Dt = input('Duration of simulation (in s) ? ')
    elif(nphot==[]):
        nphot = Dt/model.energy*model.sources.total_lum
    elif(Dt==[]):
        Dt = nphot*model.energy/model.sources.total_lum
    else:
        if(nphot!=Dt/model.energy*model.sources.total_lum):
            print('WARNING : Dt and nphot non compatible, keeping nphot value')
            #model.energy=Dt*model.sources.total_lum/nphot
            Dt = nphot*model.energy/model.sources.total_lum
    N = nphot
    Dts=Dt/nphot

    if(info.nsimu==1 or cluster==1):
        mout.plot_time(Dt,"The simulation will be equivalent to a source radiating during",hmins=1)
    if(model.usethermal==1):
        photsec=1. #phot/sec
    else:
        photsec=7. #phot/sec
    if(info.nsimu==1 or cluster==1):
        ndust=0
        for i in range(len(model.map.dust)-1):
            if(i>=3):
                i+=1
            ndust+=model.map.dust[i].usegrain
        time_prev=nphot/photsec+90*ndust #90 sec for initialisation
        mout.plot_time(time_prev,"Duration of simulation will be about",hmins=1)
    model.Dt = Dt
    if(info.nsimu==1 or cluster==1):
        print('%d photon packets will be emitted.'%N)
    if(model.usethermal==1):
        print("**** Be careful ! In order for thermal mode to work, one should use a high enough number of packets ! ****")

    if(cluster==1):
        outpath=path
        inpath='Input/'
    else:
        outpath=path+'Output/'
        inpath=path+'Input/'

    if(filename==[]):
        filename = raw_input('Output file name ? ')
    thetarec=[0]
    j=0
    dthetaobs=5
    while(thetarec[j]+dthetaobs<180):
        thetarec.append(thetarec[j]+2*dthetaobs)
        j+=1
    model.thetarec=thetarec
    info.Dt_tot=np.ones(len(model.thetarec))*Dt

    #Opening output files
    mout.openoutfile(outpath,filename,model,info)
 
    if(info.nsimu==1 or cluster==1):
        print('Computing Mueller matrices') #'Calcul des matrices de Mueller'
    #nang=999 #Résolution angulaire - Must be an odd number !
    #nforme=100 #résolution en parametre de forme

    dust_tmp=np.copy(model.map.dust)
    Rsub_tmp=np.copy(model.Rsub)
    model.map.dust=[]
    model.Rsub=[]

    i=0
    if(cluster==0 and info.nsimu==1):
        display_initphase=1
    else:
        display_initphase=0
    for grain in dust_tmp:
        if(True): #if(grain.usegrain==1):
            if(info.nsimu==1 or cluster==1):
                #poussiere=['silicates','graphites ortho','graphites para','electrons']
                print('      for '+str(grain))
            phase=mpol.fill_mueller(nang,nsize,genseed,grain,info.plotinfo,info.force_wvl,inpath)
            if(grain.typeg!='electrons'):
                phaseNorm,g,w=mpol.init_phase(nang,nsize,phase,genseed,grain,info.plotinfo,display=display_initphase)
                if(model.usethermal==1 and model.Temp_mode==1):
                    a = grain.sizevect       
                    k =  4.*np.pi / sci.simps(np.power(a,-3.5), a) # coeff normalisation MRN 
                    #print('coe : ' , k) 
                    Qabs_a2_MRN_i = []     
                    for l in range(len(grain.wvlvect)): # parcours le domaine de lambda 
                        Q = grain.Qabs[l,:]
                        Qabs_a2_MRN_i.append(mu.Qabs_a2_MRN_compute(a,Q) * k) 
                    grain.Qabs_a2_MRN = Qabs_a2_MRN_i # intégrale Qabs(lambda) 4 pi * a^2 a^-3.5 K da
                    if(info.verbose==1):
                        print('Qabs_a2_MRN :',grain.Qabs_a2_MRN)
                    EemitTi = []
                    Tdusti = []
                    for l in range(100): # parcours le domaine de température de 3. à Tsublim 
                        Tdust =  3. + (grain.Tsub - 3.)*l/100. 
                        Tdusti.append(Tdust)
                        Z = grain.Qabs_a2_MRN
                        wvl = grain.wvlvect
                        #EemitTi.append(Qabs_a2_MRN_B_compute(wvl, Z, Tdust)) # function of emission vs T proper to each type of grain 
                        EemitTi.append(mu.Qabs_a2_MRN_B_compute(wvl, Z, Tdust)*4.*np.pi*model.Dt) # function of emission vs T proper to each type of grain 
                    grain.EemitT = EemitTi
                    grain.tempRange = Tdusti
                    if(info.verbose==1):
                        print('grain.EemitT:',  grain.EemitT)
                        print('grain.tempRange :', grain.tempRange)
            else:
                phaseNorm,g,w=mpol.init_phase_electrons(nang,phase,genseed,grain,info.plotinfo)
            grain.phaseNorm = phaseNorm
            grain.w = w
            grain.g = g
            grain.Rsub_init = Rsub_tmp[i]
            model.map.dust.append(grain)
        i+=1
    dust_tmp=[]

    #sorting grains
    if('dust' in info.debug):
        print(mu.sort_grains(model.map.dust))
    model.map.dust=mu.sort_grains(model.map.dust)[0]
    for grain in model.map.dust:
        if('dust' in info.debug or 'Rsub' in info.debug):
            print(grain.Rsub_init)
        model.Rsub.append(grain.Rsub_init)
        if('dust' in info.debug or 'Rsub' in info.debug):
            print(model.Rsub)

    #preparing the recording of phase functions if required
    if(info.plotinfo!=[]):
        ndust=len(model.map.dust)
        info.alpha = np.zeros([nang,ndust])
        info.beta = np.zeros([nang,5,ndust])
        info.xstat = np.zeros([nsize,ndust])
        info.pstat = np.zeros([nang,5,2,ndust])
        info.albedo = np.zeros([nsize,ndust])

    start_phot=time.time()
    if(info.nsimu==1 or cluster==1):
        print("*********************************************")
        print("Launching simulation")
        print("*********************************************")

    nabs=0
    naffiche=1
    print("Run %d started"%(info.nsimu))
    #tau1,tau2=model.info("x","y",plot="T")
    if(cluster==0 and info.nsimu==1):
        if(N<100):
            progressb=g_pgb.Patiencebar(valmax=N,up_every=1)
        else:
            progressb=g_pgb.Patiencebar(valmax=N-1,up_every=1)
    for i in range(N):
        if(cluster==1):
            if (i+1)%(1000*naffiche)==0:
                print("photon %d"%(i+1))
                naffiche=naffiche*10
        elif(info.nsimu==1):
            if ((i+1)<=N/100):
                if ((i+1)<N/100):
                    if (i+1)%(1000*naffiche)==0:
                        print("photon %d"%(i+1))
                        naffiche=naffiche*10
                else:
                    progressb.update((N-1)/100)
            else:
                progressb.update()
                #naffiche=naffiche*10

        #ph,x,y,z,p,u = model.sources.emission_photon(genseed) #old, now p and u are integrated
        ph,x,y,z = model.sources.emission_photon(genseed,enpaq=model.energy)
        #ph.Ss=[[1.0],[0.0],[0.0],[0.0]] #done in the emission_photon routine
        if(info.force_wvl!=[]):
            ph.wvl=info.force_wvl
        #ph.p=p
        #ph.u=u
        #ph.E=model.energy
        #X = genseed.uniform()
        #ph.tau = -np.log(X)

        # MAIN LOOP
        model.N_ph_em += 1
        ph,x,y,z = MC_loop(ph,x,y,z,model,info,genseed,cst)

        # OUT
        theta=np.arccos(ph.p[2][0])
        phi=np.arctan2(ph.p[1][0],ph.p[0][0])
        #betastar=np.arctan2(float(u[2][0]),float((u[1][0]*p[0][0]-u[0][0]*p[1][0]))) #old --> still correct ?
        betastar=np.arctan2(float(ph.u[2][0]),float((ph.u[1][0]*ph.p[0][0]-ph.u[0][0]*ph.p[1][0])))
        phiqu=betastar
        ph.theta=theta
        ph.phi=phi
        ph.phiqu=phiqu
        x=np.copy(ph.x_inter)
        y=np.copy(ph.y_inter)
        z=np.copy(ph.z_inter)
        if ph.reem>0.5:
            nabs=nabs+1
        

        #recording
        for j in range(len(model.thetarec)):
            if(theta>np.pi/180*(model.thetarec[j]-model.dthetaobs) and theta<np.pi/180*(model.thetarec[j]+model.dthetaobs) and ph.write==True):
                mout.write_photon(info.ftheta[j],ph,x,y,z,i+1,Dts,info.Dt_tot[j],cst)
                info.Dt_tot[j]=0.0

    #End of Simulation
    for j in range(len(model.thetarec)):
        info.ftheta[j].close()
    if(model.usethermal==1):
        info.fT.close()
    
    if(info.force_wvl==[]):
        wvl_tau=2.2*1e-6
    else:
        wvl_tau=info.force_wvl
    tau_max,tau,tau_species=mout.compute_tau(model,wvl_tau,cst=cst)
    tauV_max,tauV,tau_speciesV=mout.compute_tau(model,0.5*1e-6,cst=cst)
    if(info.nsimu==1 or cluster==1):
        #print "Penser a verifier r !"
        print(" ")
        if(model.usethermal==1):
            print("Thermalisation was used")
            print("Absorption of",100*nabs/N,"% of photons")
        else:
            print("No absorption/emission was used")
        print("Total effective tau in V :",tauV_max,"and tau RSublimation in V :",tauV)
        
        band=str(wvl_tau)
        print("Total effective tau max at "+band+" m :",tau_max, "and tau RSublimation at "+band+" m :",tau)
        if(info.plotinfo!=[]):
            mout.displaystats(info,model,tau_speciesV,tau_species,band)
 
    print_warning(info)
    if(ret==1):
        return model,start_phot


def print_warning(info):
    """Print the encountered warnings"""
    nwar=0
    for i in info.nwarning:
        if(info.nwarning[i].n>0):
            if(nwar==0):
                print("*********************************************")
                print(" ")
                nwar+=1
            if(info.nwarning[i].n>1):
                fois=" times"
            else:
                fois=" time"
            print(info.nwarning[i].warningtxt)
            print("----- encountered "+str(info.nwarning[i].n)+fois+" -----")
    if(nwar>0):
        print(" ")
        print("*********************************************")


def montAGN(usethermal = 1, nphot = [], path='', filename = 'test', paramfile = [], usemodel = [], ask = 1, add = 0, model = [], nscattmax = 10, force_wvl = [], wvl_max = 1e5, nang = 999 , nsize = 100 , nsimu = 1,  display = 1, cluster = 0, cmap = [], vectormode = 0, plotinfo = [], debug = [], verbose = 0, unit=''):
    """
    montAGN(usethermal = 1, nphot = [], path='', filename = 'test', paramfile = [], usemodel = 0., ask = 1, add = 0, model = [], nscattmax = 50, force_wvl = [], wvl_max = 1e5, nang = 999 , nsize = 100 , nsimu = 1,  display = 1, cluster = 0, cmap = [], vectormode = 0, plotinfo = [], debug = [], verbose = 0, unit='')

    Launch a radiative transfer simulation
    if a mandatory parameter is not given, it will be asked
    
    MAIN INPUTS:
    [add          = add the output to an existing file ([0] or 1)]
    [ask          = enabled or disabled interactive mode (0 or [1])]
    [cluster      = launching or not on a cluster (disable some displays [0] or 1)]
    [filename     = prefixe of output file (string or set of strings ['test'])]
    [model        = already existing model (model object [])
    [nphot        = number of photons packets to launch (any integer [])]
    [paramfile    = name of the parameter file to be load (string or set of strings [], ask need to be set to 0)]
    [path         = relative path where to locate Input/paramfile and to record output files (string or set of strings [''])]
    [usemodel     = use a predefined model, in that case, ask should be set to 0 and no paramfile should be given ([], -1, -2, -3, -4, -5, -6, 1)]
    [usethermal   = use the absorption/thermalisation/reemission process (0 or [1])]

    MODEL TUNING KEYWORDS:
    [force_wvl    = force the emitted photons to be at this wavelentgh (in m) (positive float [])]
    [nang         = angular resolution - Must be an odd number ! (integer [999])]
    [nscattmax    = number of scattering limit (integer [10])
    [nsize        = size parameter resolution on phases functions (integer [100])]
    [wvl_max      = consider that photons above the indicated wavlength (in m) propagate without interaction (positive float [1e5])]

    DISPLAY KEYWORDS:
    [cmap         = colormap to be used (string or set of string [])]
    [display      = enabled or disabled display - for use on a cluster or without matplotlib (0 or [1])]
    [unit         = distance unit to be used (string or set of strings, 'AU' or 'pc'). If not specified, pc will be used.]

    ADVANCED KEYWORDS:
    [debug        = the list of important steps to be printed (among 'T', 'Rsub', 'tau', 'scat', 'polar', 'dust', [] by default)]
    [nsimu        = simulation number - setting other than 1 will disabled some display (integer [1])]
    [plotinfo     = plot graphs of phase function, albedo, x... (list of integers [[]])]
    [vectormode   = plot a map with polarization vectors ([0] or 1)]
    [verbose      = gives or not further informations on the computations (1 or [0])]

    OUTPUTS:
    -gives as an output a montagn model object, containing the main parameters of the simulation (cf montagn_class.py)
    -some *xxx_phot.dat file containing (one for each viewing angles xxx from 0 to 180 degrees):
    	  - photon launching number			[]
	  - theta - altitude angle			[rad]
	  - phi - azimutal angle			[rad]
	  - Stokes Q					C[0,1]
	  - Stokes U					C[0,1]
	  - Stokes V    				C[0,1]
	  - ref angle of the scattering plan    	[rad]
	  - position x of last diffusion		[pc]
	  - position y of last diffusion		[pc]
	  - position z of last diffusion   		[pc]
	  - number of interactions    			[]
	  - number of reemissions    			[]
	  - wavelentgh    				[m]
	  - source name   				['']
	  - paquets energy    				[J]
	  - Emission time of the packet    		[s]
	  - Total emission time				[s]
    -a file *_T_update.dat listing all the temperature updates that occured during the simulation
    -some files *_density_xy.dat giving the density map of the dust grains in the xy plan
    -some files *_density_xz.dat giving the density map of the dust grains in the xz plan

    (for a complete description, please consider reading the MontAGN manual)


    EXAMPLES:

    $ipython
    >import montagn
    >mymodel=montagn.MontAGN(nphot=1000,paramfile='test.txt',usethermal=0)

    will run a simulation with 1000 photon packets (ie. rather short, ideal for tests), without dust thermalisation and with importing model from 'test.txt' parameter file, assumed to be located in the current directory.

    Here is a very simple example of parameter file, please check the MontAGN user manual for more information about model building and parameter files.
    
    PARAMETER FILE:
    dust silicates silicates 0.005e-6 0.25e-6 -3.5 0.05AU
    spherepower [silicates] [5800.0] 0 1.0AU
    res_map 1AU
    rmax_map 25AU
    source AGN spectre_NIRpeak.dat 3.846e26 0 0 0 0 0 0
    af 0.
    enpaq 3.846e26


    """

    if(nsimu==1 or cluster==1):
        print("*********************************************")
        print("             MontAGN    V ",nver(),"               ")
        print("*********************************************")
        print("Initialisation")
        print("*********************************************")

    modelmont=model
    start=time.time()
    cst=mc.Constant()
    if(ask==1 and paramfile!=[]):
        ask=0
    info=mc.Info(ask=ask,add=add,force_wvl=force_wvl,wvl_max=wvl_max,plotinfo=plotinfo,display=display,nsimu=nsimu,nang=nang,nsize=nsize,cluster=cluster,debug=debug,verbose=verbose,unit=unit)
    if(modelmont==[]):
        if(nsimu==1 or cluster==1):
            print("Defining a new model")
            if(info.ask<1):
                if(paramfile!=[]):
                    print("From parameter file ",paramfile)
                else:
                    print("From existing parameters")

        if(cluster==1):
            modelmont=ml.makemodel(info=info,usemodel=usemodel,paramfile=paramfile,path='Input/')
        else:
            modelmont=ml.makemodel(info=info,usemodel=usemodel,paramfile=paramfile,path=path+'Input/')
    else:
        if(nsimu==1 or cluster==1):
            print("using an existing model")
        mout.plotT(modelmont,'i',filename,rec=info.nsimu,display=info.display)
    if(info.unit==''):
        if(modelmont.map.Rmax<0.01*cst.pc):
            unity='AU'
            obj="star"
        else:
            unity='pc'
            obj="AGN"
    else:
        unity=info.unit
        obj="object"
    mout.save3d_rho(modelmont,filename,nsimu=nsimu,cluster=cluster)
    mout.plotrhodustype(modelmont,filename,unity=unity,rec=info.nsimu,display=info.display,nsimu=nsimu,cluster=cluster)
    if(nsimu==1 or cluster==1):
        print("Grid size : ",2*modelmont.map.N,"x",2*modelmont.map.N,"x",2*modelmont.map.N)
    modelmont.nscattmax=nscattmax
    #modelmont.thetaobs=thetaobs
    #modelmont.dthetaobs=dthetaobs
    if(info.ask==1):
        usethermal = input('Enabled dust re-emission ? (1/0)')
    #if(usethermal==2):
    #    print 'Using old temperature update routine'
    #    modelmont.Temp_mode=2
    #    usethermal=1
    modelmont.usethermal=usethermal
    #start_simu=time.time()
    if(nphot==[]):
        nphot = input('Number of photons to launch ? ')
    #modelmont2=run_simulation(modelmont,add=add,usethermal=usethermal,nphot=nphot,filename=filename,thetaobs=thetaobs,dthetaobs=dthetaobs,force_wvl=force_wvl,rgmin=rgmin,rgmax=rgmax,alphagrain=alphagrain,display=display,nsimu=nsimu,plotinfo=plotinfo,cluster=cluster,ret=1)
    modelmont2,start_phot=run_simulation(modelmont,info=info,nphot=nphot,filename=filename,nang=nang,nsize=nsize,cluster=cluster,ret=1,cst=cst,path=path)
    if(nsimu==1 or cluster==1):
        print('Warning : probleme de distance intersection trop faible !')
        print("*********************************************")
        print("End of simulation")
        print("*********************************************")
        durees=(time.time() - start_phot)
        duree=(time.time() - start)
        mout.plot_time(duree,"Duration :",star=1,hmins=1)
        print("*** ",'%.4f' % (nphot/durees),"phot/s  ***")
        #print "averaged cross section :",modelmont2.map.dust[0].av_sec,"m2 or radius of",np.sqrt(modelmont2.map.dust[0].av_sec/np.pi)*1e6,"µm"
        print("*********************************************")
    if(modelmont2.usethermal==1):
        print('Number of updates of Rsub :',modelmont2.nRsub_update)
        for i in range(len(modelmont2.map.dust)):
            if(unity=='AU' or unity=='au'):
                print('Final Rsub :',(np.array(modelmont2.Rsub)/cst.AU)[i],'AU for '+modelmont2.map.dust[i].name)
            else:
                print('Final Rsub :',(np.array(modelmont2.Rsub)/cst.pc)[i],'pc for '+modelmont2.map.dust[i].name)
        mout.save3d_T(modelmont2,filename)
        mout.plotT(modelmont2,'f',filename,unity=unity,display=info.display)
    if(info.display==1):
        print("Displaying outputs :")
        print("")
        if(unity=='AU' or unity=='au'):
            res=modelmont2.map.res/cst.AU
        else:
            res=modelmont2.map.res/cst.pc
        if(modelmont2.usethermal==1):
            mout.plot_Tupdate(filename,unity=unity)
        for i in range(len(modelmont2.thetaobs)):
            if(modelmont2.thetaobs[i] in modelmont2.thetarec):
                if(modelmont2.thetaobs[i]<10):
                    thetastr='_00'+str(modelmont2.thetaobs[i])
                elif(modelmont2.thetaobs[i]<100):
                    thetastr='_0'+str(modelmont2.thetaobs[i])
                else:
                    thetastr='_'+str(modelmont2.thetaobs[i])
                mout.plot_image(filename+thetastr+"_phot",thetaobs=modelmont2.thetaobs[i],dtheta=modelmont2.dthetaobs,obj=obj,resimage=res,resunit=unity,cmap=cmap,vectormode=vectormode)

        if(usemodel!=[]):
            if(usemodel<0):
                nphotdet=mout.nphot_detected(filename+"_000_phot",path="Output/")
                if(info.display==1):
                    mout.plot_diff_angle(filename,suffixe="_phot",path="Output/")
                #return nphotdet/nphot,np.sqrt(nphotdet)/nphot,nphotdet
                return nphotdet, nphot, modelmont2
    return modelmont2
