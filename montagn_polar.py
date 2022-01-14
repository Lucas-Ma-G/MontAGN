# -*- coding: utf-8 -*-
import numpy as np
import scipy.integrate as sci
import scipy.optimize as opt
import matplotlib.pyplot as plt
import time
import tarfile
import os
import csv

try:
    import patiencebar as g_pgb
except:
    import mgui_progbar as g_pgb

#from montagn_utils import *
import bhmie_herbert_kaiser_july2012 as mie

########    Constants (USI)    ########
b = 2.8977721e-3 #Wien's displacement cst
h = 6.62606957e-34
c = 299792458
kB = 1.3806488e-23
sigma = 5.670373e-8 #Stefan-Boltzmann constant, in USI
sigt = 6.65246e-29 #Thomson electron cross section, in m2
mass_el = 9.109e-31 #Electron mass, in kg
r_el = 4.60167772e-15 #sqrt(sigt/pi)

pc = 3.08567758e16
nm = 1.0e-9


def fill_mueller(nang,nsize,genseed,grain,plotinfo,force_wvl,inpath):
    #Fonction de création des éléments S11 à S34 des matrices de Mueller
    #en utilisant BHMIE
    #Retourne la liste des paramètres de forme x pour lesquels les éléments
    #ont été calculés ainsi que S11,S12,S33 et S34.
    #Un plafond a été fixé pour x à 2*pi*2000 --> optique géométrique plus que Mie


    ### Déclaration des paramètres ###
    poussiere=['silicates','graphites ortho','graphites para','electrons','pah']

    #milieu
    if(grain.typeg=='silicates'):
        f=open(inpath+'silicates.dat','r')
    elif(grain.typeg=='graphites_ortho' or grain.typeg=='graphites_para' or grain.typeg=='pah_neutral' or grain.typeg=='pah_ionised'):
        f=open(inpath+'graphites.dat','r')
    i=1
    wvl=[]
    ndiff=[]
    ndiff2=[]
    #wvlsil=[]
    #wvlgra=[]
    #ndiffs=[]
    #ndiffgo=[]
    #ndiffgp=[]
    #ndiffgo2=[]
    #ndiffgp2=[]
    if(grain.typeg=='silicates'):
        for data in f.readlines():
            if(i>1):
                tmp=[float(data.split()[1]),float(data.split()[2])]
                wvl.append(float(data.split()[0]))
                nsil=np.sqrt(np.sqrt(tmp[0]*tmp[0]+tmp[1]*tmp[1])+tmp[0])/np.sqrt(2.)
                ksil=np.sqrt(np.sqrt(tmp[0]*tmp[0]+tmp[1]*tmp[1])-tmp[0])/np.sqrt(2.)
                ndiff.append(complex(nsil,ksil))
            else:
                i+=1
    elif(grain.typeg=='graphites_ortho'):
        for data in f.readlines():
            if(i>2):
                tmp=[float(data.split()[2]),float(data.split()[4]),float(data.split()[6]),float(data.split()[8])]
                wvl.append(float(data.split()[0]))
                ngra=np.sqrt(np.sqrt(tmp[0]*tmp[0]+tmp[1]*tmp[1])+tmp[0])/np.sqrt(2.)
                kgra=np.sqrt(np.sqrt(tmp[0]*tmp[0]+tmp[1]*tmp[1])-tmp[0])/np.sqrt(2.)
                ndiff.append(complex(ngra,kgra))
                ngra=np.sqrt(np.sqrt(tmp[2]*tmp[2]+tmp[3]*tmp[3])+tmp[2])/np.sqrt(2.)
                kgra=np.sqrt(np.sqrt(tmp[2]*tmp[2]+tmp[3]*tmp[3])-tmp[2])/np.sqrt(2.)
                ndiff2.append(complex(ngra,kgra))
            else:
                i+=1
    elif(grain.typeg=='graphites_para'):
        for data in f.readlines():
            if(i>2):
                tmp=[float(data.split()[1]),float(data.split()[3]),float(data.split()[5]),float(data.split()[7])]
                wvl.append(float(data.split()[0]))
                #ngra=np.sqrt(np.sqrt(tmp[1]*tmp[1]+tmp[3]*tmp[3])+tmp[1])/np.sqrt(2.)
                #kgra=np.sqrt(np.sqrt(tmp[1]*tmp[1]+tmp[3]*tmp[3])-tmp[1])/np.sqrt(2.)
                #ndiff.append(complex(ngra,kgra))
                #ngra=np.sqrt(np.sqrt(tmp[5]*tmp[5]+tmp[7]*tmp[7])+tmp[5])/np.sqrt(2.)
                #kgra=np.sqrt(np.sqrt(tmp[5]*tmp[5]+tmp[7]*tmp[7])-tmp[5])/np.sqrt(2.)
                #ndiff2.append(complex(ngra,kgra))
                ngra=np.sqrt(np.sqrt(tmp[0]*tmp[0]+tmp[1]*tmp[1])+tmp[0])/np.sqrt(2.)
                kgra=np.sqrt(np.sqrt(tmp[0]*tmp[0]+tmp[1]*tmp[1])-tmp[0])/np.sqrt(2.)
                ndiff.append(complex(ngra,kgra))
                ngra=np.sqrt(np.sqrt(tmp[2]*tmp[2]+tmp[3]*tmp[3])+tmp[2])/np.sqrt(2.)
                kgra=np.sqrt(np.sqrt(tmp[2]*tmp[2]+tmp[3]*tmp[3])-tmp[2])/np.sqrt(2.)
                ndiff2.append(complex(ngra,kgra))
            else:
                i+=1
    elif(grain.typeg=='pah_neutral' or grain.typeg=='pah_ionised'):
        ratio_o=2./3.
        ratio_p=1./3.
        for data in f.readlines():
            if(i>2):
                tmpo=[float(data.split()[2]),float(data.split()[4]),float(data.split()[6]),float(data.split()[8])]
                tmpp=[float(data.split()[1]),float(data.split()[3]),float(data.split()[5]),float(data.split()[7])]
                wvl.append(float(data.split()[0]))
                #r=0.01
                ngrao=np.sqrt(np.sqrt(tmpo[0]*tmpo[0]+tmpo[1]*tmpo[1])+tmpo[0])/np.sqrt(2.)
                kgrao=np.sqrt(np.sqrt(tmpo[0]*tmpo[0]+tmpo[1]*tmpo[1])-tmpo[0])/np.sqrt(2.)
                ngrap=np.sqrt(np.sqrt(tmpp[0]*tmpp[0]+tmpp[1]*tmpp[1])+tmpp[0])/np.sqrt(2.)
                kgrap=np.sqrt(np.sqrt(tmpp[0]*tmpp[0]+tmpp[1]*tmpp[1])-tmpp[0])/np.sqrt(2.)
                ndiff.append(complex(ratio_o*ngrao+ratio_p*ngrap,ratio_o*kgrao+ratio_p*kgrap))
                #r=0.1 : useless
                #ngrao=np.sqrt(np.sqrt(tmpo[2]*tmpo[2]+tmpo[3]*tmpo[3])+tmpo[2])/np.sqrt(2.)
                #kgrao=np.sqrt(np.sqrt(tmpo[2]*tmpo[2]+tmpo[3]*tmpo[3])-tmpo[2])/np.sqrt(2.)
                #ngrap=np.sqrt(np.sqrt(tmpp[2]*tmpp[2]+tmpp[3]*tmpp[3])+tmpp[2])/np.sqrt(2.)
                #kgrap=np.sqrt(np.sqrt(tmpp[2]*tmpp[2]+tmpp[3]*tmpp[3])-tmpp[2])/np.sqrt(2.)
                #ndiff2.append(complex(ratio_o*ngrao+ratio_p*ngrap,ratio_o*kgrao+ratio_p*kgrap))
            else:
                i+=1
    if(grain.typeg=='silicates' or grain.typeg=='graphites_ortho' or grain.typeg=='graphites_para' or grain.typeg=='pah_neutral' or grain.typeg=='pah_ionised'):
        f.close()
            

    #nsil=0.6851 #Indices pour des silicates a 1.65um (H)
    #ksil=0.03246
    #print 'Penser a changer les proprietes des graphites !!'
    #ngra=0.6851
    #kgra=0.03246
    #ndiffs=complex(nsil,ksil)
    #ndiffg=complex(ngra,kgra)
    nvide=1.0
    #rmin=0.005 #um Rayon min des grains 0.005 0.2
    #rmax=0.25 #um Rayon max des grains
    
    ## This routine needs r in um
    #rmin=rmin*1e6
    #rmax=rmax*1e6

    rmin=grain.rmin*1e6*0.9
    rmax=grain.rmax*1e6*1.1
    #rmins=dust[0].rmin*1e6
    #rmaxs=dust[0].rmax*1e6
    #rmingo=dust[1].rmin*1e6
    #rmaxgo=dust[1].rmax*1e6
    #rmingp=dust[2].rmin*1e6
    #rmaxgp=dust[2].rmax*1e6
    #rming=min(rmingo,rmingp)
    #rmaxg=max(rmaxgo,rmaxgp)
    
    #rayonnement
    #if(force_wvl==[]):
    #    wvl_min=1.0*1e-3 #um -3
    #    wvl_max=1.0*1e6 #um +6
    wvl_min=2.0*1e-2 #um -3
    wvl_max=2.0*1e3 #um +6
    #else:
    #    #wvl_min=1.0*1e-1 #um -3
    #    #wvl_max=1.0*1e1 #um +6
    #    wvl_min=0.1*force_wvl*1e6 #um -3
    #    wvl_max=10.0*force_wvl*1e6 #um +6
    
    #nsize=100 #nombre de pas de taille de grain pour facteurs de forme calculés
    nwvl=np.shape(ndiff)[0]
    nan=int((nang+1)//2) #car bhmie retourne 2x plus de valeurs !
    if(grain.typeg=='electrons'):
        nsize=2
    #rstep=(np.log(rmax)-np.log(rmin))/nsize #interp log
    rstep=(np.log(rmax)-np.log(rmin))/(nsize-1) #interp log
    #rstep=(rmax-rmin)/(nsize-1) #interp lineaire

    #if(grain.typeg==0):
    #    #step=(np.log(min(rmax/wvl_min,2000))-np.log(rmin/wvl_max))/nstep #pas pour x sil
    #    step=(np.log(rmax)-np.log(rmin))/nstep #pas pour x sil
    #elif(grain.typeg==1):
    #    #stepgo=(np.log(min(rmaxgo/wvl_min,2000))-np.log(rmingo/wvl_max))/nstep #pas pour x gra o
    #    #stepg=(np.log(min(rmaxg/wvl_min,2000))-np.log(rming/wvl_max))/nstep #pas pour x gra o
    #    step=(np.log(rmax)-np.log(rmin))/nstep
    #elif(grain.typeg==2):
    #    #stepgp=(np.log(min(rmaxgp/wvl_min,2000))-np.log(rmingp/wvl_max))/nstep #pas pour x gra p
    #    step=(np.log(rmax)-np.log(rmin))/nstep

    if(grain.typeg!='electrons'): #grain.typeg==0 or grain.typeg==1 or grain.typeg==2):
        #Allocation des tableaux
        S11=np.zeros((nang,nwvl,nsize)) #,ndust))
        S12=np.zeros((nang,nwvl,nsize)) #,ndust))
        S33=np.zeros((nang,nwvl,nsize)) #,ndust))
        S34=np.zeros((nang,nwvl,nsize)) #,ndust))
        phase=np.zeros((nang,nwvl,nsize)) #,ndust))

        x_M=np.zeros((nwvl,nsize))
        #x_M=np.zeros((nstep,ndust))
        albedo=np.zeros((nwvl,nsize)) #,ndust))
        Qexttab=np.zeros((nwvl,nsize)) #,ndust))
        Qabstab=np.zeros((nwvl,nsize)) #,ndust))
        
        #computing sizes :
        sizegrain=np.zeros(nsize)
        wvlvect=np.zeros(nwvl)

        #read of the pah absorption
        wvl_pah=np.zeros([1201,30])
        Qext_pah=np.zeros([1201,30])
        Qabs_pah=np.zeros([1201,30])
        Qsca_pah=np.zeros([1201,30])
        g_pah=np.zeros([1201,30])
        size_pah=np.zeros([30])
        if(grain.typeg=='pah_neutral' or grain.typeg=='pah_ionised'):
            #abs pah 
            q_gra=0.01
            ax=5.e-9 #m       
            if(grain.typeg=='pah_neutral'):
                f=open(inpath+'PAHneu_30.out','r')
            else:
                f=open(inpath+'PAHion_30.out','r')
            i=1
            j=0
            k=0
            for data in f.readlines():
                if(i==11):
                    size_pah[k]=float(data.split()[0])
                if(i>12):
                    if(j<1201):
                        wvl_pah[1200-j,k]=float(data.split()[0])
                        Qext_pah[1200-j,k]=float(data.split()[1])
                        #Qabs_pah[1200-j,k]=float(data.split()[2])
                        Qsca_pah[1200-j,k]=float(data.split()[3])
                        #g_pah[1200-j,k]=float(data.split()[4])
                        j+=1
                    else:
                        k+=1
                        j=0
                        i=11
                else:
                    i+=1
            f.close()
            #print size_pah


        ### Calcul des matrices ##
        #for j in range(max(nstep,nsteps,nstepg)): #boucle sur x
        #sizegrain=rmin+np.linspace(0,nsize-1,nsize)*rstep #lin
        sizegrain=np.exp(np.log(rmin)+np.linspace(0,nsize-1,nsize)*rstep) #log
        for k in range(nsize): #boucle sur x
            #ancienne version
            #xs=2*np.pi*np.exp(np.log(rmins/wvl_max)+j*steps)
            #xg=2*np.pi*np.exp(np.log(rming/wvl_max)+j*stepg)
            ##print 'pour j = ',j,', x = ',x
            #x_M[j,0]=2*np.pi*np.exp(np.log(rmins/wvl_max)+j*steps)
            #x_M[j,1]=2*np.pi*np.exp(np.log(rming/wvl_max)+j*stepg)
            #refrels=ndiffs/nvide
            #refrelg=ndiffg/nvide
            
            #nouvelle version
            #sizegrain=np.exp(np.log(rmin)+k*rstep)
            #sizegrain[k]=rmin+k*rstep
            for jj in range(nwvl):
                j=nwvl-jj-1
                x=2*np.pi*sizegrain[k]/wvl[j]
                x_M[jj,k]=x
                wvlvect[jj]=wvl[j]*1e-6
                if(grain.typeg=='graphites_ortho' or grain.typeg=='graphites_para'):
                    ndif=complex(np.interp(sizegrain[k],[0.01,0.1],[np.real(ndiff[j]),np.real(ndiff2[j])]),np.interp(sizegrain[k],[0.01,0.1],[np.imag(ndiff[j]),np.imag(ndiff2[j])]))
                else:
                    ndif=ndiff[j]
                refrel=ndif/nvide
                
                #print xs,refrels,nan
                #S1s,S2s,Qexts,Qscas,Qbacks,gscas=mie.bhmie(xs,refrels,nan) #appel à BHMIE silicate
                #S1g,S2g,Qextg,Qscag,Qbackg,gscag=mie.bhmie(xg,refrelg,nan) #appel à BHMIE graphite
                
                S1,S2,Qext,Qsca,Qback,gsca=mie.bhmie(x,refrel,nan) #appel à BHMIE silicates - graphites

                if(grain.typeg=='pah_neutral' or grain.typeg=='pah_ionised'):
                    #abs pah

                    #for mix pah-carbon
                    #xhi=(1-q_gra)*min(1.,(ax/sizegrain[k])**3)
                    #Qabs_n=xhi*Cpah[jj,k]/sigma+(1-xhi)*Qabs
                    
                    #ir=get_indice(size_pah,sizegrain[k])
                    ir=get_closest_indice(size_pah,sizegrain[k])
                    #iwvl=get_indice(wvl_pah[:,ir],wvl[j])
                    iwvl=get_closest_indice(wvl_pah[:,ir],wvl[j])
                    if(iwvl>=1200):
                        iwvl=1199

                    #print iwvl,wvl[j],wvl_pah[iwvl,ir]
                    #print ir,sizegrain[k],size_pah[ir]
                    Qext=Qext_pah[iwvl,ir]
                    Qsca=Qsca_pah[iwvl,ir]

                    #Qm=np.interp(wvl[jj],[wvl_pah[iwvl,ir],wvl_pah[iwvl+1,ir]],[Qext_pah[iwvl,ir],Qext_pah[iwvl+1,ir]])
                    #Qp=np.interp(wvl[jj],[wvl_pah[iwvl,ir+1],wvl_pah[iwvl+1,ir+1]],[Qext_pah[iwvl,ir+1],Qext_pah[iwvl+1,ir+1]])
                    #Qext=np.interp(sizegrain[k],[size_pah[ir],size_pah[ir+1]],[Qm,Qp])
                    #Qm=np.interp(wvl[jj],[wvl_pah[iwvl,ir],wvl_pah[iwvl+1,ir]],[Qsca_pah[iwvl,ir],Qsca_pah[iwvl+1,ir]])
                    #Qp=np.interp(wvl[jj],[wvl_pah[iwvl,ir+1],wvl_pah[iwvl+1,ir+1]],[Qsca_pah[iwvl,ir+1],Qsca_pah[iwvl+1,ir+1]])
                    #Qsca=np.interp(sizegrain[k],[size_pah[ir],size_pah[ir+1]],[Qm,Qp])
                    #Qback=
                    #gsca=

                C=0.5*(x*x)*Qsca
                albedo[jj,k]=Qsca/Qext
                Qexttab[jj,k]=Qext
                Qabstab[jj,k]=Qext-Qsca
                #Cs=0.5*(xs*xs)*Qscas
                #Cg=0.5*(xg*xg)*Qscag
                #Ce=1.0
                #albedo[j,0]=Qscas/Qexts
                #albedo[j,1]=Qscag/Qextg
                #Qexttab[j,0]=Qexts
                #Qexttab[j,1]=Qextg
                #Qabstab[j,0]=Qexts-Qscas
                #Qabstab[j,1]=Qextg-Qscag
                #print 'j =',j
                #for i in range(nang): #boucle sur les angles
                S11[:,jj,k]=0.5*(np.abs(S2[:])*np.abs(S2[:])+np.abs(S1[:])*np.abs(S1[:]))
                S12[:,jj,k]=0.5*(np.abs(S2[:])*np.abs(S2[:])-np.abs(S1[:])*np.abs(S1[:]))
                S33[:,jj,k]=np.real(S2[:]*np.conjugate(S1[:]))
                S34[:,jj,k]=np.imag(S2[:]*np.conjugate(S1[:]))
                phase[:,jj,k]=S11[:,jj,k]/C
                #    #S11[i,j,0]=0.5*(abs(S2s[i])**2+abs(S1s[i])**2)
                #    #S12[i,j,0]=0.5*(abs(S2s[i])**2-abs(S1s[i])**2)
                #    #S33[i,j,0]=np.real(S2s[i]*np.conjugate(S1s[i]))
                #    #S34[i,j,0]=np.imag(S2s[i]*np.conjugate(S1s[i]))
                #    #phase[i,j,0]=S11[i,j,0]/Cs
                #    #S11[i,j,1]=0.5*(abs(S2g[i])**2+abs(S1g[i])**2)
                #    #S12[i,j,1]=0.5*(abs(S2g[i])**2-abs(S1g[i])**2)
                #    #S33[i,j,1]=np.real(S2g[i]*np.conjugate(S1g[i]))
                #    #S34[i,j,1]=np.imag(S2g[i]*np.conjugate(S1g[i]))
                #    #phase[i,j,1]=S11[i,j,1]/Cg
                #    ##S11[i,j,3]=0.5*(abs(S2e[i])**2+abs(S1e[i])**2)
                #    ##S12[i,j,3]=0.5*(abs(S2e[i])**2-abs(S1e[i])**2)
                #    ##S33[i,j,3]=np.real(S2e[i]*np.conjugate(S1e[i]))
                #    ##S34[i,j,3]=np.imag(S2e[i]*np.conjugate(S1e[i]))
                #    ##phase[i,j,3]=S11[i,j,3]/Ce
                #print('x, S1 and S2 :',x_M[j],S1[50],S2[50])
            #polar=Polar(x_M,S11,S12,S33,S34,albedo)

        ## Cas des electrons
    elif(grain.typeg=='electrons'):
        S11=np.zeros([nang,2,2])
        S12=np.zeros([nang,2,2])
        S33=np.zeros([nang,2,2])
        S34=np.zeros([nang,2,2])
        phase=np.zeros([nang,2,2])
        S1,S2=thomson(nan) #appel à Thomson electrons
        C=1.0
        for i in range(2):
            for j in range(2):
                #for i in range(nang): #boucle sur les angles
                #mu=np.cos(i/nang*np.pi) # A VERIFIER
                #S1=1.0
                #S2=mu
                S11[:,i,j]=0.5*(np.abs(S2[:])*np.abs(S2[:])+np.abs(S1[:])*np.abs(S1[:]))
                S12[:,i,j]=0.5*(np.abs(S2[:])*np.abs(S2[:])-np.abs(S1[:])*np.abs(S1[:]))
                S33[:,i,j]=np.real(S2[:]*np.conjugate(S1[:]))
                S34[:,i,j]=np.imag(S2[:]*np.conjugate(S1[:]))
                phase[:,i,j]=S11[:,i,j]/C
                #S11[i]=0.5*(1+mu*mu)
                #S12[i]=0.5*(mu*mu-1)
                #S33[i]=mu
                #S34[i]=0
                #phase[i]=S11[i]/C
        albedo=np.array([[1.0,1.0],[1.0,1.0]])
        Qexttab=np.array([[1.0,1.0],[1.0,1.0]])
        Qabstab=np.array([[0.0,0.0],[0.0,0.0]])
        #x_M=np.array([[2*np.pi*rmin*1e-6,2*np.pi*rmax*1e3],[2*np.pi*rmin*1e-6,2*np.pi*rmax*1e3]])
        #x_M=np.array([[2*np.pi*rmin*1e-3,2*np.pi*rmax*1e-3],[2*np.pi*rmin*1e-6,2*np.pi*rmax*1e-6]])
        x_M=np.array([[2*np.pi*rmin*1e-3*1e10,2*np.pi*rmax*1e-3*1e10],[2*np.pi*rmin*1e-6,2*np.pi*rmax*1e-6]])
        #rstep=grain.rmax-grain.rmin
        wvlvect=np.array([1e-3,1e6])*1e-6
        sizegrain=np.array([rmin,rmax])
    #print x_M

    grain.x_M = x_M
    grain.S11 = S11
    grain.S12 = S12
    grain.S33 = S33
    grain.S34 = S34
    grain.albedo = albedo
    grain.Qext = Qexttab
    grain.Qabs = Qabstab
    #grain.wvlvect = np.array(wvl)*1e-6
    grain.wvlvect = wvlvect
    grain.sizevect = sizegrain*1e-6

    if(plotinfo!=[]):
        #for i in range(ndust):
        #if(grain.typeg!=3):
        #i=int(floor((np.log(0.01)-np.log(rmin))/rstep))      #size=np.exp(np.log(rmin)+k*steps)
        #j=int(floor((np.log(0.1)-np.log(rmin))/rstep))
        typegn={'silicates':0,
                'graphites_ortho':1,
                'graphites_para':2,
                'electrons':3,
                'pah_neutral':4,
                'pah_ionised':5}
        nfig=100*(typegn[grain.typeg]+1)+1
        if(grain.typeg!='electrons'):
            #i=int(np.floor((0.01-rmin)/rstep))      #size=np.exp(np.log(rmin)+k*steps)
            #j=int(np.floor((0.1-rmin)/rstep))
            #i=np.argsort(np.argsort(np.append(grain.sizevect,0.01*1e-6)))[len(grain.sizevect)]-1
            if(grain.typeg=='pah_neutral' or grain.typeg=='pah_ionised'):
                i=get_indice(grain.sizevect,0.001*1e-6)
                j=get_indice(grain.sizevect,0.01*1e-6)
                lab=['r=0.001 um','r=0.01 um']
            else:
                i=get_indice(grain.sizevect,0.01*1e-6)
                #j=np.argsort(np.argsort(np.append(grain.sizevect,0.1*1e-6)))[len(grain.sizevect)]-1
                j=get_indice(grain.sizevect,0.1*1e-6)
                lab=['r=0.01 um','r=0.1 um']
            plotphase(S11[:,:,i],wvlvect,plotinfo,title='S11 for '+grain.name,label=lab[0],n=nfig,legendpos=3,logr=1) #[1,25,50,75,nforme-1]
            plotphase(S12[:,:,i],wvlvect,plotinfo,title='S12 for '+grain.name,label=lab[0],n=nfig+1,legendpos=3,logr=1)
            plotphase(S11[:,:,i],wvlvect,plotinfo,title='S11 for '+grain.name,label=lab[0],n=nfig+2,legendpos=3)
            plotphase(S12[:,:,i],wvlvect,plotinfo,title='S12 for '+grain.name,label=lab[0],n=nfig+3,legendpos=3)
            plotphase(S12[:,:,i]/S11[:,:,i],wvlvect,plotinfo,title='S12/S11 for '+grain.name,label=lab[0],n=nfig+4,legendpos=3)
            plotphase(S11[:,:,j],wvlvect,plotinfo,title='S11 for '+grain.name,label=lab[1],n=nfig+5,legendpos=3,logr=1) #[1,25,50,75,nforme-1]
            plotphase(S12[:,:,j],wvlvect,plotinfo,title='S12 for '+grain.name,label=lab[1],n=nfig+6,legendpos=3,logr=1)
            plotphase(S11[:,:,j],wvlvect,plotinfo,title='S11 for '+grain.name,label=lab[1],n=nfig+7,legendpos=3)
            plotphase(S12[:,:,j],wvlvect,plotinfo,title='S12 for '+grain.name,label=lab[1],n=nfig+8,legendpos=3)
            plotphase(S12[:,:,j]/S11[:,:,j],wvlvect,plotinfo,title='S12/S11 for '+grain.name,label=lab[1],n=nfig+9,legendpos=3)
            #fig=plt.figure()
            #ax = fig.add_subplot(2,1,1)
            ##ax.plot(x_M[:,i],albedo[:,i],label='Albedo for r=0.01 and '+poussiere[grain.typeg])
            #ax.plot(wvl,albedo[:,i],label='Albedo for r=0.01 and '+poussiere[grain.typeg])
            #ax.set_xscale('log')
            #ax.set_yscale('log')
            #plt.legend()
            #fig=plt.figure()
            #ax = fig.add_subplot(2,1,1)
            ##ax.plot(x_M[:,j],albedo[:,j],label='Albedo for r=0.1 and '+poussiere[grain.typeg])
            #ax.plot(wvl,albedo[:,j],label='Albedo for r=0.1 and '+poussiere[grain.typeg])
            #ax.set_xscale('log')
            #ax.set_yscale('log')
            #plt.legend()
            #fig=plt.figure()
            #ax = fig.add_subplot(2,1,1)
            ##ax.plot(x_M[:,i],Qexttab[:,i],label='Qext for r=0.01 and '+poussiere[grain.typeg])
            #ax.plot(wvl,Qexttab[:,i],label='Qext for r=0.01 and '+poussiere[grain.typeg])
            #ax.set_xscale('log')
            #ax.set_yscale('log')
            #plt.legend()
            #fig=plt.figure()
            #ax = fig.add_subplot(2,1,1)
            ##ax.plot(x_M[:,i],Qabstab[:,i],label='Qabs for r=0.01 and '+poussiere[grain.typeg])
            #ax.plot(wvl,Qabstab[:,i],label='Qabs for r=0.01 and '+poussiere[grain.typeg])
            #ax.set_xscale('log')
            #ax.set_yscale('log')
            #plt.legend()
            plot_1D(grain.wvlvect,albedo[:,i],title='Albedo',label=lab[0]+' and '+grain.name,logx=1,logy=1,n=50,xlabel='wvl (m)',ylabel='albedo',legendpos=3)
            plot_1D(grain.wvlvect,albedo[:,j],title='Albedo',label=lab[1]+' and '+grain.name,logx=1,logy=1,n=50,xlabel='wvl (m)',ylabel='albedo',legendpos=3)
            plot_1D(grain.wvlvect,Qexttab[:,i],title='Qext',label=lab[0]+' and '+grain.name,logx=1,logy=1,n=51,xlabel='wvl (m)',ylabel='Qext',legendpos=3)
            plot_1D(grain.wvlvect,Qexttab[:,j],title='Qext',label=lab[1]+' and '+grain.name,logx=1,logy=1,n=51,xlabel='wvl (m)',ylabel='Qext',legendpos=3)
            plot_1D(grain.wvlvect,Qabstab[:,i],title='Qabs',label=lab[0]+' and '+grain.name,logx=1,logy=1,n=52,xlabel='wvl (m)',ylabel='Qabs',legendpos=3)
            plot_1D(grain.wvlvect,Qabstab[:,j],title='Qabs',label=lab[1]+' and '+grain.name,logx=1,logy=1,n=52,xlabel='wvl (m)',ylabel='Qabs',legendpos=3)
        else:
            i=0
            j=1
            plotphase_xfix(grain.S11[:,i,i],title='S11 for '+grain.name,label=grain.name,n=nfig,legendpos=3) #[1,25,50,75,nforme-1]
            plotphase_xfix(grain.S12[:,i,i],title='S12 for '+grain.name,label=grain.name,n=nfig+1,legendpos=3)
            plotphase_xfix(grain.S12[:,i,i]/S11[:,i,i],title='S12/S11 for '+grain.name,label=grain.name,n=nfig+2,legendpos=3)
        #plotphase(S11e,x_M,[0],title='S11 for electrons')
        #plotphase(S12e,x_M,[0],title='S12 for electrons')
        #plotphase(S12e/S11,x_M,[0],title='S12/S11 for electrons')
    #return x_M,S11,S12,S33,S34,phase,albedo,Qexttab,Qabstab,S11e,S12e,S33e,S34e,phasee
    #print wvl,sizegrain
    #print type(wvl),type(sizegrain)
    #print np.shape(wvl),np.shape(sizegrain)
    #return x_M,S11,S12,S33,S34,phase,albedo,Qexttab,Qabstab,np.array(wvl)*1e-6,sizegrain*1e-6
    return phase

def read_Jnu_Draine():
    fm1=open('jpah_draine.dat','r')
    jnu_pah = []
    for li in fm1:
        ln = li.split()
        lnum = [float(x) for x in ln]
        jnu_pah.append(lnum)     
    fm1.close()
    jnupah = np.array(jnu_pah)
    
    return jnupah
	


def init_phase(nang,nsize,phase,genseed,grain,plotinfo,ret=0,display=0):
    nwvl=len(grain.wvlvect)
    x_M=grain.x_M
    phaseNorm=np.zeros([nang,nwvl,nsize])
    w=np.zeros([nwvl,nsize])
    g=np.zeros([nwvl,nsize])
    #for j in range(ndust):
    #phaseEnv=np.zeros([nang,nwvl,nsize])
    phaseEnv=np.zeros([nang])
    if(ret==0):
        phaseEnv_svg=np.zeros([nang,len(plotinfo),2])
    else:
        phaseEnv_svg=np.zeros([nang,nwvl,nsize])
    l=0
    tmp=np.zeros(nang)
    npoly=3
    alpha=np.linspace(0,1.0*np.pi,nang)
    gg=np.zeros(4)
    eff=np.zeros(4)
    #for x in range(nforme):
    phaseEnv_Ray=0.375*(1.0+np.cos(alpha)*np.cos(alpha))
    phaseEnv_pol1=(1.0+np.cos(alpha))**(npoly-1)
    phaseEnv_pol2=(1.0-np.cos(alpha))**(npoly-1)
    phaseEnv_pol3=npoly/(2.0**npoly)
    racine5=np.sqrt(5.0)
    plotinfo_int=[]
    #n=np.argsort(np.argsort(np.append(grain.sizevect,0.01*1e-6)))[len(grain.sizevect)]-1
    #m=np.argsort(np.argsort(np.append(grain.sizevect,0.1*1e-6)))[len(grain.sizevect)]-1
    n=get_indice(grain.sizevect,0.01*1e-6)
    m=get_indice(grain.sizevect,0.1*1e-6)
    for i in range(len(plotinfo)):
        #plotinfo_int.append(int(np.argsort(np.argsort(np.append(grain.wvlvect,plotinfo[i])))[len(grain.wvlvect)]-1))
        plotinfo_int.append(int(get_indice(grain.wvlvect,plotinfo[i])))
        #plotinfo_int.append(int(np.floor(plotinfo[i]*np.shape(phaseNorm)[1]/100.01)))
    #print plotinfo_int
    if(display==1):
        progressb=g_pgb.Patiencebar(valmax=nsize,up_every=1)
    for x in range(nsize):
        if(display==1):
            progressb.update()
        nj1=nwvl-get_indice(x_M[:,x],0.1) # from Rayleigh to polyn
        nj2=nwvl-get_indice(x_M[:,x],2.5) # from Rayleigh to polyn
        #Warning, x_M is a decreasing table so nj1>nj2
        #print len(x_M[:,x]), nj1,nj2
        #print x_M[:,x]
        for j in range(nwvl):
            if(j<nj2-2):
                choix=0 # Full HG
            elif(j<nj2-1):
                choix=1 # HG (& polyn)
            elif(j<nj2):
                choix=2 # polyn (& HG)
            elif(j<nj1-2):
                choix=3 # Full polyn
            elif(j<nj1-1):
                choix=4 # polyn (& Rayleigh)
            elif(j<nj1):
                choix=5 # Rayleigh (& polyn)
            else: # j>nj1
                choix=6 # Full Rayleigh

            #x=2*np.pi*sizegrain[k]/x_M[wvl]
            #for a in range(nang):
            #if(x_M[j,x]<0.1):  #Rayleigh
            if(choix==5 or choix==6):  #Rayleigh
                #phaseEnv[:,j,x]=0.375*(1.0+np.cos(alpha)*np.cos(alpha))                             #cf page 40
                phaseEnv[:]=phaseEnv_Ray
            #if(x_M[j,x]<3):  #Polyn
            if(choix==1 or choix==2 or choix==3 or choix==4 or choix==5):  #Polyn
                gg[0]=0.0                #Bornes de l'intervalle possible pour w
                gg[3]=1.0
                gg[2]=0.5*(gg[0]*(3.0-np.sqrt(5.0))+gg[3]*(np.sqrt(5.0)-1.0))    #Point au milieu cf methode de la section doree (l/L=L/(l+L))
                gg[1]=gg[3]+gg[0]-gg[2]                                                   #Symétrique de gg(2) par rapport au milieu de l'intervalle
                while (gg[3]-gg[0]>0.0001):                            #Precision souhaitee
                    for i in range(4):
                        ##phaseEnv[:,j,x]=npoly*(gg[i]*(1.0+np.cos(alpha))**(npoly-1)+(1.0-gg[i])*(1.0-np.cos(alpha))**(npoly-1))/(2.0**npoly)              #cf page 67
                        ##phaseEnv[:]=npoly*(gg[i]*(1.0+np.cos(alpha))**(npoly-1)+(1.0-gg[i])*(1.0-np.cos(alpha))**(npoly-1))/(2.0**npoly)              #cf page 67
                        #phaseEnv[:]=(gg[i]*phaseEnv_pol1+(1.0-gg[i])*phaseEnv_pol2)*phaseEnv_pol3
                        phaseEnvtmp=(gg[i]*phaseEnv_pol1+(1.0-gg[i])*phaseEnv_pol2)*phaseEnv_pol3
                        ##if(not(list(phaseEnv[:,j,x])>1e-5)):
                        ##    print 'WARNING : phaseEnv low !',phaseEnv[:,j,x]
                        #tmp = phase[:,j,x]/(phaseEnv[:]+1e-6)                    #Mie/Enveloppe(polyn) : ecart
                        tmp = phase[:,j,x]/(phaseEnvtmp+1e-6)                    #Mie/Enveloppe(polyn) : ecart
                        eff[i] = 1.0/(max(tmp))                     #efficacite non normalisee: 1/valeur max de l'ecart Mie-Enveloppe
                        tmp = tmp*eff[i]                         #Nouveau calcul de Mie/Enveloppe normalisee
          
                    #Exclusion de la valeur la moins efficace de w
                    if(eff[1]>eff[2]):
                        gg[3]=gg[2]
                    elif(eff[2]>=eff[1]):
                        gg[0]=gg[1]
                    else:
                        print("Probleme dans le calcul de la section doree")
          
                    #Calcul des nouveaux points intermediaires
                    #gg[2]=0.5*(gg[0]*(3.0-np.sqrt(5.0))+gg[3]*(np.sqrt(5.0)-1.0))
                    gg[2]=0.5*(gg[0]*(3.0-racine5)+gg[3]*(racine5-1.0))
                    gg[1]=gg[3]+gg[0]-gg[2]
                #Determination du meilleur w (au milieu du dernier intervalle trouve)
                ww=(gg[1]+gg[2])/2.0
                #phaseEnv[:,j,x]=npoly*(w*(1.0+np.cos(alpha))**(npoly-1)+(1.0-w)*(1.0-np.cos(alpha))**(npoly-1))/(2.0**npoly)
                #phaseEnv[:]=npoly*(w*(1.0+np.cos(alpha))**(npoly-1)+(1.0-w)*(1.0-np.cos(alpha))**(npoly-1))/(2.0**npoly)
                #if(x_M[j,x]>0.1):
                if(choix==2 or choix==3 or choix==4):
                    phaseEnv[:]=(ww*phaseEnv_pol1+(1.0-ww)*phaseEnv_pol2)*phaseEnv_pol3
                w[j,x]=ww
            #else:  #HG
            #if(x_M[j,x]>2):
            if(choix==0 or choix==1 or choix==2):
                #cf case(2) pour les commentaires (vraiment identique)
                gg[0]=-0.99999999
                gg[3]=0.99999999
                gg[2]=0.5*(gg[0]*(3.0-np.sqrt(5.0))+gg[3]*(np.sqrt(5.0)-1.0))
                gg[1]=gg[3]+gg[0]-gg[2]
                while (gg[3]-gg[0]>0.0001):
                    for i in range(4):
                        #phaseEnv[:] = 0.5*(1.0-gg[i]*gg[i])/((1.0-2.0*gg[i]*np.cos(alpha)+gg[i]*gg[i])**1.5)       #cf page 58
                        phaseEnvtmp = 0.5*(1.0-gg[i]*gg[i])/((1.0-2.0*gg[i]*np.cos(alpha)+gg[i]*gg[i])**1.5)       #cf page 58
                        ##if(not(list(phaseEnv[:,j,x])>1e-5)):
                        ##    print 'WARNING : phaseEnv low !',phaseEnv[:,j,x]
                        #tmp = phase[:,j,x]/(phaseEnv[:]+1e-6)
                        tmp = phase[:,j,x]/(phaseEnvtmp+1e-6)
                        eff[i] = 1.0/(max(tmp))
                        tmp=tmp*eff[i]
                    if(eff[1]>eff[2]):
                        gg[3]=gg[2]
                    elif(eff[2]>eff[1]):
                        gg[0]=gg[1]
                    else:
                        print("Probleme dans le calcul de la section doree")
                    gg[2]=0.5*(gg[0]*(3.0-np.sqrt(5.0))+gg[3]*(np.sqrt(5.0)-1.0))
                    gg[1]=gg[3]+gg[0]-gg[2]
                gtmp=(gg[1]+gg[2])/2.0
                #if(x_M[j,x]>2.5):
                if(choix==0 or choix==1):
                    phaseEnv[:]=0.5*(1.0-gtmp*gtmp)/((1.0-2.0*gtmp*np.cos(alpha)+gtmp*gtmp)**1.5)   
                g[j,x]=gtmp

            #print "x: ",x_M[j,x],"; w: ",w[j,x],"; g: ",g[j,x]

            tmp = phase[:,j,x]/phaseEnv[:]
            Mopt = max(tmp)
            phaseNorm[:,j,x] = tmp/Mopt
            if(j in plotinfo_int):
                if(x==n):
                    phaseEnv_svg[:,l,0]=phaseEnv[:]*Mopt
                    #l+=1
                if(x==m):
                    #l-=len(plotinfo)
                    phaseEnv_svg[:,l,1]=phaseEnv[:]*Mopt
                    l+=1
            elif(ret!=0):
                phaseEnv_svg[:,j,x]=phaseEnv[:]*Mopt

    #print " max w: ",np.max(w[:,:]),"; max g: ",np.max(g[:,:])

    if(plotinfo!=[]):
        poussiere=['silicates','graphites ortho','graphites para','electrons']
        #i=int(floor((0.01-grain.rmin*1e6)/rstep))      #size=np.exp(np.log(rmin)+k*steps)
        #i=np.argsort(np.argsort(np.append(grain.sizevect,0.01*1e-6)))[len(grain.sizevect)]-1
        #j=int(floor((0.1-grain.rmin*1e6)/rstep))
        #j=np.argsort(np.argsort(np.append(grain.sizevect,0.1*1e-6)))[len(grain.sizevect)]-1
        i=get_indice(grain.sizevect,0.01*1e-6)
        j=get_indice(grain.sizevect,0.1*1e-6)
        #x_svg=np.zeros([len(plotinfo),2])
        x_svg=np.zeros([len(plotinfo)])
        for k in range(len(plotinfo)):
            #x_svg[k,0]=x_M[plotinfo_int[k],i]
            #x_svg[k,1]=x_M[plotinfo_int[k],j]
            #x_svg[k,0]=grain.wvlvect[plotinfo_int[k]]
            #x_svg[k,1]=grain.wvlvect[plotinfo_int[k]]
            x_svg[k]=grain.wvlvect[plotinfo_int[k]]
        #if(grain.typeg!=3):
        #plotinfo=[1,25,50,75,nsize-1]
        #print x_svg
        typegn={'silicates':0,
                'graphites_ortho':1,
                'graphites_para':2,
                'electrons':3,
                'pah_neutral':4,
                'pah_ionised':5}
        nfig=100*(typegn[grain.typeg]+1)+11
        plotphase(phase[:,:,i],grain.wvlvect,plotinfo,label='Phase function',title='r=0.01 um and '+grain.name,n=nfig,legendpos=3)
        plotphase(phase[:,:,j],grain.wvlvect,plotinfo,label='Phase function',title='r=0.1 um and '+grain.name,n=nfig+1,legendpos=3)
        #plotphase_xfix(phaseEnv,title='Envelope phase function for r=0.01 and '+poussiere[grain.typeg])
        #plotphase_xfix(phaseEnv,title='Envelope phase function for r=0.1 and '+poussiere[grain.typeg])
        #plotphase(phaseEnv_svg[:,:,0],x_svg[:,0],np.linspace(0,l-1,l),label='Envelope phase function',title='r=0.01 um and '+poussiere[grain.typeg],n=nfig,icolor=len(plotinfo),legendpos=3,percent=0)
        #plotphase(phaseEnv_svg[:,:,1],x_svg[:,1],np.linspace(0,l-1,l),label='Envelope phase function',title='r=0.1 um and '+poussiere[grain.typeg],n=nfig+1,icolor=len(plotinfo),legendpos=3,percent=0)
        plotphase(phaseEnv_svg[:,:,0],x_svg,np.linspace(0,l-1,l),label='Envelope phase function',title='r=0.01 um and '+grain.name,n=nfig,icolor=len(plotinfo),legendpos=3,percent=0)
        plotphase(phaseEnv_svg[:,:,1],x_svg,np.linspace(0,l-1,l),label='Envelope phase function',title='r=0.1 um and '+grain.name,n=nfig+1,icolor=len(plotinfo),legendpos=3,percent=0)
        plotphase(phaseNorm[:,:,i],grain.wvlvect,plotinfo,label='Normalised phase function',title='r=0.01 um and '+grain.name,n=nfig+2,legendpos=3)
        plotphase(phaseNorm[:,:,j],grain.wvlvect,plotinfo,label='Normalised phase function',title='r=0.1 um and '+grain.name,n=nfig+3,legendpos=3)

    if(ret==0):
        return phaseNorm,w,g#,phaseEnv
    else:
        return phaseNorm,w,g,phaseEnv_svg


def init_phase_electrons(nang,phase,genseed,grain,plotinfo):
    phaseNorm=np.zeros([nang,2,2])
    gw=np.array([[[0.00,0.00],[0.01,0.01]],[[0.00,0.00],[0.01,0.01]]])
    phaseEnv=np.zeros([nang,2,2])
    tmp=np.zeros(nang)
    alpha=np.linspace(0,1.0*np.pi,nang)
    
    for i in range(2):
        for j in range(2):
            #phaseEnv[:,i,j]=0.375*(1.0+np.cos(alpha)*np.cos(alpha)) #phase e-
            phaseEnv[:,i,j]=0.5*(1.0+np.cos(alpha)*np.cos(alpha)) #phase e-
            tmp = phase[:,i,j]/phaseEnv[:,i,j]
            Mopt = max(tmp)
            phaseNorm[:,i,j] = tmp/Mopt
    if(plotinfo!=[]):
        nfig=100*(3+1)+4
        poussiere=['silicates','graphites ortho','graphites para','electrons']
        plotphase_xfix(phase[:,0,0],label='Phase function',title=grain.name,n=nfig,legendpos=3)
        plotphase_xfix(phaseEnv[:,0,0],label='Envelope phase function',title=grain.name,n=nfig,icolor=1,legendpos=3)
        plotphase_xfix(phaseNorm[:,0,0],label='Normalised phase function',title=grain.name,n=nfig+1,legendpos=3)
        #plotphase(phase[:,:,j],x_M[:,j],[1,25,50,75,nforme-1],title='Phase function for '+poussiere[grain.typeg])
        #plotphase(phaseEnv,x_M[:,j],[1,25,50,75,nforme-1],title='Envelope phase function for '+poussiere[grain.typeg])
        #plotphase(phaseNorm[:,:,j],x_M[:,j],[1,25,50,75,nforme-1],title='Normalised phase function for '+poussiere[grain.typeg])
    return phaseNorm,gw[:,:,0],gw[:,:,1]


def diffpolar(r, wvl,photon,grain,genseed,eff_mes=[],force_alpha=[]):#polar,,dth,ph):
    nang=np.shape(grain.S11)[0]
    #nang=polar.nang
    #if(dust.typeg==2):
    #    print 'electron diffusion'
    #    S11=polar.S11e
    #    S12=polar.S12e
    #    S33=polar.S33e
    #    S34=polar.S34e
    #    phaseNorm=polar.phasenorm
    #    alpha,beta=thomsondiff(nang,phaseNorm,genseed)
    #else:
    #ir=np.argsort(np.argsort(np.append(grain.sizevect,r)))[len(grain.sizevect)]-1
    #ir=get_indice(grain.sizevect,r)
    ir=get_closest_indice(grain.sizevect,r)
    #S11=(grain.S11[:,:,ir]+grain.S11[:,:,ir+1])*0.5 #polar.S11[:,:,dust.typeg]
    #S12=(grain.S12[:,:,ir]+grain.S12[:,:,ir+1])*0.5 #polar.S12[:,:,dust.typeg]
    #S33=(grain.S33[:,:,ir]+grain.S33[:,:,ir+1])*0.5 #polar.S33[:,:,dust.typeg]
    #S34=(grain.S34[:,:,ir]+grain.S34[:,:,ir+1])*0.5 #polar.S34[:,:,dust.typeg]
    S11=grain.S11[:,:,ir]
    S12=grain.S12[:,:,ir]
    S33=grain.S33[:,:,ir]
    S34=grain.S34[:,:,ir]
    #phaseNorm=(grain.phaseNorm[:,:,ir]+grain.phaseNorm[:,:,ir+1])*0.5 #polar.phasenorm[:,:,dust.typeg]  #interp lin
    phaseNorm=grain.phaseNorm[:,:,ir]
    #x_M=(grain.x_M[:,ir]+grain.x_M[:,ir+1])*0.5 #polar.x_M[:,dust.typeg]  #interp lin
    x_M=grain.x_M[:,ir]
    #if(2*np.pi*r/wvl<2.5):
    #w=(grain.w[:,ir]+grain.w[:,ir+1])*0.5 #polar.gw[:,dust.typeg]  #interp lin
    w=grain.w[:,ir]
    #    g=[]
    #else:
    #    w=[]
    #g=(grain.g[:,ir]+grain.g[:,ir+1])*0.5 #polar.gw[:,dust.typeg]  #interp lin
    g=grain.g[:,ir]
    #print g
    #phaseNorm=loginterp(r,[grain.sizevect[ir],grain.sizevect[ir+1]],[grain.phaseNorm[:,:,ir],grain.phaseNorm[:,:,ir+1]])
    #x_M=loginterp(r,[grain.sizevect[ir],grain.sizevect[ir+1]],[grain.x_M[:,ir],grain.x_M[:,ir+1]])
    #gw=loginterp(r,[grain.sizevect[ir],grain.sizevect[ir+1]],[grain.gw[:,ir],grain.gw[:,ir+1]])
    
    if(eff_mes!=[]):
        alpha,eff_mes=diffalpha(r,wvl,nang,phaseNorm,x_M,w,g,genseed,grain.typeg,eff_mes=eff_mes)
    else:
        alpha=diffalpha(r,wvl,nang,phaseNorm,x_M,w,g,genseed,grain.typeg)
    if(force_alpha!=[]):
        alpha=force_alpha
    dth=alpha
    ang=int(np.floor(dth*(nang-1)/(1.0*np.pi)))
    #print "alpha :",alpha*180/np.pi,", ang :",ang,", nang :",nang,", x :",2*np.pi*r/wvl
    #print "x et S12 :",x_M[50],np.transpose(np.transpose(S12)[50,])
    if (wvl>1e-8 and wvl<1e-5):
        # *1e-6 ?!
        #S11a=[np.interp(2*np.pi*r/wvl,x_M,S11[ang,]),np.interp(2*np.pi*r/wvl,x_M,S11[ang+1,])]
        #S12a=[np.interp(2*np.pi*r/wvl,x_M,S12[ang,]),np.interp(2*np.pi*r/wvl,x_M,S12[ang+1,])]
        #S33a=[np.interp(2*np.pi*r/wvl,x_M,S33[ang,]),np.interp(2*np.pi*r/wvl,x_M,S33[ang+1,])]
        #S34a=[np.interp(2*np.pi*r/wvl,x_M,S34[ang,]),np.interp(2*np.pi*r/wvl,x_M,S34[ang+1,])]
        S11a=[np.interp(wvl,grain.wvlvect,S11[ang,]),np.interp(wvl,grain.wvlvect,S11[ang+1,])]
        S12a=[np.interp(wvl,grain.wvlvect,S12[ang,]),np.interp(wvl,grain.wvlvect,S12[ang+1,])]
        S33a=[np.interp(wvl,grain.wvlvect,S33[ang,]),np.interp(wvl,grain.wvlvect,S33[ang+1,])]
        S34a=[np.interp(wvl,grain.wvlvect,S34[ang,]),np.interp(wvl,grain.wvlvect,S34[ang+1,])]
        
        S11b=np.interp(dth*(nang-1)/(1.0*np.pi),[ang,ang+1],S11a)
        S12b=np.interp(dth*(nang-1)/(1.0*np.pi),[ang,ang+1],S12a)
        S33b=np.interp(dth*(nang-1)/(1.0*np.pi),[ang,ang+1],S33a)
        S34b=np.interp(dth*(nang-1)/(1.0*np.pi),[ang,ang+1],S34a)
        
        beta=diffbeta(r,wvl,photon.Ss[1][0],photon.Ss[2][0],S11b,S12b,genseed)
        ph=beta

        #Definition de la matrice de Mueller
        MMueller=np.zeros([4,4])         
        MMueller[0,0]=S11b
        MMueller[1,0]=S12b
        MMueller[1,1]=S11b
        MMueller[0,1]=S12b
        MMueller[2,2]=S33b
        MMueller[3,3]=S33b
        MMueller[3,2]=-S34b
        MMueller[2,3]=S34b
        
        MM=np.matrix(MMueller)
        
        #Definition de la matrice de rotation d'angle beta
        Rot=np.zeros([4,4])
        Rot[0,0]=1.0
        Rot[1,1]=np.cos(2.0*ph)
        Rot[2,1]=-np.sin(2.0*ph)
        Rot[1,2]=np.sin(2.0*ph)
        Rot[2,2]=np.cos(2.0*ph)
        Rot[3,3]=1.0
    
        Mrot=np.matrix(Rot)
        
        photon.Ss=MM*Mrot*photon.Ss
        photon.Ss=photon.Ss/photon.Ss[0] #Normalisation intensite a 1

        #print 'dth : ',dth,', S12a : ',S12a
        #print 'alpha : ',dth,', beta : ',ph,np.cos(2.0*ph)
        #print 'Q et U :',photon.Ss[1][0],photon.Ss[2][0],np.sqrt(photon.Ss[1][0]**2+photon.Ss[2][0]**2)
        #print 'M : ',MM,', R :',Mrot
        #print "alpha :",alpha*180/np.pi,", x :",2*np.pi*r/wvl,",P :",np.sqrt(photon.Ss[1][0]**2+photon.Ss[2][0]**2)[0][0]
        #print ang,dth*(nang-1)/(1.0*np.pi) ,ang+1
    else:
        ph=2*np.pi*genseed.uniform()
    if(eff_mes!=[]):
        return dth, ph, eff_mes
    return dth, ph #alpha, beta


def diffalpha(r,wvl,nang,phaseNorm,x_M,w,g,genseed,typeg,eff_mes=[]):
    #deltalpha=nang/(1.0*np.pi)
    x=2*np.pi*r/wvl #A changer !!! x=where(x_M...)
    #x=2*np.pi*r/wvl*1e-6 #A changer !!! x=where(x_M...)
    #nx=np.floor((x-min(x_M))/((max(x_M)-min(x_M))/np.size(x_M)))
    #print x_M
    #print x,r,wvl
    #try:
    #    nx=np.where(x_M>x)[0][np.argmin(x_M[np.where(x_M>x)])]
    #except:
    #    nx=np.argmax(x_M)
    #    print 'warning ! x value too high to be used !'
    #    print 'for '+typeg+' , x : ',x
    nx=get_closest_indice(x_M,x)
    #print nx,np.size(x_M)-2

    #print('x :',x_M,', gw :',gw)


    npoly=3
    wtmp=[]
    gtmp=[]
    if (x<0.1 or typeg=='electrons'):
        choix2=0 # Rayleigh
    elif (x<2.5):
        choix2=1 # Poly de degré n
        if(nx<0):
            nx=0
            #gwtmp=gw[min(x_M)]
            wtmp=w[nx]
        elif(nx>np.size(x_M)-2):
            nx=np.size(x_M)-2
            #gwtmp=gw[max(x_M)]
            wtmp=w[nx]
        else:
            #print x_M[nx],x_M[nx+1]
            #wtmp=w[nx]+(w[nx+1]-w[nx])/(x_M[nx+1]-x_M[nx])*(x-x_M[nx])
            wtmp=w[nx]
    else:
        choix2=2 # fonction de HG
        if(nx<0):
            nx=0
            gtmp=g[nx]
        elif(nx>np.size(x_M)-2):
            nx=np.size(x_M)-2
            gtmp=g[nx]
        else:
            #gtmp=g[nx]+(g[nx+1]-g[nx])/(x_M[nx+1]-x_M[nx])*(x-x_M[nx])
            gtmp=g[nx]


    #Etape Mie 31 avec déclaration préalable des fonctions de phase
    Deltalpha=(1.0*np.pi)/(nang-1)
    #Deltalpha2=1.0/deltalpha
    U2=genseed.uniform()                     #Pour entrer dans la boucle
    proba_acceptation=0.0
    j=0
    while(U2>proba_acceptation):
        mu = SimuEnv(choix2, wtmp, gtmp, npoly,genseed)                  #simulation de mu=cos(alpha) suivant l'enveloppe
        if(mu>1 or mu<-1):
            print("Uncorrect value of mu : ",mu)
            print("get from SimuEnv with following parameters :")
            print("   selection ",choix2," ; x ",x," ; w ",wtmp," ; g ",gtmp," ; n poly ",npoly," and seed ",genseed)
        alphadiff=np.arccos(mu)

        #Calcul de Palpha correspondant à notre angle alphadiff par interpollation lineaire
        x1=float(np.floor(alphadiff/Deltalpha))*Deltalpha                     #angle alpha inferieur a alphadiff
        x2=float(np.floor(alphadiff/Deltalpha)+1)*Deltalpha                   #angle alpha superieur a alphadiff
        #if(x1<0):
        #    alphadiff=alphadiff+2*np.pi
        #if(x2>2*np.pi):
        #    alphadiff=alphadiff-2*np.pi
            
        if(np.floor(alphadiff/Deltalpha)==999):
            print("Valeur limite de alphadiff :",alphadiff/np.pi,"pi",x1/np.pi,"pi",x2/np.pi,"pi",np.floor(alphadiff/Deltalpha))
        #    print "Valeur de alphadiff trop proche de pi :",x1/np.pi,"pi",x2/np.pi,"pi"
        #    y1=phaseNorm[int(np.floor(alphadiff/Deltalpha)-1),nx]
        #    y2=phaseNorm[int(np.floor(alphadiff/Deltalpha)),nx]
        #    x2=np.copy(x1)
        #    x1=x1-Deltalpha
        #else :
        y1=phaseNorm[int(np.floor(alphadiff/Deltalpha)),nx] #+1?                       #1/eff de x1
        y2=phaseNorm[int(np.floor(alphadiff/Deltalpha)+1),nx] #+2?                       #1/eff de x2
        a1=(y2-y1)/(x2-x1)                                                #calcul du coef d'accroissement lineaire
        b1=y1-x1*a1                                                       #calcul de l'ordonne a l'origine
        proba_acceptation=a1*alphadiff+b1                        #Palpha
        U2=genseed.uniform()
        j+=1
        #print *, "test numero ", j, "proba : ", proba_acceptation
        #print alphadiff,x1,x2,proba_acceptation
    if(eff_mes!=[]):
        eff_mes[0]+=1
        eff_mes[1]+=j
        return alphadiff, eff_mes
    return alphadiff

def diffbeta(r,wvl,Ss1,Ss2,S11a,S12a,genseed):
    #Etape Mie 32
    plin=np.sqrt(Ss1*Ss1+Ss2*Ss2)
    if(plin!=0):
        psy=0.5*np.arctan2(Ss2,Ss1)
    else:
        psy=0.0

    #print 'psy : ',psy,', Q : ',Ss1,', U : ',Ss2
    U2=genseed.uniform()

    AA=0.5*(S12a/S11a)*plin
    BB=AA*np.sin(2.0*psy)-2*np.pi*U2

    #Determination de beta par dichotomie d'apres psy A et B (cf Mie page 70)

    betatest=[0.0,0.0,0.0]
    fzero=[0.0,0.0,0.0]
    betatest[0]=0.0
    betatest[2]=2.0*np.pi
    #print 'AA : ,', AA, ', BB : ', BB, ', psy : ',psy
    while (betatest[2]-betatest[0]>0.001):
        betatest[1]=(betatest[2]+betatest[0])*0.5
        for i in range(3):
            fzero[i]=betatest[i]+AA*np.sin(2*(betatest[i]-psy))+BB
        if (fzero[0]*fzero[2]>0):
            print('AA : ,', AA, ', BB : ', BB, ', psy : ',psy)
            print('beta : ',betatest)
            print('fzero : ',fzero)
            print('Probleme dans lintervalle de dichotomie !! (Beta)')
        else:
            if (fzero[0]*fzero[1]>0):
                betatest[0]=betatest[1]
            else:
                betatest[2]=betatest[1]
        #print 'beta : ',betatest[1]
      
    beta=(betatest[0]+betatest[2])*0.5
    return beta


def SimuEnv(choix, w, g, npoly,genseed):
    if(choix==0): #Rayleigh
        U1=genseed.uniform()
        U2=genseed.uniform()
        if(U2<=0.25):
            mu=(2.0*U1-1.0)
            mu=np.sign(mu)*abs(mu)**(1.0/3.0)
        else:
            mu=2.0*U1-1.0
    elif(choix==1):
        #U=np.random.random(npoly)
        U=[]
        for i in range(npoly):
            U.append(genseed.uniform())
        U=np.array(U)
        U1=genseed.uniform()
        mu=2.0*max(U)-1.0
        if (U1>w):
            mu=-mu
    elif(choix==2):
        c0=0.5*g*(3.0-g*g)
        c1=1.0+g*g
        c2=0.5*g*(1.0+g*g)
        U1=genseed.uniform()
        U1=2.0*U1-1.0
        mu=(c0+U1*(c1+c2*U1))/((1.0+g*U1)**2)
    else: 
        print("Pas de diffusion")
        mu=0
    return mu

def thomson(nan):
    S1=np.zeros(2*nan-1)
    S2=np.zeros(2*nan-1)
    #sigt=6.65246e-29 #m2
    for i in range(2*nan-1):
        theta=i*np.pi/(2*nan-1)
        S1[i]=1.0
        S2[i]=np.cos(theta)
    return S1,S2

def thomson2(nan):
    S1=np.zeros(2*nan-1)
    S2=np.zeros(2*nan-1)
    #sigt=6.65246e-29 #m2
    for i in range(2*nan-1):
        theta=i*np.pi/nan
        S1[i]=1.0
        S2[i]=np.cos(theta)
    return S1,S2

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2)
    b, c, d = -axis*np.sin(theta/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def plotphase(phaseNorm,x_M,xp,title=[],label=[],n=[],icolor=0,legendpos=0,logr=0,percent=1):
    """
    plotphase(phaseNorm,x_M,xp)
    Function for ploting in a polar graph the phase function of scattering for a given wvl
    """
    #color=[['b','g','r','c','m','y','k','w'],['b--','g--','r--','c--','m--','y--','k--','w--']]
    color=['b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k']
    if(n==[]):
        plt.figure()
    else:
        plt.figure(n)
    i=icolor
    for j in xp:
        if(percent==1):
            #x=int(np.argsort(np.argsort(np.append(x_M,j)))[len(x_M)]-1)
            x=int(get_indice(x_M,j))
            #x=int(np.floor(j*np.shape(phaseNorm)[1]/100.01))
            #x=int(np.argsort(np.argsort(np.append(x_M[0,:],wvl)))[len(x_M[0,:])]-1)
        else:
            x=j
        #x=int(np.argsort(np.argsort(np.append(x_M,j)))[len(x_M)]-1)
        #print j,x,np.shape(phaseNorm)[1]
        phase=np.abs(phaseNorm[:,x])
        nang=len(phase)
        theta=np.linspace(0,1*np.pi,nang)
        theta2=np.linspace(2*np.pi,1*np.pi,nang)
        #theta=[i*np.pi/(nang-1) for i in range(nang)]
        #theta2=[2*np.pi-i*np.pi/(nang-1) for i in range(nang)]
        #print(len(theta),len(phase))
        #plt.plot(theta,phase)
        if(label==[]):
            #plt.polar(theta,phase)
            #plt.polar(theta2,phase)
            labelf='Phase function for wvl = '+str(x_M[x])
        else:
            labelf=label+' for wvl = '+str(x_M[x])
        if(logr==1):
            eps=1e-99
            bias=100
            phase[np.where(phase<eps)]=eps
            phase=np.log10(phase)+bias
            #plt.ylabel('log10(phase)+100')
        else:
            eps=1e-6
            phase[np.where(phase<eps)]=eps
            #phase[np.where(phase==0.0)]=np.min(phase[np.where(np.abs(phase)==phase)])
        #plt.xlabel('Theta (degrees)')
        plt.polar(theta,phase,color[i],label=labelf)
        plt.polar(theta2,phase,color[i]+'--')
        #if(title==[]):
        #    plt.title('Phase function for x ='+str(x_M[x]))
        #else:
        #    plt.title(title+' for x ='+str(x_M[x]))
        i+=1
        #if(logr==1):
        #    plt.yscale('log')
        plt.legend(loc=legendpos)
    if(title!=[]):
        plt.title(title)
    if(logr==1):
        plt.ylabel('log10(phase)+100')
    plt.xlabel('Theta (degrees)')
    plt.show()

def plotphase_xfix(phase,normsin=0,twopi=0,title=[],label=[],n=[],icolor=0,legendpos=0,logr=0,rrange=[],rec=0,outname='test',saveformat='pdf',ax=[]):
    """
    plotphase(phaseNorm,x_M,xp)
    Function for ploting in a polar graph the phase function of diffusion for a given wvl
    """
   
    #color=['b','g','r','c','m','y','k','w']
    color=['b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k']
    #phase=phaseNorm[:,x]
    nang=len(phase)
    phase=np.abs(phase)
    if(twopi==1):
        theta=np.linspace(0,2.*np.pi,nang)
    else:
        theta=np.linspace(0,1*np.pi,nang)
        theta2=np.linspace(2*np.pi,1*np.pi,nang)
    if(normsin==1):
        phase[1:nang-1]=phase[1:nang-1]/np.sin(theta[1:nang-1])
    if(normsin==-1):
        phase[1:nang-1]=phase[1:nang-1]*np.sin(theta[1:nang-1])
    if(ax==[]):
        if(n==[]):
            plt.figure()
        else:
            plt.figure(n)
    #else:
    #    fig,ax=plt.subplots(1,subplot_kw=dict(polar=True))
    if(logr==1):
        eps=1e-99
        bias=100
        phase[np.where(phase<eps)]=eps
        phase=np.log10(phase)+bias
        #plt.ylabel('log10(phase)+100')
    #plt.xlabel('Theta (degrees)')
    else:
        eps=1e-6
        phase[np.where(phase<eps)]=eps
        #phase[np.where(phase==0.0)]=np.min(phase[np.where(np.abs(phase)==phase)])
    if(label==[]):
        if(ax!=[]):
            ax.plot(theta,phase,color[icolor])
        else:
            plt.polar(theta,phase,color[icolor])
    else:
        if(ax!=[]):
            ax.plot(theta,phase,color[icolor],label=label)
        else:
            plt.polar(theta,phase,color[icolor],label=label)
    if(twopi==0):
        if(ax==[]):
            plt.polar(theta2,phase,color[icolor]+'--')
        else:
            ax.plot(theta2,phase,color[icolor]+'--')
    if(title!=[]):
        if(ax==[]):
            plt.title(title)
        else:
            ax.set_title(title)
    #if(logr==1):
    #    plt.yscale('log')
    if(logr==1):
        if(ax==[]):
            plt.ylabel('log10(phase)+100')
        else:
            ax.set_ylabel('log10(phase)+100')
    if(ax==[]):
        plt.xlabel('Theta (degrees)')
        plt.legend(loc=legendpos)
    else:
        ax.set_xlabel('Theta (degrees)')
        ax.legend(loc=legendpos)
    if(rrange!=[]):
        if(ax==[]):
            plt.axis([0,2*np.pi,rrange[0],rrange[1]])
        else:
            ax.axis([0,2*np.pi,rrange[0],rrange[1]])
    plt.show()
    if(rec==1):
        if(saveformat=='png'):
            fig.savefig(outname+'.png',dpi=600)
        else:
            fig.savefig(outname+'.pdf',rasterized=True,dpi=300)
        

def plot_1D(x,y,symbol='',n=[],title='',label='',logx=0,logy=0,xlabel='',ylabel='',legendpos=0,color=[],linestyle = '-'):
    if(n==[]):
        fig=plt.figure()
    else:
        fig=plt.figure(n)
    #ax=fig.add_subplot(2,1,1)
    #ax.plot(model.map.dust[0].x_M,info.xstat)
    if(label==''):
        if(symbol==''):
            if(color==[]):
                plt.plot(x,y,linestyle=linestyle)
            else:
                plt.plot(x,y,color=color,linestyle=linestyle)
        else:
            if(color==[]):
                plt.plot(x,y,symbol,linestyle=linestyle)
            else:
                plt.plot(x,y,symbol,color=color,linestyle=linestyle)
    else:
        if(symbol==''):
            if(color==[]):
                plt.plot(x,y,label=label,linestyle=linestyle)
                plt.legend(loc=legendpos)
            else:
                plt.plot(x,y,label=label,color=color,linestyle=linestyle)
                plt.legend(loc=legendpos) 
        else:
            if(color==[]):
                plt.plot(x,y,symbol,label=label,linestyle=linestyle)
                plt.legend(loc=legendpos)
            else:
                plt.plot(x,y,symbol,label=label,color=color,linestyle=linestyle)
                plt.legend(loc=legendpos)
    if(logx==1):
        plt.xscale('log')
    if(logy==1):
        plt.yscale('log')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def loginterp(x,y,x0):
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
                    C1=x[i]
                    C2=x[i+1]
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
                C1=x[i]
                C2=x[i+1]
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


def get_indice_int(xvect,x):
    """
    get_indice(xvect,x)

    give the indice of the closest inferior value to x in xvect.
    xvect has to be sorted, strictly increasing.
    """
    #ind = np.argsort(np.argsort(np.append(xvect,x)))[len(xvect)]-1
    ind = np.argmin(np.abs(xvect-x+0.5))
    return ind

def get_closest_indice(xvect,x):
    """
    get_indice(xvect,x)

    give the indice of the closest value to x in xvect.
    xvect has to be sorted, strictly increasing.
    """
    #ind = np.argsort(np.argsort(np.append(xvect,x)))[len(xvect)]-1
    ind = np.argmin(np.abs(xvect-x))
    return ind

def get_indice(xvect,x):
    """
    get_indice(xvect,x)

    give the indice of the closest inferior value to x in xvect.
    xvect has to be sorted, strictly increasing.
    """
    ind = np.argsort(np.argsort(np.append(xvect,x)))[len(xvect)]-1
    #ind = np.argmin(np.abs(xvect-x+0.5))
    return ind

def get_indice_altalt(xvect,x):
    """
    get_indice(xvect,x)

    give the indice of the closest inferior value to x in xvect.
    xvect has to be sorted, strictly increasing.
    """
    try:
        ind=np.where(xvect>x)[0][np.argmin(xvect[np.where(xvect>x)])]-1
    except:
        ind=np.argmax(xvect)
    return ind
