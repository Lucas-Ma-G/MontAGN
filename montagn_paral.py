# -*- coding: utf-8 -*-
import numpy as np
import scipy.integrate as sci
import scipy.optimize as opt
import time
import tarfile
import os
import csv
import argparse
from montagn import *
import gui_getoption
from sys import argv, exit

#from IPython import parallel
#import thread
from multiprocessing import Pool
#from multiprocessing import Process


def usemontagn(liste):
    #filename="test_"+str(int(n))

    #Translation of the list params
    n=liste[0]
    ask_in=liste[1]
    usemodel_in=liste[2]
    add_in=liste[3]
    usethermal_in=liste[4]
    nphot_in=liste[5]
    filename_in=liste[6]
    nscattmax_in=liste[7]
    modelmont_in=liste[8]
    display_in=liste[9]
    cluster_in=liste[10]
    thetaobs_in=liste[11]
    dthetaobs_in=liste[12]
    path_in=liste[13]
    cmap_in=liste[14]
    force_wvl_in=liste[15]
    paramfile_in=liste[16]

    #adapting the filenames for multpile runs
    if(n<10):
        number='00'+str(int(n))
    elif(n<100) :
        number='0'+str(int(n))
    else:
        number=str(int(n))
    if(filename_in is None):
        filename="test_"+number
    else:
        filename=filename_in+'_'+number
    
    #managing the ask, paramfile and usemodel parameters
    ask=0 #ask unavailable in parallel launch
    if(paramfile_in=='void' or paramfile_in=='None' or paramfile_in==None):
        if(usemodel_in==None):
            print('No paramfile or model specified')
            print('Using model 0')
            ask=0
            usemodel=0
        else:
            usemodel=float(usemodel_in)
            ask=0
    else:
        usemodel=None

    #Setting defaults values if none
    if(add_in==None):
        add=0
    else:
        add=int(add_in)
    if(usethermal_in==None):
        usethermal=0
    else:
        usethermal=int(usethermal_in)
    if(nphot_in==None):
        print('No photon number given')
        print('launching 100 packets')
        nphot=100
    else:
        nphot=int(nphot_in)
    if(nscattmax_in==None):
        nscattmax=50
    else:
        nscattmax=int(nscattmax_in)
    modelmont=[] #no model already computed usable
    if(display_in==None):
        display=0 #No display available
    else:
        display=int(display_in)
    if(cluster_in==None):
        cluster=0
    else:
        cluster=int(cluster_in)
    #if(thetaobs : unused keyword)
    #if(sthetaobs : unused keyword)
    if(path_in==None):
        path=''
    else:
        path=path_in
    if(display_in==1):
        if(cmap_in==None):
            cmap=[] #No cmap available because no display
    else:
        cmap=[]
    if(force_wvl_in==None or force_wvl_in=='None' or force_wvl_in=='none'):
        force_wvl=[]
    else:
        if(float(force_wvl_in)<1e-9):
            force_wvl=[]
        else:
            force_wvl=float(force_wvl_in)
    if(paramfile_in=='void' or paramfile_in==None):
        paramfile=[]
    else:
        paramfile=paramfile_in

    #Launching
    #model=montAGN(ask=int(liste[1]),usemodel=float(liste[2]),add=int(liste[3]),usethermal=int(liste[4]),nphot=int(liste[5]),filename=filename,ndiffmax=int(liste[7]),display=int(liste[9]),cluster=int(liste[10]),path=path,force_wvl=liste[15],paramfile=paramfile,nsimu=int(n))
    model=montAGN(ask=ask,usemodel=usemodel,add=add,usethermal=usethermal,nphot=nphot,filename=filename,nscattmax=nscattmax,display=display,cluster=cluster,path=path,force_wvl=force_wvl,paramfile=paramfile,nsimu=int(n))


def main():
    
    # initialisation
    sc = gui_getoption.getoption(opt='wes', optval='r', param=['add','display','cluster'], paramval=['usemodel','nphot','filename','nscattmax','nlaunch','thetaobs','dthetaobs','cmap','force_wvl','paramfile','ask','usethermal','path'], man='This is the manual page. Thanks you for using getoption')
    
    # appel de la commande
    sc(argv)
    
    # test option de commande
    if sc['r'] is not None: print("r option was given")
    
    if 'h' in sc.optkey:
        print(sc.man)
        #exit()

    


    #for tycho usage
    if(sc['cluster']==1):
        #n=os.getenv("i")
        if(sc['nlaunch'] is None): 
            n=1
        else:
            n=int(sc['nlaunch'])
        if(n is None): n=1
        liste=[n,sc['ask'],sc['usemodel'],sc['add'],sc['usethermal'],sc['nphot'],sc['filename'],sc['nscattmax'],sc['modelmont'],sc['display'],sc['cluster'],sc['thetaobs'],sc['dthetaobs'],sc['path'],sc['cmap'],sc['force_wvl'],sc['paramfile']]
        #print liste
        usemontagn(liste)
        
        
        
        #for normal usage
    else:        
        if(sc['nlaunch'] is None): 
            n=1
        else:
            n=int(sc['nlaunch'])
        #name=np.linspace(1,n,n)
        liste=[]
        for i in range(n):
            liste.append([i+1,sc['ask'],sc['usemodel'],sc['add'],sc['usethermal'],sc['nphot'],sc['filename'],sc['nscattmax'],sc['modelmont'],sc['display'],sc['cluster'],sc['thetaobs'],sc['dthetaobs'],sc['path'],sc['cmap'],sc['force_wvl'],sc['paramfile']])
            
        #print liste
        #Pool(n).map(usemontagn,name)
        if(n>1):
            Pool(n).map(usemontagn,liste)
        else:
            liste=['1',sc['ask'],sc['usemodel'],sc['add'],sc['usethermal'],sc['nphot'],sc['filename'],sc['nscattmax'],sc['modelmont'],sc['display'],sc['cluster'],sc['thetaobs'],sc['dthetaobs'],sc['path'],sc['cmap'],sc['force_wvl'],sc['paramfile']]
            usemontagn(liste)
        #Pool(n).apply_async(usemontagn[,scriptcall])


if __name__ == '__main__':

    main()
