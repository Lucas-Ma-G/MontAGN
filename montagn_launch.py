# -*- coding: utf-8 -*-
#from __future__ import division
import numpy as np
import scipy.integrate as sci
import scipy.optimize as opt
import time
import tarfile
import os
import csv
from collections import defaultdict

import montagn_utils as mut
import montagn_geom as mgeom
import montagn_classes as mc
import montagn_polar as mpol
import matplotlib.pyplot as plt
import mgui_random as g_rand

######################################################
### Parameter defining file for montAGN simulation ### 

def makemodel(info=[],usemodel=[],paramfile=[],path='Input/'):
    """Fonction de création d'un modèle pour MontAGN.
    Renvoie un objet de type "model" cf montagn_class"""
    if(info==[]):
        info=Info()
    
    cst=mc.Constant()


    ### map asked properties ###
    if(info.ask==0): # Parameter file  ##
        if(paramfile==[] and usemodel!=[]):
            dust_pop,param_fill_map2,model_param,sources_param,emission_param=mgeom.read_defined_models(usemodel,info)
        elif(paramfile==[] and usemodel==[]):
            paramfile = input('Parameter file:')  ####################
        else:
            dust_pop,param_fill_map2,model_param,sources_param,emission_param=mgeom.read_paramfile(path+paramfile,info)
        
    elif(info.ask==1): # Reading inputs from user #
        dust_pop,param_fill_map2,model_param,sources_param,emission_param=mgeom.read_paramuser()
        
    #print emission_param
    #elif(info.ask==-1): # Loading some pre-defined models #
    #    dust_pop,param_fill_map2,model_param,sources_param=mgeom.read_defined_models(usemodel,info)

    ######## other parameters ########

    ### dust basic properties ###
    #silicate :
    typesil=0
    typesilname='silicates'
    tsubsilicate=1400 #K Temp de sublimation 1997-Thatte
    rho_sil=3.3e3 #kg/m3 # ou 0.29*1e-3 ? ##cf Vincent Guillet 2008 p120 (Jones et al., 1994)

    #electrons :
    typeel=3
    tsubelectron=1e10 #K Temp de sublimation, inutilisée
    typeelname='electrons'
    rsubelectron=0.0
    rmin_e=4.600*1.0e-15 #m Rayon min des e- get from sqrt(sigma_T/pi) with Sigma_T=6.65e-29 m2
    rmax_e=4.602*1.0e-15 #m Rayon max des e-
    alpha_e=0 #exposant de la loi de puissance des grains (<=0)
    rho_el=cst.mass_el
    
    #graphite :
    typegrao=1
    typegraoname='graphites_ortho'
    typegrap=2
    typegrapname='graphites_para'
    tsubgraphite=1700 #K Temp de sublimation 1997-Thatte
    #ortho
    rho_grap_ortho=2.2e3 #kg/m3
    #para :
    rho_grap_para=2.2e3 #kg/m3

    #PAH
    typepah=4
    tsubpah=1000 #K Temp de sublimation TODO
    rho_pah=2.2e3 #kg/m3 # Draine et Li (2001)

    #global
    tsub={
        'silicates':tsubsilicate,
        'electrons':tsubelectron,
        'graphites_ortho':tsubgraphite,
        'graphites_para':tsubgraphite,
        'pah_neutral':tsubpah,
        'pah_ionised':tsubpah
    }
    rho={
        'silicates':rho_sil,
        'electrons':rho_el,
        'graphites_ortho':rho_grap_ortho,
        'graphites_para':rho_grap_para,
        'pah_neutral':rho_pah,
        'pah_ionised':rho_pah
    }

    ### translation for input ###
    #############################

    rsub=[]
    dust=[]
    ispah=0
    for i in range(np.shape(dust_pop)[0]):
        if(dust_pop[i][1]=='electrons'):
            rmin_e=4.600*1.0e-15 #m Rayon min des e- get from sqrt(sigma_T/pi) with Sigma_T=6.65e-29 m2
            rmax_e=4.602*1.0e-15 #m Rayon max des e-
            alpha_e=0 #exposant de la loi de puissance des grains
            dust_pop[i][2]=rmin_e
            dust_pop[i][3]=rmax_e
            dust_pop[i][4]=alpha_e
        if('pah' in dust_pop[i][1]):
            ispah=1
        size,avsec=mut.grain_size_surf(dust_pop[i][2],dust_pop[i][3],dust_pop[i][4])
        grain=mc.Grain(dust_pop[i][0],tsub[dust_pop[i][1]],mut.grain_size_surf(dust_pop[i][2],dust_pop[i][3],dust_pop[i][4])[0],avsec,dust_pop[i][2],dust_pop[i][3],dust_pop[i][4],rho[dust_pop[i][1]],dust_pop[i][1],1,i)
        if('pah' in grain.typeg):
            wvl,CC=mut.read_CC_PAH_V2()
            grain.CC_wvl=wvl
            grain.CC=CC
            grain.jnupah=mut.read_Jnu_Draine()
            mut.av_Nc(grain)
        dust.append(grain)
        rsub.append(dust_pop[i][5])
    sources=[]
    source_dust=mc.Source(0.,[],[],t='dust')
    sources.append(source_dust)
    emission_list = defaultdict(lambda:mc.Emission_properties())
    emission_list['default']=mc.Emission_properties()
    emdir = defaultdict(lambda:'default')
    polardir = defaultdict(lambda:'default')
    polar = defaultdict(lambda:'default')
    #print emission_param
    #print emission_param[1]
    emdir_param=emission_param[1]
    polardir_param=emission_param[2]
    polar_param=emission_param[3]
    emission_param=emission_param[0]
    #print emission_param
    for i in range(len(emdir_param)):
        emdir[emdir_param[i][0]] = [emdir_param[i][1],emdir_param[i][2]]
    for i in range(len(polardir_param)):
        polardir[polardir_param[i][0]] = polardir_param[i][1]
    for i in range(len(polar_param)):
        polar[polar_param[i][0]] = [polar_param[i][1],polar_param[i][2],polar_param[i][3]]
    for i in range(len(emission_param)):
        emission_list[emission_param[i][0]] = mc.Emission_properties(emdir=emdir[emission_param[i][1]],polardir=polardir[emission_param[i][2]],polar=polar[emission_param[i][3]])
    for i in range(len(sources_param)):
        spectre=mut.load_spectrum(path+sources_param[i][1])
        emission_name=sources_param[i][3]
        source=mc.Source(sources_param[i][2],spectre,mut.spdist_centre,t=sources_param[i][0],emission_prop=emission_list[emission_name])
        #if(emission_dir!=[[[0],[0],[0]],[[0],[0],[0]]]):
        #    source.emission_dir=emission_dir
        sources.append(source)
    #print sources
    #print model_param
    #print dust
    #print ispah
    map1=mc.Map(model_param[3],model_param[2],dust,pah=ispah)
    #param_fill_map2=mut.translate_map_param(param_fill_map2,dust) #before adapting all the fill_map routines
    if(info.ask==1):
        mgeom.fill_map(map1)
    elif(info.ask==0 or info.ask==-1):
        mgeom.fill_map2(map1,cst,display=info.display,**param_fill_map2)
    else:
        print("Warning wrong ask value, no ask")
        mgeom.fill_map2(map1,cst,display=info.display,**param_fill_map2)
    sources=mc.Source_total(sources) #Concaténation de toutes les sources
    model1=mc.Model(sources,map1,model_param[1],rsub,model_param[0],usemodel,ispah=ispah)
    return model1

