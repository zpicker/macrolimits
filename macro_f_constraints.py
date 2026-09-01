#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:35:47 2026

@author: zpicker
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.colors as colors
import scipy.interpolate as interpolate
import scipy.integrate as integrate

class macro:
    
    #####################initial functions#######################3

    def __init__(self):
        self.mmin=1e0
        self.mmax=1e50
        self.resm = 1000
        self.marray = np.logspace(np.log10(self.mmin),np.log10(self.mmax),self.resm)
        self.ress = 999
        self.smin=1e-40
        self.smax=1e20
        self.sarray = np.logspace(np.log10(self.smin),np.log10(self.smax),self.ress)
        self.clist = [self.asteroid,self.saturn,self.skylab,self.ohya,
                      self.fireballs,self.WD,self.mica,self.xenon,
                      self.deap,self.gas,self.HSC,self.cdms,self.chicago,self.dama,self.rg]
        
    def gev2gram(self,m):
        return m*1.78e-24
        
    ############ plotting functions #########################
    #some code graciously stolen from Ciaran O'Hare

    ### base function for creating the matplotlib figure/axes
    def plotting(self):
        plt.style.use('sty.mplstyle')
        fig, ax = plt.subplots()
        ax.tick_params(labelsize=24)
            # Finishing touches
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim([self.mmin,self.mmax])
        ax.set_ylim([self.smin,self.smax])
        ax.set_xlabel('Dark matter mass [GeV/$c^2$]')
        ax.set_ylabel(r'Geometric cross section [cm$^{2}$]')
        ax.yaxis.set_major_locator(ticker.LogLocator(base=1e5,subs=(1.0,),numticks=100))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10,subs=(1.0,),numticks=100))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.xaxis.set_major_locator(ticker.LogLocator(base=1e5,subs=(1.0,),numticks=100))
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10,subs=(1.0,),numticks=100))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        # ax.axvline(1.22e19,color='gray',linestyle='-',zorder=-100,lw=2)
        # ax.text(1.22e19*1.05,1e-28,r'$M_{\rm Pl}$',fontsize=20,ha='left',va='center',color='gray')
        self.UpperAxis_grams(ax)
        self.UpperAxis_Msun(ax)
        
        # locmaj_x = ticker.LogLocator(base=10.0, subs=(1.0,), numticks=100)
        # ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0]))
        # ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0]))
        
        #earth interactions
        km2pc=3.24078e-14
        rho_dm=0.01#msun pc^-3
        r_earth=6378#km
        v_dm=220#km/s
        msun = 2e33 #g
        g2GeV = 1/(1.8e-24)
        Rate_array = np.array([1,1/3.15e7,1e-9/3.15e7]) #s^-1
        m_array = rho_dm*np.pi*(r_earth**2)*v_dm*(km2pc**3)*msun*g2GeV/Rate_array #GeV
        ###### uncomment for ticks on bottom axes indicating the rate of collisions with earth
        # plt.vlines(m_array,self.smin*1000,self.smin,colors=['black','black','black','white'],linestyles='solid',linewidth=2,alpha=1)
        # plt.text(2*m_array[0],self.smin*5,r'1/s',rotation=0)
        # plt.text(2*m_array[1],self.smin*5,r'1/yr',rotation=0)
        # plt.text(2*m_array[2],self.smin*5,r'1/Gyr',rotation=0,color='black')  
        return fig,ax
    
    #quick square plotting function
    def MySquarePlot(self,xlab='',ylab='',title='',\
                      lw=2.5,lfs=35,tfs=25,size_x=13,size_y=12,Grid=False):
         plt.style.use('sty.mplstyle')
         fig = plt.figure(figsize=(size_x,size_y))
         ax = fig.add_subplot(111)
         ax.set_xlabel(xlab,fontsize=lfs,labelpad=15)
         ax.set_ylabel(ylab,fontsize=lfs,labelpad=15)
         ax.set_title(title,pad=20)
         ax.tick_params(which='major',direction='in',width=2,length=13,right=False,top=True,pad=12)
         ax.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)
         if Grid:
             ax.grid()
         return fig,ax
     
    #creates a number of plots in vertical formation
    def MyVertPlots(self,xlab='',ylab='',title='',\
                      lw=2.5,lfs=35,tfs=25,size_x=13,size_y=12,Grid=False,number=2):
         plt.style.use('sty.mplstyle')
         fig = plt.figure(figsize=(size_x,size_y))
         gs = fig.add_gridspec(number,hspace=0)
         ax = gs.subplots(sharex=True, sharey=True)
         ax[2].set_xlabel(xlab,fontsize=lfs,labelpad=15)
         for i in np.arange(number):
             ax[i].set_ylabel(ylab,fontsize=lfs,labelpad=15)
             ax[i].tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=12)
             ax[i].tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)
         if Grid:
             ax.grid()
         return fig,ax
     
    def MyHorPlots(self,xlab='',ylab='',title='',\
                      lw=2.5,lfs=35,tfs=25,size_x=30,size_y=12,Grid=False,number=2):
         plt.style.use('sty.mplstyle')
         fig = plt.figure(figsize=(size_x,size_y))
         gs = fig.add_gridspec(1,number,wspace=0)
         ax = gs.subplots(sharex=True, sharey=True)
         ax[0].set_ylabel(ylab,fontsize=lfs,labelpad=15)
         for i in np.arange(number):
             ax[i].set_xlabel(xlab,fontsize=lfs,labelpad=15)
             ax[i].tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=12)
             ax[i].tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)
         if Grid:
             ax.grid()
         return fig,ax

    #plots density bands on final figure
    def densities(self,ax):
        # Ice
        rho = .92/(1.8e-24) #GeV cm^-3
        m = np.array([1e0,1e50]) #g
        sigma_ice = np.pi*((3/(4*np.pi))*(m/rho))**(2/3)
        l1 = ax.plot(m,sigma_ice,color='darkorange',alpha=1,linestyle='dashed',linewidth=3,label='Ice')
    
        # Nuclear density
        rho = 2.8e14/(1.8e-24) #GeV cm^-3
        sigma_nuclear = np.pi*((3/(4*np.pi))*(m/rho))**(2/3)
        l2 = ax.plot(m,sigma_nuclear,color='firebrick',linestyle='-.',alpha=1,linewidth=3,label='Nuclear')

    #adds solar mass ticks to upper axis
    def UpperAxis_Msun(self,ax,tickdir='out',xtick_rotation=0,labelsize=25,xlabel=None,lfs=40,tick_pad=0,tfs=25,xlabel_pad=10,label_shift=[1,1],mfact=1):
        m_min,m_max = ax.get_xlim()
        GeV_2_g = 1/5.62e23 # convert GeV to grams
        g_2_Msun = 1/2e33
        ax2 = ax.twiny()
        ax2.set_xscale('log')
        ax2.set_xlabel(xlabel,fontsize=lfs,labelpad=xlabel_pad)
        ax2.tick_params(which='major',direction=tickdir,width=2.5,length=13,pad=tick_pad,rotation=xtick_rotation,labelsize=labelsize)
        ax2.tick_params(which='minor',direction=tickdir,width=1,length=10)
        ax2.set_xticks(10.0**np.arange(-35,-5,5))
        ax2.set_xticks(10.0**np.arange(-35,-5,1), minor=True)
        ax2.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax2.tick_params(which='minor',direction='out',width=1,length=10,right=True,top=True)
        ax2.set_xlim([m_min*GeV_2_g*g_2_Msun,m_max*GeV_2_g*g_2_Msun])
        ax2.text(1e-36/100*label_shift[0],ax2.get_ylim()[1]/(label_shift[1]*mfact/100),r'$M_\odot:$',ha='right',fontsize=labelsize)
        plt.sca(ax)
        return
    
    #adds gram ticks to upper axis
    def UpperAxis_grams(self,ax,tickdir='in',xtick_rotation=0,labelsize=25,xlabel=None,lfs=40,tick_pad=-40,tfs=25,xlabel_pad=10):
        m_min,m_max = ax.get_xlim()
        GeV_2_g = 1/5.62e23 # convert GeV to grams
        ax2 = ax.twiny()
        ax2.set_xscale('log')
        ax2.set_xlabel(xlabel,fontsize=lfs,labelpad=xlabel_pad)
        ax2.tick_params(labelsize=tfs)
        ax2.tick_params(which='major',direction=tickdir,width=2.5,length=13,pad=tick_pad)
        ax2.tick_params(which='minor',direction=tickdir,width=1,length=10)    
        ax2.set_xticks(10.0**np.arange(-18,18,3))
        ax2.set_xticklabels(['ag','fg','pg','ng',r'\textmu g','mg','g','kg','Mg','Gg','Tg','Pg']);
        ax2.set_xticks(10.0**np.arange(-18,18-2,1), minor=True)
        ax2.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax2.set_xlim([m_min*GeV_2_g,m_max*GeV_2_g])
        plt.sca(ax)
        return
    
    #main function for actually plotting full constraints
    def plot_constraints(self,lines=True,text=True): #call everything
    
        fig,ax = self.plotting()
        self.densities(ax)
        
        #load data in
        ast = self.asteroid()
        sat = self.saturn()
        sky = self.skylab()
        ohya = self.ohya()
        fire = self.fireballs()
        wd = self.WD()
        mica = self.mica()
        xenon = self.xenon()
        deap = self.deap()
        gas = self.gas()
        hsc = self.HSC()
        dama = self.dama()
        chic = self.chicago()
        cdms = self.cdms()
        rg = self.rg()

        alist = [ast,sat,sky,ohya,fire,wd,mica,xenon,deap,gas,hsc,dama,chic,cdms,rg]
        
        allconstraints = np.min(alist,axis=0)            
        cmap = plt.get_cmap('viridis').copy()
        color=(0,0,0,0)
        cmap.set_over(color)
        cmap.set_under(color)

        plot = ax.pcolormesh(self.marray,self.sarray,allconstraints,cmap=cmap,norm=colors.LogNorm(vmax=1.1,vmin=1e-15))
        fig.colorbar(plot, ax=ax,label=r'$f_{\mathrm{DM}}$',ticks=ticker.LogLocator(base=10.0, subs=(1.0,), numticks=99))
        ax.contourf(self.marray,self.sarray*1e15,gas,cmap='Greys',levels=[1e-50,1],alpha=0.5)

        ### lines around constraints:
        if lines:
            style = 'solid'
            lw = 1
            alph=0.5
            ax.contour(self.marray,self.sarray,ast,levels=[2],linewidths=lw,linestyles=style,colors='black',alpha=alph)
            ax.contour(self.marray,self.sarray,sat,levels=[2],linewidths=lw,linestyles=style,colors='black',alpha=alph)
            ax.contour(self.marray,self.sarray,sky,levels=[2],linewidths=lw,linestyles=style,colors='black',alpha=alph)
            ax.contour(self.marray,self.sarray,ohya,levels=[1],linewidths=lw,linestyles=style,colors='black',alpha=alph)
            ax.contour(self.marray,self.sarray,fire,levels=[2],linewidths=lw,linestyles=style,colors='black',alpha=alph)
            ax.contour(self.marray,self.sarray,wd,levels=[2],linewidths=lw,linestyles=style,colors='black',alpha=alph)
            ax.contour(self.marray,self.sarray,mica,levels=[2],linewidths=lw,linestyles=style,colors='black',alpha=alph)
            ax.contour(self.marray,self.sarray,xenon,levels=[2],linewidths=lw,linestyles=style,colors='black',alpha=alph)
            ax.contour(self.marray,self.sarray,deap,levels=[2],linewidths=lw,linestyles=style,colors='black',alpha=alph)
            ax.contour(self.marray,self.sarray,gas,levels=[2],linewidths=lw,linestyles=style,colors='black',alpha=alph)
            ax.contour(self.marray,self.sarray,hsc,levels=[2],linewidths=lw,linestyles=style,colors='black',alpha=alph)
            ax.contour(self.marray,self.sarray,dama,levels=[2],linewidths=lw,linestyles=style,colors='black',alpha=alph)
            ax.contour(self.marray,self.sarray,chic,levels=[2],linewidths=lw,linestyles=style,colors='black',alpha=alph)
            ax.contour(self.marray,self.sarray,cdms,levels=[2],linewidths=lw,linestyles=style,colors='black',alpha=alph)
            ax.contour(self.marray,self.sarray,rg,levels=[2],linewidths=lw,linestyles=style,colors='black',alpha=alph)

        ### text labels on constraints:
        if text:
            ax.text(2e28,1e5,r'Asteroids',fontsize=24,rotation=24)
            ax.text(1e16,7e-4,r'Saturn',fontsize=24,rotation=28)
            ax.text(1e7,2e-15,r'Skylab',fontsize=24,rotation=40)
            ax.text(1e12,5e-14,r'Ohya',fontsize=24,rotation=40)
            ax.text(1e26,3e-7,r'Fireballs',fontsize=24,rotation=30)
            ax.text(1e35,2e-8,r'White Dwarfs',fontsize=24,rotation=0)
            ax.text(1e27,1e-16,r'Mica',fontsize=24,rotation=0)
            ax.text(1e3,1e-39,r'Xenon1T',fontsize=24,rotation=36)
            ax.text(1e19,2e-21,r'DEAP3600',fontsize=24,rotation=0)
            ax.text(1e37,1e8,'Gas clouds',fontsize=24,rotation=36)
            ax.text(3e47,3e8,'HSC',fontsize=24,rotation=90)
            ax.text(1e3,1e-34,r'CDMS',fontsize=24,rotation=36)
            ax.text(2e3,3e-28,r'DAMA',fontsize=24,rotation=17)
            ax.text(3e0,1e-25,r'Chicago',fontsize=24,rotation=40)
            ax.text(1e40,1e5,r'RG',fontsize=24,rotation=42)




        #black hole density region
        m = np.array([1e2,1e50])
        GeV_2_g = 1/5.62e23 
        sigma_BH = np.pi*(100*2*6.67e-11*m*GeV_2_g*1e-3/3e8**2)**2
        ax.plot(m,sigma_BH,color='black',alpha=1,linewidth=4)
        ax.fill_between(m,sigma_BH,[self.smin,self.smin],facecolor='black',hatch='.',edgecolor='dimgrey',alpha=1,label='Black holes')
        plt.legend(loc=4,fontsize=24,facecolor='white',frameon=True)
    

    #################### contour bounds ######################

    #little function to findest closest value in array
    def find_closest(self,arr, val):
           idx = np.abs(arr - val).argmin()
           return idx  
      
    #define your mass-cross-section relation in here. input is mass array and list of parameters
    #output is density array
    #basic density function already included
    def cross_section_func(self,marray,parameters):
        
        #basic density:
        density = parameters[0]
        rho = density/self.gev2gram(1) #GeV cm^-3
        sigma_out = np.pi*((3/(4*np.pi))*(marray/rho))**(2/3)
        
        return sigma_out
       
    #returns f values along the contour
    def linevalues(self,parameters,constraint,smoothing=True): #g cm^{-3}
        m,sig,f2d = constraint(original=True)
        yarr = self.cross_section_func(m,parameters)
        values=np.zeros(len(yarr))
        for i,y in enumerate(yarr):
            yind = self.find_closest(sig,y)
            values[i] = f2d[yind,i]
            if values[i]>10:
                values[i]*=(np.random.rand()+1)*1e10
        if smoothing:
            midinds = self.mids(values)
            m = m[midinds]
            values=values[midinds]
        return m, values
    
    #calculates the midpoints of straight sections (artefact of sampling) to use for plotting smooth lines
    def mids(self,arr):
        changes = np.concatenate(([True], arr[:-1] != arr[1:], [True]))
        boundaries = np.where(changes)[0]
        middle_indices = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            middle = (start + end - 1) // 2
            middle_indices.append(middle)
        return np.array(middle_indices)

    #main plotting function for constraints along contour
    def fplot(self,parameters,ax):

        ax.set_ylim([1e-10,2])        
        alph=0.3
        
        m,values = self.linevalues(parameters,self.gas)
        ax.loglog(m,values,color='purple',linewidth=4,label='Gas Clouds')
        ax.fill_between(m,values,2,color='purple',alpha=alph)
        
        m,values = self.linevalues(parameters,self.asteroid)
        ax.loglog(m,values,color='red',linewidth=4,label='Asteroids')
        ax.fill_between(m,values,2,color='red',alpha=alph)
        
        m,values = self.linevalues(parameters,self.saturn)
        ax.loglog(m,values,color='lightcoral',linewidth=4,label='Saturn')
        ax.fill_between(m,values,2,color='lightcoral',alpha=alph)
        
        m,values = self.linevalues(parameters,self.fireballs)
        ax.loglog(m,values,color='orange',linewidth=4,label='Fireballs')
        ax.fill_between(m,values,2,color='orange',alpha=alph)

        m,values = self.linevalues(parameters,self.ohya)
        ax.loglog(m,values,color='blue',linewidth=4,label='Ohya')
        ax.fill_between(m,values,2,color='blue',alpha=alph)
        
        m,values = self.linevalues(parameters,self.skylab)
        ax.loglog(m,values,color='lightblue',linewidth=4,label='Skylab')
        ax.fill_between(m,values,2,color='lightblue',alpha=alph)

        m,values = self.linevalues(parameters,self.mica)
        ax.loglog(m,values,color='darkblue',linewidth=4,label='mica')
        ax.fill_between(m,values,2,color='darkblue',alpha=alph)
           
        m,values = self.linevalues(parameters,self.xenon)
        ax.loglog(m,values,color='green',linewidth=4,label='Xenon100')
        ax.fill_between(m,values,2,color='green',alpha=alph)

        m,values = self.linevalues(parameters,self.deap)
        ax.loglog(m,values,color='darkgreen',linewidth=4,label='DEAP3600')
        ax.fill_between(m,values,2,color='darkgreen',alpha=alph)
        
        m,values = self.linevalues(parameters,self.dama)
        ax.loglog(m,values,color='mediumseagreen',linewidth=4,label='DAMA')
        ax.fill_between(m,values,2,color='mediumseagreen',alpha=alph)
        
        m,values = self.linevalues(parameters,self.chicago)
        ax.loglog(m,values,color='mediumaquamarine',linewidth=4,label='Chicago')
        ax.fill_between(m,values,2,color='mediumaquamarine',alpha=alph)
        
        m,values = self.linevalues(parameters,self.cdms)
        ax.loglog(m,values,color='limegreen',linewidth=4,label='CDMS')
        ax.fill_between(m,values,2,color='limegreen',alpha=alph)

        m,values = self.linevalues(parameters,self.WD)
        ax.loglog(m,values,color='grey',linewidth=4,label='WD1CO')
        ax.fill_between(m,values,2,color='grey',alpha=alph)

        m,values = self.linevalues(parameters,self.HSC)
        ax.loglog(m,values,color='black',linewidth=4,label='HSC')
        ax.fill_between(m,values,2,color='black',alpha=alph)

        ax.set_xlim(1e0,1e50)
        
    #basic plotting function to produce a plot with m-f_dm axes
    def plot_contour(self):
        xlab = r'Dark matter mass $[\mathrm{GeV}/c^2]$'
        ylab = r'$f_{\mathrm{DM}}$'
        ##modify self.cross_section_func and set parameters here
        parameters = [1]
        fig,ax=self.MySquarePlot(xlab,ylab)
        self.fplot(parameters,ax)
        self.UpperAxis_grams(ax)
        self.UpperAxis_Msun(ax,mfact=4.5e1)
        
        #zac TODO: automatically filter out labels which aren't visible 
        ax.legend(fontsize=20,loc='lower right')
    
    #3 vertical plots, as used in paper for density example
    def plot_contour_3(self):
        xlab = r'Dark matter mass $[\mathrm{GeV}/c^2]$'
        ylab = r'$f_{\mathrm{DM}}$'
        dlist = [[1e1],[1e7],[1e14]] #g cm^-3
        fig,ax=self.MyVertPlots(xlab,ylab,size_y=14,number=len(dlist))
        for i,d in enumerate(dlist):
            self.fplot(d,ax[i])
            density=d[0]
            ax[i].text(2e35,1e-9,r'$\rho_{\chi}=$ '+f"{density:.0E}"+' g/cm$^3$',fontsize=24,color='black')
            #formatting tricks to get minor ticks displaying:
            ax[i].yaxis.set_minor_locator(ticker.LogLocator(base=10.0,numticks=100))
            ax[i].xaxis.set_minor_locator(ticker.LogLocator(base=10.0,numticks=100))
            ax[i].yaxis.set_minor_formatter(ticker.NullFormatter())
            ax[i].xaxis.set_minor_formatter(ticker.NullFormatter())
            ax[i].xaxis.set_major_locator(ticker.FixedLocator(np.logspace(0,50,11)))
        self.UpperAxis_grams(ax[0])
        self.UpperAxis_Msun(ax[0],mfact=1.5e1)
        # ax[0].legend(fontsize=20,loc='center left', bbox_to_anchor=(1, 0.2))
        
        text=True
        if text:
            ax[0].text(1e27,1e-3,r'Asteroids',fontsize=30,rotation=0,color='red')
            ax[0].text(1e20,3e-6,r'Saturn',fontsize=30,rotation=25,color='lightcoral')
            ax[0].text(1e10,1e-9,'Gas clouds',fontsize=30,rotation=32,color='purple')
            ax[0].text(1e45,1e-3,'HSC',fontsize=30,rotation=0)

            ax[1].text(1e29,1e-2,r'Fireballs',fontsize=30,rotation=0,color='orange')
            ax[1].text(3e18,1e-3,r'Ohya',fontsize=30,rotation=62,color='blue')
            ax[1].text(1e11,1e-7,r'Skylab',fontsize=30,rotation=60,color='teal')

            ax[2].text(1e42,3e-4,r'WDs',fontsize=30,rotation=60,color='grey')
            ax[2].text(1e19,3e-8,r'Mica',fontsize=30,rotation=60,color='darkblue')
            ax[2].text(1e9,1e-7,r'DEAP3600',fontsize=30,rotation=90,color='darkgreen')
            # ax[2].text(1e3,1e-34,r'CDMS',fontsize=30,rotation=36,color='limegreen')
            ax[2].text(1e13,1e-9,r'DAMA',fontsize=30,rotation=0,color='mediumseagreen')
            ax[2].text(1e3,1e-6,r'Chicago',fontsize=30,rotation=90,color='mediumaquamarine')

    ################ extended mass functions ##########################################

    #define EMF here, where params is a list. if normalize==True, the function is automatically normalized
    def psi(self,m,params):
        
        ###lognormal example:
        mc,zeta = params[0],params[1]
        out = (np.sqrt(2*np.pi)*zeta*m)**(-1) * np.exp((-np.log(m/mc)**2)/(2*(zeta**2)))
        
        ###Press-schechter example:
        # ms,mlow = params[0],params[1]
        # marr = np.logspace(np.log10(mlow),np.log10(1000*ms),100)
        # ##to normalize it:
        # outa = (np.sqrt(np.pi)*marr)**(-1) *(marr/ms)**(1/2) *np.exp(-marr/ms)*np.heaviside(marr-mlow,1)
        # factor = integrate.trapezoid(outa,x=marr)
        # out = (np.sqrt(np.pi)*m)**(-1) *(m/ms)**(1/2) *np.exp(-m/ms)*np.heaviside(m-mlow,1)*np.heaviside(m-mlow,1)/factor
        
        ###bimodal example:
        # m1,f = params[0],params[1]
        # #narrow lognormal approximation for each
        # zeta = 0.1
        # out = f*(np.sqrt(2*np.pi)*zeta*m)**(-1) * np.exp((-np.log(m/m1)**2)/(2*(zeta**2)))        
        return out
        
    #integrate contour for individual constraint
    def contour_constraint(self,constraint,sigma_params,EMF_params):
        m,values = self.linevalues(sigma_params,constraint,smoothing=False)
        psiarr = self.psi(m,EMF_params)
        integrand = psiarr/values
        res = integrate.trapezoid(integrand,x=m)
        return res**2
    
    #compute constraint on f_psi,dm for EMF
    def fpsidm(self,sigma_params,EMF_params):
        total=0
        for i,constraint in enumerate(self.clist):
            total+=self.contour_constraint(constraint, sigma_params, EMF_params)
        if total == 0:
            return 99999 #arbitrary large number >1
        else:
            return total**(-1/2)
        
    #compute constraints when you have N independent DM candidates
    #parameter lists must be list of sigma_params for each of the individual candidates
    def fpsidm_N(self,sigma_params_list,EMF_params_list):
        total=0
        for i,constraint in enumerate(self.clist):
            for j in range(len(sigma_params_list)):
                total+=self.contour_constraint(constraint, sigma_params_list[j], EMF_params_list[j])
        if total == 0:
            return 99999 #arbitrary large number >1
        else:
            return total**(-1/2)
    
    #ploting examples in paper. nested loops are slow!
    def plot_EMF(self,loop=True):
        
        ###lognormal parameters:
        mc = np.logspace(10,40,10*5)
        zeta = np.logspace(-1.3,1.3,9*5)
        sigma_params_3=[[1.0e1],[1.0e7],[1.0e14]] #g cm^-3
        EMF_values = np.zeros((3,len(zeta),len(mc)))
        xlab = r'$m_c[\mathrm{GeV}/c^2]$'
        ylab = r'$\zeta$'
        filename = 'EMF_values_ln.npy'
        
        ###Press-schechter parameters:
        # ms = np.logspace(0,50,10*5)
        # mlow = np.logspace(0,50,10*5)
        # sigma_params_3=[[1.0e1],[1.0e7],[1.0e14]] #g cm^-3
        # EMF_values = np.zeros((3,len(mlow),len(ms)))
        # xlab = r'$m_*[\mathrm{GeV}/c^2]$'
        # ylab = r'$m_{\rm low}[\mathrm{GeV}/c^2]$'
        # filename = 'EMF_values_ps.npy'

        ###bimodal parameters:
        # m1 = np.logspace(0,50,10*5)
        # m2 = np.logspace(0,50,10*5)
        # f12 = 0.5
        # sigma_params_list1 = [[1.0e1],[1.0e7]]
        # sigma_params_list2 = [[1.0e1],[1.0e14]]
        # sigma_params_list3 = [[1.0e7],[1.0e14]]
        # sigma_params_3 = [sigma_params_list1,sigma_params_list2,sigma_params_list3]
        # EMF_values = np.zeros((3,len(m2),len(m1)))
        # xlab = r'$m_1[\mathrm{GeV}/c^2]$'
        # ylab = r'$m_{2}[\mathrm{GeV}/c^2]$'
        # filename = 'EMF_values_bi.npy'
        
        if loop:
            for i in range(3):
                for j,m in enumerate(mc):
                # for j,m in enumerate(ms):
                # for j,ma in enumerate(m1):
                    print(j)
                    for k,z in enumerate(zeta):
                    # for k,ml in enumerate(mlow):
                    # for k,mb in enumerate(m2):
                        ###lognormal:
                        EMF_values[i,k,j]=self.fpsidm(sigma_params_3[i],[m,z]) 
                        ###PS
                        # if m>ml:
                            # EMF_values[i,k,j]=self.fpsidm(sigma_params_3[i],[m,ml])
                        # else:
                            # EMF_values[i,k,j]=np.nan
                        ###bimodal:
                        # psi_params_list = [[ma,f12],[mb,1-f12]]
                        # EMF_values[i,k,j]=self.fpsidm_N(sigma_params_3[i],psi_params_list) 
                        
                #save array so it doesn't have to do this loop every time
                
                np.save(filename,EMF_values)
        else:
            EMF_values=np.load(filename)
          
        fig,ax=self.MyHorPlots(xlab,ylab,size_y=10,number=3)
        cmap = plt.get_cmap('magma').copy()
        color=(0,0,0,0)
        cmap.set_over(color)
        cmap.set_under(color)
        cmap.set_bad(color='grey')

        for i in range(3):

            ax[i].set_yscale('log')
            ax[i].set_xscale('log')
            
            ###lognormal
            plot = ax[i].pcolormesh(mc,zeta,EMF_values[i,:,:],cmap=cmap,norm=colors.LogNorm(vmax=1.1,vmin=np.min(EMF_values)))
            ax[i].set_xlim([1e10,1e40])
            ax[i].set_ylim([.8e-1,2e1])
            ax[i].xaxis.set_major_locator(ticker.FixedLocator(np.logspace(10,40,7)))
            density=sigma_params_3[i][0]
            ax[i].text(3e23,1.1e-1,r'$\rho_{\chi}=$ '+f"{density:.1}"+' g/cm$^3$',bbox=dict(facecolor='white',alpha=0.9))
            ticksx = ax[i].get_xticks()
            ax[i].set_xticks(ticksx[:-1])
            
            ###press-schechter
            # plot = ax[i].pcolormesh(ms,mlow,EMF_values[i,:,:],cmap=cmap,norm=colors.LogNorm(vmax=1.1,vmin=np.nanmin(EMF_values)))                        
            # ax[i].set_xlim([ms[0],ms[-1]])
            # ax[i].set_ylim([mlow[0],mlow[-1]])
            # ax[i].xaxis.set_major_locator(ticker.FixedLocator(np.logspace(0,50,6)))
            # ax[i].yaxis.set_major_locator(ticker.FixedLocator(np.logspace(0,50,11)))
            # ticksx,ticksy = ax[i].get_xticks(),ax[i].get_yticks()
            # ax[i].set_xticks(ticksx[:-1])
            # ax[i].set_yticks(ticksy[:-1])
            # density=sigma_params_3[i][0]
            # ax[i].text(3e22,5e2,r'$\rho_{\chi}=$ '+f"{density:.1}"+' g/cm$^3$',bbox=dict(facecolor='white',alpha=0.9))
            
            ###bimodal
            # plot = ax[i].pcolormesh(m1,m2,EMF_values[i,:,:],cmap=cmap,norm=colors.LogNorm(vmax=1.1,vmin=np.nanmin(EMF_values)))
            # ax[i].set_xlim([m1[0],m1[-1]])
            # ax[i].set_ylim([m2[0],m2[-1]])
            # ax[i].xaxis.set_major_locator(ticker.FixedLocator(np.logspace(0,50,6)))
            # ax[i].yaxis.set_major_locator(ticker.FixedLocator(np.logspace(0,50,11)))
            # ticksx,ticksy = ax[i].get_xticks(),ax[i].get_yticks()
            # ax[i].set_xticks(ticksx[:-1])
            # ax[i].set_yticks(ticksy[:-1])        
            # d1,d2 = sigma_params_3[i][0][0],sigma_params_3[i][1][0],
            # ax[i].text(3e22,5e2,r'$\rho_{1}=$ '+f"{d1:.1}"+' g/cm$^3$'+'\n'+ r'$\rho_{2}=$ '+f"{d2:.1}"+' g/cm$^3$',bbox=dict(facecolor='white',alpha=0.9))

        fig.colorbar(plot, ax=ax,label=r'$f_{\psi,\mathrm{DM}}$',pad=3e-2)
               
    ############### individual constraints ################################
    #data for individual constraints is called by these functions. 
    #getting the m and sig arrays correct for each is crucial

    def resize(self,m,sig,mask):
        mask[mask == 0] = 99
        func = interpolate.RectBivariateSpline(np.log10(sig),np.log10(m),np.log10(mask))
        out = 10**func(np.log10(self.sarray),np.log10(self.marray))
        return out

    def asteroid(self,original=False):
        m = np.logspace(24,40,1000)
        sig = np.logspace(-4,12,999)
        mask = np.load('macro_constraints/asteroid.npy')
        mask2 = self.resize(m,sig,mask)
        if not original:
            return mask2
        else:
            return m,sig,mask
        
    def saturn(self,original=False):
        m = np.logspace(12,30,1000)
        sig = np.logspace(-12,6,999)
        mask = np.load('macro_constraints/saturn.npy')
        mask2 = self.resize(m,sig,mask)
        if not original:
            return mask2
        else:
            return m,sig,mask
    
    def skylab(self,original=False):
        m = np.load('macro_constraints/skylab_masses.npy')
        sig = np.load('macro_constraints/skylab_sigmas.npy')
        mask = np.load('macro_constraints/skylab_logFraction.npy')
        mask2 = self.resize(m,sig,10**mask.T)
        if not original:
            return mask2
        else:
            return m,sig,10**mask.T
    
    def ohya(self,original=False):
        m = np.load('macro_constraints/ohya_masses.npy')
        sig = np.load('macro_constraints/ohya_sigmas.npy')
        mask = np.load('macro_constraints/ohya_logFraction.npy')
        mask2 = self.resize(m,sig,10**mask.T)
        if not original:
            return mask2
        else:
            return m,sig,10**mask.T
        
    def WD(self,original=False):
        sig = np.load('macro_constraints/WDsigmas.npy')
        m = np.load('macro_constraints/WDmasses.npy')
        mask = np.load('macro_constraints/WD1COlims.npy')
        mask[mask==0]=99
        mask2 = self.resize(m,sig,mask)
        if not original:
            return mask2
        else:
            return m,sig,mask
    
    def fireballs(self,original=False):
        m = np.logspace(24,31,1000)
        sig = np.logspace(-6,4,999)
        mask = np.load('macro_constraints/fireballs.npy')
        mask2 = self.resize(m,sig,mask)
        if not original:
            return mask2
        else:
            return m,sig,mask
        
    def mica(self,original=False):
        data = np.load('macro_constraints/mica.npz')
        m = 10**data['log_masses']
        sig = 10**data['log_sigmas']
        N=data['N_events']
        N[N==0]=1/99
        mask = data['Nc']/N
        mask[mask>1]=99
        mask[-1,:]=99
        mask2 = self.resize(m,sig,mask)
        if not original:
            return mask2
        else:
            return m,sig,mask
        
    def xenon(self,original=False):
        m = np.logspace(0,16,1000)
        sig = np.logspace(-40,-22,999)
        mask = np.load('macro_constraints/xenon.npy')
        mask2 = self.resize(m,sig,mask)
        if not original:
            return mask2
        else:
            return m,sig,mask
        
    def deap(self,original=False):
        m = np.logspace(6,20,1000)
        sig = np.logspace(-24,-17,999)
        mask = np.load('macro_constraints/deap.npy')
        mask2 = self.resize(m,sig,mask)
        if not original:
            return mask2
        else:
            return m,sig,mask
        
    def dama(self,original=False):
        m = np.logspace(2,17,1000)
        sig = np.logspace(-30,-12,999)
        mask = np.load('macro_constraints/dama.npy')
        mask2 = self.resize(m,sig,mask)
        if not original:
            return mask2
        else:
            return m,sig,mask
        
    def chicago(self,original=False):
        m = np.logspace(4,13,1000)
        sig = np.logspace(-24,-14,999)
        mask = np.load('macro_constraints/chicago.npy')
        mask2 = self.resize(m,sig,mask)
        if not original:
            return mask2
        else:
            return m,sig,mask
        
    def cdms(self,original=False):
        m = np.logspace(0,15,1000)
        sig = np.logspace(-35,-21,999)
        mask = np.load('macro_constraints/cdms.npy')
        mask2 = self.resize(m,sig,mask)
        if not original:
            return mask2
        else:
            return m,sig,mask
    
    def gas(self,original=False):
        m = np.logspace(-6,51,1000)
        sig = np.logspace(-28,30,999)
        mask = np.load('macro_constraints/gas.npy')
        mask2 = self.resize(m,sig,mask)
        if not original:
            return mask2
        else:
            return m,sig,mask      
        
    def HSC(self,original=False):
        m = np.logspace(45,55,100)
        sig = np.logspace(-9,22,99)
        mask = np.load('macro_constraints/HSC26.npy')
        mask2 = self.resize(m,sig,mask)
        if not original:
            return mask2
        else:
            return m,sig,mask    
        
    def rg(self,original=False):
        m = np.logspace(16,21,1000)/self.gev2gram(1)
        sig = np.logspace(1,8,999)
        mask = np.load('macro_constraints/rg.npy')
        mask2 = self.resize(m,sig,mask)
        if not original:
            return mask2
        else:
            return m,sig,mask    
                
#%%
#call the class:
    
# a = macro()

