"""
Created 7/5/23
File is a copy of xps_individual in the GA folder, but with some modifications to make it work with xps analysis file:
-Evan Restuccia (evan@restuccias.com)
"""
from tkinter import N

from matplotlib.bezier import get_parallels
from xes_fit import peak,background

class Individual():
    def __init__(self,backgrounds,peaks,pars_range=''):
        """
        backgrounds (array) Array where each element is the name of the background type desired
        peaks (array) array where each element is the name of the desired peakType
        pars_range (dict) each key is the name of a parameter with a tuple that contains the range the parameter is allowed to explore
        """

        self.pars_range = pars_range
        #both peaks and backgrounds are arrays of strings which represent the type of the background of the peak/bkgn
        self.nPeaks = len(peaks)
        self.nBackgrounds = len(backgrounds)
        self.peakArr = [None]* self.nPeaks
        self.bkgnArr = [None] * self.nBackgrounds
        
        """
        the Binding Energy needs to be personalized
        we take the range, which right now is something like (0,.2), i.e. the allowed variance in BE
        Then we add our binding energy to it to move the range to the right spot
        """
        try:
            peak_energy_range = pars_range['Peak Energy']
            PeakEnergy1,PeakEnergy2 =peak_energy_range[0],peak_energy_range[1]
            binding_energy = pars_range['Peak Energy Guess']
            pars_range['Peak Energy'][0],pars_range['Peak Energy'][1] = [peak_energy_range[0] + binding_energy, peak_energy_range[1] + binding_energy,]
        except:
            #Special option if youre creating a custom individual(i.e. for analysis)
            if pars_range =='':
                pass
        
        #each index in the peaks/background array is the name of the peak/background type to be used
        for i in range(self.nPeaks):
            self.peakArr[i] = peak(pars_range,peaks[i])
        for i in range(self.nBackgrounds):
            self.bkgnArr[i] = background(pars_range,backgrounds[i])
        
        print(self.peakArr)
        if pars_range != '':
            pars_range['Peak Energy'][0],pars_range['Peak Energy'][1] = PeakEnergy1,PeakEnergy2
        



    def add_peak(self,peakType):
        self.peakArr.append(peak(self.pars_range,peakType))
    def add_bkgn(self,bkgnType):
        self.bkgnArr.append(background(self.pars_range,bkgnType))

    
    #adds all backgrounds and peaks as one y value array
    def getFit(self,x,y):
        yFit = [0]*len(x)
        print(self.peakArr)
        for i in range(self.nPeaks):
            yFit += self.peakArr[i].peakFunc(x)
        for i in range(self.nBackgrounds):
            yFit += self.bkgnArr[i].getY(x,y)
        
        return yFit

    def getFitWithComponents(self,x,y):
        yFit = [0]*len(x)
        print(self.peakArr)
        bkgn_components_arr = []
        peak_components_arr = []
        for i in range(self.nPeaks):
            peakComp =  self.peakArr[i].peakFunc(x)
            yFit += peakComp
            peak_components_arr.append(peakComp)
        for i in range(self.nBackgrounds):
            bkgnComp = self.bkgnArr[i].getY(x,y)
            yFit += bkgnComp
            bkgn_components_arr.append(bkgnComp)
        
        #inefficient but only needs to work a few times so it should be fine
        for i in range(len(peak_components_arr)):
            for l in range(len(bkgn_components_arr)):
                for k in range(len(x)):
                    peak_components_arr[i][k] += bkgn_components_arr[l][k]
        return yFit,peak_components_arr,bkgn_components_arr

    def get(self):
        """
        Get the whole set
        """
        return (self.peakArr + self.bkgnArr)
    
    def get_params(self):
        params = []
        for i in range(len(self.peakArr)):
            params.append(self.peakArr[i].get())
        for i in range(len(self.bkgnArr)):
            params.append(self.bkgnArr[i].get())
        return params

    def get_peak(self,i):
        return self.peakArr[i].get()
    
    def get_peaks(self):
        return self.peakArr
    
    def get_background(self,i):
        return self.bkgnArr[i]
    def get_backgrounds(self):
        return self.bkgnArr

    def mutate_(self,chance):
        for peak in self.peakArr:
            peak.mutate(chance)
    
    def setPeak(self,i,BE,gauss,lorentz,amp):
        self.peakArr[i].set(BE,gauss,lorentz,amp)
        


    def verbose(self):
        """
        Print out the Populations
        """
        for i in range(self.npaths):
            self.Population[i].verbose()

    ''' Could be useful later I just dont want to write it
    def set_peak(self,i,A,h_f,m):
        self.Population[i].set(A,h_f,m)
    '''

