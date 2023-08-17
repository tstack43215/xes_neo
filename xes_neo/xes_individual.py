"""
Created 7/5/23
Created the file, left out set peak to manually set a peak and its
parameters, it seems non essential for now
-Evan Restuccia (evan@restuccias.com)
"""
from tkinter import N

from matplotlib.bezier import get_parallels
from .xes_fit import peak,background

class Individual():

    def __init__(self,backgrounds,peaks,pars_range={}):
        """
        Creates an individual with the given backgrounds and peaks

        Args:
            backgrounds (list): List of background types
            peaks (list): List of peak types
            pars_range (dict), optional): dictionary of the paras limits.
        """
        self.parsRange = pars_range
        #both peaks and backgrounds are arrays of strings which represent the type of the background of the peak/bkgn
        self.nPeaks = len(peaks)
        self.nBackgrounds = len(backgrounds)
        self.peakArr = [None]* self.nPeaks
        self.bkgnArr = [None] * self.nBackgrounds

        self.peakDict = {}

        """
        the Binding Energy needs to be personalized
        we take the range, which right now is something like (0,.2), i.e. the allowed variance in BE
        Then we add our binding energy to it to move the range to the right spot
        """
        if 'SVSC_shirley' in backgrounds:
            self.SVSC_toggle = True
        else:
            self.SVSC_toggle = False
        for i in range(pars_range['npeaks']):

            range_key = 'Peak Energy'
            guess_key = 'Peak Energy Guess'
            peak_energy_range = pars_range[range_key]
            PeakEnergy1,PeakEnergy2 =peak_energy_range[0],peak_energy_range[1]
            peak_energy = pars_range[guess_key][i]
            pars_range[range_key][0],pars_range[range_key][1] = [peak_energy_range[0] + peak_energy, peak_energy_range[1] + peak_energy]

            # pars_range['PeakEng_Scale'] =  [peak_energy_range[0] + peak_energy, peak_energy_range[1] + peak_energy]
            self.peakArr[i] = peak(pars_range,peaks[i])
            if self.SVSC_toggle:
                self.peakArr[i].SVSC_toggle(self.SVSC_toggle) #activate peak shirley


            pars_range[range_key][0],pars_range[range_key][1] = PeakEnergy1,PeakEnergy2
        '''
        except:
            #Special option if youre creating a custom individual(i.e. for analysis)
            if pars_range =='':
                pass
            else:
                print("Error modding guesses")
                exit()
        '''


        #each index in the peaks/background array is the name of the peak/background type to be used
        for i in range(self.nBackgrounds):
            self.bkgnArr[i] = background(pars_range,backgrounds[i])


        # TODO: change it to dictionary
        for i in range(self.nPeaks):
            self.peakDict[f'peak_{i}'] = self.peakArr[i]

        for i in range(self.nBackgrounds):
            self.peakDict[f'bkgn_{i}'] = self.bkgnArr[i]


    def add_peak(self,peakType):
        self.peakArr.append(peak(self.parsRange,peakType))

    def add_bkgn(self,bkgnType):
        self.bkgnArr.append(background(self.parsRange,bkgnType))


    #adds all backgrounds and peaks as one y value array
    def getFit(self,x,y):
        yFit = [0]*len(x)
        for i in range(self.nPeaks):
            if self.SVSC_toggle:
                peak_y,svsc_y = self.peakArr[i].getY(x)
                yFit += peak_y
                yFit += svsc_y
            else:
                yFit += self.peakArr[i].getY(x)
        for i in range(self.nBackgrounds):
            yFit += self.bkgnArr[i].getY(x,y)

        return yFit

    def get(self):
        """
        Get the whole set
        """
        return (self.peakArr + self.bkgnArr)

    def get_params(self,verbose_type=list):
        if verbose_type == list:
            params = []
            #fetches all the params as independent lists
            for i in range(len(self.peakArr)):
                params.append(self.peakArr[i].get())
            for i in range(len(self.bkgnArr)):
                params.append(self.bkgnArr[i].get())

            #puts it in one array
            for i in range(1,len(params)):
                for k in range(len(params[i])):
                    params[0].append(params[i][k])

            #print("Params : " + str(params[0]))
            return params[0]
        elif verbose_type == dict:
            params = {}
            for i in range(len(self.peakArr)):
                temp_dicts = self.peakArr[i].get(verbose_type=dict)
                params[f'peak_{i}'] = temp_dicts
            for i in range(len(self.bkgnArr)):
                temp_dicts = self.bkgnArr[i].get(verbose_type=dict)
                params[f'bkgn_{i}'] = temp_dicts
            return params
    def get_peak(self,i:int):
        """Get specific peaks

        Args:
            i (int): Index of the peak to get

        Returns:
            peaks: Peaks object
        """
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
        for bkgn in self.bkgnArr:
            bkgn.mutate(chance)

    def setPeak(self,i,param_arr) -> int:
        """Forces a given peak to have the given values, returns 0 on success, -1 on failure

        Note: param array comes in with its last element indicating its type

        Args:
            i (int): peak range
            param_arr (list): List of parameters with the last elements as the peak type
                i.e. [6000,200,0.5,1,'Voigt']

        Returns:
            int: Return code
        """

        peakType = param_arr[len(param_arr)-1]
        #if param_array is voigt, it comes in form [BE,Gauss,Lorentz,Amplitude,'Voigt']
        if peakType.lower() == 'voigt':
            self.peakArr[i].set_voigt(param_arr)
            return 0
        else:
            return -1

    def setPeaks(self,param_arr):
        """Set a bunch of peaks value

        Args:
            param_arr (List): list of peaks
        """
        self.peakArr = param_arr

    def setBkgn(self,i,param_arr):
        bkgnType = param_arr[len(param_arr)-1]
        #if param_array is voigt, it comes in form [BE,Gauss,Lorentz,Amplitude,'Voigt']
        if bkgnType.lower() == 'shirley-sherwood':
            self.bkgnArr[i].set_shirley_sherwood(param_arr)
        if bkgnType.lower() == 'linear':
            self.bkgnArr[i].set_linear(param_arr)

    def verbose(self):
        """Print out the populations
        """
        for i in range(self.npaths):
            self.Population[i].verbose()

    ''' Could be useful later I just dont want to write it
    def set_peak(self,i,A,h_f,m):
        self.Population[i].set(A,h_f,m)
    '''

    def __len__(self):
        """Returns the length of the independent parameters

        Returns:
            int: length of list of parameters
        """
        return len(self.get_params())

    def checkBound(self):
        """
        Check if the parameters are within the bounds
        """

        for i in self.peakArr:
            i.checkOutbound()


    def __getitem__(self,i):
        """Gets the ith peaks

        Args:
            i (int): index of the parameter

        Returns:
            float: value of the parameter
        """
        return self.peakArr[i]


    def getPeaksType(self):
        """Gets the type of the peaks

        Returns:
            list: list of the peak types
        """
        return [i.peakType for i in self.peakArr]

    def setParams(self,params,paramType=list):
        """Sets the parameters of the individual

        Args:
            params (list): list of parameters
        """
        # for i in range(len(self.peakArr)):
        #     self.peakArr[i].set(params[i],paramType=paramType)
        # for i in range(len(self.bkgnArr)):
        #     self.bkgArr[i].set(params[i],paramType=paramType)
        for i in self.peakDict:
            self.peakDict[i].set(params[i],paramType=paramType)

