"""
Author: Evan Restuccia (evan@restuccias.com)
"""
from turtle import back
from matplotlib import get_backend
import numpy as np
import scipy as scipy
from scipy import signal
import random


class peak():
    #Probably default these to -1 later as a test condition, but not yet sure
    def __init__(self,paramRange,peakType,singlet_or_doublet='singlet'):
        """
        takes in dictionary paramRange, with names of parameters and their corresponding allowed values
        takes the dictionary and sets the ranges for each
        Two types of params: Free and semi-free
        Free params get fully random values within their range
        Semi-Free take in a starting guess and modify it within an allowed range

        Then initalizes the peakType and hooks the proper function up to the correspond peakFunc

        Args:
            paramRange (dicts): Dictionary of the peak and background parameters and their allowed ranges
            peakType (list): list of peak types
            singlet_or_doublet (str, optional): singlet or doublet. Defaults to 'singlet'.
        """
        self.paramRange= paramRange
        self.gaussRange = np.arange(paramRange['Gaussian'][0],paramRange['Gaussian'][1],paramRange['Gaussian'][2])
        self.lorentzRange = np.arange(paramRange['Lorentzian'][0],paramRange['Lorentzian'][1],paramRange['Lorentzian'][2])
        self.bindingEnergyRange = np.arange(paramRange['Peak Energy'][0],paramRange['Peak Energy'][1],paramRange['Peak Energy'][2])
        self.ampRange = np.arange(paramRange['Amplitude'][0],paramRange['Amplitude'][1],paramRange['Amplitude'][2])
        self.asymmetryRange = np.arange(paramRange['Asymmetry'][0],paramRange['Asymmetry'][1],paramRange['Asymmetry'][2])
        try:
            self.s_o_splittingRange = paramRange['Spin-Orbit Splitting']
        except:
            self.s_o_splittingRange = [0,0,0]


        # Fully free within their range
        self.gaussian = np.random.choice(self.gaussRange)
        self.lorentz = np.random.choice(self.lorentzRange)
        self.amp = np.random.choice(self.ampRange)
        self.asymmetry = np.random.choice(self.asymmetryRange)


        #the range is a modifier on the input value
        self.bindingEnergy= np.random.choice(self.bindingEnergyRange)
        self.s_o_split = np.random.choice(self.s_o_splittingRange)

        self.peakType = peakType
        self.peak_y = []

        #singlet or doublet baby - not yet functional in voigt peak func
        self.singlet_or_doublet = singlet_or_doublet
        self.SVSC = False

        self.peakType = peakType
        if(self.peakType.lower() == "voigt"):
            self.func = self.voigtFunc
        elif(self.peakType.lower() == "double lorentzian"):
            self.func = self.doubleLorentzFunc

        else:
            print("Error assigning peak type")
            print("Peaktype found is: " + str(self.peakType))
            exit()

    def peakFunc(self,x):
        return self.func(x)

    def get(self,verbose_type=list):
        params = []
        # ANDY: Keep the old function of returning list...
        if verbose_type == list:
            if self.peakType.lower() == 'voigt':
                params = [self.bindingEnergy,self.gaussian,self.lorentz,self.amp] #mutate relies on the order here, so to change this you need to change mutate
            if self.peakType.lower() == 'double lorentzian':
                params = [self.bindingEnergy,self.gaussian,self.lorentz,self.amp,self.asymmetry]
                #params = [self.bindingEnergy,self.gaussian,self.lorentz,self.amp,self.asymmetry,self.peakType]
            params.append(self.peakType)

        elif verbose_type == dict:
            params = {}
            params['peakType'] = self.peakType.lower()
            if self.peakType.lower() == 'voigt':
                params['bindingEnergy'] = self.bindingEnergy
                params['gaussian'] = self.gaussian
                params['lorentz'] = self.lorentz
                params['amp'] = self.amp
            if self.peakType.lower() == 'double lorentzian':
                params['bindingEnergy'] = self.bindingEnergy
                params['gaussian'] = self.gaussian
                params['lorentz'] = self.lorentz
                params['amp'] = self.amp
                params['asymmetry'] = self.asymmetry
                #params = [self.bindingEnergy,self.gaussian,self.lorentz,self.amp,self.asymmetry,self.peakType]
        '''
        else:
            if len(params) == 0:
                print("Cant do 'def get' in peaks class in XPS_FIT, most likely a new peak was added and needs to be added to the get options")
                exit()
            else:
                return params
        '''
        return params

    def getGaussian(self):
        return self.gaussian

    def getLorenztian(self):
        return self.lorentz

    def getAmplitude(self):
        return self.amp
    def getAsymmetry(self):
        return self.asymmetry

    def getBindingEnergy(self) -> dict:
        return self.bindingEnergy

    def getParams(self):
        """Get the params of the peak, similar to `def get` but returns a dictionary instead of a list

        Raises:
            Exception: _description_

        Returns:
            dicts: dictionary of the parameters
        """
        temp_dicts = {}
        if self.peakType.lower() == 'voigt':
            temp_dicts['bindingEnergy'] = self.bindingEnergy
            temp_dicts['gaussian'] = self.gaussian
            temp_dicts['lorentz'] = self.lorentz
            temp_dicts['amp'] = self.amp
            temp_dicts['peakType'] = self.peakType

            return temp_dicts
        elif self.SVSC:
            SVSC_params = self.SVSC_background.getParams()
            for key in SVSC_params:
                temp_dicts[key] = SVSC_params[key]
        else:
            raise Exception("Cant do 'def getParams' in peaks class in XPS_FIT, most likely a new peak was added and needs to be added to the get options")

        return temp_dicts

    def getY(self,x):
        self.peakFunc(x)
        if self.SVSC:
            SVSC_vals = self.SVSC_background.getY(x,self.peak_y)
            return self.peak_y,SVSC_vals
        return self.peak_y

    def SVSC_toggle(self,boolVal):
        self.SVSC = boolVal
        self.SVSC_background = background(self.paramRange,'SVSC_shirley')
    '''
    def set(self,BE,gauss,lorentz,amp):
        self.bindingEnergy = BE
        self.gaussian = gauss
        self.lorentz = lorentz
        self.amp = amp
    '''

    def set(self,params,paramType=list):
        """
        This is the main differentiator between the fit in gui and xpsfolders
        The set function takes in a list of params, and should expect them to come in the same order that they were pushed out
        in the get function.

        """
        if(self.peakType.lower() == "voigt"):
            self.set_voigt(params,paramType = paramType)
        elif(self.peakType.lower() == "double lorentzian"):
            self.set_doubleLorentz(params,paramType = paramType)


    def setGaussian(self,newVal):
        self.gaussian = newVal
    def setLorentzian(self,newVal):
        self.lorentz = newVal

    def setAmplitude(self,newVal):
        self.amp = newVal
    def setAsymmetry(self,newVal):
        self.asymmetry = newVal
    def setBindingEnergy(self,newVal):
        self.bindingEnergy = newVal

    def set_voigt(self,paramList,paramType=list):
        if paramType == list:
            self.bindingEnergy = paramList[0]
            self.gaussian = paramList[1]
            self.lorentz = paramList[2]
            self.amp = paramList[3]
        elif paramType == dict:
            self.bindingEnergy = paramList['bindingEnergy']
            self.gaussian = paramList['gaussian']
            self.lorentz = paramList['lorentz']
            self.amp = paramList['amp']

    def set_doubleLorentz(self,paramList,paramType=list):
        if paramType == list:
            self.bindingEnergy = paramList[0]
            self.gaussian = paramList[1]
            self.lorentz = paramList[2]
            self.amp = paramList[3]
            self.asymmetry = paramList[4]
        elif paramType == dict:
            self.bindingEnergy = paramList['bindingEnergy']
            self.gaussian = paramList['gaussian']
            self.lorentz = paramList['lorentz']
            self.amp = paramList['amp']
            self.asymmetry = paramList['asymmetry']



    def mutate(self,chance):
        self.mutateGauss(chance)
        self.mutateAmplitude(chance)
        self.mutateBE(chance)
        self.mutateLorentz(chance)
        self.mutateSplitting(chance)
        self.mutateAsymmetry(chance)

    def mutateGauss(self,chance):
        if random.random()*100 < chance:
            self.gaussian = np.random.choice(self.gaussRange)

    def mutateLorentz(self,chance):
        if random.random()*100 < chance:
            self.lorentz = np.random.choice(self.lorentzRange)

    def mutateAmplitude(self,chance):
        if random.random()*100 < chance:
            self.amp = np.random.choice(self.ampRange)

    def mutateBE(self,chance):
        if random.random()*100 < chance:
            self.bindingEnergy = np.random.choice(self.bindingEnergyRange)

    def mutateSplitting(self,chance):
        if random.random()*100 < chance:
            self.s_o_split = np.random.choice(self.s_o_splittingRange)

    def mutateAsymmetry(self,chance):
        if random.random()*100 < chance:
            self.asymmetry = np.random.choice(self.asymmetryRange)


    def checkOutbound(self):
        """Check if out of bounds

        Raises:
            Exception: SVSC not implemented for checkforBound
        """
        if self.peakType.lower() == 'voigt':
            peakEnergy_min = np.min(self.paramRange['Peak Energy Guess'])
            peakEnergy_max = np.max(self.paramRange['Peak Energy Guess'])

            self.bindingEnergy = np.clip(self.bindingEnergy,peakEnergy_min + self.paramRange['Peak Energy'][0],peakEnergy_max + self.paramRange['Peak Energy'][1])
            self.gaussian = np.clip(self.gaussian,self.paramRange['Gaussian'][0],self.paramRange['Gaussian'][1])
            self.lorentz = np.clip(self.lorentz,self.paramRange['Lorentzian'][0],self.paramRange['Lorentzian'][1])
            self.amp = np.clip(self.amp,self.paramRange['Amplitude'][0],self.paramRange['Amplitude'][1])

        elif self.peakType.lower() == 'double lorentzian':
            peakEnergy_min = np.min(self.paramRange['Peak Energy Guess'])
            peakEnergy_max = np.max(self.paramRange['Peak Energy Guess'])

            self.bindingEnergy = np.clip(self.bindingEnergy,peakEnergy_min + self.paramRange['Peak Energy'][0],peakEnergy_max + self.paramRange['Peak Energy'][1])
            self.gaussian = np.clip(self.gaussian,self.paramRange['Gaussian'][0],self.paramRange['Gaussian'][1])
            self.lorentz = np.clip(self.lorentz,self.paramRange['Lorentzian'][0],self.paramRange['Lorentzian'][1])
            self.asymmetry = np.clip(self.asymmetry,self.paramRange['Asymmetry'][0],self.paramRange['Asymmetry'][1])
            self.amp = np.clip(self.amp,self.paramRange['Amplitude'][0],self.paramRange['Amplitude'][1])
        else:
            raise Exception ('SVSC not implemented for checkforBound')



    def voigtFunc(self,x):
        """
        Calculate the Voigt lineshape with variable peak position and intensity using a convolution of a Gaussian and a Lorentzian distribution.

        Peaks Parameters:
            self.bindingEnergy (float): The position of the peak.
            self.gaussian (float): The standard deviation of the Gaussian distribution.
            self.lorentz (float): The full width at half maximum (FWHM) of the Lorentzian distribution.
            self.amp (float): The amplitude of the peak.


        Args:
            x (list): The input array of independent variables.

        Returns:
            list: The output array of voigt values.
        """

        if self.gaussian == 0:
            self.gaussian = .01

        gaussian = np.exp(-np.power(x - self.bindingEnergy, 2) / (2 * np.power(self.gaussian, 2))) / (self.gaussian * np.sqrt(2 * np.pi))


        # Calculate the Lorentzian component
        # TODO: need asserations to make sure the stepsize is the same across the array
        stepSize = x[2]-x[1]

        # energy_step = x[1] - x[0]
        decimal_places = 2
        step = round(stepSize, decimal_places)
        energy_min = min(x)
        energy_max = max(x)
        lorentzian_range =  energy_max - energy_min

        #Check whether the energy range is odd or even because it affects the lorentzian x-value array cutoff values

        #if odd
        if (lorentzian_range % 0.002 == 0):
            lorentzian_x_min = -(lorentzian_range/2)
            lorentzian_x_max = (lorentzian_range/2) + step
        #else even
        else:
            lorentzian_x_min = -(lorentzian_range/2) + (0.5*step)
            lorentzian_x_max = (lorentzian_range/2) + (0.5*step)

        z = np.arange(lorentzian_x_min, lorentzian_x_max, step)
        # z = np.arange(-xRange, xRange+step,step)

        lorentzian = (self.lorentz / (2 * np.pi)) / (np.power(z, 2) + np.power(self.lorentz / 2, 2))

        # Perform the convolution using the Fourier transform
        convolve = scipy.signal.convolve(gaussian, lorentzian, mode = 'same')
        #convolve = scipy.signal.fftconvolve(gaussian, lorentzian, mode = 'same')
        #convolve = scipy.signal.oaconvolve(gaussian, lorentzian, mode = 'full')
        #write function to center x points at 0
        #6460-6505

        #convolution = np.real(np.fft.ifft(np.fft.fft(gaussian) * np.fft.fft(lorentzian)))

        # TODO: Scale the intensity of the peak, does automatically right now based on center, this will need to be changed
        voigt = convolve * self.amp

        #returns, but also updates the yValues of the fit to improve efficiency, we can call that instead of recalculating every time
        self.peak_y = voigt
        return voigt


    def doubleLorentzFunc(self,x):
        #Same as voigt but with an added asymmetry factor to the width (lorentz)in the lorentzian equation
        global bindingEnergy
        # Calculate the Gaussian component
        if self.gaussian == 0:
            self.gaussian = .01


        data_range= max(x) - min(x)
        data_range /= 2

        middle = min(x)+data_range
        offset = self.bindingEnergy-middle
        num_points = len(x)
        x_values, dx = np.linspace(-data_range,data_range,num_points,retstep=True)

        #gaussian = np.exp(-np.power(x - self.bindingEnergy, 2) / (2 * np.power(self.gaussian, 2))) / (self.gaussian * np.sqrt(2 * np.pi))
        gaussian = np.exp(-np.power(x_values, 2) / ((np.power(self.gaussian, 2)))) / (self.gaussian * np.sqrt(np.pi)) #Took away 2*np.power(self.gaussian,2) and np.sqrt(2*np.pi)

        # Calculate the Lorentzian component

        #Added +offset to all Lorentzian functions instead of Gauss --> It has to be inside the equation to make the leftside of the peak more lorentzian
        numP = len(x)
        HWHM = self.lorentz/2
        if HWHM == 0:
            HWHM = .01

        #print("The asymmetry is:", self.asymmetry)

        lorentzLeft = HWHM*self.asymmetry #Width of left side of peak due to asymmetry
        #z = np.arange(-xRange, xRange+stepSize,.05)
        yDoubleL = [0]*numP
        #Double Lorentzian formula taken from Aanalyzer code in PUnit1 line 7157
        for i in np.arange(1, numP, 1):
            if x[i] < self.bindingEnergy:
                yDoubleL[i] = 1 / ( 1 + np.power( (x_values[i] - offset)/lorentzLeft, 2 ) ) / np.pi
            else:
                yDoubleL[i] = 1 / ( 1 + np.power( (x_values[i] - offset)/HWHM, 2 ) ) / np.pi

            #yDoubleL[2*i] = 0;
        lorentzian = yDoubleL


        #lorentzian = (self.amp * self.asymmetry / (2 * np.pi)) / (np.power(z, 2) + np.power(self.lorentz * self.asymmetry / 2, 2)) #Double Lorentzian

        # Perform the convolution using the Fourier transform
        convolve = scipy.signal.convolve(gaussian,lorentzian,'same')

        #normalize the height so that intensity is the height of the max of the peak
        #scale = max(doubleLorentz)
        #for i in range(len(doubleLorentz)):
            #doubleLorentz[i] *= (self.amp/scale)

        doubleLorentz = convolve * self.amp
        self.peak_y = doubleLorentz

        return doubleLorentz




class background():

    def __init__(self,paramRange,bkgnType):
        self.bkgnType = bkgnType

        if self.bkgnType == 'Shirley-Sherwood':
            self.bkgn = self.shirley_Sherwood
            #self.bkgn = self.better_shirley

            try:
                k_range = np.arange(paramRange['k_range'][0],paramRange['k_range'][1],paramRange['k_range'][2])
                self.k = np.random.choice(k_range)
            except:
                self.k = -1
        elif self.bkgnType.lower() == 'linear':
            self.bkgn = self.linear_background
            self.backgroundRange = np.arange(paramRange['Background'][0],paramRange['Background'][1],paramRange['Background'][2])
            self.slopeRange = np.arange(paramRange['Slope'][0],paramRange['Slope'][1],paramRange['Slope'][2])

            #self.background is the b value in y = mx+b
            self.background = np.random.choice(self.backgroundRange)
            #self.slope = np.random.choice(self.slopeRange)
            self.slope = 0
        elif self.bkgnType == 'Exponential':
            self.bkgn = self.exponential_bkgn
        elif self.bkgnType == 'SVSC_shirley':
            self.bkgn = self.SVSC_shirley
            #func y vals tells you what to integrate over
            try:
                k_range = np.arange(paramRange['k_range'][0],paramRange['k_range'][1],paramRange['k_range'][2])
                self.k = np.random.choice(k_range)
            except:
                self.k = -1

        else:
            print("Error Choosing Background in init of xes_fit")
            print("Background read as: " + str(self.bkgnType))
            exit()
        self.yBkgn = []

    def getY(self,x,y):
        self.get_Background(x,y)
        return self.yBkgn
    def get_Background(self,x,y):
        self.bkgn(x,y)


    def mutate(self,chance):
        self.mutate_background_val(chance)

    def mutate_background_val(self,chance):
        if random.random()*100 < chance:
        	#print(self.background)
            self.background = np.random.choice(self.backgroundRange)
            #print(self.background)


    #Make sure to add in each background here
    def get(self,verbose_type=list):
        if verbose_type == list:
            if self.bkgnType == 'Shirley-Sherwood' or self.bkgnType == 'SVSC_shirley':
                return [self.k, self.bkgnType]
            elif self.bkgnType.lower() == 'linear':
                return [self.background,self.bkgnType,self.slope]
            elif self.bkgnType == 'Exponential':
                return [self.bkgnType]
        if verbose_type == dict:
            params = {}
            params['bkgnType'] = self.bkgnType
            if self.bkgnType.lower() == 'linear':
                params['background'] = self.background
                params['slope'] = self.slope
            return params
        # elif verbose_type == dict:
        #     if self.bkgnType == 'Shirley-Sherwood' or self.bkgnType == 'SVSC_shirley':
        #         return []
        #     elif self.bkgnType.lower() == 'linear':
        #         return [self.background,self.bkgnType]
        #     elif self.bkgnType == 'Exponential':
        #         return [self.bkgnType]

    def getParams(self):
        temp_dict = {}
        temp_dict['type'] = self.bkgnType
        if self.bkgnType == 'Shirley-Sherwood' or self.bkgnType == 'SVSC_shirley':
            temp_dict['k'] = self.k
        elif self.bkgnType.lower() == 'linear':
            temp_dict['background'] = self.background
        elif self.bkgnType == 'Exponential':
            # return [self.bkgnType]
            pass
        return temp_dict


    def set(self,params,paramType=list):
        if self.bkgnType.lower() == 'Shirley-Sherwood' or self.bkgnType.lower() == 'SVSC_shirley':
            self.set_shirley_sherwood(params,paramType)
        elif self.bkgnType.lower() == 'linear':
            self.set_linear(params,paramType)
        elif self.bkgnType == 'Exponential':
            pass

    def getType(self):
        return self.bkgnType

    def set_k(self,newVal):
        self.k = newVal

    def set_shirley_sherwood(self,params,paramType):
        if paramType == 'list':
            self.k = params[0]
        elif paramType == 'dict':
            self.k = params['k']

    def set_linear(self,paramArr,paramType):
        if paramType == 'list':
            self.background = paramArr[0]
            self.slope = paramArr[1]
        # self.background = paramArr[0]
        elif paramType == 'dict':
            self.bkgnType = paramArr['bkgnType']
            self.background = paramArr['background']
            self.slope = paramArr['slope']


    def exponential_bkgn(self,x,y):
        self.y = y
        self.x = x
        #need to make these parameters outside background functions
        numPeaksUsed = 1
        maxPeaks = 1
        numberBackgrounds = 1
        ma = maxPeaks + numberBackgrounds

        numP = len(y)
        funcs =[0]*len(y)

        #Taken from Aanalyzer code --> not sure why exponent is initially set to 1 or 0
        exponent = 1
        deltaExponent = max(abs(exponent / 100), 0.001)
        exponent += deltaExponent


        ma += 1  # create exponential
        #Not sure if the x data needs to be flipped for BE instead of KE --> The exponential should be on the left side of the peak not the right
        for j in range(1, numP): #Cut off before numP so the end point is off. Need to fix this in order to scale down the righthand side of the background to the data
            gar = -exponent * (x[j] - x[numP // 2])
            if gar > 30:
                gar = 30
            elif gar < -30:
                gar = -30
            funcs[j] = -(np.exp(gar)) #Added negative sign to flip exponential to be in the -xy plane instead of +xy plane

        self.yBkgn = funcs
        return self.yBkgn






     #Integral slope background works for now but is bad. Left side of data is not scaling properly
    def linear_background(self,x,y):
        self.y = y
        self.x = x
        num_points = len(self.x)
        self.yBkgn = [0]*num_points

        #slope = (self.y[-1] - self.y[0])/(self.x[-1] - self.x[0])
        for i in range(num_points):
            self.yBkgn[i] = self.linear(self.slope,self.x[i],self.background)
        #print(self.yBkgn)
        return self.yBkgn

    def linear(self,slope,x,b):
        return (slope*x)+b

    def better_shirley(self, x, y):
        #Not sure where this code came from. May need to cite it later. It has similar structure to the shirley formula described in Herrera's paper
        self.y = y
        self.x = x
        E = x
        J = y


        def integralOne(E, J, B, E1=0, E2=-1):
            integral = []
            if E2 < 0:
                E2 = len(J) + E2
            integral = sum([J[n] - B[n] for n in range(E,E2)])
            return integral

        def integralTwo(E, I, B, E1=0, E2=-1):
            integral = []
            if E2 < 0:
                E2 = len(I) + E2
            integral = sum([I[n] - B[n] for n in range(E1,E2)])
            return integral

        def getBn(E,I,B,E1=0,E2=-1):
            I2 = I[E2]
            I1 = I[E1]
            value = I2 + (I1 - I2)/(integralTwo(E,I,B,E1,E2))*integralOne(E,I,B,E1,E2)
            return value

        def iterateOnce(I,B,E1=0,E2=-1):
            b = [getBn(E,I,B,E1,E2) for E in range(len(I))]
            return b

        Bn = [0 for i in range(len(J))]
        Bn = iterateOnce(J,Bn)
        for i in range(6): #how many iterations it's doing
            B_temp = Bn
            Bn = iterateOnce(J,Bn)
            B_diff = [Bn[j] - B_temp[j] for j in range(len(Bn))] #Could make a check to see if the iterations are getting better. Usually little difference after 7 iterations

        self.yBkgn = Bn


        return self.yBkgn









    def SVSC_shirley(self, x, y):

        self.y = y
        self.x = x
        #Should probably declare these items outside of each background type
        numPeaksUsed = 1
        maxPeaks = 1
        numberBackgrounds = 1
        ma = maxPeaks + numberBackgrounds #will need to change to make it able to use multiple peaks/backgrounds
        numP = len(self.y)
        a =[0]*len(self.x) #dont know if we need a anymore?

        #voigt = peak.voigt
        funcs = y #len = numP -1 #recover initial peak curve --> How to get y points of just one peak??? Use BE range + some delta
        backgroundFromPeakShirleyFix = [0]*(len(self.y)-1) #not sure why its one less than the number of points
        SVSC_bkgn = backgroundFromPeakShirleyFix #easier to write --> original name comes from aanalyzer code

        a_old = 0.3
        a_new = 0.5 #are these initial values too large?
        old_fit = 10000
        best_fit = funcs #setting initial best fit --> just equal to y originally
        SVSC_diff = 1
        while a_new >= 0: #Iterates until a = 0, but keeps track of std of background to voigt fit. Need to find a better way for the GA to optimize a
            i = 1
            for i in range(maxPeaks):#calculates background for each peak then iterates
                #a_ratio is some parameter ratio --> I think it is the ratio of one parameter of different correlated peaks, unsure as to which parameter is being correlated
                a_ratio_b4 = a_old #Right now these are just random --> real code: a[ peakShirleyma[ peakShirelyCorrTo] ] / a[ mama[peakShirelyCorrTo] ]
                a_ratio_after = a_new #defined on line 15233 in PUnit1 --> Values are a[] before and after lfitmod is called
                peakShirleyBackground = 0.8*a_ratio_b4 + 0.2*a_ratio_after #I think this is supposed to be the scattering factor? Now sure how it is optimized
                #Maybe for now we should treat peakShirleyBackground as the scattering factor?

                for j in np.arange(numP -2, 0, -1):
                    SVSC_bkgn[j-1] = self.y[j-1]*-(self.x[j+1]-self.x[j])*peakShirleyBackground + SVSC_bkgn[j] #isnt this just what we already had but now with a wider range?
                    funcs[j] += SVSC_bkgn[j-1]
                #should write array in here to store each peak curves background --> will sum these up later
                i +=1

                iteration_diff = np.subtract(voigt, funcs) #need to change voigt to whatever the curve fit y array is
                new_fit = np.std(iteration_diff)
                new_fit_array = funcs
                if new_fit < old_fit:
                    old_fit = new_fit
                    best_fit = new_fit_array
                a_old = a_new
                a_new -= 0.01 #slow decrease for now --> NEED TO FIND BETTER WAY TO OPTIMIZE a_new
                #lfitmod caluculated here --> Calcualtes parameters between iterations: This is what makes the background active
                #Should we call class Peak here to recalculate the fit with the new background? Active curve fitting

        funcs = best_fit
        #return funcs #Not sure how we are calling this (self.yBKgn?)
        self.yBkgn = funcs

        return self.yBkgn





    def shirley_Sherwood(self,x,y):
        #too lazy to find all the x and ys and get rid of the self
        self.y = y
        self.x = x
        #need to make these parameters outside background functions
        numPeaksUsed = 1
        maxPeaks = 1
        numberBackgrounds = 1
        ma = maxPeaks + numberBackgrounds
        useIntegralBkgn=True
        numP = [0]*len(self.y) #we are using this as an array right now but it should just be the number of data points
        a =[0]*len(self.x)


        def iterations(self,x,y):
            numPeaksUsed = 1
            maxPeaks = 1
            numberBackgrounds = 1
            ma = maxPeaks + numberBackgrounds
            useIntegralBkgn=True
            numP = [0]*len(self.y) #we are using this as an array right now but it should just be the number of data points
            a =[0]*len(self.x)
            #need this to find the correct data points in which the bakcground will be removed
            numPointsAroundBackgroundLimitsLocal = 5
            nRightLocal = numPointsAroundBackgroundLimitsLocal // 2
            nLeftLocal = numPointsAroundBackgroundLimitsLocal // 2

            yRightLocal = 0
            yLeftLocal = 0

            for j in range(-(numPointsAroundBackgroundLimitsLocal // 2), numPointsAroundBackgroundLimitsLocal // 2 + 1):
                #yRightLocal += datos[dataNumber].ModifiedCurve.y[nRightLocal + j]
                #yLeftLocal += datos[dataNumber].ModifiedCurve.y[nLeftLocal + j]
                yRightLocal += self.y[len(self.y) - nRightLocal-1 + j]
                yLeftLocal += self.y[nLeftLocal + j]

            yLeftLocal /= (numPointsAroundBackgroundLimitsLocal // 2) * 2 + 1
            yRightLocal /= (numPointsAroundBackgroundLimitsLocal // 2) * 2 + 1



            nLeft = nLeftLocal
            nRight = len(x)-nRightLocal
            global yRight
            yRight = yRightLocal
            yLeft = yLeftLocal


            iterationsIntegralBkgn = 6
            BkgdCurve = []
            funcs = numP
            #funcs = np.zeros((ma, numP))

            if useIntegralBkgn: #from Aanalyzer code line 14867
                ma += 1 #Need to add in array to store each peak background peak[ma] --> sum in other backgrounds? peak[ma] += funcs...
                #funcs[ma][numP] = 0
                #print("K is " + str(self.k))
                for j in range(nRight-1, nLeft-1, -1):
                    #funcs[ma][j] = (self.y[j] - yRight[j]) * (self.x[j+1] - self.x[j]) + funcs[ma][j+1]
                    #print(self.y[j] - yRight)
                    funcs[j] = (self.y[j] - yRight) *self.k* -(self.x[j+1] - self.x[j]) + funcs[j+1] #assumes x is in KE, not sure if that changes anything


                for j in range(0, nLeft):
                    #funcs[ma][j] = funcs[ma][nLeft]
                    funcs[j] = yLeft-yRight

                integralma = ma

            '''
            #iterates shirley background
            if useIntegralBkgn: #from Aanalyzer code line 15140
                for l in range(iterationsIntegralBkgn):
                    for j in range(nRight-1, nLeft, -1):
                        #funcs[integralma][j] = (self.y[j] - yRight[j] - a[integralma] * funcs[integralma][j]) * (self.x[j+1] - self.x[j]) + funcs[integralma][j+1]
                        funcs[j] = (self.y[j] - yRight - funcs[j]) * (self.x[j+1] - self.x[j]) + funcs[j+1]
                    for j in range(1, nLeft):
                        #funcs[integralma][j] = funcs[integralma][nLeft]
                        funcs[j] = funcs[nLeft]
                        #calls lfitmod here -->calculates chisq and deletes all parameters

                    l += 1
            '''
            return funcs
        for i in range(6): #How many iterations it is performing
            funcs = iterations(self,x,y)
        self.yBkgn = funcs
        for i in range(len(self.yBkgn)):
            self.yBkgn[i] += yRight

        return self.yBkgn
    '''
    Just barely started on peak shirley, commented out so it wont cause a compilation error
    def peak_shirley(self,x,y,peak):
        peak.getY
    '''


