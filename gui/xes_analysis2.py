import sys,glob,re,os
import numpy as np
import fnmatch
import xes_data
import matplotlib.pyplot as plt
import copy
from xes_individual import *
from xes_fit import *

"""
Author: Evan Restuccia evan@restuccias.com
"""


def sort_fold_list(dirs):
    fold_list = list_dirs(dirs)
    fold_list.sort(key=natural_keys)
    return fold_list
## Human Sort
def list_dirs(path):
    return [os.path.basename(x) for x in filter(
        os.path.isdir, glob.glob(os.path.join(path, '*')))]


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

class xes_analysis():
    def __init__(self,dirs,params):
        self.dirs = dirs
        fileName = params['fileName']
        self.bkgns = params['bkgns']     
        self.peaks = params['peaks']  
        self.peak_options = self.peaks
        for i in range(len(self.peaks)):
            self.peak_options[i] = self.peak_options[i].lower()
        #self.numPeaks = len(self.peaks)
        for i in range(len(self.bkgns)):
            self.bkgns[i] = self.bkgns[i].lower()
        '''
        for i in range(len(self.peaks)):
            self.peaks[i] = self.peaks[i].lower()
        '''

        self.data_obj = params['data obj']

            
        
        self.x = self.data_obj.get_x()
        self.y = self.data_obj.get_y()
        self.title = 'Fit'
        self.xlabel = 'Energy (eV)'
        self.ylabel= 'Counts/s'

        #number of parameters that the GA fit
        #self.number_of_parameters = self.calculate_number_of_params(self.peaks,self.bkgns)

    def get_param_num(self,type):
        """
        (helper) takes in a peak/background type and matches it with the number of parameters it has
        always has the number of parametes + 1, the +1 is for it's type
        """
        if type == 'voigt':
        	print("I selected Voigt")
        	return 5
        if type == 'double lorentzian':
        	print("I selected Double Lorentzian")
        	return 6
        if type == 'shirley-sherwood':
            return 2
        if type == 'exponential':
            return 1
        if type == 'linear':
            return 2
        if type == 'peak-shirley':
            return len(self.peaks)+1
        else:
            print("Warning, need to add parameter number to get_param_num in analysis file")

    def calculate_number_of_params(self,types):
        """
        peaks(array) array containing strings which represent the type of each peak, length is equal to the number of peaks
        bkgns(array) length is number of backgrounds, each element is a string representing the type of background

        (helper)
        Adds the number of parameters each type has together so the analysis will know how many columns to have in arrays of the parameters
        """
        
        num = 0
        for i,one_type in enumerate(types):
            #voigt has 4 params
            num += self.get_param_num(one_type.lower().strip())

        
        return num
    
    def extract_data(self,plot_err=True,passin=''):
        """
        Extract data value using array data

        plot_err(bool) whether or not to plot the error
        passin(num?) Optional- the best fit
        """
        #gets the parameters and best fit if they are not passed in
        if passin == '':
            full_param_list = []
            files = [] 
            for r, d, f in os.walk(self.dirs):
                f.sort(key = natural_keys)
                for file in f:
                    if fnmatch.fnmatch(file,'*_data.csv'):
                        files.append(os.path.join(r, file))
            files.sort(key=natural_keys)

            for i in range(len(files)):
                final_values,self.types = self.extract_from_result_file(files[i])
                full_param_list.append(final_values)
                if i == 0:
                    stack = full_param_list[i]
                stack = np.vstack((stack,full_param_list[i]))

            for i in range(len(self.types)):
                self.types[i]= self.types[i].strip()
            
            self.number_of_parameters = self.calculate_number_of_params(self.types)
            #print(full_param_list)
            self.full_matrix = stack
            #print(stack)
            bestFit,err = self.construct_bestfit_err_mat(stack,plot_err)
            best_Fit = np.reshape(copy.deepcopy(bestFit),(1, -1))
            #print(best_Fit)
        else:
            err = np.zeros([self.number_of_parameters,1])
            bestFit = np.zeros([self.number_of_parameters,1])
            best_Fit = passin
        
        self.bestFit = best_Fit.flatten()
        self.err = err
        #should print error here later, but need to customize it to number of params


        self.bestFit_mat = best_Fit

        # print("Not Normalized:")
        self.best_Fit_n = copy.deepcopy(best_Fit)
        #print(self.best_Fit_n)
        # self.best_Fit_n[:,0] -= self.scaler.min_
        # self.best_Fit_n[:,0] /= self.scaler.scale_
        # print(self.best_Fit_n)
        # self.x_model = np.arange
        self.best_ind = self.create_individual_from_bestfit(self.best_Fit_n[0])
        print("peak attributes read are:")
        #print(self.best_Fit_n)
        print(self.best_ind.get_params())

        print(self.best_ind.getFit(self.x,self.y))
        self.y_model,self.peak_components,self.bkgn_components = self.best_ind.getFitWithComponents(self.x,self.y)
        '''
        self.y_model_components = self.best_ind.get_peaks()
        self.bkgn_components = self.best_ind.get_backgrounds()
        self.totalbackground = [0] * len(self.x)
        for i in range(len(self.bkgn_components)):
            self.bkgn_components[i] = self.bkgn_components[i].getY(self.x,self.y)
            for k in range(len(self.x)):
                self.totalbackground[k] += self.bkgn_components[i][k]

        for i in range(len(self.y_model_components)):
            self.y_model_components[i] = self.y_model_components[i].getY(self.x) + self.totalbackground
        '''
        
    def extract_from_result_file(self,file):
        try:
            os.path.exists(file)

            csv_unflatten = []
            csv_numbers = []
            gen_csv = np.genfromtxt(file,delimiter=',',dtype = None,encoding= None)
            for row in gen_csv:
                temp_row = []
                temp_numbers = []
                types = []
                for k in range(len(row)):
                    try:
                        temp_numbers.append(np.float64(row[k]))
                    except:
                        types.append(row[k])
                    temp_row.append(row[k])
                if len(csv_unflatten) == 0:
                        csv_unflatten = temp_row
                        csv_numbers = temp_numbers
                else:
                    csv_unflatten = np.vstack((csv_unflatten,temp_row))
                    csv_numbers = np.vstack((csv_numbers,temp_numbers))
                
            #csv_unflatten = gen_csv.reshape((len(gen_csv),self.number_of_parameters+1))
            #print("Type is " + str(gen_csv[0]))
            return csv_numbers,types
        except OSError:
            print(" " + str(file) + " Missing")
            pass
    
    def  construct_bestfit_err_mat(self,full_matrix,plot=False):
        """
        Construct the average best fit matrix using the sum of the files, and
        generate the corresponding labels using the paths provided.

        full_matrix <matrix>: n by m where n is the number of samples and m is the number of parameters
        npaths <int>: number of paths
        plot <bol>: if it plot or not

        """
        score = []
        full_mat_var_cov = np.cov(full_matrix.T)
        full_mat_diag = np.diag(full_mat_var_cov)
        err = np.sqrt(full_mat_diag)

        labels = self.generate_labels()
#         print(full_mat)
#         bestFit = np.mean(full_mat,axis=0)
#         bestFit = full_mat[-1,:]
    
        for i in self.full_matrix:
            #arr = i.reshape((1,-1))
            ind = self.create_individual_from_bestfit(i)  
            score.append(self.fitness(ind))
            
        bestScore = np.nanargmin(score)
        bestFit = full_matrix[bestScore]
        if plot:
            plt.figure(figsize=(7,5))
            plt.xticks(np.arange(len(full_mat_diag)),labels[0],rotation=70);
            plt.bar(np.arange(len(full_mat_diag)),np.sqrt(full_mat_diag))

        return bestFit,err
    
    def create_individual_from_bestfit(self, best_fit):
        pars_range = '' #dummy range, since we're going to set the params again later

        #construct peaks and backgrounds list for individual constructor
        peaks = []
        bkgns = []
        #print(self.types)
        for i,Type in enumerate(self.types):
            if Type.lower().strip() in self.peaks:
                peaks.append(Type)
            if Type.lower().strip() in self.bkgns:
                bkgns.append(Type)

        #creates dummy individual, with random attributes
        ind = Individual(bkgns,peaks,pars_range)
        peaks2 = ind.get_peaks()
        bkgns2 = ind.get_backgrounds()

        #going to custom set the attributes of the individual to match the ones in the best fit
        current_index = 0
        peak_index = 0
        bkgn_index = 0
        for i,Type in enumerate(self.types):
            num_params = self.get_param_num(Type.lower())
            param_list = []
            for k in range(current_index,current_index+num_params-1):
                param_list.append(best_fit[k])
                current_index = k+1
            '''
            if Type.lower() == 'voigt':
                peaks2[peak_index].set_voigt(param_list)
                peak_index += 1
            '''
            if Type.lower() in self.peak_options:
                peaks2[peak_index].set(param_list)
                peak_index += 1
            if Type.lower() == 'shirley-sherwood':
                bkgns2[bkgn_index].set_shirley_sherwood(param_list)
                bkgn_index +=1
            if Type.lower() == 'linear':
                bkgns2[bkgn_index].set_linear(param_list)
                bkgn_index +=1
            

        '''    
        for i in range(len(self.peaks)):
            if self.peaks[i].lower() == 'voigt':
                #needs to be adjusted, best_ft[0...etc is a hard coded value]
                peaks[i].setBindingEnergy(best_fit[0])
                peaks[i].setGaussian(best_fit[1])
                peaks[i].setLorentzian(best_fit[2])
                peaks[i].setAmplitude(best_fit[3])
        for i in range(len(self.bkgns)):
            if bkgns[i].getType().lower() == 'shirley-sherwood':
                print("setting k to be " + str(best_fit[5]))
                bkgns[i].set_k(best_fit[5])
        '''
        return ind
    

        
    def generate_labels(self):
        label=[]
        amp_label = []
        center_label = []
        sigma_label = []
        gamma_label = []
        asymmetry_label = []

        for i in range(1,self.number_of_parameters+1):
            label.append('amp_' + str(i))
            amp_label.append('amp_' + str(i))

            label.append('center_' + str(i))
            center_label.append('center_' + str(i))

            label.append('sigma_' + str(i))
            sigma_label.append('sigma_' + str(i))

            label.append('gamma_' + str(i))
            gamma_label.append('gamma_' + str(i))

            label.append('asymmetry_' + str(i))
            asymmetry_label.append('asymmetry_' + str(i))

        return label,amp_label,center_label,sigma_label,gamma_label,asymmetry_label

    def fitness(self,indObj):
        """
        Evaluate fitness of a individual
        """
        loss = 0
        Individual = indObj
        #print(indObj.get_peaks())
        yTotal = np.zeros(len(self.x))


        yTotal = Individual.getFit(self.x,self.y)
        for j in range(len(self.x)):

           # loss = loss + (yTotal[j]*self.x_array[j]**2 - self.y_array[j]* self.x_array[j]**2 )**2
            #loss = loss + (((yTotal[j]- self.y[j])**2) * self.y[j])
            loss = loss + ((yTotal[j]- self.y[j])**2)*np.sqrt(self.y[j])
        # if loss == np.nan:
            # print(individual[0].verbose())
        return loss
    
    def score(self,verbose=False):
        loss =self.fitness(self.best_ind)
        print('Fitness Score (Chi2):',loss)
        # print('Fitness Score (ChiR2):', loss/(len(self.x_raw)-4*self.npaths))
        # self.fwhm(verbose=verbose)
        # self.cal_area(verbose=verbose)
        # print(self.err_full)

    def analyze(self):
        #should calculate area and other params
        print("Full analysis not yet functional")

    def plot_data(self,title='Test',fig_gui=None):
        if fig_gui == None:
            plt.rc('xtick', labelsize='12')
            plt.rc('ytick', labelsize='12')
            # plt.rc('font',size=30)
            plt.rc('figure',autolayout=True)
            plt.rc('axes',titlesize=12,labelsize=12)
            # plt.rc('figure',figsize=(7.2, 4.45))
            plt.rc('axes',linewidth=1)

            plt.rcParams["font.family"] = "Times New Roman"
            # fig,ax = plt.subplots(1,1,figsize=(6,4.5))
            # ax.plot(self.x_raw,self.y_raw,'ko-',linewidth=1,label='Data')
            plt.plot(self.x,self.y,'b--',linewidth=1,label='Data')
            # ax.plot(self.x_slice,self.y_slice,'r--',linewidth=1.2,label='Slice Data')
            plt.plot(self.x,self.y_model,'r',linewidth=1,label='Fit')
            #plt.plot(self.x_linear,self.y_linear,'--',color='tab:purple')
            plt.title(title)
            plt.xlabel('Energy (eV)')
            plt.ylabel('Counts/s')
            plt.legend()
        else:
            ax  = fig_gui.add_subplot(111)
            ax.plot(self.x,self.y,'b.',markersize=5,linewidth=1,label='Data')
            ax.plot(self.x,self.y_model,'r--',linewidth=1.2,label='Fit')
            if(len(self.peak_components)>1):
                for i,peak in enumerate(self.peak_components):
                    ax.plot(self.x,peak,'--',markersize=2,linewidth = 1,label=('Peak ' + str(i)))
            if(len(self.bkgn_components)>1):
                for i,bkgn in enumerate(self.bkgn_components):
                    ax.plot(self.x,bkgn,'b--',linewidth = 1,label=('Bkgn ' + str(i)))

            bkgns = self.best_ind.get_backgrounds()
            self.background = [] * len(self.x)
            for i in range(len(bkgns)):
                self.background = bkgns[i].getY(self.x,self.y)
            ax.plot(self.x,self.background,'c-',linewidth =1, label='background')
            ax.set_xlabel('Energy (eV)')
            ax.set_ylabel('Counts/s')
            # ax.title(str('Fit'))
            # ax.xlabel(str('Energy (eV)'))
            # ax.ylabel(str('Counts/s'))
            #ax.invert_xaxis()

            #ax.plot(self.x_linear,self.y_linear,'--',color='tab:purple')
            ax.legend()
            fig_gui.tight_layout()
   
    def get_params(self):
        return self.params
