import enum
from . import helper
from .import_lib import *
from .ini_parser import *
#from pathObj import OliverPharr
#from .individual import Individual
# from .pathrange import Pathrange_limits  #may be deletable/ built for XAFS
# from .nano_neo_data import NanoIndent_Data


from .xes_individual import Individual
from .xes_fit import peak,background
from .xes_data import xes_data

from copy import deepcopy #fixes bug at line 70ish with deepcopy
"""
Author: Andy Lau
"""

# Setup some default constraints
MAX = 3.40282e+38
MIN = 1.17549e-38

class XES_GA:

    def initialize_params(self,verbose = False):
        """Initalize the params

        Args:
            verbose (bool, optional): verbosity. Defaults to False.
        """
        print("Initializing Params")
        self.intervalK = 0.05
        self.tol = np.finfo(np.float64).resolution

    def initialize_variable(self):
        """Initialize variables
        """
        print("Initializing Variables")
        self.genNum = 0
        self.nChild = 4
        self.globBestFit = [0,np.inf]
        self.currBestFit = [0,np.inf]
        self.bestDiff = np.inf
        self.bestBest = np.inf
        self.diffCounter = 0

        self.pathDictionary = {}
        self.data_file = data_file

        # Paths
        self.npaths = npaths
        #self.fits = fits

        # Populations
        self.npops = size_population
        self.ngen = number_of_generation
        self.steady_state = steady_state

        # Mutation Parameters
        self.mut_opt = mutated_options
        self.mut_chance = chance_of_mutation
        # self.mut_chance_e0 = chance_of_mutation_e0

        # Crosover Parameters
        self.n_bestsam = int(best_sample*self.npops*(0.01))
        self.n_lucksam = int(lucky_few*self.npops*(0.01))

        # Time related
        self.time = False
        self.tt = 0


        # DE related:
        self.F = 0.5
        self.cR = 0.3


    def initialize_file_path(self):
        """Initalize file paths for output and log files
        """
        print("Initializing file paths")
        self.base = os.getcwd()
        self.output_path = os.path.join(self.base,output_file)
        self.check_output_file(self.output_path)
        self.log_path = os.path.splitext(copy.deepcopy(self.output_path))[0] + ".log"
        self.check_if_exists(self.log_path)

        # Initialize logger
        self.logger = logging.getLogger('')
        # Delete handler
        self.logger.handlers=[]
        file_handler = logging.FileHandler(self.log_path,mode='a+',encoding='utf-8')
        stdout_handler = logging.StreamHandler(sys.stdout)

        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        stdout_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stdout_handler)

        self.logger.setLevel(logging.INFO)
        self.logger.info(banner())

    @staticmethod
    def check_if_exists(path_file):
        """Check if the file exists, if it does, delete it

        Args:
            path_file (str): File path
        """
        if os.path.exists(path_file):
            os.remove(path_file)
        # Make Directory when its missing
        path = pathlib.Path(path_file)
        path.parent.mkdir(parents=True, exist_ok=True)

    def check_output_file(self,file):
        """
        check if the output file for each of the file
        """
        file_base = os.path.splitext(file)[0]
        XES_GA.check_if_exists(file)
        self.file = file

        self.file_initial = open(self.output_path,"a+")
        self.file_initial.write("Gen,TPS,FITTNESS,CURRFIT,CURRIND,BESTFIT,BESTIND\n")  # writing header
        self.file_initial.close()

        file_data = os.path.splitext(file)[0] + '_data.csv'
        XES_GA.check_if_exists(file_data)
        self.file_data = file_data


    def initialize_range(self,i=0,BestIndi=None):
        """
        Initalize range

        # TODO: Initalize range wi            self.crossoverPopulation()

            i (int, optional): _description_. Defaults to 0.
            BestIndi (_type_, optional): _description_. Defaults to None.
        """
        self.logger.info("Initializing ranges")

        # data = np.genfromtxt(self.data_file,delimiter=',',skip_header=1)
        self.data_obj = xes_data(self.data_file,0)

        self.x_slice = self.data_obj.get_x()
        self.y_slice = self.data_obj.get_y()
        self.x_array = self.x_slice
        self.y_array = self.y_slice

        yAvg = 0
        yTot = 0
        j=0
        for i,yVal in enumerate(self.data_obj.get_y()[-10:]): #[-10:] gets last 10 items in array.
            yTot += yVal
            j=i
        yAvg = yTot/(j+1)
        background_range[1] = yAvg

        amp_range[1] = max(self.data_obj.get_y())
        self.pars_range = {
            'Peak Energy': peak_energy_range,
            'Peak Energy Guess': peak_energy_guess,
            'Gaussian': sigma_range,
            'Lorentzian': fwhm_range,
            'Amplitude' : amp_range,
            'Background' : background_range,
            'Slope' : slope_range,
            'npeaks' : npaths
        }
        self.peak_type = peak_type
        self.backgrounds = background_type


    def generateIndividual(self):
        """Generate single individual

        Returns:
            Individual: Individual
        """

        ind = Individual(self.backgrounds,self.peak_type,self.pars_range)
        return ind

    def generateFirstGen(self):
        print("generating first gen")
        self.Populations=[]

        for i in range(self.npops):
            self.Populations.append(self.generateIndividual())

        self.eval_Population()
        self.globBestFit = self.sorted_population[0]

        print("First gen generated")

    def fitness(self,indObj):
        """
        Evaluate fitness of a individual
        """
        loss = 0
        Individual = indObj
        yTotal = np.zeros(len(self.x_slice))

        yTotal = Individual.getFit(self.x_array,self.y_array)
        for j in range(len(self.x_array)):

           # loss = loss + (yTotal[j]*self.x_array[j]**2 - self.y_array[j]* self.x_array[j]**2 )**2
            #loss = loss + (((yTotal[j]- self.y_array[j])**2)*self.y_array[j]
            loss = loss + (((yTotal[j]- self.y_array[j])**2))*np.sqrt(self.y_array[j])
        # if loss    # @profile

    def eval_Population(self)-> list:
        """Evaluate Populations

        Returns:
            list: List of score
        """
        score = []
        populationPerf = {}
        self.nan_counter = 0
        for i,individual in enumerate(self.Populations):

            temp_score = self.fitness(individual)
            # Calculate the score, if encounter nan, discard and generate new individual later
            if np.isnan(temp_score):
                self.nan_counter +=1
            else:
                score.append(temp_score)
                populationPerf[individual] = temp_score
        self.sorted_population = sorted(populationPerf.items(), key=operator.itemgetter(1), reverse=False)
        '''
        for a,b in self.sorted_population:
            print(str(b) + " " + str(a.get_params()))
        '''
        ''' Debugging again
        for i in range(len(self.sorted_population)):
            print(self.sorted_population[i][0].get_peak(0))
        '''
        self.currBestFit = self.sorted_population[0]

        return score


    def next_generation(self):
        """Calculate next generations
        """
        self.st = time.time()
        self.logger.info("---------------------------------------------------------")
        self.logger.info(datetime.datetime.fromtimestamp(self.st).strftime('%Y-%m-%d %H:%M:%S'))
        self.logger.info(f"{helper.bcolors.BOLD}Gen: {helper.bcolors.ENDC}{self.genNum+1}")

        self.genNum += 1

        # Evaluate Fitness
        score = self.eval_Population()
        self.bestDiff = abs(self.globBestFit[1]-self.currBestFit[1])
        if self.currBestFit[1] < self.globBestFit[1]:
            self.globBestFit = self.currBestFit

        with np.printoptions(precision=5, suppress=True):
            self.logger.info("Different from last best fit: " +str(self.bestDiff))
            self.logger.info(helper.bcolors.BOLD + "Best fit: " + helper.bcolors.OKBLUE + str(self.currBestFit[1]) + helper.bcolors.ENDC)
            self.logger.info("Best fit combination:\n" + str((self.sorted_population[0][0].get_params())))
            self.logger.info(helper.bcolors.BOLD + "History Best: " + helper.bcolors.OKBLUE + str(self.globBestFit[1]) + helper.bcolors.ENDC)
            self.logger.info("NanCounter: " + str(self.nan_counter))
            self.logger.info("History Best Indi:\n" + str((self.globBestFit[0].get_params())))

        nextBreeders = self.selectFromPopulation()

        self.mutatePopulation()
        if self.mut_opt != 3:
            self.selectFromPopulation()
            self.createChildren()
            self.logger.info("Number of Breeders: " + str(len(self.parents)))
            self.logger.info("DiffCounter: " + str(self.diffCounter))
            self.logger.info("Diff %: " + str(self.diffCounter / self.genNum))
            self.logger.info("Mutation Chance: " + str(self.mut_chance))
        # DE
        else:
            self.crossoverPopulation()

        self.et = timecall()
        self.tdiff = self.et - self.st
        self.tt = self.tt + self.tdiff
        self.logger.info("Time: "+ str(round(self.tdiff,5))+ "s")

    def crossoverPopulation(self):
        self.trialPopulations = []
        for i in range(self.npops):
            self.trialPopulations.append(self.crossoverDE(self.mutated_Populations[i],self.Populations[i],self.cR))

    def crossoverDE(self,mutateInd: Individual, popInd: Individual, cR: float) -> Individual:
        """Uniform crossover of DE using `self.cR`

        Args:
            mutateInd (Individual): Individual for the mutated population
            popInd (Individual): Individual for the original population
            cR (float): crossover rate

        Returns:
            Individual: crossovered individual
        """

        p = np.random.rand(len(mutateInd))
        mutatePars = mutateInd.get_params()
        popPars = popInd.get_params()
        tempPars = []
        for i in range(len(mutateInd)):
            if p[i] < cR:
                tempPars.append(mutatePars[i])
            else:
                tempPars.append(popPars[i])

        # tempInd.
        split_full_list = XES_GA.split_into_x(tempPars)
        temp_individual = self.generateIndividual()
        XES_GA.set_pars(temp_individual,split_full_list)

        return temp_individual

    def adjust_DE_parameters(self,on=True):
        """Adjust the DE parameters using jDE algorithm
        """
        if on:
            rand_val = np.random.rand(4)
            tau_1 = 0.1
            tau_2 = 0.1
            if rand_val[1] < tau_1:
                self.F = 0.1 + rand_val[0] * 0.9
                self.logger.info(f"F has been adjusted to {np.round(self.F,4)}")

            if rand_val[3] < tau_2:
                self.cR = rand_val[2]
                self.logger.info(f"Cr has been adjusted to {np.round(self.cR,4)}")

    def mutatePopulation(self):
        """
        # Mutation operators
        # 0 = original: generated a new versions:
        # 1 = mutated every genes in the total populations
        # 2 = mutated genes inside population based on secondary probability
        # 3 = Differntial Evolution
        """
        self.nmutate = 0

        "DE mutation"
        if self.mut_opt == 3:
            self.mutated_Populations = []
            for i in range(self.npops):
                candidates = [candidate for candidate in range(self.npops) if candidate != i]
                a,b,c = np.random.choice(candidates,3,replace=False)
                mutation_vectors = [self.Populations[a],self.Populations[b],self.Populations[c]]
                temp_individual = self.mutateIndi_DE(mutation_vectors,self.F)
                temp_individual.checkBound()
                self.mutated_Populations.append(temp_individual)

        else:
            # Rechenberg mutation
            if self.bestDiff < 0.1:
                self.diffCounter += 1
            else:
                self.diffCounter -= 1
            if (abs(self.diffCounter)/ float(self.genNum)) > 0.2:
                self.mut_chance += 0.5
                self.mut_chance = abs(self.mut_chance)
            elif (abs(self.diffCounter) / float(self.genNum)) < 0.2:
                self.mut_chance -= 0.5
                self.mut_chance = abs(self.mut_chance)

        for i in range(self.npops):
            if random.random()*100 < self.mut_chance:
                self.nmutate += 1
                self.Populations[i] = self.mutateIndi(i)

        self.logger.info("Mutate Times: " + str(self.nmutate))

    @staticmethod
    def setPars(individual,split_full_list):
        """Static Method to set the parameters

        Args:
            individual (individuals): individuals type
            split_full_list (list): multidims list containing each peaks type and
                parameters, and bg peaks.
        """
        full_list = copy.copy(split_full_list)

        bg = full_list.pop()
        for i in range(len(full_list)):
            individual.setPeak(i,full_list[i])

    @staticmethod
    def split_into_x(full_list):
        """Split a full parameters list into multidmenisonal list

        Args:
            full_list (list): full list of parameters
        """
        split_list = []
        temp_list = []
        for i in full_list:
            if isinstance(i,str) == False:
                temp_list.append(i)
            else:
                temp_list.append(i)
                split_list.append(temp_list)
                temp_list = []

    def mutateIndi_DE(self,mutateIndividuals:list,F:float) -> Individual:
        """
        Mutate the individuals using DE mutation

        Args:
            mutated_individuals (list): Input list of individuals
            F (float): Mutation Factors
        """
        length = len(mutateIndividuals[0])
        assert all(len(lst) == length for lst in mutateIndividuals)

        x_Pars = mutateIndividuals[0].get_params()
        y_Pars = mutateIndividuals[1].get_params()
        z_Pars = mutateIndividuals[2].get_params()

        # Convert to the extracted full list with peaks and BG
        full_list = []
        for i in range(len(x_Pars)):
            if isinstance(x_Pars[i],str) == False:
                full_list.append(x_Pars[i]  + F*(y_Pars[i]  - z_Pars[i]))
            else:
                full_list.append(x_Pars[i])

        split_full_list = XES_GA.split_into_x(full_list)
        temp_individual = self.generateIndividual()
        XES_GA.setPars(temp_individual,split_full_list)

        return temp_individual



    def mutateIndi(self,indi :int) -> Individual:
        """Mutate the Individual

        Args:
            indi (int): index of the populations

        Returns:
            Individual: Mutated Individuals
        """
        if self.mut_opt == 0:
            # Create a new individual with Rechenberg
            newIndi = self.generateIndividual()
        # Random pertubutions
        if self.mut_opt == 1:
            # Random Pertubutions
            self.Populations[indi].mutate_(self.mut_chance)
            newIndi = self.Populations[indi]
            # Mutate every gene in the Individuals

        if self.mut_opt == 2:
            n_success = 0
            og_individual = self.generateIndividual()
            # Create a new individual with the same parameters
            og_pars = copy.copy(self.Populations[indi].get_peaks())


            og_individual.setPeaks(og_pars)
            og_score = self.fitness(og_individual)

            new_individual = self.generateIndividual()
            mut_score = self.fitness(new_individual)

            T = - self.bestDiff/(np.log(1-(self.genNum/self.ngen)+MIN))
            if mut_score < og_score:
                n_success = n_success + 1

                newIndi = new_individual
            elif np.exp(-(mut_score-og_score)/(T+MIN)) > np.random.uniform():
                n_success = n_success + 1
                newIndi = new_individual
            else:
                newIndi = og_individual

        return newIndi

    def selectFromPopulation(self):
        self.parents = []

        select_val = np.minimum(self.n_bestsam,len(self.sorted_population))
        self.n_recover = 0
        if len(self.sorted_population) < self.n_bestsam:
            self.n_recover = self.n_bestsam - len(self.sorted_population)
        for i in range(select_val):
            self.parents.append(self.sorted_population[i][0])

    def crossover(self,individual1: Individual, individual2: Individual) -> Individual:
        """Crossover between two individuals, uniform crossover

        Args:
            individual1 (Individual): First Individual
            individual2 (Individual): Second Indivudal

        Returns:
            Individual: crossovered individual
        """
        # TODO: Rewrite this function to use the new code. this is too complicated...
        #    The `get_params` method needs to calculate a bunch of stuff, but this you have to divided
        #    FIX: use a dictionary to setup the code.
        #
        #
        child = self.generateIndividual()

        individual1_path = individual1.get_params()
        individual2_path = individual2.get_params()
        #print("Ind 1 : " + str(individual1_path))
        #print("Ind 2 : " + str(individual2_path))
        temp_path = []
        dividers = [] # markers where the strings are in the list of params, this indicates where the array switches to a new peak or background
        #crossover for peak vars
        for j in range(len(individual1_path)):
            if (isinstance(individual1_path[j],str)):
                dividers.append(j)
            if np.random.randint(0,2) == True:
                temp_path.append(individual1_path[j])
            else:
                temp_path.append(individual2_path[j])
            '''
        for j in range(1):
            if np.random.randint(0,2) == True:
                temp_path.append(individual1_path[1][j])
            else:
                temp_path.append(individual2_path[1][j])
        '''
        #print("Temp Path: " + str(temp_path))
        temp_peak = []
        #print(temp_path)
        divider = 0
        peakNum = 0
        bkgnNum = 0
        for k in range(len(dividers)):
            for j in range(divider,dividers[k]+1):
                temp_peak.append(temp_path[j])
            if i < self.npaths:
                #print()
                #print("Child pre-write: " + str(child.get_params()))
                #print("temp peak : " + str(temp_peak))
                if child.setPeak(peakNum,temp_peak) == -1:
                    if bkgnNum<len(background_type):
                        #print("Bkgn")
                        child.setBkgn(bkgnNum,temp_peak)
                        bkgnNum += 1
                else:
                    #print("wrote peak")
                    peakNum +=1
                #print("Child after write")
                #print(child.get_params())
                #print()
                temp_peak = []
            divider = j + 1

        #print("Child : " + str(child.get_params()))
        '''
        child.setPeak(i,temp_path[0],temp_path[1],temp_path[2],temp_path[3])
        child.get_background(0).set_k(temp_path[4])
        '''
        '''
        print(temp_path)
        print("Child:")
        print(child.get_params())
        exit()
        '''
        return child

    def createChildren(self):
        """
        Generate Children
        """
        self.nextPopulation = []
        # --- append the breeder ---
        for i in range(len(self.parents)):
            self.nextPopulation.append(self.parents[i])
        # print(len(self.nextPopulation))
        # --- use the breeder to crossover
        # print(abs(self.npops-self.n_bestsam)-self.n_lucksam)

        for i in range(abs(self.npops-self.n_bestsam)-self.n_lucksam):
            par_ind = np.random.choice(len(self.parents),size=2,replace=False)
            child = self.crossover(self.parents[par_ind[0]],self.parents[par_ind[1]])
            self.nextPopulation.append(child)
        # print(len(self.nextPopulation))

        for i in range(self.n_lucksam):
            self.nextPopulation.append(self.generateIndividual())
        # print(len(self.nextPopulation))

        for i in range(self.n_recover):
            self.nextPopulation.append(self.generateIndividual())

        random.shuffle(self.nextPopulation)
        self.Populations = self.nextPopulation

    def run_verbose_start(self):
        self.logger.info("-----------Inputs File Stats---------------")
        self.logger.info(f"{helper.bcolors.BOLD}File{helper.bcolors.ENDC}: {self.data_file}")
        #self.logger.info(f"{bcolors.BOLD}File Type{bcolors.ENDC}: {self.data_obj._ftype}")
        self.logger.info(f"{helper.bcolors.BOLD}Fits{helper.bcolors.ENDC}: {peak_type}")
        # self.logger.info(f"{helper.bcolors.BOLD}Peak Energy Range{helper.bcolors.ENDC}: {self.pars_range}")
        self.logger.info(f"{helper.bcolors.BOLD}Backgrounds{helper.bcolors.ENDC}: {background_type}")
        self.logger.info(f"{helper.bcolors.BOLD}File{helper.bcolors.ENDC}: {self.output_path}")
        self.logger.info(f"{helper.bcolors.BOLD}Population{helper.bcolors.ENDC}: {self.npops}")
        self.logger.info(f"{helper.bcolors.BOLD}Num Gen{helper.bcolors.ENDC}: {self.ngen}")
        self.logger.info(f"{helper.bcolors.BOLD}Mutation Opt{helper.bcolors.ENDC}: {self.mut_opt}")
        self.logger.info("-------------------------------------------")

    def run_verbose_end(self):
        self.logger.info("-----------Output Stats---------------")
        # self.logger.info(f"{bcolors.BOLD}Total)
        self.logger.info(f"{helper.bcolors.BOLD}Total Time(s){helper.bcolors.ENDC}: {round(self.tt,4)}")
        self.logger.info("-------------------------------------------")

    def run(self):
        self.run_verbose_start()
        self.historic = []
        self.historic.append(self.Populations)

        for i in range(self.ngen):
            temp_gen = self.next_generation()
            self.output_generations()
        #print(self.globBestFit[0].getFit(self.x_array,self.y_array))
        self.run_verbose_end()
        # test_y = self.export_paths(self.globBestFit[0])
        # plt.plot(self.data_obj.get_raw_data()[:,0],self.data_obj.get_raw_data()[:,1],'b-.')
        # plt.plot(self.x_slice,self.y_slice,'o--',label='data')
        # plt.plot(self.x_slice,test_y,'r--',label='model')
        # plt.legend()
        # plt.show()

    def export_paths(self,indObj):
        area_list=[]
        Individual = indObj.get()

        yTotal = np.zeros(len(self.x_slice))
        plt.figure()
        for i,paths in enumerate(Individual):
            y = paths.getY()

            yTotal += y
            # area = np.trapz(y.flatten(),x=self.x_slice.flatten())
            # component = paths.get_func(self.x_slice).reshape(-1,1)

            # area_list.append(area)

        Total_area = np.sum(area_list)
        return yTotal

    def output_generations(self):
        """
        Output generations result into two files
        """
        with open(self.output_path,"a") as f1:
            f1.write(str(self.genNum) + "," + str(self.tdiff) + "," +
                str(self.currBestFit[1]) + "," + str(self.currBestFit[0].get_params()) +")," +
                str(self.globBestFit[1]) + "," + str(self.globBestFit[0].get_params()) +"\n")
        with open(self.file_data,"a") as f2:
            write = csv.writer(f2)
            bestFit = self.globBestFit[0]
            #write.writerow((bestFit[i][0], bestFit[i][1], bestFit[i][2]))
            str_pars = bestFit.get_params()
            write.writerow(str_pars)
            f2.write("#################################\n")


    def __init__(self):
        """
        Steps to Initalize EXAFS
            EXAFS
        """
        # initialize params
        self.initialize_params()
        # variables
        self.initialize_variable()
        # initialze file paths
        self.initialize_file_path()
        # initialize range
        self.initialize_range()
        # Generate first generation
        self.generateFirstGen()
        # Actually run
        self.run()

def main():
    XES_GA()

if __name__ == "__main__":
    main()
