"""
Authors     Miu Lun Lau, Megan Burrill
Email       mburrill@hawk.iit.edu, andylau@u.boisestate.edu
Version     0.2
Date        Jul 28, 2022
"""
'''
Edited by: Alaina Thompson, Evan Restuccia, Tim Stack
Email:
Date: July 24, 2023
'''
"""
TODO
- analysis [Done]
- improve graphing
- preprocessing?
- connect calibration to nano-indent
"""

from tkinter import *
from threading import Thread
from tkinter import ttk, Tk, N, W, E, S, StringVar, IntVar, DoubleVar, BooleanVar, Checkbutton, NORMAL, DISABLED, \
    scrolledtext, filedialog, messagebox, LabelFrame, Toplevel, END, TOP
from tkinter.font import Font
from tokenize import Double

# import matplotlib
import matplotlib
from matplotlib.hatch import HorizontalHatch
import numpy as np
import configparser

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
# from threading import *
# import os
import os, subprocess, asyncio
import signal
import pathlib
import multiprocessing as mp
# custom Libraries
from xes_plot import Data_plot, Analysis_plot
#import preprocess_data
import xes_data as data
import platform
import xes_analysis2

def get_platform():
    os_name = platform.system()
    return os_name


class App():
    """
    Start of the application
    """

    def __init__(self):
        self.__version__ = 0.1
        self.root = Tk(className='XES Neo GUI')
        self.root.wm_title("XES GUI (Pre-pre-pre-Beta)")
        self.root.geometry("875x650") #Changed size so that select folder button on Analysis tab is fully visible
        self.mainframe = ttk.Notebook(self.root, padding='5')
        self.mainframe.grid(column=0, row=0, sticky=(N, W, E, S), columnspan=5)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.resizable(True, True)
        self.padx = 5
        self.pady = 3

        self.os = get_platform()


        # Specify standard font
        self.entryFont = Font(family='TkTextFont', size=11)
        self.labelFont = Font(family='TkMenuFont', size=11)
        s = ttk.Style()
        # s.configure('my.TButton', font=labelFont)

        # Set multiprocessing run type
        mp.set_start_method('spawn')

        # initialize variables
        self.initialize_var()
        self.initialize_tabs()
        self.build_tabs()

    def initialize_var(self):
        """
        Initalize all possible variables in the GUI
        """
        # Inputs Tab (column 0)______________________________________________________________________
        # Averaging Section of Input Tab:
        self.all_files_display = StringVar(self.root, 'Please choose file/s')  #Files that you want to average
        self.all_files = np.array([])
        self.averaged_file = StringVar(self.root, 'Please choose file/s') #File to save the averaged data to
        self.num_files = DoubleVar(self.root, 0) #Number of files to be averaged after selection
        self.num_points = DoubleVar(self.root, 0) #Number of points in each data file
        self.xPoints_for_data_plot = np.array([])
        self.yPoints_for_data_plot = np.array([])
        self.xPoints_avg = np.array([]) #Averaged x points for plotting
        self.yPoints_avg = np.array([]) #Averaged y points for plotting
        self.plot_data = np.array([],[])

        # Input Section of Input Tab:
        # There is a string var to give the prompt on selection, then a path saved to allow easier and safer manipulation
        self.csv_file = StringVar(self.root, "Please choose a data file")
        self.csv_generate_from = pathlib.Path()
        self.file_name = "Data file"
        self.csv_folder = StringVar(self.root, "Please choose a folder containing data files")
        self.csv_folder_path = pathlib.Path()
        self.output_folder = StringVar(self.root, 'Please choose a folder to save outputs')
        self.output_folder_path = pathlib.Path()
        self.output_file = pathlib.Path()
        #File reading options
        self.skipLn = StringVar(self.root,"0")

        # Waiting to create path variable for calibration because I think it will be deleted
        #self.csv_calibration_file = StringVar(self.root, "Please choose a calibration file") #--------------Will need to comment out-----------
        self.yes_folder = IntVar()
        self.multi_known = BooleanVar(value = False)
        self.multi_known = False  # This will change to true when the user has already said if they do/do not want to generate/run multiple files in the folder so generate ini does nto keep asking
        self.filelist = []
        self.command_list = mp.Queue()
        self.proc_list = []
        self.pid_list = []
        # Preprocessing
        self.preprocess_file = pathlib.Path()
        self.stop_not_pressed = True

        # Variables for the dropdown menu
        self.file_menu = ttk.Combobox()
        self.data_obj = 0

        # Populations Tab (column 2)________________________________________________________________________
        self.population = IntVar(self.root, 2000)
        self.num_gen = IntVar(self.root, 10)
        self.best_sample = IntVar(self.root, 20)
        self.lucky_few = IntVar(self.root, 10)

        # Mutations Tab (column 3)________________________________________________________________________
        self.chance_of_mutation = IntVar(self.root, 20)
        self.original_chance_of_mutation = IntVar(self.root, 20)
        self.mutated_options = IntVar(self.root, 0)

        # Fitting parameters(column 4)________________________________________________________________________
        #New XES parameters
        self.number_of_peaks = DoubleVar(self.root, 1) #Selection for how many peaks you want the algorithm to fit
        self.peak_Energy_guesses = [] #Array to hold Peak Energy guesses
        for i in range(10):
            self.peak_Energy_guesses.append(DoubleVar(self.root, 6489.0))
        self.so_split = DoubleVar(self.root, 0.00)
        self.peaks = [] #Array to hold what type of peaks selected for each peak
        for i in range(10):
            self.peaks.append(StringVar(self.root, "Select a peak type"))
        self.background_types = []
        #self.path_bkgn = StringVar(self.root, "Select a background type") #Dont think we need anymore because we want to use an array now
        #self.path_branching = StringVar(self.root, "Select branching ratio")

        #Ranges for different variables
        self.peak_energy_min,self.peak_energy_max,self.peak_energy_delta = -1.5,1.5,.01
        self.sigma_min,self.sigma_max,self.sigma_delta = 0,4,.001
        self.fwhm_min,self.fwhm_max,self.fwhm_delta = 0,2.5,.001
        self.amp_min,self.amp_max,self.amp_delta = 0,5000, 0.05
        self.background_min,self.background_max,self.background_delta = 0,1.25,0.001
        self.slope_min,self.slope_max,self.slope_delta = 0,0.01, 0.00001 #Allows for a slope to be applied to the background, currently hard set to 0

        # Plot Tab (column 5)________________________________________________________________________
        self.print_graph = BooleanVar(self.root, False)
        self.num_output_paths = BooleanVar(self.root, True)
        self.steady_state_exit = BooleanVar(self.root, True)

        # Output Tab (column 6)________________________________________________________________________
        self.print_graph = BooleanVar(self.root, False)
        self.num_output_paths = BooleanVar(self.root, True)
        self.steady_state_exit = BooleanVar(self.root, True)
        self.n_ini = IntVar(self.root, 100)
        self.pop_min = IntVar(self.root, 100)
        self.pop_max = IntVar(self.root, 5001)
        self.gen_min = IntVar(self.root, 20)
        self.gen_max = IntVar(self.root, 501)
        self.mut_min = IntVar(self.root, 20)
        self.mut_max = IntVar(self.root, 51)
        self.run_folder = BooleanVar(self.root, False)
        self.pertub_check = IntVar(self.root, 0)
        self.checkbutton_whole_folder = ttk.Checkbutton()

        self.analysis_dir = StringVar(self.root, "Please choose a data file")

        # Analysis Tab (column 7)________________________________________________________________________
        # I don't know what goes here so leave blank instead

    def initialize_tabs(self):
        """
        Initialize tabs for the main frame
        """
        s = ttk.Style()
        s.configure('TNotebook.Tab', font=('TkHeadingFont', '11'))
        height = 1
        # Creating tabs
        self.input_tab = ttk.Frame(self.mainframe, height=height)
        self.population_tab = ttk.Frame(self.mainframe, height=height)
        self.mutation_tab = ttk.Frame(self.mainframe, height=height)
        self.fitting_param_tab = ttk.Frame(self.mainframe, height=height)
        self.plot_tab = ttk.Frame(self.mainframe, height=height)
        self.output_tab = ttk.Frame(self.mainframe, height=height)
        self.analysis_tab = ttk.Frame(self.mainframe, height=height)

        # Adding tabs
        self.mainframe.add(self.input_tab, text="Inputs")
        self.mainframe.add(self.population_tab, text='Populations')
        self.mainframe.add(self.mutation_tab, text="Mutations")
        self.mainframe.add(self.fitting_param_tab, text="Fitting Parameters")
        self.mainframe.add(self.plot_tab, text="Plots")
        self.mainframe.add(self.output_tab, text="Output")
        self.mainframe.add(self.analysis_tab, text="Analysis")

    def build_tabs(self):
        """
        Build tabs. Will call function for each tab
        """
        self.build_global()
        self.build_inputs_tab()
        self.build_population_tab()
        self.build_mutations_tab()
        self.build_fitting_param_tab()
        self.build_plot_tab()
        self.build_output_tab()
        self.build_analysis_tab()

        self.mainframe.grid_rowconfigure(0, weight=1)
        self.mainframe.grid_columnconfigure(0, weight=1)
        self.mainframe.grid_columnconfigure(1, weight=1)
        self.mainframe.grid_rowconfigure(2, weight=1)
        self.mainframe.grid_columnconfigure(3, weight=1)
        self.mainframe.grid_columnconfigure(4, weight=1)

    def description_tabs(self, arr, tabs, sticky=(W, E), row=None, column=None, return_description=False):
        # Rows = index of rows
        # Loops through array of descriptors to be added to the tabs
        description_list = []
        if row is not None:
            assert len(row) == len(arr)
        for i, inputs in enumerate(arr):
            entry = ttk.Label(tabs, text=inputs, font=self.labelFont)
            if row is not None:
                k = row[i]
            else:
                k = i
            entry.grid_configure(column=column, row=k, sticky=sticky, padx=self.padx, pady=self.pady)
            description_list.append(entry)
        if description_list:
            return description_list


    # When the user selects a particular file from the input directory in the dropdown menu it will be assigned to the
    # csv generate from variable so that it is used for running
    def file_selected(self, event):
        self.csv_generate_from = pathlib.Path(self.csv_folder_path.joinpath(self.file_menu.get()))
        # print("file selected: ", self.csv_generate_from)

    # Writes the ini file to filename using the user inputs or defaults if nothing is changed
    def write_ini(self, filename):
        # First select data range
        # preprocess_data.read_files(self.csv_generate_from, limits=(self.percent_min.get(), self.percent_max.get()))
        # self.preprocess_file = pathlib.Path.cwd().joinpath('example.txt')

        # print("write ini csv path generate from", str(self.csv_generate_from))
        inputs = ("[Inputs]\ndata_file = {data}\noutput_file = {out} "
                  "".format(
            data=str(self.csv_generate_from),
            out=str(self.output_file),
            #calibration=str(self.csv_calibration_file.get()),#deleted: calibration=str(self.csv_calibration_file.get())
            #dat_cutoff=", ".join(str(i) for i in [self.percent_min.get(), self.percent_max.get()])
        ))
        populations = ("\n\n[Populations]\npopulation = {pop}\nnum_gen = {numgen}\nbest_sample = {best} "
                       "\nlucky_few = {luck}".format(pop=str(self.population.get()),
                                                     numgen=str(self.num_gen.get()),
                                                     best=str(self.best_sample.get()),
                                                     luck=str(self.lucky_few.get())))
        # Sends range for power law equation as [min, max]
        guesses = []
        inputPeaks = []
        for i in range(int(self.number_of_peaks.get())):
           guesses.append(self.peak_Energy_guesses[i].get())
           inputPeaks.append(self.peaks[i].get())

        paths = ("\n\n[Paths]\nnPeaks={nPeaks} \nbackground_type = {bkgn_type} \npeak_type = {peak_type} \npeak_Energy_range = {peak_energy_range} \npeak_energy={peak_energy} \nsigma_range = "
                 "{sigma_range} \nfwhm_range = {fwhm_range} \namp_range = {amp_range} \nbackground_range = {background_range} \nslope_range = {slope_range}"
                 .format(nPeaks=int(self.number_of_peaks.get()),
                         bkgn_type = ",".join(str(i) for i in self.background_types),
                         peak_type = ",".join(str(i) for i in inputPeaks),
                         peak_energy_range=", ".join(str(i) for i in [self.peak_energy_min, self.peak_energy_max, self.peak_energy_delta]),
                         peak_energy = ", ".join(str(i) for i in guesses),
                         sigma_range=", ".join(str(i) for i in [self.sigma_min, self.sigma_max, self.sigma_delta]),
                         fwhm_range=", ".join(str(i) for i in [self.fwhm_min, self.fwhm_max, self.fwhm_delta]),
                         amp_range =",".join(str(i) for i in [self.amp_min,self.amp_max,self.amp_delta]),
                         background_range = ", ".join(str(i) for i in [self.background_min, self.background_max, self.background_delta]),
                         slope_range = ", ".join(str(i) for i in [self.slope_min, self.slope_max, self.slope_delta])
                         ))

        mutations = ("\n\n[Mutations]\nchance_of_mutation = {chance} \noriginal_chance_of_mutation = {original} "
                     "\nmutated_options = {opt}"
                     .format(chance=str(self.chance_of_mutation.get()),
                             original=str(self.original_chance_of_mutation.get()),
                             opt=str(self.mutated_options.get())))
        outputs = ("\n\n[Outputs]\nprint_graph = {graph} \nnum_output_paths = {num} "
                   .format(graph=False, num=False))
        # I have more variables but do not see them in the nano-indent ini and am going to leave for now and see
        # I think I have now deleted the extra variables other than the calibration file
        print(str(inputs))
        with open(filename, 'w') as writer:
            writer.write(str(inputs))
            writer.write(str(populations))
            writer.write(str(mutations))
            writer.write(str(paths))
            writer.write(str(outputs))
        return filename

    def loop_gen_ini_same_params(self):
        """
        Will loop through every file in the selected directory and run it with the same parameters
        """
        # file_list = [file.absolute() for file in self.output_folder_path.glob('**/*.ini') if file.is_file()]
        file_list = [filename for filename in self.csv_folder_path.glob('**/*txt') if filename.is_file()]
        # file_list = [filename for filename in self.csv_folder_path.iterdir() if filename.is_file()]
        # stem = self.output_folder_path.stem
        # print(file_list)
        for each in file_list:
            # Change the output file name to match the file name
            name_out = each.stem + '_out.txt'
            # parent = self.output_folder_path.parent
            # print("Output name ", name_out)
            output_path = self.output_folder_path.joinpath(name_out)
            # sets equal so that the name in ini file matches
            self.output_file = output_path
            # Write an ini for for it
            self.csv_generate_from = each
            name = str(each.stem) + '.ini'
            # print("ini name: ", name)
            file_each_path = self.output_folder_path.joinpath(name)
            file_each_path.touch()
            # print('file path: ', str(file_each_path))
            self.write_ini(file_each_path)
        self.multi_known = False

    def generate_directory_popup(self):
        # Popup to ask if the user wants to run all files or just the selected file
        self.multi_known = True
        directory_popup = Toplevel(self.root)
        msg = "Do you want to generate ini files for all files in this directory with the same parameters or just the selected file?"
        entry = ttk.Label(directory_popup, text=msg)
        entry.grid(column=0, row=0, columnspan=2, padx=5, pady=3)
        B1 = ttk.Button(directory_popup, text="All files",
                        command=lambda: [directory_popup.destroy(), self.loop_gen_ini_same_params(),
                                         change_multi_known()])
        B2 = ttk.Button(directory_popup, text="Just this one",
                        command=lambda: [directory_popup.destroy(), self.generate_ini(), change_multi_known()])
        B1.grid(column=0, row=1, padx=5, pady=3, sticky=E)
        B2.grid(column=1, row=1, padx=5, pady=3, sticky=W)
        directory_popup.grid_columnconfigure((0, 1), weight=1)
        directory_popup.grid_rowconfigure((0, 1), weight=1)
        directory_popup.protocol('WM_DELETE_WINDOW', directory_popup.destroy)
        directory_popup.attributes('-topmost', 'true')

        def change_multi_known():
            self.multi_known = False

    def generate_ini(self):
        if self.yes_folder.get() == 1 and self.multi_known is False:  # A folder is selected
            # pop up and ask if generate ini for all files, call appropriate loops
            self.generate_directory_popup()
        else:
            def unique_path():
                counter = 0
                while True:
                    num_name = str(name) + "_" + str(counter) + '.ini'
                    out_name = self.csv_generate_from.stem + "_" + str(counter) + '_out.txt'
                    self.output_file = self.output_folder_path.joinpath(out_name)
                    path = self.output_folder_path.joinpath(num_name)
                    if not path.exists():
                        return path
                    counter += 1

            name = self.csv_generate_from.stem
            file_path = unique_path()
            file_path.touch()
            self.write_ini(file_path)
            return file_path
            # os.chdir(pathlib.Path.cwd().parent)  # change the working directory from gui to nano-indent
            # ini_file = filedialog.asksaveasfilename(initialdir=pathlib.Path.cwd(),
            #                                       title="Choose output ini file",
            #                                      filetypes=[("ini files", "*.ini")])
            # if ini_file is None:
            #   return
            # if isinstance(ini_file, tuple) == False:
            #   if len(ini_file) != 0:
            #      self.write_ini(ini_file)
            #     messagebox.showinfo('', 'Ini file written to {fileloc}'.format(fileloc=ini_file))

            # os.chdir(pathlib.Path.cwd().joinpath('gui'))

    def select_csv_folder(self):
        os.chdir(pathlib.Path.cwd().parent)
        folder_name = pathlib.Path(filedialog.askdirectory(initialdir=pathlib.Path.cwd(), title="Choose a folder"))
        self.csv_folder.set(folder_name)
        self.csv_folder_path = folder_name  # No file has been selected yet - this is the folder
        # This calls a method to create a dropdown menu next to generate ini button of the files in the directory
        self.file_dropdown()
        os.chdir(pathlib.Path.cwd().joinpath('gui'))
        return folder_name

    def select_output_folder(self):
        """
        Select output folder
        """
        os.chdir(pathlib.Path.cwd().parent)  # change the working directory from gui to nano-indent
        folder_name = pathlib.Path(filedialog.askdirectory(initialdir=pathlib.Path.cwd(), title="Choose a folder"))
        self.output_folder.set(folder_name)
        print("select output folder, folder.get ", self.output_folder.get())
        self.output_folder_path = pathlib.Path(folder_name)
        self.output_file = self.output_folder_path.joinpath('out.txt')
        os.chdir(pathlib.Path.cwd().joinpath('gui'))
        return folder_name

    def loop_direc_same_params(self):
        """
        Will loop through every file in the selected directory and run it with the same parameters
        """
        # file_list = [file.absolute() for file in self.output_folder_path.glob('**/*.ini') if file.is_file()]
        file_list = [filename for filename in self.csv_folder_path.glob('**/*txt') if filename.is_file()]
        # file_list = [filename for filename in self.csv_folder_path.iterdir() if filename.is_file()]
        # stem = self.output_folder_path.stem
        # print(file_list)
        for each in file_list:
            # Change the output file name to match the file name
            name_out = each.stem + '_out.txt'
            # parent = self.output_folder_path.parent
            # print("Output name ", name_out)
            # Create output file
            output_path = self.output_folder_path.joinpath(name_out)
            output_path.touch()
            self.output_file = output_path

            # Write an ini for for it
            self.csv_generate_from = each
            name = str(each.stem) + '.ini'
            print("ini name: ", name)
            file_each_path = self.output_folder_path.joinpath(name)
            file_each_path.touch()
            self.write_ini(file_each_path)
            self.command_list.put(str(file_each_path))
            print("Finished adding to command_list")
            # Run the ini using the run_multi_function
            # self.run_multi_ini()
            # Use labda in button press - next go to run_multi_ini so we can stop

    def loop_direc_diff_params(self):
        # At current this just rins the single file selected
        # Would be cool but a lot of work to make it loop through files and accept new inputs
        # file_list = [filename for filename in self.folder_path.iterdir() if filename.is_file()]
        if not pathlib.Path(self.csv_folder_path.joinpath(self.file_menu.get())).is_file():
            print("No file selected from dropdown menu")
        else:
            name = self.generate_ini()
            self.stop_term()
            command = 'nano_neo -i ' + f'"{name.absolute().as_posix()}"'
            self.proc = subprocess.Popen("exec " + command, shell=True)

    def directory_popup(self):
        # Popup to ask if the user wants to run all files or just the selected file
        self.multi_known = True
        directory_popup = Toplevel(self.root)
        msg = "Do you want to run all files in this directory with the same parameters or just the selected file?"
        entry = ttk.Label(directory_popup, text=msg)
        entry.grid(column=0, row=0, columnspan=2, padx=5, pady=3)
        B1 = ttk.Button(directory_popup, text="All files",
                        command=lambda: [directory_popup.destroy(), self.loop_direc_same_params(),
                                         self.run_multi_ini()])
        B2 = ttk.Button(directory_popup, text="Just this one",
                        command=lambda: [directory_popup.destroy(), self.loop_direc_diff_params()])
        B1.grid(column=0, row=1, padx=5, pady=3, sticky=E)
        B2.grid(column=1, row=1, padx=5, pady=3, sticky=W)
        directory_popup.grid_columnconfigure((0, 1), weight=1)
        directory_popup.grid_rowconfigure((0, 1), weight=1)
        directory_popup.protocol('WM_DELETE_WINDOW', directory_popup.destroy)
        directory_popup.attributes('-topmost', 'true')

    def run_term(self):
        """
        Runs two separate methods
        if yes folder = 1 means that there is a folder selected
            leads to a popup that allows the user to run all the files with the same parameters
        else a single file is selected and run
        """
        if self.yes_folder.get() == 1:  # A folder is selected
            # pop up and ask if run all files, call appropriate loops
            self.directory_popup()
        else:
            name = self.generate_ini()
            self.stop_term()
            #command = 'python ../xes_neo/xes_neo.py -i ' + f'"{name.absolute().as_posix()}"'
            if self.os != 'Windows':
                command = 'exec xes_neo -i '+ f'"{name.absolute().as_posix()}"'
            else:
                raise NotImplementedError('Windows is not supported yet')
            print(command)
            self.proc = subprocess.Popen(''.join(command), shell=True)
            self.proc_list.append(self.proc)

    def build_global(self):
        '''
        Create global tab -  generate ini, run, about, dropdown
        '''

        def about_citation():
            popup = Toplevel()
            popup.wm_title("About: Ver: " + str(self.__version__))
            msg = 'Citation:' \
                  '\nTitle' \
                  '\n Authors' \
                  '\n[Submission], Year'
            cite = ttk.Label(popup, text='Citation:', font='TkTextFont')
            cite.grid(column=0, row=0, sticky=W, padx=self.padx, pady=self.pady)
            citation = scrolledtext.ScrolledText(popup, font="TkTextFont")
            citation.grid(column=0, row=1, padx=self.padx, pady=self.pady)
            with open('media/Citation') as f:
                citation.insert(END, f.read())

            License_Label = ttk.Label(popup, text='License:', font='TkTextFont')
            License_Label.grid(column=0, row=2, sticky=W, padx=self.padx, pady=self.pady)
            license = scrolledtext.ScrolledText(popup)
            license.grid(column=0, row=3, sticky=N + S + W + E, padx=self.padx, pady=self.pady)
            with open('../LICENSE') as f:
                license.insert(END, f.read())
            B1 = ttk.Button(popup, text="Okay", command=popup.destroy)
            B1.grid(column=0, row=4, padx=self.padx, pady=self.pady)

            popup.grid_columnconfigure((1, 3), weight=1)
            popup.grid_rowconfigure((1, 3), weight=1)
            popup.protocol('WM_DELETE_WINDOW', popup.destroy)

        # Column 2 is the dropdown list, is created later as it will only appear if needed
        self.generate_button = ttk.Button(self.root, text="Generate Input", command=self.generate_ini)
        self.generate_button.grid(column=3, row=2, sticky=E, padx=self.padx, pady=self.pady)

        self.run_button = ttk.Button(self.root, text='Run', command=self.run_term)
        self.run_button.grid(column=4, row=2, columnspan=1, sticky=E, padx=self.padx, pady=self.pady)
        self.stop_button = ttk.Button(self.root, text='Stop',
                                      command=lambda: [self.stop_term(), self.run_multi_ini()])

        self.stop_button.grid(column=1, row=2, columnspan=1, sticky=W, padx=self.padx, pady=self.pady)

        self.about_button = ttk.Button(self.root, text='About', command=about_citation)
        self.about_button.grid(column=0, row=2, columnspan=1, sticky=W, padx=self.padx, pady=self.pady)

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=2)
        self.root.grid_columnconfigure(3, weight=1)

        self.root.grid_rowconfigure(0, weight=1)

        # Create an empty frame
        self.label_frame = LabelFrame(self.root, text="Terminal", padx=5, pady=5)
        self.label_frame.grid(column=0, row=1, columnspan=5, padx=self.padx, pady=self.pady, sticky=E + W + N + S)

        # Create the textbox
        self.label_frame.rowconfigure(0, weight=1)
        self.label_frame.columnconfigure(0, weight=1)
        self.txtbox = scrolledtext.ScrolledText(self.label_frame, width=40, height=10)
        self.txtbox.grid(row=0, column=0, sticky=E + W + N + S)

    def file_dropdown(self):
        p = self.csv_folder_path
        self.filelist = [filename.name for filename in p.glob('**/*txt') if filename.is_file()]
        self.file_menu['values'] = self.filelist
        self.file_menu['state'] = 'readonly'
        # sets the width of the combobox to be the length of the first file in the directory, not perfect but best dynamic solution I could think of
        self.file_menu['width'] = len(self.filelist[0])
        self.file_menu.bind("<<ComboboxSelected>>", self.file_selected)

#___Input Tab____________________________________________________________________________________________________________
    def build_inputs_tab(self):
        # Add the tab names for
        arr_input = ["Input Files: ", "Input file", "Input Folder", "Output folder", "Average Files: ", "Files to Average", "Averaged File"]
        self.description_tabs(arr_input, self.input_tab, row=[1, 2, 3, 4, 6, 7, 8])

        self.input_tab.grid_columnconfigure(1, weight=1)

        #Entries_____________________________________________________________________________
        # Check button to choose file or folder input
        checkbutton_label = ttk.Label(self.input_tab, text="Check to select a folder instead of a file", font=self.labelFont)
        checkbutton_label.grid(column=1, row=1, sticky=W)

        # File/Folder Input and Output Folder Entries
        txt_file_entry = ttk.Entry(self.input_tab, textvariable=self.csv_file, font=self.entryFont)
        txt_file_entry.grid(column=1, row=2, sticky=(W, E))

        txt_folder_entry = ttk.Entry(self.input_tab, textvariable=self.csv_folder, font=self.entryFont)
        txt_folder_entry.grid(column=1, row=3, sticky=(W, E))

        output_folder_entry = ttk.Entry(self.input_tab, textvariable=self.output_folder, font=self.entryFont)
        output_folder_entry.grid(column=1, row=4, sticky=(W, E))

        # Line between input/output and averaging entries
        separator = ttk.Separator(self.input_tab, orient='horizontal')
        separator.grid(column=0, row=5, columnspan=4, sticky=W + E, padx=self.padx,pady=self.pady)

        # Section to average files
        entry_files_to_average = ttk.Combobox(self.input_tab, textvariable=self.all_files_display, font=self.entryFont)
        entry_files_to_average.grid(column=1, row=7, sticky=(W,E),padx=self.padx,pady=self.pady)

        entry_averaged_file = ttk.Entry(self.input_tab, textvariable=self.averaged_file, font=self.entryFont)
        entry_averaged_file.grid(column=1, row=8, sticky=(W,E),padx=self.padx,pady=self.pady)
        #______________________________________________________________________________________________________

    # Functions______________________________________________________________________________________________
        def select_folder():
            if self.yes_folder.get() == 1:  # When multiple input is checked
                csv_file_button.config(state=DISABLED)
                csv_folder_button.config(state=NORMAL)
                self.csv_folder.set("Please select a folder")
                self.csv_file.set("Folder is selected")
                if self.pertub_check.get() == 1:  # They are also running multiple instances of each file
                    self.checkbutton_whole_folder.config(state='normal')
                # self.file_dropdown()
                # Because no folder is selected yet this errors when put here
            elif self.yes_folder.get() == 0:  # Not Checked
                csv_file_button.config(state=NORMAL)
                csv_folder_button.config(state=DISABLED)
                self.csv_folder.set("File is selected")
                self.csv_file.set("Please select a file")
                if self.pertub_check.get() == 1:  # They are running multiple instances and previously may have selected a folder - disable folder button
                    self.checkbutton_whole_folder.config(state='disabled')
                    self.run_folder.set(
                        False)  # If they previously selected an entire folder need to now only run through the single file

        # functions for file paths
        def select_csv_file():
            os.chdir(pathlib.Path.cwd().parent)  # change the working directory from gui to nano-indent
            file_name = filedialog.askopenfilename(initialdir=pathlib.Path.cwd(), title="Choose txt/csv",
                                                   filetypes=(("txt files", "*.txt"), ("csv files", "*.csv"),
                                                              ("all files", "*.*")))
            if not file_name:
                self.csv_file.set('Please select a file')
            else:
                self.csv_folder.set("File is selected")
                self.csv_file.set(pathlib.Path(file_name))
                self.csv_generate_from = pathlib.Path(file_name)

                # create the data objectives
                self.data_obj = data.xes_data(pathlib.Path(file_name),int(self.skipLn.get()))

            # disable the dropdown file menu (if user had folder and changed their mind)
            if self.yes_folder.get() == 0:
                self.file_menu.configure(state="disabled")
                self.csv_folder.set("File selected")
            os.chdir(pathlib.Path.cwd().joinpath('gui'))

        def select_csv_folder():
            os.chdir(pathlib.Path.cwd().parent)
            initial_dir = pathlib.Path.cwd()
            folder_name = filedialog.askdirectory(initialdir=pathlib.Path.cwd(), title="Choose a folder")

            if not folder_name:  # They did not select a folder
                self.csv_folder.set("Please choose a folder")
            else:
                folder_path = pathlib.Path(folder_name)
                self.csv_file.set("Folder is selected")
                self.csv_folder.set(folder_path)
                self.csv_folder_path = folder_path  # No file has been selected yet - this is the folder
                # This calls a method to create a dropdown menu next to generate ini button of the files in the directory
                self.file_dropdown()
            os.chdir(pathlib.Path.cwd().joinpath('gui'))

        def select_files_to_average():
            os.chdir("..") #change the working directory from gui to XES

            all_files = filedialog.askopenfilenames(initialdir = os.getcwd(), title = "Choose txt/csv", filetypes = (("txt files", "*.txt"),("csv files","*.csv"),("all files","*.*")))

            if not all_files:
                self.all_files.set('Please choose files to average')
                #os.chdir("gui")
            else:
                self.all_files_display.set(all_files)
                self.all_files = all_files
                #os.chdir("gui")
                #print(self.all_files)
            os.chdir(pathlib.Path.cwd().joinpath('gui'))

        def select_file_to_save_averaged_data():
            os.chdir("..")
            averaged_file = filedialog.asksaveasfilename(initialdir = os.getcwd(), title = "Choose txt/csv", filetypes = (("txt files", "*.txt"),("csv files","*.csv"),("all files","*.*")))

            if not averaged_file:
                self.averaged_file.set('Please choose a file to save averaged data')
                # os.chdir("gui")
            else:
                self.averaged_file.set(averaged_file)
                # os.chdir("gui")
            os.chdir(pathlib.Path.cwd().joinpath('gui'))

        def average_selected_data():
            #Finds lines to skip at top of data that are comments
            lines_to_skip = 2

            with open(self.all_files[0]) as file:
                str=file.readline()
                while(not str.__contains__('***')):
                    lines_to_skip += 1
                    str=file.readline()

            self.num_files = len(self.all_files)
            self.num_points = len(np.loadtxt(self.all_files[0], skiprows=lines_to_skip, usecols=(0,)))

            self.xPoints_for_data_plot = np.loadtxt(self.all_files[0], skiprows=lines_to_skip, usecols=(0,))
            self.yPoints_for_data_plot = np.loadtxt(self.all_files[0], skiprows=lines_to_skip, usecols=(8,))

            for i in range(self.num_files):
                if i > 0:
                    self.xPoints_for_data_plot += np.array(np.loadtxt(self.all_files[i], skiprows=lines_to_skip, usecols=(0,)))
                    self.yPoints_for_data_plot += np.array(np.loadtxt(self.all_files[i], skiprows=lines_to_skip, usecols=(8,)))

            self.xPoints_avg = np.array(self.xPoints_for_data_plot/self.num_files)
            self.yPoints_avg = np.array(self.yPoints_for_data_plot/self.num_files)

            self.plot_data = np.column_stack((self.xPoints_avg,self.yPoints_avg))
            np.savetxt(self.averaged_file.get(), self.plot_data)
        #__________________________________________________________________________________________________
            #os.chdir(pathlib.Path.cwd())

        # _____Input/Output File/Folder Buttons______________________________________________
        multiple_input_button = ttk.Checkbutton(self.input_tab,variable=self.yes_folder,command=select_folder,offvalue=0, onvalue=1)
        multiple_input_button.grid(column=0, row=1, sticky=E)

        csv_file_button = ttk.Button(self.input_tab, text="Select File", command=select_csv_file,style='my.TButton')
        csv_file_button.grid(column=2, row=2, sticky=W,padx=self.padx,pady=self.pady)

        csv_folder_button = ttk.Button(self.input_tab, text="Select Folder", command=select_csv_folder,style='my.TButton')
        csv_folder_button.grid(column=2, row=3, sticky=W,padx=self.padx,pady=self.pady)
        csv_folder_button.config(state=DISABLED)  # Unless the multiple file button is checked this will be disabled

        output_folder_button = ttk.Button(self.input_tab, text="Select Folder", command=self.select_output_folder,style='my.TButton')
        output_folder_button.grid(column=2, row=4, sticky=W,padx=self.padx,pady=self.pady)

        self.file_menu.grid(column=2, row=3, sticky=(W, E))

        self.file_menu.grid(column=2, row=4, sticky=(W, E))

        #_______Averaging buttons________________________________________________
        button_choose_files = ttk.Button(self.input_tab, text="Select Files",command=select_files_to_average, style='my.TButton')
        button_choose_files.grid(column=2, row=7, sticky=W,padx=self.padx,pady=self.pady)

        button_output_file = ttk.Button(self.input_tab,text="Select File",command=select_file_to_save_averaged_data, style='my.TButton')
        button_output_file.grid(column=2, row=8, sticky=W,padx=self.padx,pady=self.pady)

        separator = ttk.Separator(self.input_tab, orient='horizontal')
        separator.grid(column=0, row=9, columnspan=4, sticky=W + E, padx=self.padx,pady=self.pady)

        button_average_data = ttk.Button(self.input_tab,text="Average",command=average_selected_data, style='my.TButton')
        button_average_data.grid(column=0, row=10, sticky=W,padx=self.padx,pady=self.pady)

#___Population Tab____________________________________________________________________________________________
    def build_population_tab(self):
        """
        Build population tab
        """
        arr_pop = ["Population", "Number of generations", "Best individuals (%)", "Lucky survivor (%)"]
        self.description_tabs(arr_pop, self.population_tab, row=[2, 3, 4, 5])
        population_entry = ttk.Entry(self.population_tab, width=7, textvariable=self.population, font=self.entryFont)
        population_entry.grid(column=2, row=2, sticky=W)
        num_gen_entry = ttk.Entry(self.population_tab, width=7, textvariable=self.num_gen, font=self.entryFont)
        num_gen_entry.grid(column=2, row=3, sticky=W)
        best_sample_entry = ttk.Entry(self.population_tab, width=7, textvariable=self.best_sample, font=self.entryFont)
        best_sample_entry.grid(column=2, row=4, sticky=W)
        lucky_few_entry = ttk.Entry(self.population_tab, width=7, textvariable=self.lucky_few, font=self.entryFont)
        lucky_few_entry.grid(column=2, row=5, sticky=W)

#___Mutations Tab____________________________________________________________________________________________
    def build_mutations_tab(self):
        arr_mutations = ["Mutation chance (%)", "Original chance of mutation (%)","Mutation options"]
        self.description_tabs(arr_mutations, self.mutation_tab, row=[2, 3, 4])

        mut_list = list(range(101))
        chance_of_mutation_entry = ttk.Combobox(self.mutation_tab, width=7, textvariable=self.chance_of_mutation,values=mut_list,state="readonly")
        chance_of_mutation_entry.grid(column=4, row=2, sticky=W)

        original_chance_of_mutation_entry = ttk.Combobox(self.mutation_tab, width=7,textvariable=self.original_chance_of_mutation,values=mut_list, state="readonly")
        original_chance_of_mutation_entry.grid(column=4, row=3, sticky=W)

        mutated_options_drop_list = ttk.Combobox(self.mutation_tab, width=2, textvariable=self.mutated_options,values=[0,1,2],state="readonly")
        mutated_options_drop_list.grid(column=4, row=4, sticky=W)

#___Fitting Parameters Tab____________________________________________________________________________________________
    def build_fitting_param_tab(self):
        """
        Build fitting parameters tab
        """
        arr_xes_peaks = ["Number of Peaks"]
        self.description_tabs(arr_xes_peaks, self.fitting_param_tab, row=[2]) #tabs start at row=1. Row=2 is for background and number_of_peaks
        self.peak_state = 'disabled' #initial state for so_split and path_branching

        #Peak type picker:
        self.peak_types = ['Voigt', 'Gaussian', 'Lorentzian','Double Lorentzian'] #leave it like this so it can be accessed elsewhere -evan
        #Line between selections and peak parameters:
        self.separators = ttk.Separator(self.fitting_param_tab,orient = 'horizontal')
        self.separators.grid(column= 0,row = 5,columnspan=8,sticky=W+E,padx=self.padx, pady=self.pady)
        #Peak Type Guess Label:
        path_peakType = ttk.Label(self.fitting_param_tab, text="Fit Type", font=self.labelFont)
        path_peakType.grid_configure(column=2, row=6, sticky=W, padx=self.padx, pady=self.pady)
        #Peak Energy Guess Label:
        peak_Energy = ttk.Label(self.fitting_param_tab,text = "Peak Energy",font = self.labelFont)
        peak_Energy.grid(column = 3,row = 6,sticky = W, padx= self.padx,pady= self.pady)

        self.BE_entries = [0] *10
        self.checkbutton_doublets = [0] *10
        self.peakTypes_entries = [0] *10
        self.branching_entries = [0] *10
        self.so_split_entries = [0] *10 #changed
        self.separators = [0]*10
        self.peak_labels = []
        self.oldNum = 1
        #Creates a row for each peak
        def updatePeakSelectionRows(args):
            self.num = int(self.number_of_peaks.get())
            peak_labels = []
            rows = []
            i=0
            voigt_for_peak_type = 'Voigt'
            #values = self.peak_types in self.peakTypes_entries[i] if you want multiple background types
            for row in range(7,7+(2*self.num),2):
                peak_labels.append("Peak " + str(i+1))
                rows.append(row)
                # print("i" + str(i))
                self.peakTypes_entries[i] = ttk.Combobox(self.fitting_param_tab, textvariable=self.peaks[i], font=self.entryFont, values= voigt_for_peak_type)
                self.peakTypes_entries[i].grid(column=2, row=row, sticky=W)
                #Peak Energy:
                self.BE_entries[i] = ttk.Entry(self.fitting_param_tab, textvariable=self.peak_Energy_guesses[i], font=self.entryFont)
                self.BE_entries[i].grid(column=3, row=row, sticky=(W, E))
                i+=1
            #destroy any that were leftover
            for k in range(self.num,self.oldNum):
                #print(type(self.peakTypes_entries[k]))
                #print(self.peak_labels)
                    self.peakTypes_entries[k].destroy()
                    self.sigma_entries[k].destroy()
                    self.checkbutton_doublets[k].destroy()
                    self.separators[k].destroy()

            for label in self.peak_labels:
                label.destroy()
            #print(peak_labels)
            self.peak_labels = self.description_tabs(peak_labels,self.fitting_param_tab,row = rows)
            self.oldnum = self.num

        #Nunber of Peaks:
        number_of_peaks_options = [1,2,3,4,5,6,7,8,9,10]
        number_of_peaks_entry = ttk.Combobox(self.fitting_param_tab, textvariable=self.number_of_peaks, font=self.entryFont,values= number_of_peaks_options)
        number_of_peaks_entry.grid(column=2, row=2, sticky=(W, E)) #on same row as background checkbox
        number_of_peaks_entry.bind('<<ComboboxSelected>>', updatePeakSelectionRows)

        #Backgrounds Checkboxes:
        path_bkgn = ttk.Label(self.fitting_param_tab, text="Background Type:", font=self.labelFont)
        path_bkgn.grid_configure(column=3, row=2, sticky=W, padx=self.padx, pady=self.pady)

        global background_types
        self.background_types = [] #make an array of the background types selected
        #-----Linear Background-----
        self.linear_selected = 1
        def linear_bkgn():
            global background_types
            global slope_selected
            if (self.linear_selected % 2) == 0: #state of button is off. Used when button is clicked on then off again.
                self.background_types.remove('Linear')
                self.linear_selected = 1
            else: #state of button is on
                self.background_types.append('Linear')
                self.linear_selected = 2
            return self.linear_selected

        checkbutton_doublet = ttk.Checkbutton(self.fitting_param_tab, text="Linear", command=linear_bkgn)
        checkbutton_doublet.grid(column=5, row=2, sticky=W)
        checkbutton_doublet.state(['!alternate'])

        #doublet_peak()
        updatePeakSelectionRows(0)

#___Plot Tab____________________________________________________________________________________________
    def build_plot_tab(self):
        """
        Build plot tab
        """
        self.plot_tab.columnconfigure(0, weight=1)
        self.plot_tab.rowconfigure(1, weight=1)

        # def plot_selection():
        #     #self.data_obj.pre_processing((self.percent_min.get(), self.percent_max.get()))
        #     data_plot.initial_parameters(self.data_obj)
        #     data_plot.plot_selected()

        def plot_raw():
            #self.data_obj.pre_processing((self.percent_min.get(), self.percent_max.get()))
            data_plot.initial_parameters(self.data_obj)
            x = []
            y = []
            x.append(self.data_obj.get_x())
            y.append(self.data_obj.get_y())
            data_plot.plot(x,y,'Raw Data', 'Raw Data')

        # def plot_both():
        #     self.data_obj.pre_processing((self.percent_min.get(), self.percent_max.get()))
        #     data_plot.initial_parameters(self.data_obj, title=self.csv_generate_from.stem)
        #     data_plot.plot_raw_and_selected()

        data_plot = Data_plot(self.plot_tab)
        self.plot_button = ttk.Button(self.plot_tab, text='Plot Data', command=plot_raw)
        self.plot_button.grid(column=0, row=0, columnspan=1, sticky=W, padx=self.padx, pady=self.pady)

        # self.plot_selected_button = ttk.Button(self.plot_tab, text='Plot Selected Range', command=plot_selection)
        # self.plot_selected_button.grid(column=1, row=0, columnspan=1, sticky=W, padx=self.padx, pady=self.pady)

        # self.plot_both_button = ttk.Button(self.plot_tab, text='Plot Raw and Selected', command=plot_both)
        # self.plot_both_button.grid(column=2, row=0, columnspan=1, sticky=W, padx=self.padx, pady=self.pady)

    def generate_randomized_ini(self, multifolder, i):
        pop_range = np.arange(self.pop_min.get(), self.pop_max.get(), 100)
        gen_range = np.arange(self.gen_min.get(), self.gen_max.get(), 5)
        mut_range = np.arange(self.mut_min.get(), self.mut_max.get(), 10)

        self.population.set(np.random.choice(pop_range))
        self.num_gen.set(np.random.choice(gen_range))
        self.chance_of_mutation.set(np.random.choice(mut_range))
        name = self.csv_generate_from.stem

        def unique_path():
            counter = i
            while True:
                num_name = str(name) + "_" + str(counter) + '.ini'
                path = self.output_folder_path.joinpath(num_name)
                if not path.exists():
                    return path
                counter += 1

        file_path = unique_path()
        file_path.touch()
        self.write_ini(file_path)

    def generate_multi_ini(self):
        if self.run_folder.get():  # They want to generate ini for every file in directory
            # generates list of files within the folder
            file_list = [filename for filename in self.csv_folder_path.glob('**/*txt') if filename.is_file()]
            # Loop through each file and generate proper number of ini files
            # stem = self.output_folder_path.stem
            # parent = self.output_folder_path.parent
            for i in range(len(file_list)):
                # set the generate_from file to the current file
                self.csv_generate_from = file_list[i]
                fname = file_list[i].stem
                # create specified number of iterations for this file
                for j in range(self.n_ini.get()):
                    # Gives the output path a unique file name
                    name = fname + '_' + str(j) + '_out' + '.txt'
                    output_path = self.output_folder_path.joinpath(name)
                    # output_path.touch()
                    self.output_file = output_path
                    # print("generate_multi output name: ", self.output_folder_path)
                    self.generate_randomized_ini(self.output_folder_path, j)
        else:
            # stem = self.output_folder_path.stem
            for i in range(self.n_ini.get()):
                # Gives the output path a unique file name
                name = self.csv_generate_from.stem + '_' + str(i) + '_out' + '.txt'
                # parent = self.output_folder_path.parent
                output_path = self.output_folder_path.joinpath(name)
                # output_path.touch()
                self.output_file = output_path
                # print("generate_multi output name: ", self.output_folder_path)
                self.generate_randomized_ini(self.output_folder_path, i)
        return self.output_folder_path

    def stop_all(self):
        self.stop_not_pressed = False
        for i in self.proc_list:
            i.kill()
            i.wait()
        while not self.command_list.empty():
            self.command_list.get()
        print("Stopped nano_neo")

    def run_ini_in_command_list(self, flag):
        global loop
        if self.stop_not_pressed and not self.command_list.empty():
            each = self.command_list.get()
            command = "exec nano_neo -i " + each
            self.proc = subprocess.Popen(command, shell=True)
            self.proc.wait()
            self.proc_list.append(self.proc)
            loop = self.root.after(self.run_ini_in_command_list(True))
        else:
            try:
                self.root.after_cancel(loop)
            except:
                pass

        # while self.stop_not_pressed and len(self.command_list) > 0:

    def set_command_list(self):
        self.output_folder_path = self.generate_multi_ini()
        print("in run multi. flag value", self.stop_not_pressed)
        file_list = [f'"{_file.absolute().as_posix()}"' for _file in self.output_folder_path.glob('**/*.ini') if
                     _file.is_file()]
        print("File list in run multi ", file_list)
        for i in range(len(file_list)):
            print("in for ", i)
            self.command_list.put(file_list[i])

    def run_multi_ini(self):
        # Runs all instances of a single file

        print("before loop")
        global loop
        if self.stop_not_pressed and self.command_list:
            print("Stop not pressed and files exist")
            # print("\n\n\n\n\n")
            each = self.command_list.get()
            command = "exec nano_neo -i " + each
            self.proc = subprocess.Popen(command, shell=True)
            self.proc_list.append(self.proc)
            self.pid_list.append(self.proc.pid)
            # print("Current process ID: ", self.proc.pid)
            loop = self.root.after(0, self.run_multi_ini)
            self.proc.wait()
        else:
            # print("in else")
            try:
                # print("trying to cancel")
                self.root.after_cancel(loop)
                # Empty any unrun commands so they do not run on next iteration
                while self.command_list:
                    self.command_list.get()
            except:
                print("In pass")
                pass
            self.stop_not_pressed = True

        # if self.stop_not_pressed:
        #   print("if self.stop_not_pressed is yes")
        #  self.output_folder_path = self.generate_multi_ini()
        # if self.output_folder.get() =='Please choose a folder to save outputs' or not self.output_folder_path:
        #    print("skipped to not running pls work")
        #   return
        # else:
        #   print("in else of run multi")
        # file_list = [str(filename) for filename in self.output_folder_path.glob('**/*.ini') if filename.is_file()]
        # file_list = [f'"{_file.absolute().as_posix()}"' for _file in self.output_folder_path.glob('**/*.ini') if
        #            _file.is_file()]
        #  print("File list in run multi ", file_list)
        # for i in range(len(file_list)):
        #    print("in for ", i)
        #   self.command_list.append(file_list[i])
        # pls_run(self.stop_not_pressed)
        # else:
        #   print("else of run_multi should cause stop")
        #  pls_run(self.stop_not_pressed) # looks the same but it will send in false (hopefully) & cause after_cancel

        # self.run_ini_in_command_list(True)

    def runningThread(self):
        t1 = Thread(target=self.runningmulti)
        print("In runningThread")
        t1.start()

    def runningmulti(self):
        self.set_command_list()
        self.run_multi_ini()
# Output Tab ________________________________________________________________________________________________________
    def build_output_tab(self):
        """
        Will allow for multiple iterations over the same data to be performed
        Each time create & save ini, run, save outputs
        """

        # pertub_check = IntVar(self.output_tab, 0)

        def checkbox_multi():
            widget_lists = [
                entry_n_ini,
                entry_pertub_pop_min,
                entry_pertub_pop_max,
                entry_pertub_gen_min,
                entry_pertub_gen_max,
                entry_pertub_mut_min,
                entry_pertub_mut_max,
                button_gen_nini,
                button_run_nini]
            if self.pertub_check.get() == 0:
                for i in widget_lists:
                    i.config(state='disabled')
                    self.checkbutton_whole_folder.config(state='disabled')
            elif self.pertub_check.get() == 1:
                for i in widget_lists:
                    i.config(state='normal')
                    if self.yes_folder.get() == 1:  # Check to see if folder is selected
                        self.checkbutton_whole_folder.config(state='normal')

        arr_out = ["Print graph", "Steady state exit"]
        self.description_tabs(arr_out, self.output_tab)

        checkbutton_print_graph = ttk.Checkbutton(self.output_tab, var=self.print_graph)
        checkbutton_print_graph.grid(column=1, row=0, sticky=W + E, padx=self.padx)

        checkbutton_steady_state = ttk.Checkbutton(self.output_tab, var=self.steady_state_exit)
        checkbutton_steady_state.grid(column=1, row=1, sticky=W + E, padx=self.padx)

        # Create separators
        separator = ttk.Separator(self.output_tab, orient='horizontal')
        separator.grid(column=0, row=2, columnspan=4, sticky=W + E, padx=self.padx)
        self.output_tab.columnconfigure(3, weight=1)

        arr_out = ["Create Multiple Input Files", "Number of Ini Files", "Pertubutions-Population(min,max)",
                   "Pertubutions-Generation(min,max)", "Pertubutions-Mutation(min,max)"]
        self.description_tabs(arr_out, self.output_tab, row=[3, 5, 6, 7, 8])
        # Create New pertubutuions

        checkbutton_pertub = ttk.Checkbutton(self.output_tab, var=self.pertub_check, command=checkbox_multi)
        checkbutton_pertub.grid(column=1, row=3, sticky=W + E, padx=self.padx)

        pertub_list = list(range(1, 101))

        text = 'Each entry allows user to control perturbation percentage of the desire variables.'
        entry = ttk.Label(self.output_tab, text=text, font=self.labelFont)
        entry.grid_configure(column=0, row=4, columnspan=3, sticky=W + E, padx=self.padx, pady=self.pady)

        entry_n_ini = ttk.Entry(self.output_tab, textvariable=self.n_ini, font=self.entryFont)
        entry_n_ini.grid(column=1, row=5, columnspan=2, sticky=(W, E), padx=self.padx)
        entry_n_ini.config(state='disabled')

        width = 5
        # --------------
        entry_pertub_pop_min = ttk.Entry(self.output_tab, width=width, textvariable=self.pop_min, font=self.entryFont)
        entry_pertub_pop_min.grid(column=1, row=6, sticky=(W, E), padx=self.padx)

        entry_pertub_pop_max = ttk.Entry(self.output_tab, width=width, textvariable=self.pop_max, font=self.entryFont)
        entry_pertub_pop_max.grid(column=2, row=6, sticky=(W, E), padx=self.padx)

        entry_pertub_pop_min.config(state='disabled')
        entry_pertub_pop_max.config(state='disabled')

        # --------------
        entry_pertub_gen_min = ttk.Entry(self.output_tab, width=width, textvariable=self.gen_min, font=self.entryFont)
        entry_pertub_gen_min.grid(column=1, row=7, sticky=(W, E), padx=self.padx)

        entry_pertub_gen_max = ttk.Entry(self.output_tab, width=width, textvariable=self.gen_max, font=self.entryFont)
        entry_pertub_gen_max.grid(column=2, row=7, sticky=(W, E), padx=self.padx)

        entry_pertub_gen_min.config(state='disabled')
        entry_pertub_gen_max.config(state='disabled')

        # --------------
        entry_pertub_mut_min = ttk.Entry(self.output_tab, width=width, textvariable=self.mut_min, font=self.entryFont)
        entry_pertub_mut_min.grid(column=1, row=8, sticky=(W, E), padx=self.padx)

        entry_pertub_mut_max = ttk.Entry(self.output_tab, width=width, textvariable=self.mut_max, font=self.entryFont)
        entry_pertub_mut_max.grid(column=2, row=8, sticky=(W, E), padx=self.padx)

        entry_pertub_mut_min.config(state='disabled')
        entry_pertub_mut_max.config(state='disabled')

        # --------------

        button_gen_nini = ttk.Button(self.output_tab, text="Generate Input Files", command=self.generate_multi_ini)
        button_gen_nini.grid(column=0, row=9, columnspan=3, sticky=W + E, padx=self.padx, pady=self.pady)
        button_gen_nini.config(state='disabled')

        button_run_nini = ttk.Button(self.output_tab, text="Run All Instances",
                                     command=lambda: [self.set_command_list(), self.run_multi_ini()])
        button_run_nini.grid(column=0, row=10, columnspan=3, sticky=W + E, padx=self.padx, pady=self.pady)
        button_run_nini.config(state='disabled')

        # Adding button to chose if run all files in the folder
        self.checkbutton_whole_folder = ttk.Checkbutton(self.output_tab, var=self.run_folder)
        self.checkbutton_whole_folder.grid(column=1, row=11, sticky=W + E, padx=self.padx)
        self.checkbutton_whole_folder.config(state='disabled')

        checkbutton_label = ttk.Label(self.output_tab,
                                      text="Check to generate/run iterations for each file in the directory",
                                      font=self.labelFont)
        checkbutton_label.grid(column=0, row=11, sticky=W)
#___Analysis Tab____________________________________________________________________________________________
    def build_analysis_tab(self):
        '''
        arr_col_0 = ['h_f', 'm', 'A', 'Elastic modulus (GPa)', 'Reduced Modulus', 'Stiffness (S)', 'Hardness, H (MPa)',
                     'Max Load (un)', 'Max Depth (nm)']
                     '''
        #arr_col_0 = ['Sigma','Lorentz','Amp','ChiSqr', 'Peak', 'Peak 2', 'Peak 3']
        #self.description_tabs(arr_col_0, self.analysis_tab, sticky='W', row=(1, 2, 3, 4, 5, 6, 7))

        def select_analysis_folder():
            os.chdir(pathlib.Path.cwd().parent)  # change the working directory from gui to nano-indent

            folder_name = filedialog.askdirectory(initialdir=pathlib.Path.cwd(), title="Choose a folder")
            if not folder_name:
                self.analysis_dir.set('Please choose a directory')
            else:
                # folder_name = os.path.join(folder_name,'feff')
                self.analysis_dir.set(folder_name)

            os.chdir(pathlib.Path.cwd().joinpath('gui'))

        def calculate_and_plot():
            self.background_types = ['Linear']
            params = {
                'base': pathlib.Path.cwd().parent,
                'file': self.csv_generate_from,
                'fileName' : self.csv_file.get(),
                'peaks' : self.peak_types,
                'bkgns' : self.background_types,
                'data obj' : self.data_obj

            }
            self.params = self.analysis_obj.initial_parameters(self.analysis_dir,params,title='Fit')
            #get_params_for_export()
            self.numPeaks = int(self.number_of_peaks.get())
            export_data = []
            for i in range(self.numPeaks+1):
                print(i)
                export_data.append([StringVar(value=0),StringVar(value=0),StringVar(value=0),StringVar(value=0)])

            print(export_data)
            print(self.params)
            print(len(export_data))
            for i,param in enumerate(self.params):
                if i == self.numPeaks:
                    export_data[i][0].set(param[0])
                else:
                    for j in range(4):
                        export_data[i][j].set(param[j])
            self.export_data = export_data
            updateExportDataRows(self)

        # def get_params_for_export():
        #     self.analysis_export = self.analysis_obj.get_params(self)
        #     return self.analysis_export

        # Labels
        self.peak_labels =[0]*10
        self.sigma_labels = [0]*10
        self.lorentzian_labels = [0]*10
        self.amp_labels = [0]*10
        # Entries
        self.peak_energy_entries = [0]*10
        self.sigma_entries = [0]*10
        self.lorentzian_entries = [0]*10
        self.amp_entries = [0]*10
        # Export Data Values
        self.peak_energy_export = [0]*10
        self.sigma_export = [0]*10
        self.lorentzian_export = [0]*10
        self.amp_export = [0]*10

        def updateExportDataRows(self):
            self.num = int(self.number_of_peaks.get())
            self.Peak_number = abs(self.peak_number)
            peak_labels = []
            sigma_labels = []
            lorentzian_labels = []
            amp_labels = []
            rows = []
            i=0
            voigt_for_peak_type = 'Voigt'
            #values = self.peak_types in self.peakTypes_entries[i] if you want multiple background types
            

            for row in range(1,((4*self.Peak_number)+1),4):
                peak_labels.append("Peak " + str(i+1))
                sigma_labels.append("Sigma " + str(i+1))
                lorentzian_labels.append("Lorentz " + str(i+1))
                amp_labels.append("Amp " + str(i+1))
                
                rows.append(row)
                print(rows)

                #Peak Energy:
                self.peak_energy_entries[i] = ttk.Label(self.analysis_tab, textvariable=self.export_data[i][0], font=self.entryFont)
                self.peak_energy_entries[i].grid(column=1, row=row, sticky=(W, E))

                self.sigma_entries[i] = ttk.Label(self.analysis_tab, textvariable=self.export_data[i][1], font=self.entryFont)
                self.sigma_entries[i].grid(column=1, row=row+1, sticky=(W, E))

                self.lorentzian_entries[i] = ttk.Label(self.analysis_tab, textvariable=self.export_data[i][2], font=self.entryFont)
                self.lorentzian_entries[i].grid(column=1, row=row+2, sticky=(W, E))

                self.amp_entries[i] = ttk.Label(self.analysis_tab, textvariable=self.export_data[i][3], font=self.entryFont)
                self.amp_entries[i].grid(column=1, row=row+3, sticky=(W, E))
                
                i += 1

            #print(peak_labels)
            sigma_rows = []
            lorentzian_rows = []
            amp_rows = []
            for i in rows:
                sigma_rows.append(i+1)
                lorentzian_rows.append(i+2)
                amp_rows.append(i+3)

            self.peak_labels = self.description_tabs(peak_labels,self.analysis_tab,row = rows)
            self.sigma_labels = self.description_tabs(sigma_labels,self.analysis_tab,row = sigma_rows)
            self.lorentzian_labels = self.description_tabs(lorentzian_labels,self.analysis_tab,row = lorentzian_rows)
            self.amp_labels = self.description_tabs(amp_labels,self.analysis_tab,row = amp_rows)

        """
        TODO:
        read in output file and get the relevant values to put in the boxes
        """
        self.analysis_obj = Analysis_plot(self.analysis_tab)

        # analysis_chisqr = StringVar(self.analysis_tab, 0.0)
        # analysis_background = StringVar(self.analysis_tab, 0.0)

        # For now put in placeholders
        # Select Folder Button
        analysis_button = ttk.Button(self.analysis_tab, text="Select Folder",command=select_analysis_folder)  # Add command to export data
        analysis_button.grid(column=0, row=0, sticky=(W, E), padx=self.padx, pady=self.pady, columnspan=2)

        self.peak_number = int(0)
        peak_number_options = [1,2,3,4,5,6,7,8,9,10]
        peak_number_entry = ttk.Combobox(self.analysis_tab, textvariable=self.peak_number, font=self.entryFont,values= peak_number_options)
        peak_number_entry.grid(column=1, row=1, sticky=(W, E)) #on same row as background checkbox
        peak_number_entry.bind('<<ComboboxSelected>>', updateExportDataRows)

        # Entries___________________________________________________________________________________________
        # entry_chisqr_best = ttk.Label(self.analysis_tab, textvariable=analysis_chisqr, font=self.entryFont, borderwidth=2,
        #                           relief="groove", background='#a9a9a9')
        # entry_chisqr_best.grid(column=1, row=4, sticky=(W, E), padx=self.padx)

        # entry_background = ttk.Label(self.analysis_tab, textvariable=analysis_background, font=self.entryFont,borderwidth=2,
        #                               relief="groove", background='#a9a9a9')
        # entry_background.grid(column=1, row=5, sticky=(W, E), padx=self.padx)

        # Number of Peak to Find Row
        self.numPeaks = int(self.number_of_peaks.get())

        # Chisqr Entry
        # entry_chisqr_best = ttk.Label(self.analysis_tab, textvariable=analysis_chisqr, font=self.entryFont, borderwidth=2,
        #                           relief="groove", background='#a9a9a9')
        # entry_chisqr_best.grid(column=1, row=4, sticky=(W, E), padx=self.padx)

        # Plot Best Fit Button
        button_plot = ttk.Button(self.analysis_tab,text="Plot Best Fit",command=calculate_and_plot)  # Add command to plot data using postprocessing
        button_plot.grid(column=0, row=((4*self.numPeaks)+5), sticky=(W, E), padx=self.padx, pady=self.pady, columnspan=2)

        # Export Values Button

        button_export = ttk.Button(self.analysis_tab, text="Export Values")  # Add command to export data
        button_export.grid(column=0, row=((4*self.numPeaks)+6), sticky=(W, E), padx=self.padx, pady=self.pady, columnspan=2)

        self.analysis_tab.columnconfigure(3, weight=1)
        self.analysis_tab.rowconfigure(0, weight=1)
        self.analysis_tab.rowconfigure(1, weight=1)
        self.analysis_tab.rowconfigure(2, weight=1)
        self.analysis_tab.rowconfigure(3, weight=1)
        self.analysis_tab.rowconfigure(4, weight=1)
        self.analysis_tab.rowconfigure(5, weight=1)
        self.analysis_tab.rowconfigure(6, weight=1)
        self.analysis_tab.rowconfigure(7, weight=1)
        self.analysis_tab.rowconfigure(8, weight=1)
        self.analysis_tab.rowconfigure(9, weight=1)

    def stop_term(self):
        # print("In stop term")
        # print("PID TO KILL ", self.pid_list[len(self.pid_list)-1])
        # command = 'kill -9 ', self.pid_list[len(self.pid_list)-1]
        # os.killpg(self.pid_list[len(self.pid_list)-1], signal.SIGTERM)
        # subprocess.Popen(command)
        # print("/n/n/n/n/n/n/n/n/n/n"
        #       "**************"
        #       "*******************"
        #       "(++++&****************************"
        #       ""
        #       "/n/n/n/n/n/n")
        self.stop_not_pressed = False
        if not self.command_list.empty():
            self.stop_all()
        elif hasattr(self, 'proc'):
            self.proc.kill()
            self.proc.terminate()
            print("Stopped xes_neo")
            self.proc.kill()

    def on_closing(self):
        """
        on closing function
        """
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.stop_term()
            if hasattr(self, 'terminal'):
                self.root.quit()
                self.terminal.destroy()
            else:
                self.root.quit()

    def Run(self):
        """
        Run the code
        """
        self.root.protocol('WM_DELETE_WINDOW', self.on_closing)
        self.root.mainloop()


root = App()
root.Run()
