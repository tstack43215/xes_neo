# from xes_fit import background
from .input_arg import *
from . import helper

# ------------------------------------
# Andy Lau
# 4/22/2019
# goal : Parsed the ini into each specific documents.
#-------------------------------------
def split_string(var_dict,label):
	"""
	Read the path list
	"""
	# print(num_compounds)
	arr_str = var_dict[label]
	starter = []
	end = []
	k = 0
	split_str = []
	for i in arr_str:
		if i == '[':
			starter.append(k)
		elif i == ']':
			end.append(k)
		k = k + 1


	assert(len(starter) == len(end)),'Bracket setup not right.'
	# if num_compounds > 1:
	# 	assert(num_compounds == len(starter)),'Number of compounds not matched.'
	# 	assert(num_compounds == len(end)),'Number of compounds not matched.'

	# check if both are zeros, therefore the array is one 1 dimensions
	# arr_str = optional_var(var_dict,label,[],list)
	if len(starter) == 0 and len(end) == 0:
		split_str = list(arr_str.split(","))
	f_arr = []
	for i in range(len(split_str)):
		f_arr.append(float(split_str[i]))

	return f_arr

def optional_var(dict,name_var,alt_var=None,type_var=int):
	"""
	Detections of optional variables exists within input files, and
		put in corresponding default inputs parameters.
	"""
	# boolean needs special attentions
	if type_var == bool:
		if name_var in dict:
			return_var = helper.str_to_bool(dict[name_var])
		else:
			return_var = alt_var
	elif type_var == None:
		if name_var in dict:
			return_var = dict[name_var]
		else:
			return_var = None
	else:
		if name_var in dict:
			return_var = type_var(dict[name_var])
		else:
			return_var = type_var(alt_var)
	return return_var

def optional_range(var_dict,label):
	if label not in var_dict:
		# Label is not therefore
		return_var = []
	else:
		return_var = split_string(var_dict,label)

	return return_var
def optional_range_string(var_dict, label):
    if label not in var_dict:
        return_var = []
    else:
       return_var = var_dict[label].split(',')


    return return_var


# -----
Inputs_dict = file_dict['Inputs']
Populations_dict = file_dict['Populations']
Mutations_dict = file_dict['Mutations']
Paths_dict = file_dict['Paths']
Outputs_dict = file_dict['Outputs']
# -----

# Input
data_file = Inputs_dict['data_file']
output_file = Inputs_dict['output_file']
#data_cutoff = [float(x) for x in Inputs_dict['data_cutoff'].split(',')] -Unneccesary, remove later when we have stable build
#pathrange_file = optional_var(Inputs_dict,'pathrange_file',None,None)		-evan


# population
size_population = int(Populations_dict['population'])
number_of_generation = int(Populations_dict['num_gen'])
best_sample = int(Populations_dict['best_sample'])
lucky_few = int(Populations_dict['lucky_few'])

# Mutations
chance_of_mutation = int(Mutations_dict['chance_of_mutation'])
original_chance_of_mutation = int(Mutations_dict['original_chance_of_mutation'])
mutated_options = int(Mutations_dict['mutated_options'])

# Paths  -Should change the name of this to be peak or params or something later for readability - evan
npaths = int(Paths_dict['npeaks'])
background_type = optional_range_string(Paths_dict,'background_type')
peak_type = optional_range_string(Paths_dict,'peak_type')
peak_energy_range = optional_range(Paths_dict,'peak_energy_range')
peak_energy_guess = optional_range(Paths_dict,'peak_energy')
sigma_range = optional_range(Paths_dict,'sigma_range')
fwhm_range = optional_range(Paths_dict,'fwhm_range')
amp_range = optional_range(Paths_dict,'amp_range',)
background_range = optional_range(Paths_dict,'background_range',)
slope_range = optional_range(Paths_dict,'slope_range',)

for i in range(len(peak_type)):
	peak_type[i].replace(" ","")

print("Background type is " + ', '.join(background_type))
print("Peak Type is " + ', '.join(peak_type))
print("Peak Energy Range is " + str(peak_energy_range))
print("Trying Peak Energy " + str(peak_energy_guess))
print("Sigma Range is " + str(sigma_range))
print("Fwhm Range is " + str(fwhm_range))
print("Amplitude Range is " + str(amp_range))
print("Background Range is " + str(background_range))
print("Slope Range is " + str(slope_range))

# Output
printgraph = helper.str_to_bool(Outputs_dict['print_graph'])
num_output_paths = helper.str_to_bool(Outputs_dict['num_output_paths'])
steady_state = optional_var(Outputs_dict,'steady_state_exit',False,bool)
