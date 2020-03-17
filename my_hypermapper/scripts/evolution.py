import sys
import os
import space
import numpy as np
import csv
import json
import copy
import datetime
from jsonschema import Draft4Validator, validators, exceptions
from utility_functions import *
from collections import defaultdict
from scipy import stats


# Just to see how to document properly
def arbitrary_function(arg1, arg2):
    """
    Explain what I do
    :param arg1: I do this.
    :param arg2: I do that.
    :return: What is my data type and what do I contain.
    """
    return 0


# Taken from local_search and modified
def run_objective_function(
                        configurations,
                        hypermapper_mode,
                        param_space,
                        beginning_of_time,
                        run_directory,
                        evolution_data_array,
                        fast_addressing_of_data_array,
                        enable_feasible_predictor=False,
                        evaluation_limit=float("inf"),
                        black_box_function=None,
                        number_of_cpus=0):
    """
    Evaluate a list of configurations using the black-box function being optimized.
    This method avoids evaluating repeated points by recovering their value from the history of evaluated points.
    :param configurations: list of configurations to evaluate.
    :param x hypermapper_mode: which HyperMapper mode is being used.
    hypermapper_mode == "default"
    :param param_space: a space object containing the search space.
    :param beginning_of_time: timestamp of when the optimization started.
    :param run_directory: directory where HyperMapper is running.
    :param evolution_data_array: a dictionary containing all of the configurations that have been evaluated.
    :param fast_addressing_of_data_array: a dictionary containing evaluated configurations and their index in the evolution_data_array.
    :param x enable_feasible_predictor: whether to use constrained optimization.
    :param (x) evaluation_limit: the maximum number of function evaluations allowed for the evolutionary search.
    :param black_box_function: the black_box_function being optimized in the evolutionary search.
    :param number_of_cpus: an integer for the number of cpus to be used in parallel.
    :return: data_array with evaluations for all points in configurations.
    """
    new_configurations = []
    new_evaluations = {}
    previous_evaluations = defaultdict(list)
    number_of_new_evaluations = 0
    t0 = datetime.datetime.now()
    absolute_configuration_index = len(fast_addressing_of_data_array)

    # Adds configutations to new if they have not been evaluated before
    for configuration in configurations:
        str_data = param_space.get_unique_hash_string_from_values(configuration)
        if str_data in fast_addressing_of_data_array:
            configuration_idx = fast_addressing_of_data_array[str_data]
            for key in evolution_data_array:
                previous_evaluations[key].append(evolution_data_array[key][configuration_idx])
        else:
            if absolute_configuration_index + number_of_new_evaluations < evaluation_limit:
                new_configurations.append(configuration)
                number_of_new_evaluations += 1

    # Evaluates new configurations. If there is any
    t1 = datetime.datetime.now()
    if number_of_new_evaluations > 0:
        # if hypermapper_mode = default, this equals run_configurations_with_black_box_function()
        # All new evaluations are printed, deep down
        new_evaluations = param_space.run_configurations(hypermapper_mode, new_configurations, beginning_of_time,
                                                         black_box_function, run_directory)

    # Values for all given configurations
    all_evaluations = concatenate_data_dictionaries(previous_evaluations, new_evaluations)
    all_evaluations_size = len(all_evaluations[list(all_evaluations.keys())[0]])

    for idx in range(number_of_new_evaluations):
        configuration = get_single_configuration(new_evaluations, idx)
        for key in configuration:
            evolution_data_array[key].append(configuration[key])

        str_data = param_space.get_unique_hash_string_from_values(configuration)
        fast_addressing_of_data_array[str_data] = absolute_configuration_index
        absolute_configuration_index += 1
    sys.stdout.write_to_logfile(("Time to run new configurations %10.4f sec\n" % ((datetime.datetime.now() - t1).total_seconds())))
    sys.stdout.write_to_logfile(("Total time to run configurations %10.4f sec\n" % ((datetime.datetime.now() - t0).total_seconds())))

    # return list(scalarized_values), feasibility_indicators
    return all_evaluations_size


def evolution(population_size, param_space, fast_addressing_of_data_array, optimization_function, optimization_function_parameters):

    """
    Do the entire evolutinary process from config to best config
    :param population_size: an integer for the number of configs to keep. All will be initiated randomly
    :param param_space: a space object containing the search space.
    :param optimization_function: the function that will be optimized by the evolutionary search.
    :param optimization_function_parameters: a dictionary containing the parameters that will be passed to the optimization function.
    :return: all points evaluted and the best point found by the Evolutionary Algorithm.
    """

    t0 = datetime.datetime.now()
    tmp_fast_addressing_of_data_array = copy.deepcopy(fast_addressing_of_data_array)
    # input_params = param_space.get_input_parameters()
    # feasible_parameter = param_space.get_feasible_parameter()[0]
    # How does it differ from *_data_array in the **dict?
    data_array = {}
    # end_of_search = False


    # 1. Get a population by taking random samples
    # 2. evolutionary loop
    #   3. evaluate fitness
    #   4. update population
    #       5 crossover, random sampling, mutation
    # return best config. and all seen samples


    # 1
    # data_array = concatenate_data_dictionaries(data_array, new_data_array)

    # I think the point of this is to always get the default configuration, if there is one
    default_configuration = param_space.get_default_or_random_configuration()
    str_data = param_space.get_unique_hash_string_from_values(default_configuration)
    if str_data not in fast_addressing_of_data_array:
        tmp_fast_addressing_of_data_array[str_data] = 1
        if population_size - 1 > 0:     # Will Allways be true
            configurations = [default_configuration] + param_space.random_sample_configurations_without_repetitions(
                tmp_fast_addressing_of_data_array, population_size - 1)
    else:
        configurations = param_space.random_sample_configurations_without_repetitions(tmp_fast_addressing_of_data_array,
                                                                                      population_size)

    # Passing the dictionary with ** expands the key-value pairs into function parameters
    # could this be a void function? all_evaluations_size = population_size
    function_values_size = optimization_function(configurations=configurations, **optimization_function_parameters)

    # This will concatenate the entire data array if all configurations were evaluated
    # but only the evaluated configurations if we reached the budget and did not evaluate all
    # So in my case function_values_size is useless?
    new_data_array = concatenate_list_of_dictionaries(configurations[:function_values_size])
    data_array = concatenate_data_dictionaries(data_array, new_data_array)

    # Jag tror jag har tagit det jag vill ha av delen före best improvement local search
    # 2

    best_configuration = 0      # Do we need this?

    sys.stdout.write_to_logfile(("Evolution time %10.4f sec\n" % ((datetime.datetime.now() - t0).total_seconds())))

    return data_array, best_configuration


def main(config, black_box_function=None, output_file=""):
    """
    Run design-space exploration using evolution.
    :param config: dictionary containing all the configuration parameters of this design-space exploration.
    :param output_file: a name for the file used to save the dse results.
    :return:
    """

    # Space basically turn config into a class with many handy functions
    param_space = space.Space(config)

    # I probably want to do this
    run_directory = config["run_directory"]
    application_name = config["application_name"]
    hypermapper_mode = config["hypermapper_mode"]["mode"]
    if hypermapper_mode == "default":
        if black_box_function == None:
            print("Error: the black box function must be provided")
            raise SystemExit
        if not callable(black_box_function):
            print("Error: the black box function parameter is not callable")
            raise SystemExit

    # Here they pull a lot of configurations from the config about how to execute the LS, basically its hyperparams
    # I should probably do my own for the EA. But can maybe hard code them here for now

    # this is the required param. Will probably always be value only
    optimization_metrics = config["optimization_objectives"]
    number_of_objectives = len(optimization_metrics)

    ### Assign hyperparameters for the EA here ###
    population_size = config["evolution_population_size"]

    ### End of hyperparameter assigning ###

    # This could be good to keep. but doubt I will need it
    log_file = deal_with_relative_and_absolute_path(run_directory, config["log_file"])
    sys.stdout.change_log_file(log_file)
    if hypermapper_mode == "client-server":
        sys.stdout.switch_log_only_on_file(True)

    # How to set the output path, only needed when unspecified i scenario
    if output_file == "":
        output_data_file = config["output_data_file"]
        if output_data_file == "output_samples.csv":
            output_data_file = application_name + "_" + output_data_file
    else:
        output_data_file = output_file


    # I want something equivalent to this
    absolute_configuration_index = 0
    fast_addressing_of_data_array = {}
    evolution_fast_addressing_of_data_array = {}
    evolution_data_array = defaultdict(list)

    beginning_of_time = param_space.current_milli_time()

    optimization_function_parameters = {}
    # optimization_function_parameters['hyper_parameter_x'] = hyper_parameter_x
    optimization_function_parameters['hypermapper_mode'] = hypermapper_mode
    optimization_function_parameters['param_space'] = param_space
    optimization_function_parameters['beginning_of_time'] = beginning_of_time
    optimization_function_parameters['run_directory'] = run_directory
    optimization_function_parameters['black_box_function'] = black_box_function
    optimization_function_parameters['evolution_data_array'] = evolution_data_array
    optimization_function_parameters['fast_addressing_of_data_array'] = evolution_fast_addressing_of_data_array
    #optimization_function_parameters['evaluation_limit'] = local_search_evaluation_limit

    # Unsure about these
    #optimization_function_parameters['number_of_cpus'] = number_of_cpus
    #optimization_function_parameters['enable_feasible_predictor'] = enable_feasible_predictor

    # I think this is unnecessary
    # optimization_function_parameters['exhaustive_search_data_array'] = exhaustive_search_data_array
    # optimization_function_parameters[
        # 'exhaustive_search_fast_addressing_of_data_array'] = exhaustive_search_fast_addressing_of_data_array
    #optimization_function_parameters['scalarization_weights'] = objective_weights
    #optimization_function_parameters['objective_limits'] = objective_limits
    #optimization_function_parameters['scalarization_method'] = scalarization_method

    print("Starting evolution...")
    evolution_t0 = datetime.datetime.now()
    all_samples, best_configuration = evolution(
                                                population_size,
                                                param_space,
                                                fast_addressing_of_data_array,
                                                run_objective_function,
                                                optimization_function_parameters
                                                )

    print("Local search finished after %d function evaluations"%(len(evolution_data_array[optimization_metrics[0]])))
    sys.stdout.write_to_logfile(("Local search time %10.4f sec\n" % ((datetime.datetime.now() - evolution_t0).total_seconds())))

    with open(deal_with_relative_and_absolute_path(run_directory, output_data_file), 'w') as f:
        w = csv.writer(f)
        w.writerow(list(evolution_data_array.keys()))
        tmp_list = [param_space.convert_types_to_string(j, evolution_data_array) for j in list(evolution_data_array.keys())]
        tmp_list = list(zip(*tmp_list))
        for i in range(len(evolution_data_array[optimization_metrics[0]])):
            w.writerow(tmp_list[i])

    print("### End of the evolutionary search.")

if __name__ == "__main__":

    # This handles the logger. The standard setting is that HyperMapper always logs both on screen and on the log file.
    # In cases like the client-server mode we only want to log on the file.
    # This is in local search but not random_scalarizations. I leave it out for now
    # sys.stdout = Logger()

    if len(sys.argv) == 2:
        parameters_file = sys.argv[1]
    else :
        print("Error: only one argument needed, the parameters json file.")

    if parameters_file == "--help" or len(sys.argv) != 2:
        print("################################################")
        print("### Example: ")
        print("### cd hypermapper")
        print("### python3 scripts/hypermapper.py example_scenarios/spatial/BlackScholes_scenario.json")
        print("################################################")
        raise SystemExit

    try:
        initial_directory = os.environ['PWD']
        hypermapper_home = os.environ['HYPERMAPPER_HOME']
        os.chdir(hypermapper_home)
    except:
        hypermapper_home = "."
        initial_directory = "."

    if not parameters_file.endswith('.json'):
        _, file_extension = os.path.splitext(parameters_file)
        print("Error: invalid file name. \nThe input file has to be a .json file not a %s" %file_extension)
        raise SystemExit
    with open(parameters_file, 'r') as f:
        config = json.load(f)

    json_schema_file = 'scripts/schema.json'
    with open(json_schema_file, 'r') as f:
        schema = json.load(f)

    try:
        DefaultValidatingDraft4Validator = extend_with_default(Draft4Validator)
        DefaultValidatingDraft4Validator(schema).validate(config)
    except exceptions.ValidationError as ve:
        print("Failed to validate json:")
        print(ve)
        raise SystemExit

    run_directory = config["run_directory"]
    if run_directory == ".":
        run_directory = initial_directory
        config["run_directory"] = run_directory
    log_file = config["log_file"]
    if log_file == "hypermapper_logfile.log":
        log_file = deal_with_relative_and_absolute_path(run_directory, log_file)
    sys.stdout = Logger(log_file)

    main(config)

    try:
        os.chdir(hypermapper_pwd)
    except:
        pass

    print("### End of the evolutionary script.")
