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


def evolution(param_space, fast_addressing_of_data_array, optimization_function_parameters):
    """
    Do the entire evolutinary process from config to best config
    """
    # Get a population by taking ranom samples
    # evolutionary loop
    #   evaluate fitness
    #   update population
    #       crossover, random sampling, mutation
    # return best config. and all seen samples

    all_samples = 0
    best_configuration = 0
    return all_samples, best_configuration


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
    # hyper_parameter_x = config["hyper_parameter_x"

    # this is the required param. Will probably allways be value
    optimization_metrics = config["optimization_objectives"]
    number_of_objectives = len(optimization_metrics)

    ### Assign hyperparameters for the EA here ###



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
    local_search_fast_addressing_of_data_array = {}
    local_search_data_array = defaultdict(list)

    beginning_of_time = param_space.current_milli_time()

    optimization_function_parameters = {}
    # optimization_function_parameters['hyper_parameter_x'] = hyper_parameter_x
    optimization_function_parameters['hypermapper_mode'] = hypermapper_mode
    optimization_function_parameters['param_space'] = param_space
    optimization_function_parameters['beginning_of_time'] = beginning_of_time
    optimization_function_parameters['run_directory'] = run_directory
    optimization_function_parameters['black_box_function'] = black_box_function


    print("Starting local search...")
    local_search_t0 = datetime.datetime.now()
    all_samples, best_configuration = evolution(
                                                param_space,
                                                fast_addressing_of_data_array,
                                                optimization_function_parameters
                                                )

    print("Local search finished after %d function evaluations"%(len(local_search_data_array[optimization_metrics[0]])))
    sys.stdout.write_to_logfile(("Local search time %10.4f sec\n" % ((datetime.datetime.now() - local_search_t0).total_seconds())))

    with open(deal_with_relative_and_absolute_path(run_directory, output_data_file), 'w') as f:
        w = csv.writer(f)
        w.writerow(list(local_search_data_array.keys()))
        tmp_list = [param_space.convert_types_to_string(j, local_search_data_array) for j in list(local_search_data_array.keys())]
        tmp_list = list(zip(*tmp_list))
        for i in range(len(local_search_data_array[optimization_metrics[0]])):
            w.writerow(tmp_list[i])

    print("### End of the local search.")

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
