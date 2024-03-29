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
import numpy.random as rd
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


def get_best_config(configs):
    """
    Returns the configuration with lowest Value among the given configurations
    :param configs: list of configurations.
    :return: dict, the best configuration.
    """
    leader = configs[0]
    for c in configs:
        if c['Value'] < leader['Value']:
            leader = c
    return leader


def mutation(param_space, config, mutationrate, list=False):
    """
    Mutates one configuration. This overcomplicates the procedure. But since I might
    change the functionality I leave it like this for now
    :param param_space: space.Space(), will give us imprmation about parameters
    :param configs: list of configurations.
    :param mutationrate: integer for how many parameters to mutate
    :param list: boolean whether returning one or more configs
    :return: list of dicts, list of mutated configurations
    """

    parameter_object_list = param_space.get_input_parameters_objects()
    rd_config = dict()
    for name, obj in parameter_object_list.items():
        x = obj.randomly_select()
        # sucks that there are parameters with only one value
        if x == config[name]:
            x = obj.randomly_select()
        rd_config[name] = x
    parameter_names_list = param_space.get_input_parameters()
    nbr_params = len(parameter_names_list)

    configs = []
    n_configs = nbr_params if list else 1

    for _ in range(n_configs):
        indices = rd.permutation(nbr_params)[:mutationrate]
        for idx in indices:
            mutation_param = parameter_names_list[idx]
            # Should I do something if they are the same?
            temp = config.copy()
            temp[mutation_param] = rd_config[mutation_param]
            configs.append(temp)

    return configs


# Taken from local_search and slightly modified
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
    :param fast_addressing_of_data_array: a dictionary containing evaluated configurations and their index in
    the evolution_data_array.
    :param x enable_feasible_predictor: whether to use constrained optimization.
    :param (x) evaluation_limit: the maximum number of function evaluations allowed for the evolutionary search.
    :param black_box_function: the black_box_function being optimized in the evolutionary search.
    :param number_of_cpus: an integer for the number of cpus to be used in parallel.
    :return: configurations with evaluations for all points in configurations and the number of evaluated configurations
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
        new_evaluations = param_space.run_configurations(hypermapper_mode, new_configurations, beginning_of_time,
                                                         black_box_function, run_directory)

    # Values for all given configurations
    all_evaluations = concatenate_data_dictionaries(previous_evaluations, new_evaluations)
    all_evaluations_size = len(all_evaluations[list(all_evaluations.keys())[0]])

    population = list()
    for idx in range(number_of_new_evaluations):
        configuration = get_single_configuration(new_evaluations, idx)
        population.append(configuration)
        for key in configuration:
            evolution_data_array[key].append(configuration[key])

        str_data = param_space.get_unique_hash_string_from_values(configuration)
        fast_addressing_of_data_array[str_data] = absolute_configuration_index
        absolute_configuration_index += 1

    sys.stdout.write_to_logfile(("Time to run new configurations %10.4f sec\n" %
                                 ((datetime.datetime.now() - t1).total_seconds())))
    sys.stdout.write_to_logfile(("Total time to run configurations %10.4f sec\n" %
                                 ((datetime.datetime.now() - t0).total_seconds())))

    return population, all_evaluations_size


def evolution(population_size, generations, mutation_rate, crossover, regularize, batch_size, param_space,
              fast_addressing_of_data_array, optimization_function, optimization_function_parameters):

    """
    Do the entire evolutinary process from config to best config
    :param population_size: an integer for the number of configs to keep. All will be initiated randomly
    :param generations: an integer for the number of iterations through the evolutionary loop
    :param mutation_rate: an integer for the number of parameters to change in a mutation
    :param crossover: a boolean whether to use crossover in the algorithm
    :param regularize: boolean, whether to use regularized or non-regularized evolution strategy
    :param batch_size: an integer for how many individuals to compare in a generation
    :param param_space: a space object containing the search space.
    :param fast_addressing_of_data_array: an array that keeps track of all evaluated configurations
    :param optimization_function: the function that will be optimized by the evolutionary search.
    :param optimization_function_parameters: a dictionary containing the parameters that will be passed to the
    optimization function.
    :return: all points evaluted and the best config at each generation of the Evolutionary Algorithm.
    """

    t0 = datetime.datetime.now()
    tmp_fast_addressing_of_data_array = copy.deepcopy(fast_addressing_of_data_array)
    input_params = param_space.get_input_parameters()
    data_array = {}

    ### Initialize a random population ###
    default_configuration = param_space.get_default_or_random_configuration()
    str_data = param_space.get_unique_hash_string_from_values(default_configuration)
    if str_data not in fast_addressing_of_data_array:
        tmp_fast_addressing_of_data_array[str_data] = 1
        if population_size - 1 > 0:     # Will always be true
            configurations = [default_configuration] + param_space.random_sample_configurations_without_repetitions(
                tmp_fast_addressing_of_data_array, population_size - 1)
    else:
        configurations = param_space.random_sample_configurations_without_repetitions(tmp_fast_addressing_of_data_array,
                                                                                      population_size)

    population, function_values_size = optimization_function(configurations=configurations, **optimization_function_parameters)

    # This will concatenate the entire data array if all configurations were evaluated
    new_data_array = concatenate_list_of_dictionaries(configurations[:function_values_size])
    data_array = concatenate_data_dictionaries(data_array, new_data_array)

    # A list of the best individual in the population at each generation
    best_configs = []
    best_config = get_best_config(population)
    best_configs.append(best_config)

    ### Evolutionary loop ###
    for gen in range(1, generations + 1):
        if not gen % 10:
            print('Now we are att gen: ', gen)

        # pick a random batch from the populotion and find the two best and the worst of the batch
        cand_idxs = rd.permutation(len(population))[:batch_size]
        infty = float("inf")
        best = (-1, infty)
        second = (-1, infty)
        worst = (-1, -infty)
        for ci in cand_idxs:
            val = population[ci]['Value']
            if val < best[1]:
                second = best
                best = (ci, val)
            elif val < second[1]:
                second = (ci, val)
            if val > worst[1]:
                worst = (ci, val)

        # checks that candidate loop was successful
        if min(best[0], second[0], worst[0]) < 0:
            print('failed to fined best and/or worst individual. Script will terminate')
            sys.exit()

        # Make a child by copy/crossover from parent(s)
        child = dict()
        parent = population[best[0]]
        if crossover:
            parent2 = population[second[0]]
            for param in input_params:
                if rd.uniform(0, 1) < 0.5:
                    child[param] = parent[param]
                else:
                    child[param] = parent2[param]
        else:
            for param in input_params:
                child[param] = parent[param]

        # Get mutation candidates, evaluate and add to population
        child_list = mutation(param_space, child, mutation_rate, list=True)
        need_random = True
        for c in child_list:
            evaluated_child_list, func_val_size = optimization_function(configurations=[c],
                                                                        **optimization_function_parameters)

            if evaluated_child_list:
                new_data_array = concatenate_list_of_dictionaries([c][:func_val_size])
                data_array = concatenate_data_dictionaries(data_array, new_data_array)

                population.append(evaluated_child_list[0])
                need_random = False
                break

        # If no new configs where found, draw some random configurations instead.
        if need_random:
            tmp_fast_addressing_of_data_array = copy.deepcopy(
                optimization_function_parameters['fast_addressing_of_data_array'])

            random_children = param_space.random_sample_configurations_without_repetitions(
                tmp_fast_addressing_of_data_array, 1)

            evaluated_random_children, func_val_size = optimization_function(configurations=random_children,
                                                                             **optimization_function_parameters)
            new_data_array = concatenate_list_of_dictionaries(random_children[:func_val_size])
            data_array = concatenate_data_dictionaries(data_array, new_data_array)
            population.append(evaluated_random_children[0])

        # Remove a configuration
        if regularize:     # removing oldest, which will be first as we append new last
            killed = population.pop(0)
        else:               # removing the worst in the subset
            killed = population.pop(worst[0])

        best_config = get_best_config(population)
        best_configs.append(best_config)

    sys.stdout.write_to_logfile(("Evolution time %10.4f sec\n" % ((datetime.datetime.now() - t0).total_seconds())))

    return data_array, best_configs


def main(config, black_box_function=None, output_file=""):
    """
    Run design-space exploration using evolution.
    :param config: dictionary containing all the configuration parameters of this design-space exploration.
    :param black_box_function: The function hypermapper seeks to optimize
    :param output_file: a name for the file used to save the dse results.
    :return:
    """

    # Space basically turn config into a class with many handy functions
    param_space = space.Space(config)

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


    optimization_metrics = config["optimization_objectives"]
    number_of_objectives = len(optimization_metrics)

    population_size = config["evolution_population_size"]
    generations = config["evolution_generations"]
    mutation_rate = config["mutation_rate"]
    if mutation_rate > len(param_space.get_input_parameters()):
        print("mutation rate higher than the number of parameters. makes no sense. Exiting")
        sys.exit()
    if mutation_rate < 1:
        print("mutation rate must be at least 1 for evolution to work. Exiting")
    crossover = config["evolution_crossover"]
    regularize = config["regularize_evolution"]
    batch_size = config["batch_size"]
    if batch_size > population_size:
        print("batch_size bigger than the population_size. makes no sense. Exiting")
        sys.exit()
    elif batch_size < 2 and not crossover:
        print("batch_size smaller than 2 makes no sense. Exiting")
        sys.exit()
    elif batch_size < 3 and crossover:
        print("batch_size must be at least 3 when using crossover . Exiting")
        sys.exit()

    log_file = deal_with_relative_and_absolute_path(run_directory, config["log_file"])
    sys.stdout.change_log_file(log_file)
    if hypermapper_mode == "client-server":
        sys.stdout.switch_log_only_on_file(True)

    if output_file == "":
        output_data_file = config["output_data_file"]
        if output_data_file == "output_samples.csv":
            output_data_file = application_name + "_" + output_data_file
    else:
        output_data_file = output_file

    absolute_configuration_index = 0
    fast_addressing_of_data_array = {}
    evolution_fast_addressing_of_data_array = {}
    evolution_data_array = defaultdict(list)

    beginning_of_time = param_space.current_milli_time()

    optimization_function_parameters = dict()
    optimization_function_parameters['hypermapper_mode'] = hypermapper_mode
    optimization_function_parameters['param_space'] = param_space
    optimization_function_parameters['beginning_of_time'] = beginning_of_time
    optimization_function_parameters['run_directory'] = run_directory
    optimization_function_parameters['black_box_function'] = black_box_function
    optimization_function_parameters['evolution_data_array'] = evolution_data_array
    optimization_function_parameters['fast_addressing_of_data_array'] = evolution_fast_addressing_of_data_array

    # optimization_function_parameters['number_of_cpus'] = number_of_cpus
    # optimization_function_parameters['enable_feasible_predictor'] = enable_feasible_predictor
    # optimization_function_parameters['exhaustive_search_data_array'] = exhaustive_search_data_array
    # optimization_function_parameters['exhaustive_search_fast_addressing_of_data_array'] =
    # exhaustive_search_fast_addressing_of_data_array
    # optimization_function_parameters['scalarization_weights'] = objective_weights
    # optimization_function_parameters['objective_limits'] = objective_limits
    # optimization_function_parameters['scalarization_method'] = scalarization_method

    print("Starting evolution...")
    evolution_t0 = datetime.datetime.now()
    all_samples, best_configurations = evolution(
                                                population_size,
                                                generations,
                                                mutation_rate,
                                                crossover,
                                                regularize,
                                                batch_size,
                                                param_space,
                                                fast_addressing_of_data_array,
                                                run_objective_function,
                                                optimization_function_parameters
                                                )
    # plotting the best found configurtion as a function of optimization iterations
    # r = range(len(best_configurations))
    # vals = []
    # for bc in best_configurations:
    #     vals.append(bc['Value'])
    # plt.scatter(r, vals)
    # plt.savefig('evolution_output.png')

    print("Evolution finished after %d function evaluations"%(len(evolution_data_array[optimization_metrics[0]])))
    sys.stdout.write_to_logfile(("Evolutionary search time %10.4f sec\n" % ((datetime.datetime.now() - evolution_t0).total_seconds())))

    with open(deal_with_relative_and_absolute_path(run_directory, output_data_file), 'w') as f:
        w = csv.writer(f)
        w.writerow(list(evolution_data_array.keys()))
        tmp_list = [param_space.convert_types_to_string(j, evolution_data_array) for j in list(evolution_data_array.keys())]
        tmp_list = list(zip(*tmp_list))
        for i in range(len(evolution_data_array[optimization_metrics[0]])):
            w.writerow(tmp_list[i])

    print("### End of the evolutionary search")


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
