#!/usr/bin/python3
# -*- coding: utf-8 -*-

try:
    import sys
    import os
    #from tqdm import tqdm
    import argparse
    import logging
    import subprocess
    import shlex
    import datetime
    from logging.handlers import RotatingFileHandler
    from pathlib import Path
    import itertools

except ImportError as error:
    print(error)
    print()
    print(" ")
    print()
    sys.exit(-1)

DEFAULT_VERBOSITY_LEVEL = logging.INFO
NUM_EPOCHS = 1000
TIME_FORMAT = '%Y-%m-%d_%H:%M:%S'

DEFAULT_CAMPAIGN = "demo"
PATH_LOG = 'logs'
PATH_DATASETS = 'datasets'
PATHS = [PATH_LOG]
args = None
COMMAND = "pipenv run python main.py "

datasets = [
        'datasets/defenseDroid2939_original_6000Malwares_5975Benign.csv',
        'datasets/defenseDroid2939_small_256Malwares_256Benign.csv',
        'datasets/defenseDroid2939_small_512Malwares_512Benign.csv',
        'datasets/defenseDroid2939_small_64Malwares_64Benign.csv',
        'datasets/drebin215_original_5560Malwares_6566Benign.csv',
        'datasets/drebin215_small_256Malwares_256Benign.csv',
        'datasets/drebin215_small_512Malwares_512Benign.csv',
        'datasets/drebin215_small_64Malwares_64Benign.csv']

# training_algorithm_choices = ['Adam', 'RMSprop', 'Adadelta']
campaigns_available = {}

campaigns_available['demo'] = {
    'input_dataset': ['datasets/defenseDroid2939_small_64Malwares_64Benign.csv'],
    'number_epochs' : ['1'],
    'training_algorithm': ['Adam'],
}

campaigns_available['sf23_1l_64'] = {
    'input_dataset': ['datasets/defenseDroid2939_original_6000Malwares_5975Benign.csv',
                     'datasets/drebin215_original_5560Malwares_6566Benign.csv'],
    "dense_layer_sizes_g" : ['64'],
    "dense_layer_sizes_d" : ['64'],
    'number_epochs' : ['1000'],
    'training_algorithm': ['Adam'],
}

campaigns_available['sf23_1l_128'] = {
    'input_dataset': ['datasets/defenseDroid2939_original_6000Malwares_5975Benign.csv',
                     'datasets/drebin215_original_5560Malwares_6566Benign.csv'],
    "dense_layer_sizes_g" : ['128'],
    "dense_layer_sizes_d" : ['128'],
    'number_epochs' : ['1000'],
    'training_algorithm': ['Adam'],
}

campaigns_available['sf23_1l_256'] = {
    'input_dataset': ['datasets/drebin215_original_5560Malwares_6566Benign.csv'],
    "dense_layer_sizes_g" : ['256'],
    "dense_layer_sizes_d" : ['256'],
    'number_epochs' : ['1000'],
    'training_algorithm': ['Adam'],
}

campaigns_available['sf23_1l_512'] = {
    'input_dataset': ['datasets/defenseDroid2939_original_6000Malwares_5975Benign.csv',
                     'datasets/drebin215_original_5560Malwares_6566Benign.csv'],
    "dense_layer_sizes_g" : ['512'],
    "dense_layer_sizes_d" : ['512'],
    'number_epochs' : ['1000'],
    'training_algorithm': ['Adam'],
}

campaigns_available['sf23_1l_1024'] = {
    'input_dataset': ['datasets/drebin215_original_5560Malwares_6566Benign.csv'],
    "dense_layer_sizes_g" : ['1024'],
    "dense_layer_sizes_d" : ['1024'],
    'number_epochs' : ['1000'],
    'training_algorithm': ['Adam'],
}

campaigns_available['sf23_1l_4096'] = {
    'input_dataset': ['datasets/drebin215_original_5560Malwares_6566Benign.csv'],
    "dense_layer_sizes_g" : ['4096'],
    "dense_layer_sizes_d" : ['4096'],
    'number_epochs' : ['1000'],
    'training_algorithm': ['Adam'],
}

campaigns_available['sf23_1l_8192'] = {
    'input_dataset': ['datasets/defenseDroid2939_original_6000Malwares_5975Benign.csv',
                     'datasets/drebin215_original_5560Malwares_6566Benign.csv'],
    "dense_layer_sizes_g" : ['8192'],
    "dense_layer_sizes_d" : ['8192'],
    'number_epochs' : ['1000'],
    'training_algorithm': ['Adam'],
}


campaigns_available['sf23_2l'] = {
    'input_dataset': ['datasets/defenseDroid2939_original_6000Malwares_5975Benign.csv',
                     'datasets/drebin215_original_5560Malwares_6566Benign.csv'],
    "dense_layer_sizes_g" : ['512,1024'],
    "dense_layer_sizes_d" : ['1024,512'],
    'number_epochs' : ['1000'],
    'training_algorithm': ['Adam'],
}

campaigns_available['sf23_3l'] = {
    'input_dataset': ['datasets/defenseDroid2939_original_6000Malwares_5975Benign.csv',
                     'datasets/drebin215_original_5560Malwares_6566Benign.csv'],
    "dense_layer_sizes_g" : ['512,1024,2048'],
    "dense_layer_sizes_d" : ['2048,1024,512'],
    'number_epochs' : ['1000'],
    'training_algorithm': ['Adam'],
}


campaigns_available['teste'] = {
    'input_dataset': ['datasets/defenseDroid2939_original_6000Malwares_5975Benign.csv',
                      'datasets/drebin215_original_5560Malwares_6566Benign.csv'],
    'classifier' : ['knn', 'random_forest', 'svm'],
    'training_algorithm': ['Adam', 'RMSprop', 'Adadelta'],
}


#'perceptron',
# 'training_algorithm': ['Adam', 'RMSprop', 'Adadelta'],
# 'classifier' : ['knn', 'random_forest', 'svm'],
#'data_type' : ['int8', 'float16', 'float32'],


def print_config(args):

    logging.info("Command:\n\t{0}\n".format(" ".join([x for x in sys.argv])))
    logging.info("Settings:")
    lengths = [len(x) for x in vars(args).keys()]
    max_lenght = max(lengths)

    for k, v in sorted(vars(args).items()):
        message = "\t"
        message +=  k.ljust(max_lenght, " ")
        message += " : {}".format(v)
        logging.info(message)

    logging.info("")


def convert_flot_to_int(value):

    if isinstance(value, float):
        value = int(value * 100)

    return value



# Custom argparse type representing a bounded int
# source: https://stackoverflow.com/questions/14117415/in-python-using-argparse-allow-only-positive-integers
class IntRange:

    def __init__(self, imin=None, imax=None):

        self.imin = imin
        self.imax = imax

    def __call__(self, arg):

        try:
            value = int(arg)

        except ValueError:
            raise self.exception()

        if (self.imin is not None and value < self.imin) or (self.imax is not None and value > self.imax):
            raise self.exception()

        return value

    def exception(self):

        if self.imin is not None and self.imax is not None:
            return argparse.ArgumentTypeError(f"Must be an integer in the range [{self.imin}, {self.imax}]")

        elif self.imin is not None:
            return argparse.ArgumentTypeError(f"Must be an integer >= {self.imin}")

        elif self.imax is not None:
            return argparse.ArgumentTypeError(f"Must be an integer <= {self.imax}")

        else:
            return argparse.ArgumentTypeError("Must be an integer")


def run_cmd(cmd, shell=False):
    logging.info("Command line  : {}".format(cmd))
    cmd_array = shlex.split(cmd)
    logging.debug("Command array: {}".format(cmd_array))
    if not args.demo:
        subprocess.run(cmd_array, check=True, shell=shell)


class Campaign():

    def __init__(self, datasets, training_algorithm, dense_layer_sizes_g, dense_layer_sizes_d):
        self.datasets = datasets
        self.training_algorithm = training_algorithm
        self.dense_layer_sizes_g = dense_layer_sizes_g
        self.dense_layer_sizes_d = dense_layer_sizes_d


def check_files(files, error=False):
    internal_files = files
    if isinstance(files, str):
        internal_files = [files]

    for f in internal_files:
        if not os.path.isfile(f):
            if error:
                logging.info("ERROR: file not found! {}".format(f))
                sys.exit(1)
            else:
                logging.info("File not found! {}".format(f))
                return False
        else:
            logging.info(("File found: {}".format(f)))

    return True


def main():

    parser = argparse.ArgumentParser(description='Torrent Trace Correct - Machine Learning')

    help_msg = "Campaign {} (default={})".format([x for x in campaigns_available.keys()], DEFAULT_CAMPAIGN)
    parser.add_argument("--campaign", "-c", help=help_msg, default=DEFAULT_CAMPAIGN, type=str)


    help_msg = "demo mode (default={})".format(False)
    parser.add_argument("--demo", "-d", help=help_msg, action='store_true')
    
    help_msg = "verbosity logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
    parser.add_argument("--verbosity", "-v", help=help_msg, default=DEFAULT_VERBOSITY_LEVEL, type=int)


    global args
    args = parser.parse_args()

    print("Creating the structure of directories...")
    for p in PATHS:
        Path(p).mkdir(parents=True, exist_ok=True)

    print("done.")
    print("")


    output_dir = 'outputs/out_{}'.format(
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    Path(output_dir).mkdir(parents=True, exist_ok=True)    
    logging_filename = '{}/evaluation_campaigns.log'.format(output_dir)

    logging_format = '%(asctime)s\t***\t%(message)s'
    # configura o mecanismo de logging
    if args.verbosity == logging.DEBUG:
        # mostra mais detalhes
        logging_format = '%(asctime)s\t***\t%(levelname)s {%(module)s} [%(funcName)s] %(message)s'

    # formatter = logging.Formatter(logging_format, datefmt=TIME_FORMAT, level=args.verbosity)
    logging.basicConfig(format=logging_format, level=args.verbosity)

    # Add file rotating handler, with level DEBUG
    rotatingFileHandler = RotatingFileHandler(filename=logging_filename, maxBytes=100000, backupCount=5)
    rotatingFileHandler.setLevel(args.verbosity)
    rotatingFileHandler.setFormatter(logging.Formatter(logging_format))
    logging.getLogger().addHandler(rotatingFileHandler)

    # imprime configurações para fins de log
    print_config(args)



    campaigns_chosen = []


    if args.campaign is None:
        campaigns_chosen = campaigns_available.keys()
    else:
        if args.campaign in campaigns_available.keys():
            campaigns_chosen.append(args.campaign)
            
        elif ',' in args.campaign: 
            campaigns_chosen = args.campaign.split(',')
    
        else:
            logging.error(" Campaign '{}' not found".format(args.campaign))
            sys.exit(-1)


    time_start_campaign = datetime.datetime.now()
    logging.info("\n\n\n")
    logging.info("##########################################")
    logging.info(" EVALUATION ")
    logging.info("##########################################")
    time_start_evaluation = datetime.datetime.now()
    
    count_campaign = 1
    for c in campaigns_chosen:
        logging.info("\tCampaign {} {}/{} ".format(c, count_campaign, len(campaigns_chosen)))
        count_campaign += 1

        campaign = campaigns_available[c]
        params, values = zip(*campaign.items())
        combinations_dicts = [dict(zip(params, v)) for v in itertools.product(*values)]
        campaign_dir = '{}/{}'.format(output_dir, c)

        count_combination = 1
        for combination in combinations_dicts:
            logging.info("\t\tcombination {}/{} ".format(count_combination, len(combinations_dicts)))
            logging.info("\t\t{}".format(combination))
            

            cmd = COMMAND
            cmd += " --verbosity {}".format(args.verbosity)
            
            cmd += " --output_dir {}".format(os.path.join(campaign_dir, "combination_{}".format(count_combination)))
            count_combination += 1

            for param in combination.keys():
                cmd += " --{} {}".format(param, combination[param])


            time_start_experiment = datetime.datetime.now()
            logging.info(
                "\t\t\t\t\tBegin: {}".format(time_start_experiment.strftime(TIME_FORMAT)))
            run_cmd(cmd)

            time_end_experiment = datetime.datetime.now()
            duration = time_end_experiment - time_start_experiment
            logging.info("\t\t\t\t\tEnd                : {}".format(time_end_experiment.strftime(TIME_FORMAT)))
            logging.info("\t\t\t\t\tExperiment duration: {}".format(duration))


        time_end_campaign = datetime.datetime.now()
        logging.info("\t Campaign duration: {}".format(time_end_campaign - time_start_campaign))

    time_end_evaluation = datetime.datetime.now()
    logging.info("Evaluation duration: {}".format(time_end_evaluation - time_start_evaluation))


if __name__ == '__main__':
    sys.exit(main())
