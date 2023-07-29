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

except ImportError as error:
    print(error)
    print()
    print(" ")
    print()
    sys.exit(-1)

#https://liyin2015.medium.com/tensorflow-cpus-and-gpus-configuration-9c223436d4ef
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

DEFAULT_VERBOSITY_LEVEL = logging.INFO
NUM_EPOCHS = 1000
TIME_FORMAT = '%Y-%m-%d_%H:%M:%S'

DEFAULT_CAMPAIGN = "demo"
PATH_LOG = 'logs'
PATH_DATASETS = 'datasets'
PATHS = [PATH_LOG]
  

args = None


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

    help_msg = "Campaign [demo, teste, sf23] (default={})".format(DEFAULT_CAMPAIGN)
    parser.add_argument("--campaign", "-c", help=help_msg, default=DEFAULT_CAMPAIGN, type=str)

    parser.add_argument('--use_gpu', action='store_true', default=False, help='Opção para usar a GPU do TensorFlow.')

    help_msg = "verbosity logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
    parser.add_argument("--verbosity", "-v", help=help_msg, default=DEFAULT_VERBOSITY_LEVEL, type=int)


    global args
    args = parser.parse_args()

    print("Creating the structure of directories...")
    for p in PATHS:
        Path(p).mkdir(parents=True, exist_ok=True)

    print("done.")
    print("")


    logging_filename = '{}/evaluation_campaign_{}.log'.format(PATH_LOG, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

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

    datasets =[
    'defenseDroid2939_original_6000Malwares_5975Benign.csv',
    'defenseDroid2939_small_256Malwares_256Benign.csv',
    'defenseDroid2939_small_512Malwares_512Benign.csv',
    'defenseDroid2939_small_64Malwares_64Benign.csv',
    'drebin215_original_5560Malwares_6566Benign.csv',
    'drebin215_small_256Malwares_256Benign.csv',
    'drebin215_small_512Malwares_512Benign.csv',
    'drebin215_small_64Malwares_64Benign.csv']

    # training_algorithm_choices = ['Adam', 'RMSprop', 'Adadelta']
    campaign_demo = Campaign(datasets=['drebin215_small_64Malwares_64Benign.csv'],
                             training_algorithm=['Adam'],
                             dense_layer_sizes_g=['[128, 256, 512]'],
                             dense_layer_sizes_d=['[512, 256, 128]']
                             )



    campaigns = []
    if args.campaign == "demo":
        campaigns = [campaign_demo]

    # elif args.campaign == "teste":
    #     campaigns = campaign_teste
    #

    time_start_campaign = datetime.datetime.now()
    logging.info("\n\n\n")
    logging.info("##########################################")
    logging.info(" EVALUTION ")
    logging.info("##########################################")


    count_campaign = 1
    for c in campaigns:
        logging.info("\tCampaign {}/{} ".format(count_campaign, len(campaigns)))
        count_campaign += 1
        count_dataset = 1
        for dataset in c.datasets:
            logging.info("\t\tDatasets {}/{} ".format(count_dataset, len(c.datasets)))
            count_dataset += 1

            count_training_algorithm = 1
            for training_algorithm in c.training_algorithm:
                logging.info("\t\t\ttraining_algorithm {}/{} ".format(count_training_algorithm, len(c.training_algorithm)))
                count_training_algorithm += 1

                count_dense_layer_sizes_g = 1
                for dense_layer_sizes_g in c.dense_layer_sizes_g:
                    logging.info("\t\t\tdense_layer_sizes_g {}/{} ".format(count_dense_layer_sizes_g,
                                                                          len(c.dense_layer_sizes_g)))
                    count_dense_layer_sizes_g += 1

                    count_dense_layer_sizes_d = 1
                    for dense_layer_sizes_d in c.dense_layer_sizes_d:
                        logging.info("\t\t\t\tdense_layer_sizes_g {}/{} ".format(count_dense_layer_sizes_d,
                                                                               len(c.dense_layer_sizes_d)))
                        count_dense_layer_sizes_d = +1

                        time_start_experiment = datetime.datetime.now()
                        logging.info(
                            "\t\t\t\t\tBegin: {}".format(time_start_experiment.strftime(TIME_FORMAT)))

                        cmd = "pipenv run generation2.py "
                        cmd += " --use_gpu {}".format(args.use_gpu)
                        cmd += " --dataset {}".format(os.path.join(PATH_DATASETS, dataset))
                        cmd += " --training_algorithm {}".format(training_algorithm)
                        cmd += " --dense_layer_sizes_g {}".format(dense_layer_sizes_g)
                        cmd += " --dense_layer_sizes_d {}".format(dense_layer_sizes_d)
                        #run_cmd(cmd)

                        time_end_experiment = datetime.datetime.now()
                        duration = time_end_experiment - time_start_experiment
                        logging.info("\t\t\t\t\tEnd                : {}".format(time_end_experiment.strftime(TIME_FORMAT)))
                        logging.info("\t\t\t\t\tExperiment duration: {}".format(duration))


    time_end_campaign = datetime.datetime.now()
    logging.info("\t Campaign duration: {}".format(time_end_campaign - time_start_campaign))


if __name__ == '__main__':
    sys.exit(main())