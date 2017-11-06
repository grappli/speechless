from pathlib import Path
import argparse

from speechless import tools
from speechless import configuration
from speechless.configuration import Configuration, DataDirectories

def main(train_tib, augment):
    configuration.default_data_directories = DataDirectories(Path("/data/speechless-data"))

    config = Configuration.german_tib(train_tib, augment)
    wav2letter = config.load_best_german_model()

    run_name = 'German-TIB'
    if train_tib:
        run_name += '-train'
    if augment:
        run_name += '-augment'

    tools.logLocation = '/data/{}.log'.format(run_name)

    config.train(wav2letter, run_name=run_name)

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--train-tib', dest='train_tib', action='store_true')
    parse.add_argument('--a', dest='augment', action='store_true')
    args = parse.parse_args()

    main(parse.train_tib, parse.augment)