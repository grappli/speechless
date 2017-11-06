from pathlib import Path

from speechless import configuration
from speechless.configuration import Configuration, DataDirectories

configuration.default_data_directories = DataDirectories(Path("/data/speechless-data"))

config = Configuration.german_tib(True, True)
wav2letter = config.load_best_german_model()

config.train(wav2letter, run_name='German-TIB-train')

