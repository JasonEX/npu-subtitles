"""Local NPU subtitles. Importing the package never loads a model."""

import os

# Configure before adapters import NumPy. The NPU worker serializes model work;
# host math libraries should not create a second pool of busy CPU threads.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
