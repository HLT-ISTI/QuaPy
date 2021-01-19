import multiprocessing

N_JOBS = 1  #multiprocessing.cpu_count()
ENSEMBLE_N_JOBS = -2
SAMPLE_SIZE = 100

assert N_JOBS==1 or ENSEMBLE_N_JOBS==1, 'general N_JOBS and ENSEMBLE_N_JOBS should not be both greater than 1'