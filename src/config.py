import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

LOG_DIR = os.path.join(PROJECT_DIR, 'logs')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_DIR_SEQUENTIAL = os.path.join(LOG_DIR, 'sequential')
if not os.path.exists(LOG_DIR_SEQUENTIAL):
    os.makedirs(LOG_DIR_SEQUENTIAL)
LOG_DIR_CPU = os.path.join(LOG_DIR, 'cpu')
if not os.path.exists(LOG_DIR_CPU):
    os.makedirs(LOG_DIR_CPU)
LOG_DIR_GPU = os.path.join(LOG_DIR, 'gpu')
if not os.path.exists(LOG_DIR_GPU):
    os.makedirs(LOG_DIR_GPU)

RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

