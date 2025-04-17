from temperature_max_min import main as generate_min_max_csv
from temperatures import main as get_temps
from joblib import Parallel, delayed
import os


def main (path):
    generate_min_max_csv(path)
    
    ### Mariona CSV
    
    get_temps(path)
    
    
if __name__ == "__main__":
    main("/data/uabcvmsc/shared/newborn")