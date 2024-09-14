import yaml
import os
import sys
sys.path.append('.')
from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


# # # # # # # # # # # # # # #
# EXPERIMENT LOADING UTILS  #
# # # # # # # # # # # # # # #

        
def load_config_file(EXP_NAME, path_prefix=''):
    '''
    Load YAML config file corresponding to an experiment name
    '''
    config_path = os.path.join(path_prefix, 'config/'+EXP_NAME+".yml")
    logger.debug("Opening config file "+str(config_path))
    with open(config_path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data
