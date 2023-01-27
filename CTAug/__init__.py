from .methods import *
from .preprocess import *
import logging


class Method:
    def __init__(self, args):
        self.method = args.method
        self.args = args
        self.path = self.path()
        self.logger = config_logger(self.path + '.log')
        self.logger.info(args)
        self.logger.info("Output path: " + self.path)
        self.dataset = self.get_dataset()

    def path(self):
        if 'AUG' in self.method:
            path = self.args.save_path + '_' + str(self.args.factor)
        else:
            path = self.args.save_path
        return path

    def get_dataset(self):
        if self.method == 'MVGRL':
            if self.args.feature == 'sub':
                dataset = process_sub(self.args.dataset, self.args.dataset_file)
            else:
                dataset = process(self.args.dataset)
        elif 'MVGRL' in self.method:
            if self.args.feature == 'sub':
                dataset = process_sub(self.args.dataset, self.args.dataset_file, self.args.cal_weight)
            else:
                dataset = process(self.args.dataset, self.args.cal_weight)
        elif self.args.feature == 'one':
            dataset = get_dataset_one(self.args.dataset)
        elif self.args.feature == 'deg':
            dataset = get_dataset_deg(self.args.dataset)
        elif self.args.feature == 'sub':
            dataset = get_dataset_sub(self.args.dataset_file)
        elif self.args.feature == 'sub_deg':
            dataset = get_dataset_sub_deg(self.args.dataset_file)
        else:
            raise NotImplementedError
        self.logger.info("Dataset is ok.")
        return dataset

    def train(self):
        # GraphCL
        if self.method == 'GraphCL':
            return GraphCL(self.args.times, self.args, self.path, self.logger, self.dataset, False)
        elif self.method == 'GraphCL_OGSN':
            return GraphCL(self.args.times, self.args, self.path, self.logger, self.dataset, True)
        elif self.method == 'GraphCL_AUG':
            return GraphCL_AUG(self.args.times, self.args, self.path, self.logger, self.dataset, False)
        elif self.method == 'GraphCL_AUG_OGSN':
            return GraphCL_AUG(self.args.times, self.args, self.path, self.logger, self.dataset, True)
        # JOAO
        elif self.method == 'JOAO':
            return JOAO(self.args.times, self.args, self.path, self.logger, self.dataset, False)
        elif self.method == 'JOAO_OGSN':
            return JOAO(self.args.times, self.args, self.path, self.logger, self.dataset, True)
        elif self.method == 'JOAO_AUG':
            return JOAO_AUG(self.args.times, self.args, self.path, self.logger, self.dataset, False)
        elif self.method == 'JOAO_AUG_OGSN':
            return JOAO_AUG(self.args.times, self.args, self.path, self.logger, self.dataset, True)
        # MVGRL
        elif self.method == 'MVGRL':
            return MVGRL(self.args.times, self.args, self.path, self.logger, self.dataset, False)
        elif self.method == 'MVGRL_OGSN':
            return MVGRL(self.args.times, self.args, self.path, self.logger, self.dataset, True)
        elif self.method == 'MVGRL_AUG':
            return MVGRL_AUG(self.args.times, self.args, self.path, self.logger, self.dataset, False)
        elif self.method == 'MVGRL_AUG_OGSN':
            return MVGRL_AUG(self.args.times, self.args, self.path, self.logger, self.dataset, True)
        # InfoGraph
        elif self.method == 'InfoGraph':
            return InfoGraph(self.args.times, self.args, self.path, self.logger, self.dataset)
        # JOAOv2
        elif self.method == 'JOAOv2':
            return JOAOv2(self.args.times, self.args, self.path, self.logger, self.dataset)
        else:
            raise NotImplementedError


def config_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fhandler = logging.FileHandler(log_path, mode='w')
    shandler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    shandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.addHandler(shandler)
    return logger

