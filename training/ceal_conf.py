from training.train_conf import GeneralConfig


class CealConfig(GeneralConfig):

    def __init__(self, learning_rate: float, k: int , delta: float, dr: float, max_interations :int, t: int, summary_interval=100, check_interval=200, config_name='default'
                 , model_name='dataset_default'):
        '''

        :param learning_rate: the learning rate to be used in the training
        :param k: uncertain samples selection size
        :param delta: , high conﬁdence samples selection threshold
        :param dr: threshold decay rate
        :param t: ﬁne-tuning interval
        :param max_interations: maximum iteration number
        :param summary_interval: he interval of iterations at which the summaries are going to be performed
        :param check_interval: the interval of iterations at which the evaluations and checkpoints are going to be
        :param config_name: a descriptive name for the training configuration
        :param model_name: a descriptive name for the model
        '''
        super().__init__(learning_rate, summary_interval, check_interval, config_name
                         , model_name)
        self.k = k
        self.delta = delta
        self.dr = dr
        self.t = t
        self.T = max_interations




