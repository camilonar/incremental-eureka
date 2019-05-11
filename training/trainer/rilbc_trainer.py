"""
A proposed instantiation of the CRIL algorithm: Representatives Incremental Learning with BvSB and Crowding distance
(RILBC). This uses BvSB as measure for candidate selection and Crowding distance for competition between representatives
"""
import math
import numpy as np
from training.trainer.cril_trainer import CRILTrainer, Representative


class RILBCTrainer(CRILTrainer):
    """
    Trains with the proposed algorithm that uses RMSProp, BvSB and Crowding distance for incremental learning
    """

    def _buffer_candidates(self, image_batch, target_batch, outputs, iteration, megabatch):
        """
        Adds samples to the buffer. This version buffers all the original images from a batch

        :param image_batch: the list of images of a batch
        :param target_batch: the list of one hot labels of a batch
        :param outputs: output probabilities of the neural network
        :param iteration: current iteration of training
        :param megabatch: current megabatch

        :return: None
        """
        sorted_outputs = np.sort(outputs)  # Aux. for outputs sorted by probability
        difs = np.array([i[-1] - i[-2] for i in sorted_outputs])  # Best vs. Second Best
        sort_indices = np.argsort(difs)  # Order indices (from lowest dif. to highest dif.)
        difs = difs[sort_indices]
        image_batch = np.asarray(image_batch)[sort_indices]  # The data is ordered according to the indices
        target_batch = np.asarray(target_batch)[sort_indices]  # The data labels are ordered according to the indices
        outputs = outputs[sort_indices]

        x = 0
        initial_index = max(math.floor(len(image_batch) / 2 - self.n_candidates / 2) - x, 0)
        end_index = min(math.floor(len(image_batch) / 2 + self.n_candidates / 2) - x, len(image_batch))

        for i in range(initial_index, end_index):
            self.buffered_reps.append(
                Representative(image_batch[i].copy(), target_batch[i].copy(), difs[i].copy(), iteration, megabatch,
                               outputs[i].copy()))

    def _modify_representatives(self, candidate_representatives):
        """
        Modifies the representatives list according to the new data

        :param candidate_representatives: the preselected representatives from the buffer
        :return: None
        """
        for i, _ in enumerate(candidate_representatives):
            nclass = int(np.argmax(candidate_representatives[i].label))
            self.representatives[nclass].append(candidate_representatives[i])
            self.class_count[nclass] += 1

        # self.__recalculate_metrics(self.representatives)
        # Sorts representatives of each list, corresponding to each class
        self.__calculate_crowd_distance(self.representatives)
        for i in range(len(self.representatives)):
            self.representatives[i].sort(key=lambda x: x.crowd_distance)
            # self.representatives[i].sort(key=lambda x: x.metric * (1 + min(1, (total_it - x.iteration) / 10000)),
            #                             reverse=True)
            self.representatives[i] = self.representatives[i][-min(self.memory_size, len(self.representatives[i])):]

    # TODO make more efficient
    @staticmethod
    def __calculate_crowd_distance(representatives):
        """
        Calculates or recalculates the crowd distances for all the representatives

        :param representatives: the list of representatives divided by class
        :return: None
        """
        # Resets crowd distance for everyone
        for cls in representatives:
            for rep in cls:
                rep.crowd_distance = 0

        # Recalculates crowd distance for everyone
        for cls in representatives:
            # Pass if the class doesn't have any samples
            if len(cls) == 0:
                continue
            # Travels each 'objective'
            for i in range(len(cls[0].net_output)):

                def sort(x):
                    return x.net_output[i]

                cls.sort(key=sort)
                # Calculates crowd distance for the i-th objective.
                # cls[0].crowd_distance, cls[-1].crowd_distance = math.inf, math.inf
                for j in range(1, len(cls) - 1):
                    cls[j].crowd_distance += cls[j].net_output[i] - cls[j - 1].net_output[i]
                    cls[j].crowd_distance += cls[j + 1].net_output[i] - cls[j].net_output[i]
