"""
A proposed instantiation of the CRIL algorithm: Naive Incremental Learning (NIL). This uses random selection and
competition between representatives
"""
import numpy as np
from training.trainer.crif_trainer import CRIFTrainer, Representative


class NILTrainer(CRIFTrainer):
    """
    Trains with the proposed algorithm that uses RMSProp and random representatives selection for incremental
    learning
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
        rand_indices = np.random.permutation(len(outputs))
        outputs = outputs[rand_indices]
        difs = [0 for _ in outputs]
        image_batch = np.asarray(image_batch)[rand_indices]  # The data is ordered according to the indices
        target_batch = np.asarray(target_batch)[rand_indices]
        for i in range(min(self.n_candidates, len(image_batch))):
            self.buffered_reps.append(
                Representative(image_batch[i].copy(), target_batch[i].copy(), difs[i], iteration, megabatch,
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

        for i in range(len(self.representatives)):
            rand_indices = np.random.permutation(len(self.representatives[i]))
            self.representatives[i] = [self.representatives[i][j] for j in rand_indices]
            self.representatives[i] = self.representatives[i][-min(self.memory_size, len(self.representatives[i])):]
