"""
Generates synthetic data for matrix factorization

Author: Alexander Hentschel
        alex.hentschel@axiomzen.co
"""

import numpy as np
import pandas as pd


class synthetic_data_generator():

    def __init__(self, number_users:int, number_items:int, number_hidden_classes:int):
        self.number_users = number_users
        self.number_items = number_items
        self.number_hidden_classes = number_hidden_classes
        #
        self.preference_matrix = self._generate_score_matrix([number_users,number_hidden_classes])
        self.relevance_matrix = self._generate_score_matrix([number_items,number_hidden_classes])
        self.hidden_truth = np.tensordot(self.preference_matrix, self.relevance_matrix, axes=(1, 1))
        self.full_observations = self._generate_full_observations(self.hidden_truth)

    def _generate_scores(self, number_entries, low_ranking_entires, high_ranking_entries):
        item_idcs = np.linspace(start=0, stop=number_entries, num=number_entries, endpoint=False, dtype='int')
        ranked_indices = np.random.choice(item_idcs, size=low_ranking_entires+high_ranking_entries, replace=False)
        low_scores = np.random.uniform(low=0.0, high=0.5, size=low_ranking_entires)
        high_scores = np.random.uniform(low=0.5, high=1.0, size=high_ranking_entries)
        scores = np.append(low_scores, high_scores)
        return ranked_indices, scores

    def _set_scores_in_matrix(self, matrix, row_idx, ranked_column_indices, scores):
        if len(ranked_column_indices) != len(scores):
            raise ValueError("serires of ranked indices and series of scores must have same dimension")
        for i,s in zip(ranked_column_indices, scores):
            matrix[row_idx, i] = s

    def _generate_score_matrix(self, shape):
        m = np.zeros(shape)
        number_entries = shape[1]
        for u in range(shape[0]):
            low_ranking_entires = round(number_entries * np.random.uniform(low=0.2, high=0.4))  # 20%-40% low ranking entries
            high_ranking_entries = round(number_entries * np.random.uniform(low=0.15, high=0.25))  # 15%-25% high ranking entries
            indices, scores = self._generate_scores(number_entries, low_ranking_entires, high_ranking_entries)
            self._set_scores_in_matrix(m, u, indices, scores)
        return m

    def _generate_full_observations(self, full_user_score_matrix):
        l = np.prod(full_user_score_matrix.shape)
        user_dix = np.zeros(l, dtype=np.int32)
        item_idx = np.zeros(l, dtype=np.int32)
        score = np.zeros(l)
        i = 0
        for idx_tuple in np.ndindex(*full_user_score_matrix.shape):
            user_dix[i], item_idx[i] = idx_tuple
            score[i] = full_user_score_matrix[idx_tuple[0], idx_tuple[1]]
            i += 1
        o = pd.DataFrame()
        o['user index'] = user_dix
        o['item index'] = item_idx
        o['score'] = score
        return o

