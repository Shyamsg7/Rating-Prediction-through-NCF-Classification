#CHANGE THIS FILE FOR CLASSIFICATION

import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset

random.seed(0)


class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into PyTorch Dataset"""
    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:
            target_tensor: torch.Tensor, the corresponding rating (class label) for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


class SampleGenerator(object):
    """Construct dataset for multi-class classification"""

    def __init__(self, ratings,n_test=1):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns

        self.ratings = ratings
        self.n_test = n_test
        self.preprocessed_ratings = self._preprocess_ratings(ratings)
        self.train_ratings, self.test_ratings = self._split_loo(self.preprocessed_ratings, n_test)

    def _preprocess_ratings(self, ratings):
        """
        Keep ratings as categorical classes (1-5 mapped to 0-4 for PyTorch compatibility)
        """
        ratings = deepcopy(ratings)
        ratings['rating'] = ratings['rating'].astype(int) - 1  # Map [1-5] -> [0-4]
        return ratings

    # def _split_loo(self, ratings):
    #     """Leave-One-Out train/test split"""
    #     ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
    #     test = ratings[ratings['rank_latest'] == 1]
    #     train = ratings[ratings['rank_latest'] > 1]
    #     assert train['userId'].nunique() == test['userId'].nunique()
    #     return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    def _split_loo(self, ratings, n_test=1):
        """
        Split dataset into train and test sets, with `n_test` interactions per user in the test set.

        Args:
            ratings: DataFrame, contains user-item interactions.
            n_test: int, number of recent interactions to include in the test set.

        Returns:
            train: DataFrame, contains the train set.
            test: DataFrame, contains the test set.
        """
        # Rank interactions by timestamp for each user
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)

        # Test set contains the top `n_test` interactions for each user
        test = ratings[ratings['rank_latest'] <= n_test]
        
        # Train set contains all other interactions
        train = ratings[ratings['rank_latest'] > n_test]

        # Ensure every user is in both train and test sets
        # assert train['userId'].nunique() == test['userId'].nunique()

        # Return train and test dataframes
        return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    def instance_a_train_loader(self, batch_size):
        """Instance train loader for one training epoch"""
        users = self.train_ratings['userId'].tolist()
        items = self.train_ratings['itemId'].tolist()
        ratings = self.train_ratings['rating'].tolist()

        dataset = UserItemRatingDataset(
            user_tensor=torch.LongTensor(users),
            item_tensor=torch.LongTensor(items),
            target_tensor=torch.LongTensor(ratings)  # LongTensor for classification labels
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def instance_a_test_loader(self, batch_size):
        """Instance test loader for one training epoch"""
        users = self.test_ratings['userId'].tolist()
        items = self.test_ratings['itemId'].tolist()
        ratings = self.test_ratings['rating'].tolist()

        dataset = UserItemRatingDataset(
            user_tensor=torch.LongTensor(users),
            item_tensor=torch.LongTensor(items),
            target_tensor=torch.LongTensor(ratings)  # LongTensor for classification labels
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    @property
    def evaluate_data(self):
        """Create evaluate data for classification"""
        test_users = self.test_ratings['userId'].tolist()
        test_items = self.test_ratings['itemId'].tolist()
        test_ratings = self.test_ratings['rating'].tolist()

        return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(test_ratings)]


# class UserItemRatingDataset(Dataset):
#     """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""
#     def __init__(self, user_tensor, item_tensor, target_tensor):
#         """
#         args:

#             target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
#         """
#         self.user_tensor = user_tensor
#         self.item_tensor = item_tensor
#         self.target_tensor = target_tensor

#     def __getitem__(self, index):
#         return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

#     def __len__(self):
#         return self.user_tensor.size(0)


# class SampleGenerator(object):
#     """Construct dataset for NCF """

#     def __init__(self, ratings):
#         """
#         args:
#             ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
#         """
#         assert 'userId' in ratings.columns
#         assert 'itemId' in ratings.columns
#         assert 'rating' in ratings.columns

#         self.ratings = ratings
#         # explicit feedback using _normalize and implicit using _binarize
#         # self.preprocess_ratings = self._normalize(ratings)
#         self.preprocess_ratings = self._binarize(ratings)
#         self.user_pool = set(self.ratings['userId'].unique())
#         self.item_pool = set(self.ratings['itemId'].unique())
#         # create negative item samples for NCF learning
#         self.negatives = self._sample_negative(ratings)
#         self.train_ratings, self.test_ratings = self._split_loo(self.preprocess_ratings)

#     def _normalize(self, ratings):
#         """normalize into [0, 1] from [0, max_rating], explicit feedback"""
#         ratings = deepcopy(ratings)
#         max_rating = ratings.rating.max()
#         ratings['rating'] = ratings.rating * 1.0 / max_rating
#         return ratings
    
#     def _binarize(self, ratings):
#         """binarize into 0 or 1, imlicit feedback"""
#         ratings = deepcopy(ratings)
#         ratings.loc[ratings['rating'] > 0, 'rating'] = 1.0 # replace ratings['rating'][ratings['rating'] > 0] = 1.0
#         return ratings

#     def _split_loo(self, ratings):
#         """leave one out train/test split """
#         ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
#         test = ratings[ratings['rank_latest'] == 1]
#         train = ratings[ratings['rank_latest'] > 1]
#         assert train['userId'].nunique() == test['userId'].nunique()
#         return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

#     def _sample_negative(self, ratings):
#         """return all negative items & 100 sampled negative items"""
#         interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
#             columns={'itemId': 'interacted_items'})
#         interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
#         interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(list(x), 99))
#         return interact_status[['userId', 'negative_items', 'negative_samples']]

#     def instance_a_train_loader(self, num_negatives, batch_size):
#         """instance train loader for one training epoch"""
#         users, items, ratings = [], [], []
#         train_ratings = pd.merge(self.train_ratings, self.negatives[['userId', 'negative_items']], on='userId')
#         train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(list(x), num_negatives))
#         for row in train_ratings.itertuples():
#             users.append(int(row.userId))
#             items.append(int(row.itemId))
#             ratings.append(float(row.rating))
#             for i in range(num_negatives):
#                 users.append(int(row.userId))
#                 items.append(int(row.negatives[i]))
#                 ratings.append(float(0))  # negative samples get 0 rating
#         dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
#                                         item_tensor=torch.LongTensor(items),
#                                         target_tensor=torch.FloatTensor(ratings))
#         return DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     @property
#     def evaluate_data(self):
#         """create evaluate data"""
#         test_ratings = pd.merge(self.test_ratings, self.negatives[['userId', 'negative_samples']], on='userId')
#         test_users, test_items, negative_users, negative_items = [], [], [], []
#         for row in test_ratings.itertuples():
#             test_users.append(int(row.userId))
#             test_items.append(int(row.itemId))
#             for i in range(len(row.negative_samples)):
#                 negative_users.append(int(row.userId))
#                 negative_items.append(int(row.negative_samples[i]))
#         return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(negative_users),
#                 torch.LongTensor(negative_items)]
