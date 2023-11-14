
import scipy.sparse as sp
import numpy as np
import pandas as pd


class Data(object):

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix, self.train_num = self.load_rating_file_as_matrix(path + "/train.csv")
        self.testRatings, self.test_num = self.load_rating_file_as_matrix(path + "/test.csv")
        self.textualfeatures,self.visualfeatures, = self.load_features(path)
        self.num_users, self.num_items = self.trainMatrix.shape
        print(self.train_num+self.test_num)

    def load_rating_file_as_matrix(self, filename):
        '''
        construct a sparse matrix to represent the user's
        rating of the item from the file
        '''

        num_users, num_items, num_total = 0, 0, 0
        df = pd.read_csv(filename, index_col=None, usecols=None)
        for index, row in df.iterrows():
            u, i = int(row['userID']), int(row['itemID'])
            num_users = max(num_users, u)
            num_items = max(num_items, i)

        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        for index, row in df.iterrows():
            user, item ,rating = int(row['userID']), int(row['itemID']) ,1.0
            if (rating > 0):
                mat[user, item] = 1.0
                num_total += 1
        return mat, num_total

    def load_features(self, path):
        '''
        Extract feature vectors from the text feature
        data and the image feature data
        '''

        import os
        doc2vec_model = np.load(os.path.join(path, 'review.npz'), allow_pickle=True)['arr_0'].item()
        vis_vec = np.load(os.path.join(path, 'image_feature.npy'), allow_pickle=True).item()
        filename = path + '/train.csv'
        filename_test =  path + '/test.csv'
        df = pd.read_csv(filename, index_col=None, usecols=None)
        df_test = pd.read_csv(filename_test, index_col=None, usecols=None)
        num_items = 0
        asin_i_dic = {}
        for index, row in df.iterrows():
            asin, i = row['asin'], int(row['itemID'])
            asin_i_dic[i] = asin
            num_items = max(num_items, i)
        for index, row in df_test.iterrows():
            asin, i = row['asin'], int(row['itemID'])
            asin_i_dic[i] = asin
            num_items = max(num_items, i)
        features = []
        image_features = []
        for i in range(num_items+1):
            features.append(doc2vec_model[asin_i_dic[i]][0])
            image_features.append(vis_vec[asin_i_dic[i]])

        return np.asarray(features,dtype=np.float32),np.asarray(image_features,dtype=np.float32)
