#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 17:42:02 2017

@author: sean
"""

import pandas as pd
import numpy as np
import scipy.sparse as sps



class SongUser:
    s2i = None
    u2i = None

    def __init__(self, filename, nrows=None):
        self.filename = filename
        self.df = pd.read_csv(filename,
                              header=None, sep='\t', \
                              names=['user', 'song', 'cnt'], nrows=nrows)

        songs = self.df.song.value_counts()
        songs.name = 'count_users'


        self.songs = songs

        users = self.df.user.value_counts()
        users.name = 'count_songs'

        self.users = users

        def create_index(cv, ov, name):
            if cv is None:
                cv = pd.Series(data=np.arange(ov.shape[0]),
                               index=ov.index,
                               name=name)
            else:
                miss_index = ov.index.difference(cv.index)
                data = np.arange(miss_index.shape[0]) + cv.shape[0]
                cv = cv.append(pd.Series(data=data,
                                         index=miss_index,
                                         name=name))
            return cv

        SongUser.s2i = create_index(SongUser.s2i, songs, 's2i')
        SongUser.u2i = create_index(SongUser.u2i, users, 'u2i')

        self.m_songuser = \
            sps.csr_matrix(\
                           (np.repeat(1, self.df.shape[0]), \
                           (self.df.song.map(SongUser.s2i),
                            self.df.user.map(SongUser.u2i)))
                          )
        # shape only set so don't create 1 d array

# shape =(48373586, 3)

def vrange(starts, stops):
    """Create concatenated ranges of integers for multiple start/stop

    Parameters:
        starts (1-D array_like): starts for each range
        stops (1-D array_like): stops for each range (same shape as starts)

    Returns:
        numpy.ndarray: concatenated ranges

    For example:

        >>> starts = [1, 3, 4, 6]
        >>> stops  = [1, 5, 7, 6]
        >>> vrange(starts, stops)
        array([3, 4, 4, 5, 6])

    """
    stops = np.asarray(stops)
    l = stops - starts # Lengths of each range.
    return np.repeat(stops - l.cumsum(), l) + np.arange(l.sum()), l.cumsum()




def mask_sparse(csr_mat, rows):
    indptr = csr_mat.indptr[rows]
    # rows could be boolean or indices
    # ASSUME INDICES ie rows + 1
    indptr_n = csr_mat.indptr[rows+1]
    sel, l_cum = vrange(indptr, indptr_n)

    data = csr_mat.data[sel]
    indices = csr_mat.indices[sel]
    indptr_new = np.insert(l_cum, 0, 0)
    extra_rows = csr_mat.shape[0] - len(rows)
    import pdb; pdb.set_trace()
    if extra_rows > 0:
        indptr_new = np.append(indptr_new,np.repeat(l_cum[-1],extra_rows))
    #have to insert dummy matrices if want same shape
    return sps.csr_matrix((data, indices, indptr_new), shape=csr_mat.shape)


def match_np(all_songs, u_songs, u2i,  A):
    l1 = all_songs.sum(axis=1)
    l2 = u_songs.sum(axis=1)
    n_users_all_songs = all_songs.shape[1]
    n_users_u_songs = u_songs.shape[1]
    n_users = len(u2i)

    if n_users_all_songs < n_users:
        # https://stackoverflow.com/questions/26462617/adding-a-column-of-zeroes-to-a-csr-matrix
        all_songs= sps.hstack((all_songs,
                               sps.csr_matrix((all_songs.shape[0], 
                                               n_users - n_users_all_songs),
                                              dtype=all_songs.dtype)))
    if n_users_u_songs < n_users:
        u_songs = sps.hstack((u_songs,
                              sps.csr_matrix((u_songs.shape[0],
                                              n_users - n_users_u_songs),
                                             dtype=u_songs.dtype)))
    up=all_songs.dot(u_songs.T)
    #up[up>0]/= np.dot(np.float_power(l1,A),np.float_power(l2.T,(1.0-A)))
    return up


def Score(self, all_songs, user_songs):
    s_scores={}
    all_song
    s_scores[s] = 0.0
    s2u
    s_match = self.Match(s, u_song)
    s_match = np.power(s_match, Q).sum(axis=1)

    return s_scores


def RecommendToUser(self, user, u2s_v):
    songs_sorted = []
    for p in self.predictors:
        ssongs = []
        if user in u2s_v:
            ssongs = MSD_util.sort_dict_dec(p.Score(self.all_songs, u2s_v[user]))
        else:
            ssongs = list(self.all_songs)

        cleaned_songs = []
        for x in ssongs:
            if len(cleaned_songs) >= self.tau:
                break
            if x not in u2s_v[user]:
                cleaned_songs.append(x)

        songs_sorted += [cleaned_songs]

    return self.GetStochasticRec(songs_sorted, self.Gamma)

def RecommendToUsers(self, l_users, u2s_v):
    sti = time.clock()
    rec4users = []
    for i,u in enumerate(l_users):
        if not (i+1)%10:
            if u in u2s_v:
                print "%d] %s w/ %d songs"%(i+1, l_users[i], len(u2s_v[u])),
            else:
                print "%d] %s w/ 0 songs"%(i+1, l_users[i]),
            fl()
        rec4users.append(self.RecommendToUser(u, u2s_v))
        cti = time.clock() - sti
        if not (i+1)%10:
            print " tot secs: %f (%f)"%(cti, cti/(i+1))
        fl()
    return rec4users






