# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 11:55:07 2016

@author: sean
"""
import pandas as pd
import numpy as np

import MSD_rec_np


f_triplets_tr = '../millionsong/data/train_triplets.txt'
f_triplets_tev = '../millionsong/data/kaggle_visible_evaluation_triplets.txt'
f_kaggle_users = '../millionsong/data/kaggle_users.txt'


users_v = pd.Index(pd.read_csv(f_kaggle_users, header=None, names=['user']).user)
tr = MSD_rec_np.SongUser(f_triplets_tr)
te = MSD_rec_np.SongUser(f_triplets_tev)

te_songs = MSD_rec_np.SongUser.s2i[te.songs.index]
te_songs = te_songs[te_songs<384546]
tr_songuser_test = mask_sparse(tr.m_songuser,te_songs.values)

print 'song to users on %s'%f_triplets_tr

kaggle_users_i = MSD_rec_np.SongUser.u2i[users_v]
#s2u_tr=MSD_util.song_to_users(f_triplets_tr)

_A = 0.15
_Q = 3


a = tr.m_songuser[te_songs.values,:]

import os
import psutil
process = psutil.Process(os.getpid())
print(process.memory_info().rss/1e9)

slices =np.append(np.arange(tr.m_songuser.shape[0],step=1000),tr.m_songuser.shape[0])
zs=[]
for i in range(len(slices)-1):
    print i
    z = MSD_rec_np.match_np(tr.m_songuser[slices[i]:slices[i+1],], 
                         tr_songuser_test, tr.u2i, _A) 
    zs.append(z)
    print(process.memory_info().rss/1e9)


# problem want the slice to preserve the user index 

miss_users = users_v.difference(te.users.index)

print 'Creating predictor..'
### calibrated
### pr=MSD_rec.PredSIc(s2u_tr, _A, _Q, "songs_scores.txt")

### uncalibrated
pr=MSD_rec.PredSI(s2u_tr, _A, _Q)

print 'Creating recommender..'
cp = MSD_rec.SReco(songs_ordered)
cp.Add(pr)
cp.Gamma=[1.0]

r=cp.RecommendToUsers(users_v[user_min:user_max],u2s_v)
MSD_util.save_recommendations(r,"kaggle_songs.txt",osfile)



