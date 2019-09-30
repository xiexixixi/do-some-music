# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 11:11:18 2019

@author: Lenovo
"""


import pandas as pd
import crawl_music as cm

to_path = r'E:\NetEase_cloud_music\healing'
to_path = r'/home/public/zhang_xie/xie_task/music_crawl/sad' 

albums_file = r'E:\NetEase_cloud_music\albums\playlist_500.csv'
albums_file = r'/home/public/zhang_xie/xie_task/music_crawl/albums/sad_playlist_400.csv'

albums_df = pd.read_csv(albums_file,header = 0)
sr = albums_df.link
c=0
for album_url in sr:
    print('------------finish album%d------------'%c)

    try:
        cm.download_wav_in_album(album_url,to_path)
    except:
        pass
    
    c+=1
    










