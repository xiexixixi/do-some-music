
import os
import re
import time
import numpy as np
import pandas as pd


path = '/home/aaa/output'
label = 'healing'




'''输入歌词文本，输出时间序列time_list，歌词序列 lyrics_sentence'''
def parse_lyrics(text):
    """
    解析歌词， 输出歌词时间和歌词
    ：param needs：歌词文本
    ：return time_list 时间序列 lyrics_sentence 歌词序列
    """
    segs = text.split('\n')
    time_list = []
    lyrics_sentence = []
    for need in segs:
        if need == "":
            continue
        else:
            need = need.strip("\n")
            need = need.split("]")

            if need[0][1].isdigit():
                time_list.append(need[0][1:])
                lyrics_sentence.append(need[1])
    return time_list,lyrics_sentence

def gen_df(path):
    
    
    text = open(path).read()
    time_seg,lyr_seg = parse_lyrics(text)
    df = pd.DataFrame({'time_stamp':time_seg,'lyric':lyr_seg})
    return df





def _cal_last_time(ss, ee):
    """
    计算起始时间和终止时间的时间差，得到持续时间。
    :param ss:  起始时间str: hh:mm:ss.mm
    :param ee:  终止时间str: hh:mm:ss.mm
    :return:  差值，持续时间str: hh:mm:ss.mm
    """
    a = [float(i) for i in ss.split(':')]
    s = a[0] * 60 + a[1]

    a = [float(i) for i in ee.split(':')]
    e = a[0] * 60 + a[1]

    duration = e - s

    return s,duration


def generate_all_segs(path,label):
    liss = os.listdir(os.path.join(path,label))
    
    for song in liss:
        try:
            song_dir = os.path.join(path,label,song)
            print(song_dir)

            os.system('mkdir {}/drums'.format(song_dir))
            os.system('mkdir {}/other'.format(song_dir))

            output_other_path = os.path.join(song_dir,'drums')
            output_other_path = os.path.join(song_dir,'other')

            string = open(os.path.join(song_dir,song+".txt")).read()
            time_seg,lyr_seg = parse_lyrics(string)


            txt_path = os.path.join(song_dir,song)
            df = gen_df(txt_path+'.txt')
            df.to_pickle(txt_path+'.pkl')


            drums_original_wav = os.path.join(song_dir,'drums.wav')
            other_original_wav = os.path.join(song_dir,'other.wav')

            drums_prelude_derived_wav = os.path.join(song_dir,'drums','prelude.wav')
            other_prelude_derived_wav = os.path.join(song_dir,'other','prelude.wav')

            command = 'ffmpeg -i {} -ss 0 -t 15 {}'.format(drums_original_wav,drums_prelude_derived_wav)
            os.system(command)
            command = 'ffmpeg -i {} -ss 0 -t 15 {}'.format(other_original_wav,other_prelude_derived_wav)
            os.system(command)
        except:
            print("exception in song")
            continue
        
        
        for i in range(len(time_seg)-1):
            try:
                ss,t = _cal_last_time(time_seg[i],time_seg[i+1])

                drums_original_wav = os.path.join(song_dir,'drums.wav')
                other_original_wav = os.path.join(song_dir,'other.wav')

                drums_derived_wav = os.path.join(song_dir,'drums','{}.wav'.format(i))
                other_derived_wav = os.path.join(song_dir,'other','{}.wav'.format(i))
                command = 'ffmpeg -i {} -ss {} -t {} {}'.format(drums_original_wav,ss,t,drums_derived_wav)
                os.system(command)
                command = 'ffmpeg -i {} -ss {} -t {} {}'.format(other_original_wav,ss,t,other_derived_wav)
                os.system(command)    
            
            
            except:
                print("exception in song")
                continue


for label in ["happy","sad","quiet","healing"]:
    generate_all_segs(path,label)


