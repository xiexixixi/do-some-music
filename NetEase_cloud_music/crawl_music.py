# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:45:22 2019

@author: Lenovo
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
import urllib.request
import json
import re
import os


#to_path = r'E:\NetEase_cloud_music\healing'
 
def Header(url):
    #创建请求头部
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.104 Safari/537.36 Core/1.53.4882.400 QQBrowser/9.7.13059.400"}
    req = urllib.request.Request(url, headers=headers)
    #打开url
    reponse = urllib.request.urlopen(req)
    lycJson = reponse.read().decode("utf-8","ignore")
    return lycJson
 
def lyricsCrawer(music_id):
    url = 'http://music.163.com/api/song/lyric?' + 'id=' + str(music_id) + '&lv=1&kv=1&tv=-1'  # 括号中填入歌曲id
    lyc=Header(url)
    #进行json的解码
    l=json.loads(lyc)
    #l是字典类型的  l字典里面读取键lyc得到一个value 而value又是一个字典类型的 再读取键lyric得到时间戳和歌词
    data=l["lrc"]["lyric"]
    #利用正则表达式去掉歌词前面的时间戳
    re_lyrics=re.compile(r"\[.*\]")
    #将data字符串中的re_lyrcs替换成空
    #lyc=re.sub(re_lyrics,"",data)
    #lyc=lyc.strip()
    #print(lyc)
#    return lyc
    return data
 
def get_music_name_id(url):
    HtmlStr=str(Header(url))
    #第一次正则表达式筛选
    pat1=r'<ul class="f-hide"><li><a href="/song\?id=\d*?">.*</a></li>'
    #<ul class="f-hide"><li><a href="/song\?id=\d*?">.*</a></li></ul>
    Html=re.compile(pat1,re.S)
    list1=Html.findall(HtmlStr)
    result=list1[0]
 
    #第二次正则表达式筛选出name id
    pat2=r'<li><a href="/song\?id=\d*?">(.*?)</a></li>'#歌名
    pat3=r'<li><a href="/song\?id=(\d*?)">.*?</a></li>'#歌id
    Mname=re.compile(pat2)
    namelist2=Mname.findall(result)
    #print(namelist2)
 
    Mid=re.compile(pat3)
    idlist3=Mid.findall(result)
 
    return namelist2,idlist3


def write_to_file(name,id_,path):
    Mlyc=lyricsCrawer(id_)

    filepath = os.path.join(path, name + ".txt")
    print("开始下载",name)
    try:
        f=open(filepath,"w",encoding='utf-8')
    except FileNotFoundError:
        print('FileNotFoundError')
        return False
    if len(Mlyc) > 100 and '[' in Mlyc:
        f.write(Mlyc)
        f.close()
        print('歌词下载成功')
        return True
    else:
        print('歌词下载失败')
        return False


def write_to_wav(name,id_,to_path):
    musicUrl='http://music.163.com/song/media/outer/url?id='+id_+'.wav'      
    print(musicUrl)      
    try:
#        print('正在下载',name)
#        urllib.request.urlretrieve(musicUrl,r'E:\NetEase_cloud_music\quiet\%s.wav'% name)
#        urllib.request.urlretrieve(musicUrl,r'/home/public/zhang_xie/xie_task/music_crawl/happy/%s.wav'% name)
        res = urllib.request.urlopen(musicUrl)
        html = res.read()
        with open(to_path+'/%s.wav'%name,mode='wb') as file:
            file.write(html)
        
        
        print('歌曲下载成功\n')
        return True
    except:
        print('歌曲下载失败\n')
        return False




def download_Lyrics_in_music(music_url,path):
    #输入歌曲的lrc_url，要保存的path
    music_name,music_id=get_music_name_id(music_url)
    write_to_file(music_name,music_id,path)
    



def download_wav_in_album(play_url,to_path):

    music_name,music_id = get_music_name_id(play_url)
    
    for name,id_ in zip(music_name,music_id):
#        print('fuck',name,id_)
        if write_to_file(name,id_,to_path):
            write_to_wav(name,id_,to_path)
    






to_path = r'/home/public/zhang_xie/xie_task/music_crawl/happy' 
to_path = r'E:\NetEase_cloud_music\happy'

albums_file = r'/home/public/zhang_xie/xie_task/music_crawl/albums/happy_playlist_200.csv'
albums_file = r'E:\NetEase_cloud_music\albums\happy_playlist_200.csv'

albums_df = pd.read_csv(albums_file,header = 0)
sr = albums_df.link
c=0
for album_url in sr:
    print('------------finish album%d------------'%c)

    try:
        download_wav_in_album(album_url,to_path)
    except:
        pass
    
    
    c+=1













# =============================================================================
#         
# if __name__ == '__main__':
#     
#     url=r'https://music.163.com/playlist?id=2335548255'
# 
#     download_wav_in_album(url,to_path)
#     
# =============================================================================
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    