# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 11:11:18 2019

@author: Lenovo
"""

from selenium import webdriver
import csv

to_path = r'E:\NetEase_cloud_music\albums\playlist_400.csv'
url = 'https://music.163.com/#/discover/playlist/?order=hot&cat=%E6%B2%BB%E6%84%88&limit=35&offset=35'

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
browser = webdriver.Chrome(r"C:\Users\Lenovo\Downloads\chromedriver.exe",chrome_options=chrome_options)
csv_file = open(to_path, 'w', newline='', encoding='utf-8')
writer = csv.writer(csv_file)
writer.writerow(['标题', '播放数', '链接'])
#url = 'http://music.163.com/#/discover/playlist/?order=hot&cat=%E5%85%A8%E9%83%A8&limit=35&offset=0'

i = 0
while url != 'javascript:void(0)':
    print(i)
    i+=1
    browser.get(url)
    browser.switch_to.frame('contentFrame')
    data = browser.find_element_by_id('m-pl-container').find_elements_by_tag_name('li')
    for i in range(len(data)):
        nb = data[i].find_element_by_class_name('nb').text
        if '万' in nb and int(nb.split('万')[0]) > 400:
            msk = data[i].find_element_by_css_selector('a.msk')
            writer.writerow([msk.get_attribute('href')])
    url = browser.find_element_by_css_selector('a.zbtn.znxt').get_attribute('href')
csv_file.close()
