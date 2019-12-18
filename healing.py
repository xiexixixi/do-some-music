import os
import re
import time

input_path = '/home/dutir/xieyuning/music_crawl/healing'
input_path = '/home/aaa/healing'
output_path = '/home/aaa/output/healing'

def movefile(file_path):
    filename = file_path+'.txt'
    command = 'cp {} {}'.format(os.path.join(input_path,filename),os.path.join(output_path,file_path))
    print(command)
    os.system(command)
    


def divide(input_path,output_path):
    
    lis = os.listdir(input_path)
    lis.sort()
    #print(lis)
    
    wav_files = [i for i in lis if i.split('.')[-1] == 'wav']
    totle_num = len(wav_files)
    time_init = time.time()
    num = 0
    liss = os.listdir(input_path)[::-1]
    for file in liss: 

        try:
            fname = file.split('.')[0]
            if file.split('.')[-1] == 'wav':
                time_start=time.time()
                command = 'spleeter separate -i {}/{} -p spleeter:4stems -o {}'.format(input_path,file,output_path)
                print(command)
                
                os.system(command)
                
                movefile(fname)
                num+=1
                time_end=time.time()
                
                print("time_consume:\t",time_end-time_start)
                print("totle_time:\t",time_end-time_init)
                print("{}/{}".format(num,totle_num))
                
                
        except:
            continue
divide(input_path,output_path)
