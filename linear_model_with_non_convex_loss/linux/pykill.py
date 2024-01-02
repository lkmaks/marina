#!python3
import os

res = os.system('ps aux | grep python3.8 > tmp.txt')
lines = open('tmp.txt', 'r').readlines()

for line in lines:
    if line.find('python3.8 ./experiment') != -1:
        i = 0
        while not line[i].isdigit():
            i += 1
        pid = int(line[i:i+4])
        os.system('kill ' + str(pid))

