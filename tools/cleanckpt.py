import os 
import shutil

pwd = os.path.abspath(os.getcwd())

path = pwd + '/ckpts'


for date in os.listdir(path):
    datepath = os.path.join(path, date)
    for step in sorted(os.listdir(datepath)):
        steppath = os.path.join(datepath, step)
        if steppath.split('/')[-1] == 'last.ckpt':
            continue
        if len(os.listdir(steppath)) == 0:
            os.removedirs(steppath)