from glob import glob
from os.path import join
from pathlib import Path
import os
import shutil
import zipfile


predictions = 'predictions'
codalabfiles = 'codalab'

if os.path.exists(codalabfiles):
    shutil.rmtree(codalabfiles)
os.makedirs(codalabfiles)

def dozip(filein, zipout):
    with zipfile.ZipFile(zipout, 'w') as myzip:
        #myzip.write(open(filein, 'rt').read())
        myzip.write(filein, arcname='answer.csv')


for file in Path(predictions).rglob('*.csv'):
    taskname = str(file.parent).split('/')[1]
    taskname = taskname.replace('_raw', '').replace('_npy', '')
    name = file.name.replace('.csv', '')
    if '_' in taskname:
        task, proc = taskname.split('_')
        if name != 'MLPE':
            name += '-'+proc
    else:
        task = taskname

    if '.reg' in name:
        continue


    os.makedirs(join(codalabfiles, task), exist_ok=True)

    outpath = join(codalabfiles, task, name+'.zip')
    dozip(file, outpath)

    print(task, name)
    