from glob import glob
from os import path
from random import sample

rootDir = '../data/'
view = 'FRONT'
pics = glob( path.join( rootDir, '**' ,'CAM_{}.jpeg'.format(view)), recursive = True )
val_ratio = 0.1
outDir = '.'

n_scene = 134
n_sample = 126

train_sence = sample(list(range(n_scene)), int((1-val_ratio) * n_scene))


with open(path.join(outDir, 'train.txt'), 'w') as tf:
    with open(path.join(outDir,  'val.txt'), 'w') as vf:
        for pic in pics:
            scene_id = int(pic.split('scene_')[1].split('/')[0])
            sample_id = int(pic.split('sample_')[1].split('/')[0])
            print(sample_id)
            if (sample_id < n_sample - 2):
                if (scene_id not in train_sence):
                    vf.write('{}\n'.format(pic[len(rootDir):]))
                else:
                    tf.write('{}\n'.format(pic[len(rootDir):]))
