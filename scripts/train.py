import os
import sys

# os.chdir('/home/jovyan/Co-Speech-Motion-Generation/src')  

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
sys.path.append(os.getcwd())
from trainer import Trainer,Trainer_vq

if __name__ == '__main__':

    trainer = Trainer()
    trainer.train()





