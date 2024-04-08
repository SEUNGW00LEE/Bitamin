import os
import json
from glob import glob

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

import cv2


class EggDatasetBasic(Dataset):
    def __init__(self, base_PATH, transforms=None):
        
        self.transforms=transforms
        
        self.main_df=pd.DataFrame(columns=['image_path','json_path','label'])
        image_path=glob(os.path.join(base_PATH, '*', '*.jpg'))
        json_path=[path.replace('.jpg','.json') for path in image_path]
        label=[int(path.split('/')[-2]) for path in image_path]
        
        self.main_df['image_path']=image_path
        self.main_df['json_path']=json_path
        self.main_df['label']=label
        self.main_df.reset_index(drop=True, inplace=True)       

        
    def __len__(self):
        return len(self.main_df)
    
    def __getitem__(self, idx):
        img_path=self.main_df['image_path'][idx]           
        
        label=self.main_df[self.main_df['image_path']==img_path]['label'].unique().item()          
        
        img=cv2.imread(img_path)
        tmp=np.nan   
        
        if img is None:            
            return torch.empty((3,512,512)).fill_(tmp), torch.tensor(99.)
        
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)           

            img = black_crop(img, self.main_df['json_path'][idx])

            if img is None:            
                return torch.empty((3,512,512)).fill_(tmp), torch.tensor(99.)

            else:         
                img/=255.0
                if self.transforms:                        
                    img=self.transforms(image=img)['image']

                return img.float(), torch.tensor(label).float()
   
        
        
def black_crop(original_image, json_path):    
    
    json_=open(json_path).read()
    json_=json.loads(json_)
    
    ls=[]
    for i in json_['label_info']['shapes']:
        label=i['label']
        ls.append(label)
        if (label=="egg_top_albumen") or (label=="egg_top_yolk"):  # top_result
            kp=cv2.KeyPoint_convert(i['points'])
            pt_list=[]
            for item in kp:
                pt_list.append(item.pt)
            contours=np.array(pt_list, dtype=np.int)

            xmin=contours[:,1].min()
            xmax=contours[:,1].max()
            ymin=contours[:,0].min()
            ymax=contours[:,0].max()

            stencil=np.zeros(original_image.shape).astype(original_image.dtype)        
            _=cv2.fillPoly(stencil, pts=[contours], color=(255,255,255))             
            result=cv2.bitwise_and(original_image, stencil)
            result=result[xmin:xmax, ymin:ymax, :]
            result=result.astype(np.float32)
            
            if (result.shape[0]==0) or (result.shape[1]==0):
                return None
            else:
                return result
    return None

