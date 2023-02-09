# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import transforms
#import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger
from core.inference import get_final_preds
from utils.transforms import transform_preds
import dataset
import models
import cv2
from PIL import Image as im
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import ast
import pandas as pd
import scipy as sp    
import glob
import math
from scipy import spatial 

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))   
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )
    model = torch.nn.DataParallel(model, device_ids = (0,)).cuda()

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        
        #model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE,map_location=lambda storage, loc: storage.cuda(0)), strict=True)
        checkpoint = torch.load(cfg.TEST.MODEL_FILE,map_location=lambda storage, loc: storage.cuda(0))
        
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file,map_location=lambda storage, loc: storage.cuda(0)))

    #model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    #model = torch.nn.DataParallel(model, device_ids = (0,)).cuda()
    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize =  transforms.Compose([transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),transforms.ToTensor()]) 

    
    #normalize = transforms.Normalize(
    #    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
   # )
    
    '''

    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    validate(cfg, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir)

'''
    
    
    '''
    def getpoint(mat):
        height, width = mat.shape
        mat = mat.reshape(-1)
        idx = np.argmax(mat)
    '''    
    from cv2 import *
  
# initialize the camera
# If you have multiple camera connected with 
# current device, assign a value in cam_port 
# variable according to that
    cam_port = 0
    cam = VideoCapture(cam_port)
  
# reading the input using the camera
    result, image = cam.read()
  
# If image will detected without any error, 
# show result
    if result:
  
    # showing result, it take frame name and image 
    # output
    #imshow("GeeksForGeeks", image)
  
    # saving image in local storage
        imwrite("images/GeeksForGeeks.png", image)
  
    # If keyboard interrupt occurs, destroy image 
    # window
        waitKey(0)
        destroyWindow("GeeksForGeeks")
  





    #image = cv2.imread("image.jpg", cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    #test_image = im.open("/home/cslabdp/Downloads/deepfashion2-kps-agg-finetune-master/HRNet-Human-Pose-Estimation/128B.jpg").convert('RGB')
    path  = "images/"
    result = glob.glob(path + '/*.png')
    print(result)
    for t , path in enumerate(result):

        test_image = cv2.imread(path,1)

        image = np.asarray(test_image)
        print(image.shape)

        image = cv2.resize(image,(256,256))
       #image = cv2.resize(img, (3, 3))
        #image = transforms.Resize((224,244))
        transforms.Compose([transforms.Normalize(
        mean=[0.485, 0.456, 0.406,0.406], std=[0.229, 0.224, 0.225,0.225]),transforms.ToTensor()]) 

        #image.astype("float32")
        #cv2.imshow("image_resize", image)
        #x = transforms.ToTensor()
      #t= torch.tensor(image)
        #r = torch.unsqueeze(t,0)
        i = transforms.ToTensor()(image).unsqueeze_(0)
        #t = torch.tensor(image)
        #i = r.to(torch.float)
        #r = normalize(i)
        
        #r.unsqueeze(0)
        
        #print(i.shape)
        transforms.Lambda(lambda i: i.repeat(256,1,1,1))
        i = i.repeat(256,1,1,1)
        
    #print(i)
        #rint(r.shape)
        x,y,z = np.shape(image)
        print(x,y,z)
        heatmaps = np.zeros((x,y),dtype = 'float32')
    #print(heatmaps)

        with torch.no_grad():
            model.eval()
            res = model.forward(i)
            output = model(i)
            score_map = output.data.cpu()
            print(score_map.shape)
        #print(output)
        preds, _ = get_final_preds(
            cfg,
            output.clone().cpu().numpy())

        #print(preds)
        print(preds.shape) 


    #res.detach()
    #res=res.to("cpu").numpy().squeeze()
    
        res=res.to("cpu")


        res = np.array(res.detach().squeeze())
  
    #print(res)
    #print(res.shape)   
    #image = cv2.resize(res, (64, 64))
    #print(res.shape)
        def getpoint(mat):

        #print(mat)
        #return mat

            width,height=  mat.shape
        
            print(width)
            print(height)
        
        #width= mat.shape
        #size = mat.shape 
            print(mat.shape)

            mat = mat.reshape(-1)
            idx = np.argmax(mat)
            return mat

        
        for mat in preds:
            #print(mat)

            for i in mat:
                #print(i)


  
  #    size = getpoint(mat)
        
        
        
        
        #n,x,y = getpoint(mat)
        #print(type(mat))
        #x,y =mat
            #sparr2  = ', '.join(map(str, i))
            
                splitted = np.array_split(i,2)
                print(splitted)

            #splitted1 = [item.replace("]", "") for item in splitted]            
            #print(splitted[0])


                with open("x.txt",'a') as z:
                    z.write(str(splitted[0]))
                    z.write("\n")

            
                file_x = pd.read_csv("x.txt")
            
            
                file_x.to_csv("file_x.csv",header  = ['x'],index = False)
                file_x1 = pd.read_csv("file_x.csv")
           

                with open("y.txt",'a') as g:
                    g.write(str(splitted[1]))
                    g.write("\n")
                file_y  =pd.read_csv("y.txt")
                file_y.to_csv("file_y.csv",header = ['y'], index = False)
                x = pd.read_csv("file_x.csv")
                y = pd.read_csv("file_y.csv")

                x['x'] =  x['x'].apply(lambda x: x.replace('[','').replace(']','')) 
                y['y'] =  y['y'].apply(lambda y: y.replace('[','').replace(']','')) 
                x.to_csv("finalx.csv",index  =False)
                y.to_csv("finaly.csv",index = False)


#convert the string columns to int
#x['x'] = x['x'].astype(int)
                print(x)
                print(y)




                finalx = pd.read_csv("finalx.csv")
                finaly = pd.read_csv("finaly.csv")
                finalx1 = pd.DataFrame(finalx)
                finaly1 = pd.DataFrame(finaly)

#file1 = open("finalx.csv",'a',newline = '')
#file2 = open("finaly.csv",'r')

                frames  = [finalx1,finaly1]
                file = pd.concat(frames,axis = 1,join = 'inner')
                print(file)
                file.to_csv("coord5.csv",index  = False)



                df = pd.read_csv("coord5.csv")
                
                data = [(float(x),float(y)) for x, y in df[['x', 'y']].values ]

                nearest_points = []
                for point in data:
    # Compute the distance between the current point and all others
                    distances = [math.sqrt((point[0]-x[0] )**2+ (point[1]-x[1])**2) for x in data]
    # Use np.argsort() to sort the array and keep the three closest points
                    nearest_points.append([data[i] for i in np.argsort(distances)[1:4]])
                #print(distances)
                    print(distances)

                    with open("distances"+str(t)+'.txt','a') as p:
                        p.write(str(distances))
                        p.write("\n")
                
                
                    
                #row1 = df.loc[0]
                #print(row1)
            '''
                with open("coord5.csv",'r') as f:

                    for row in f:
                        #print(type(row))
                        #row1 = pd.DataFrame(row)
                        #row1.to_csv("row"+str(t)+'.csv')
                        #print(row)
                        #return row
                        #image = cv2.imread("/home/cslabdp/Downloads/deepfashion2-kps-agg-finetune-master/HRNet-Human-Pose-Estimation/128B.jpg")
                    #print(image.shape)
                    #image = cv2.resize(image,(60,60))
                    #file_x2 = pd.read_csv("file_x.csv")
                        
                        pts = list([[row]])
                        print(pts)
                        #distance_matrix = sp.spatial.distance_matrix(pts, pts)
                        #plt.xlim(0,60)
                        #plt.ylim(0,60)
                        #plt.imshow(image)

                    
            #file1 = pd.DataFrame(splitted[0])

            #file1.to_csv("x.csv")
            #print(splitted[1])
            #file2 = pd.DataFrame(splitted[1])
            #file2.to_csv("y.csv")
            #print(splitted)
            #print(i)
            '''


            
'''             #print(type(i))
            '''
           
'''
'''
    
'''
        
        #point1 = np.array([34.25, 6.75])

        #point2 = np.array([30.25, 4.75])
                    plt.plot(640, 570, "og", markersize=10)  # og:shorthand for green circle
                    plt.scatter(pts[:, 0], pts[:, 1], marker="x", color="red", s=200)
                    plt.show()  
    '''
'''

                data = pd.read_csv("coord5.csv")
                data_df  = pd.DataFrame(data)
  
                points = pd.DataFrame(my_data_df, columns=["x", "y"]).astype(float)
                distance_matrix = sp.spatial.distance_matrix(points, points)
                plt.savefig("final_image.jpg",distance_matrix)
    '''
'''        
        lst  = []
        for i in range(preds):
            lst.append(i)
            print(i)
            print(lst)


        #,y = preds

        
        #print(x,y)

        
            
       #rint(size)
        #image = np.toarray(i)
        
        #y1 =  cv2.circle(image, mat[0], 2, ( 0, 0,255), 2)
        #print(x,y)
        #print(mat[1])
        #print(mat[2])
        point1 = np.array(mat[0])
        point2 = np.array(mat[1])
'''
 
# calculating Euclidean distance
# using linalg.norm()
        #dist = np.linalg.norm(point1 - point2)
 
# printing Euclidean distance
        #print(dist)
        #blue=np.array([0,0,255],dtype=np.uint8)
       #v2.circle(image,blue,2,( 0, 0,255), 2)
        #blues=np.where(np.all((image==blue),axis=-1))
        #dist = blues[0] - blues[1]
        #print(dist)
        #print(blues)

        #return idx % width, idx // width
          
    #plt.show(image)
   # cv2.imwrite("image"+'.jpg',y1)
    #current_frame = current_frame + 1

'''
'''
if __name__ == '__main__':
    main()