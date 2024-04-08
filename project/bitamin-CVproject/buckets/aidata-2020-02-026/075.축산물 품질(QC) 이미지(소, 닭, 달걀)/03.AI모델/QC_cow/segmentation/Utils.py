import json, os, cv2
import numpy as np

import torch
import time

import matplotlib.pyplot as plt

from skimage.io import imread
from tqdm import tqdm

# 주소에 폴더가 없으면 폴더를 생성하는 함수
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        
def interpolation_classifier(label):  
    assert np.array_equal(label[:, :, 0], label[:, :, 1])
    assert np.array_equal(label[:, :, 1], label[:, :, 2])
    assert np.array_equal(label[:, :, 0], label[:, :, 2])
    
    temp = label[:, :, 0]
    temp[np.where(temp > 127.5)] = 255
    temp[np.where(temp != 255)] = 0
    
    return np.repeat(temp[:, :, np.newaxis], 3, axis=2)
    
# 특정 Path의 파일들을 읽어와서 전처리 후 파일들을 저장하는 함수
def transform_folders(LOAD_PATH, SAVE_PATH, Test_folders, IMG_WIDTH=512, IMG_HEIGHT=512, IMG_CHANNELS=3):
    # 처리된 결과를 저장할 Path를 지정
    results_train_img_path = SAVE_PATH + "/train/img/"
    results_test_img_path = SAVE_PATH + "/test/img/"
    results_train_label_path = SAVE_PATH + "/train/label/"
    results_test_label_path = SAVE_PATH + "/test/label/"
    
    # 해당 Path에 폴더를 생성함
    createFolder(results_train_img_path)
    createFolder(results_test_img_path)
    createFolder(results_train_label_path)
    createFolder(results_test_label_path)
    
    # Raw path내 폴더 개수
    folder_list = sorted(os.listdir(LOAD_PATH))

    # Train 대상 폴더를 구분함
    Train_folders = [x for x in folder_list if x not in Test_folders]
        
    # Train 파일들에 대해서 처리함
    print("For Train set!!")
    time.sleep(2)
    for train_folder in tqdm(Train_folders):
        transform_files(LOAD_PATH, train_folder, results_train_img_path, results_train_label_path, IMG_WIDTH=IMG_WIDTH, IMG_HEIGHT=IMG_HEIGHT, IMG_CHANNELS=IMG_CHANNELS)
        
    # Test 파일들에 대해서 처리함
    print("For Test set!!")
    time.sleep(2)
    for test_folder in tqdm(Test_folders):
        transform_files(LOAD_PATH, test_folder, results_test_img_path, results_test_label_path, IMG_WIDTH=IMG_WIDTH, IMG_HEIGHT=IMG_HEIGHT, IMG_CHANNELS=IMG_CHANNELS)
        

def transform_files(LOAD_PATH, EACH_FOLDER, SAVE_IMG_PATH, SAVE_LABEL_PATH, IMG_WIDTH=512, IMG_HEIGHT=512, IMG_CHANNELS=3):

    # Segmentation 정보 파일 Json 파일들의 이름을 가져옴
    list_files = os.listdir(LOAD_PATH + EACH_FOLDER)
    list_files = sorted([f.replace(".json", "") for f in list_files if f.endswith('json')])
    
    # 이미지와 Segmentation 정보가 동시에 존재하면
    for file in list_files:
        img_file_extension = ""
        if os.path.isfile(LOAD_PATH + EACH_FOLDER + "/" + file + ".jpg"):
            img_file_extension = ".jpg"
        elif os.path.isfile(LOAD_PATH + EACH_FOLDER + "/" + file + ".png"):
            img_file_extension = ".png"
        else:
            continue
        
        if os.path.isfile(LOAD_PATH + EACH_FOLDER + "/" + file + img_file_extension) and os.path.isfile(LOAD_PATH + EACH_FOLDER + "/" + file + ".json"):
            
            try:
                # Segmenatation 정보를 읽어옴
                with open(LOAD_PATH + EACH_FOLDER + "/" + file + ".json") as json_file:
                    json_data = json.load(json_file)
            except:
                print("Unable to read", LOAD_PATH + EACH_FOLDER + "/" + file + ".json")
                continue
                
            # Segmentation 좌표를 가져옴
            pixel_points_list = np.array(json_data["shapes"][0]["points"], dtype=np.int32)
            
            # 이미지 사이즈에 맞는 Numpy Array을 생성 후
            empty_polygon_array = np.zeros((json_data["imageHeight"], json_data["imageWidth"], IMG_CHANNELS), dtype=np.uint8)

            # 가져온 좌표에 맞춰서 Polygon 생성            
            polygon_array = cv2.fillPoly(empty_polygon_array, [pixel_points_list], (255, 255, 255))
            polygon_array = cv2.resize(polygon_array, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            polygon_array = interpolation_classifier(polygon_array)
            
            # Polygon 영역이 존재한다면
            if (polygon_array != 0).sum() != 0 :
                
                # 이미지를 읽어와서 모델에 넣을 사이즈로 Resizing하여 저장함
                img = imread(LOAD_PATH + EACH_FOLDER + "/" + file + img_file_extension)
                img = img[:,:,:IMG_CHANNELS] # PNG는 Alpha 채널을 날려줌
                
                # 이미지가 어둡지 않을때
                if np.mean(img) > 50 and np.std(img) > 25:
                    # 라벨 정보를 저장
                    np.save(SAVE_LABEL_PATH + EACH_FOLDER + "_" + file + ".npy", polygon_array)
                    # 이미지를 리사이징 하여 저장
                    img = cv2.resize(img, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
                    np.save(SAVE_IMG_PATH + EACH_FOLDER + "_" + file + ".npy", img)
                else:
                    continue

## Intersection Over Union (IOU) 구현
def IOU_Numpy(outputs: torch.Tensor, labels: torch.Tensor, reduction='mean', SMOOTH=1e-6):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    #outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    results = []

    outputs = outputs.squeeze(1)
    labels = labels.squeeze(1)
    
    outputs = outputs.to('cpu').detach().numpy()
    labels = labels.to('cpu').detach().numpy()
    
    batch_size = labels.shape[0]
    for batch in range(batch_size):
        t, p = labels[batch], outputs[batch]
        true = np.sum(t)
        pred = np.sum(p)
        
        # non empty mask case.  Union is never empty 
        # hence it is safe to divide by its number of pixels
        intersection = np.sum(t * p)
        union = true + pred - intersection
        #iou = (intersection + SMOOTH) / (union + SMOOTH)
        iou = intersection / union        
        results.append(iou)

    if reduction == 'mean':
        return np.mean(results)  # Or thresholded.mean() if you are interested in average across the batch
    else:
        return results  

# 예측 이미지의 확률을
fn_classifier = lambda x :  1.0 * (x > 0.5)  # threshold 0.5 기준으로 indicator function으로 classifier 구현
# 모델에서 출력된 Output은 배치 차원이 들어가기 때문에 다음과 같이 차원을 변경함
fn_tonumpy_output = lambda x : x.to('cpu').detach().numpy().transpose(0,2,3,1) # device 위에 올라간 텐서를 detach 한 뒤 numpy로 변환
# 라벨은 결과 하나를 가져오기 때문에 배치 차원이 안들어감. 따라서 다음과 같이 차원을 변경함
fn_tonumpy_label = lambda x : x.to('cpu').detach().numpy().transpose(1,2,0) # device 위에 올라간 텐서를 detach 한 뒤 numpy로 변환

def Draw_Image(eval_net, dataset, test_iou_np, save_path, file_name, idx=0, device="cpu"):
    image_idx = np.argsort(test_iou_np)[-1 - idx]
    iou = round(test_iou_np[image_idx], 4)
    
    with torch.no_grad():  # test 이기 때문에 backpropa 진행 x, 학습된 네트워크가 정답과 얼마나 가까운지 loss만 계산
        eval_net.eval()  # 네트워크를 evaluation 용으로 선언

        ## 전체 PLT를 구성함
        fig = plt.figure(figsize=(12,12))
        fig.suptitle("IOU (" + str(iou) + ")", fontsize=22, y=0.95)
        fig.patch.set_facecolor('white')

        rows = 2
        cols = 2

        # 특정 Index의 아이템을 가져옴
        data = dataset.__getitem__(image_idx)
        label = data['label'].to(device)
        
        ## 정답 이미지를 보여줌
        img = np.load(data['file_name'])
        ax1 = fig.add_subplot(rows, cols, 1)
        ax1.imshow(img)
        ax1.set_title("Original image", fontsize=18)
        ax1.set_xticks([]), ax1.set_yticks([])

        ## 라벨 이미지를 불러와서 겹쳐 보여줌
        label = (fn_tonumpy_label(label) * 255)
        zero = np.zeros(shape=(512, 512, 2), dtype=np.uint8)
        label_image = np.concatenate((label, zero), axis=2).astype(np.uint8)
        label_overlay_image = cv2.addWeighted(img, 0.4, label_image, 0.6, 0)

        ax2 = fig.add_subplot(rows, cols, 2)
        ax2.imshow(label_overlay_image)
        ax2.set_title("Answer image", fontsize=18)
        ax2.set_xticks([]), ax2.set_yticks([])

        ## 예측 이미지를 불러와서 겹쳐 보여줌
        inputs = data['input'].to(device)
        inputs = inputs.reshape(1, 3, 512, 512)    
        output = eval_net(inputs)
        predict = (fn_tonumpy_output(fn_classifier(output)) * 255).reshape(512, 512, 1)
        predict_image = np.concatenate((zero, predict), axis=2).astype(np.uint8)
        predict_overlay_image = cv2.addWeighted(img, 0.4, predict_image, 0.6, 0)

        ax3 = fig.add_subplot(rows, cols, 3)
        ax3.imshow(predict_overlay_image)
        ax3.set_title("Predict image", fontsize=18)
        ax3.set_xticks([]), ax3.set_yticks([])

        ## Overlay 이미지를 생성하여 보여줌
        overlay_image = cv2.addWeighted(label_overlay_image, 0.5, predict_overlay_image, 0.5, 0)
        ax4 = fig.add_subplot(rows, cols, 4)
        ax4.imshow(overlay_image)
        ax4.set_title("Overlay image", fontsize=18)
        ax4.set_xticks([]), ax4.set_yticks([])
        
        # 생성된 이미지를 저장
        plt.savefig(save_path + file_name + "_" + str(idx) + ".png", transparent=False, dpi=300)
        # 생성된 이미지를 보여줌
        plt.show()