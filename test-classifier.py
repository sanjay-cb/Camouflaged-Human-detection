from skimage.transform import pyramid_gaussian
from skimage.feature import hog
import joblib
import cv2
import os
import numpy as np
from skimage import feature
from imutils.object_detection import non_max_suppression
import warnings
warnings.filterwarnings("ignore")
import winsound
frequency=1500
duration=1000

def describe(image, radius, n_points,METHOD,eps=1e-7):
      
        lbp = feature.local_binary_pattern(image, n_points,
            radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, n_points + 3),
            range=(0, n_points + 2))
 
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
 
        
        return hist
    
def rgb2gray(im):
    gray = im[:, :, 0]*0.2989+im[:, :, 1]*0.5870+im[:, :, 2]*0.1140
    return gray

def sliding_window(image, window_size, step_size):
    
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

if __name__ == "__main__":

    test_images_path = 'Video/video2/I_CA_01'
    gt_images_path = 'Video/video2/I_CA_01-GT'
    test_dir_list = os.listdir(test_images_path)
    gt_dir_list = os.listdir(gt_images_path)
    test_length = len(test_dir_list)
    gt_length = len(gt_dir_list)
    print("Total Images = ",test_length)        

    for i in range(0,test_length):
        downscale=1.25
        visualize = True
    
        im= cv2.imread(test_images_path+'/' + test_dir_list[i])
        im= cv2.resize(im,(352,288))
        img = im.copy()
        print("Processing Image ",i+1," of ",test_length)
        min_wdw_sz = (100, 200)
        step_size = (30, 30)
        visualize_det = visualize
        radius = 3
        n_points = 8 * radius
        METHOD = 'uniform'
        model_path = "svm_model.npy"   
    
    
        # Load the classifier
        clf = joblib.load(model_path)
    
        detections = []
       
        scale = 0
        
        for im_scaled in pyramid_gaussian(im, downscale=downscale):
            
            cd = []
            for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
                if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                    continue
                # Calculate the HOG features
                im_window = cv2.resize(im_window,(64,128))
                im_window = rgb2gray(im_window)
                fd_hog = hog(im_window,orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
                fd_lbp = describe(im_window, radius, n_points,METHOD)
                fd = np.hstack([fd_hog,fd_lbp])
                fd = fd.reshape(1,-1)
                pred = clf.predict(fd)
                if pred == 1:
                   
                        detections.append((x, y,clf.decision_function(fd),
                        int(min_wdw_sz[0]*(downscale**scale)),
                        int(min_wdw_sz[1]*(downscale**scale))))                                         
                        cd.append(detections[-1])
            
            scale+=1
    
       
        clone = im.copy()
    
        rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
        sc = [score[0] for (x, y, score, w, h) in detections]
        sc = np.array(sc)
        pick = non_max_suppression(rects, probs = None, overlapThresh = 0.1)
    
        for(xA, yA, xB, yB) in pick:
            cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)
      
        
        gtImg = cv2.imread(gt_images_path+'/' + gt_dir_list[i])
        gtImg = cv2.resize(gtImg,(352,288))
        gtImg = cv2.cvtColor(gtImg,cv2.COLOR_BGR2GRAY)
        _,gtThresh = cv2.threshold(gtImg,50,255,0)
        contours, hierarchy = cv2.findContours(gtThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt = contours[max_index]
            xc, yc, wc, hc = cv2.boundingRect(cnt)
            inc = 30
            im = cv2.rectangle(im, (xc-10, yc-10), (xc + wc+10, yc + hc+10), 255, 2)
        cv2.imshow(" input image", img)
       
        cv2.imshow("Final Detections after applying NMS",im)
        cv2.waitKey(1)
            
        if len(contours) > 0 and len(pick) > 0:
            if (xA in range(xc-inc,xc+wc//2) and yA in range(yc-inc,yc+inc)) \
                    or (xB in range(xc+wc-wc//2,xc+wc+inc) and yB in range(yc+hc-inc,yc+hc+inc)) :
                print("Camouflage Instance Detected ")
                winsound.Beep(frequency, duration)
                print("Alert !!!!!")