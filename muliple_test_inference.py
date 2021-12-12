from inference import image_haze_removel
from PIL import Image
import torchvision
import os
import argparse

def multiple_dehaze_test(directory):

    print(directory)
    images = []
    for filename in os.listdir(directory):
        img = Image.open(os.path.join(directory,filename))
        if img is not None:
            images.append(img)
    

#data_folder = "E:/Light-DehazeNet Implementation/query hazy images/outdoor natural/"

    
    print(len(images))

    c=0
    for i in range(len(images)):
        img = images[i]
        dehaze_image = image_haze_removel(img)
        torchvision.utils.save_image(dehaze_image, "vis_results/dehaze_img("+str(c+1)+").jpg")
        c=c+1
    
if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-td", "--test_directory", required=True, help="path to test images directory")
    args = vars(ap.parse_args())
    
    multiple_dehaze_test(args["test_directory"])


