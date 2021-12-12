import torchvision
from PIL import Image
from inference import image_haze_removel
import argparse

def single_dehaze_test(input):
    hazy_input_image = Image.open(input)
    dehaze_image = image_haze_removel(hazy_input_image)
    torchvision.utils.save_image(dehaze_image, "dehaze.jpg")

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    args = vars(ap.parse_args())
    
    single_dehaze_test(args["image"])

