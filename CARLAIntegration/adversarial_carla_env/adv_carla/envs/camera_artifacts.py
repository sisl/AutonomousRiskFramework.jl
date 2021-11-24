import cv2
import numpy as np
from numpy.core.numeric import False_
import json
import torch
from torchvision.utils import save_image

def read_camera_config(filepath,width,height,depth,config_file_name="camera_config.json"):
    with open(config_file_name,"r") as f:
        x = json.load(f)
    img = np.zeros((width,height,depth))
    ppm_dead_pixels = x['ppm_dead']
    ppm_red = x['ppm_red']
    ppm_green = x['ppm_green']
    ppm_blue = x['ppm_blue']
    exposure_compensation = x['exp_comp']
    dynamic_noise_std = x['dynamic_noise']
    lensflare = x['lensflare']
    vignette = x['vignette']
    blurred_corners = x['blurred_corners']
    chromatic_abberation = x['chromatic_abberation']

    return SimulatedCamera(img,ppm_dead_pixels=ppm_dead_pixels,ppm_red=ppm_red,ppm_green=ppm_green,ppm_blue=ppm_blue,dynamic_noise_std=dynamic_noise_std,exposure_compensation=exposure_compensation,vignette=vignette,blurred_corners=blurred_corners,lensflare=lensflare,chromatic_abberation=chromatic_abberation)

class SimulatedCamera():
    def __init__(self,img,ppm_dead_pixels=0,ppm_red=0,ppm_green=0,ppm_blue=0,exposure_compensation=0,dynamic_noise_std=(0,0,0),lensflare=False,vignette=False,blurred_corners=False,chromatic_abberation=False):
        self.static_noise = self.create_static_noise(img,ppm_dead_pixels,ppm_red,ppm_green,ppm_blue,exposure_compensation)
        self.dynamic_noise_std = dynamic_noise_std
        self.lensflare = lensflare
        self.vignette = vignette
        self.blurred_corners = blurred_corners
        self.chromatic_abberation = chromatic_abberation
    
    def simulate(self,img):
        img = img/255
        img = self.camera_artifacts(img,self.static_noise,self.dynamic_noise_std,self.lensflare,self.vignette,self.blurred_corners,self.chromatic_abberation)
        img = (img*255).astype('uint8')
        return img

    def camera_artifacts(self,img,static_noise=False,dynamic_noise_std=(0,0,0),lenseflare=False,vignette=False,blurred_corners=False,chromatic_abberation=False):
        '''
        input: This function expects as an input an image in the form of an np array of size HxWxC with C in the order BGR.
        output: The output is an np array of the same size as the input where the original image has been modified according to the options
        static_noise: Bias of image, requires np array of same size as image, positive values are added to the pixel values, while negtive values are subtracted
        dynamic_noise_var: Determines the variance of the dymic (changing for frame to frame) for the image. This simulates high ISO settings for cameras. To turn this effect off, set to 0. The tuple is for BGR noise.
        lensflare: If set to true, a random sun angle will be used for the simulation, however the angle can also be specified by using a number instead of "True".
        vignette: An effect seen in lenses that the picture gets darker twoards the edges, especially towards the corners. False disables, the effect, a number in the range between 0 and 1 specifies the strength of the effect.
        blurred_corners: Mimics the effects of lenses that have weak performance in the corners of the image, meaning that the image there gets blurred. A tuple is required as input where the first entry is the strength of the size of the blurred are (ranging from 0 to 1) and how blurred the image is going be (expressed in pixels, must be an odd number).
        chromatic_abberation: Simulates the effect of refraction of light based on their wave length. Light with short waves (blue colors) is refracted in a stronger way then light with long waves. The input is a tuple in order BGR that contains a scaling coefficient for the respective color channels.
        '''
        def center_crop(img, dim):
            """Returns center cropped image
            Args:
            img: image to be center cropped
            dim: dimensions (width, height) to be cropped
            """
            width, height = img.shape[1], img.shape[0]

            # process crop width and height for max available dimension
            crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
            crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
            mid_x, mid_y = int(width/2), int(height/2)
            cw2, ch2 = int(crop_width/2), int(crop_height/2) 
            crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
            
            return crop_img

        H,W,_ = img.shape

        # lense flare
        if lenseflare is not False:

            lensflare_base = np.zeros_like(img)

            if lenseflare==True:
                #randomly generate lensflare angle 
                lensflareangle = 180*np.random.rand()-90
            else:
                lensflareangle = lenseflare

            lensflareangle = lensflareangle*np.pi/180

            # iteratively create the lensflare
            no_flares = np.random.randint(10,25)
            for i in range(no_flares):
                single_flare = np.zeros_like(img)
                
                a = 1.5*np.random.rand()-0.75
                x_ctr = int((W/2)+np.sin(lensflareangle)*a*W)
                y_ctr = int((H/2)+np.cos(lensflareangle)*a*W)
                center_point = (x_ctr,y_ctr)
                radius = int(np.random.rand()*W*0.1) #maximum size of flares is 20% of the image width
                single_flare = cv2.circle(single_flare, center_point, radius, (1,1,1), -1)
                alpha = 0.3*np.random.rand()
                lensflare_base += alpha*single_flare
            
            img = img + lensflare_base
        
        # blurred corners
        if blurred_corners != False:
            blurred_img = np.copy(img)
            blurred_img = cv2.GaussianBlur(blurred_img,(blurred_corners[1],blurred_corners[1]),0)
            
            gaussian_std_lower_limit = W*80/400
            gaussian_std_spread = 2*W-gaussian_std_lower_limit
            gaussian_std = gaussian_std_lower_limit + gaussian_std_spread* blurred_corners[0]   #linear transformation for gaussian kernel std
            kernel_x = cv2.getGaussianKernel(W,gaussian_std)
            kernel_y = cv2.getGaussianKernel(H,gaussian_std)
            kernel = kernel_y * kernel_x.T
            mask = kernel / np.linalg.norm(np.max(kernel))
            mask = np.reshape(mask,(H,W,1))
            mask = np.tile(mask,(1,1,3))

            img = mask*img + (1-mask)*blurred_img

        # vignette
        # generating vignette mask using Gaussian kernels
        if vignette != False:
            gaussian_std_lower_limit = W*80/400
            gaussian_std_spread = 2*W-gaussian_std_lower_limit
            gaussian_std = gaussian_std_lower_limit + gaussian_std_spread* vignette   #linear transformation for gaussian kernel std
            kernel_x = cv2.getGaussianKernel(W,gaussian_std)
            kernel_y = cv2.getGaussianKernel(H,gaussian_std)
            kernel = kernel_y * kernel_x.T
            mask = kernel / np.linalg.norm(np.max(kernel))
            # print(np.min(mask),np.max(mask))
            # applying the mask to each channel in the input image
            for i in range(3):
                img[:,:,i] = img[:,:,i] * mask

        # chromatic abberation
        if np.any(chromatic_abberation != False): 
            scale_blue = chromatic_abberation[0]
            scale_green = chromatic_abberation[1]
            scale_red = chromatic_abberation[2]
            
            img[:,:,0] = center_crop(cv2.resize(img[:,:,0],(int(scale_blue*W),int(scale_blue*H))),(W,H))
            img[:,:,1] = center_crop(cv2.resize(img[:,:,1],(int(scale_green*W),int(scale_green*H))),(W,H))
            img[:,:,2] = center_crop(cv2.resize(img[:,:,2],(int(scale_red*W),int(scale_red*H))),(W,H))


        # dynamic noise
        if np.max(dynamic_noise_std)>0:
            dyn_noise = np.zeros_like(img)
            for i in range(3):
                dyn_noise[:,:,i] = dynamic_noise_std[i] * np.random.randn(H,W)
            
            img += dyn_noise

        # static noise
        if np.any(static_noise != False):
            img += static_noise

        #cap intensities between 0 and 1
        img[img>1] = 1
        img[img<0] = 0

        return img

    def create_static_noise(self,img,ppm_dead_pixels=0,ppm_red=0,ppm_green=0,ppm_blue=0,exposure_compensation=0):    
        '''
        output: mask for static noise
        inputs:
        img: original image
        ppm_dead_pixels: parts per million of completly black pixels
        ppm_red: parts per million of dead pixels that show up red in the image
        ppm_green: parts per million of dead pixels that show up green in the image
        ppm_blue: parts per million of dead pixels that show up blue in the image
        exposure_compensation: -1 all black and 1 all white
        '''
        H,W,_ = img.shape
        no_pixels = H*W

        #exposure compensation
        mask = exposure_compensation * np.ones_like(img)

        #dead pixels
        no_dead = int(ppm_dead_pixels/1000000*no_pixels)
        for i in range(no_dead):
            x = np.random.randint(0,W)
            y = np.random.randint(0,H)
            mask[y,x,:] = [-1,-1,-1]
        
        #red pixels
        no_red = int(ppm_red/1000000*no_pixels)
        for i in range(no_red):
            x = np.random.randint(0,W)
            y = np.random.randint(0,H)
            mask[y,x,:] = [0,0,1]

        #green pixels
        no_green = int(ppm_green/1000000*no_pixels)
        for i in range(no_green):
            x = np.random.randint(0,W)
            y = np.random.randint(0,H)
            mask[y,x,:] = [0,1,0]

        #blue pixels
        no_blue = int(ppm_blue/1000000*no_pixels)
        for i in range(no_blue):
            x = np.random.randint(0,W)
            y = np.random.randint(0,H)
            mask[y,x,:] = [1,0,0]
        return mask


def save_camera_image(img, filename="tmp_img.png"):
    save_image((torch.tensor(img).float().permute(2,0,1)/255)[[2,1,0],:,:],filename)


if __name__=="__main__":
    img = cv2.imread("carla_onboard_camera.png")/255
    # img = cv2.imread("Screenshot from 2021-08-09 22-37-06.png")/255
    cam = SimulatedCamera(img,ppm_dead_pixels=100,ppm_red=300,ppm_green=50,ppm_blue=100,exposure_compensation=0.1,dynamic_noise_std=(0.05,0.05,0.05),lensflare=True,vignette=0.2,blurred_corners=(0.5,11),chromatic_abberation=(1,1.01,1.02))
    output = cam.simulate(img)


    # mask = create_static_noise(img,100,100,100,100,0.1)
    # img = camera_artifacts(img,static_noise=mask,dynamic_noise_std=(0.06,0.06,0.06),lenseflare=True,vignette=0.2,blurred_corners=(0.5,11),chromatic_abberation=(1,1.01,1.02))
    cv2.imwrite("all.jpg",output*255)
    cv2.imshow("test",output)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 