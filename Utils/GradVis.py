import cv2 as cv
from PIL import Image  
import numpy
#https://www.cnblogs.com/FHC1994/p/9130184.html
def main():
    image = cv.imread('/data/fjsdata/CVTEDR/images/DR200721006.jpeg')
    
    #grad_x = cv.Sobel(image, cv.CV_32F, 1, 0)   #First order derivative of X 
    #grad_y = cv.Sobel(image, cv.CV_32F, 0, 1)   #First order derivative of Y
    grad_x = cv.Scharr(image, cv.CV_32F, 1, 0)   #First order derivative of X 
    grad_y = cv.Scharr(image, cv.CV_32F, 0, 1)   #First order derivative of Y
    gradx = cv.convertScaleAbs(grad_x)  #turn to unit8
    grady = cv.convertScaleAbs(grad_y)
    #cv.imwrite("/data/pycode/ChestXRay/Imgs/gradient_x.jpg", gradx) 
    #cv.imwrite("/data/pycode/ChestXRay/Imgs/gradient_y.jpg", grady) 
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0) #merge
    #cv.imwrite("/data/pycode/ChestXRay/Imgs/gradient.jpg", gradxy)

    gradxy_pil = Image.fromarray(cv.cvtColor(gradxy,cv.COLOR_BGR2RGB)) 
    gradxy_pil.save("/data/pycode/ChestXRay/Imgs/gradient_pil_0.jpg", quality=95)

    """
    dst = cv.Laplacian(image, cv.CV_32F) #Laplace Operator, second order derivative
    lpls = cv.convertScaleAbs(dst)
    cv.imwrite("/data/pycode/ChestXRay/Imgs/lpls.jpg", lpls)
    """


if __name__ == "__main__":
    #for debug   
    main()