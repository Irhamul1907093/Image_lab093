import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_pdf(img):
    img_ht, img_wt = img.shape
    size = img_ht * img_wt
    freq = np.zeros(256, dtype=int)

    for i in range(img_ht):
        for j in range(img_wt):
            intensity = img[i, j]
            freq[intensity] += 1

    pmf = freq / size

    return pmf

def get_cdf(pmf):
    cmf = np.zeros_like(pmf)
    cmf[0] = pmf[0]

    for i in range(1, len(pmf)):
        cmf[i] = cmf[i - 1] + pmf[i]

    return cmf

def histogram_equalization(img, cmf):
    img_ht, img_wt = img.shape
    out = np.zeros_like(img)

    for i in range(img_ht):
        for j in range(img_wt):
            intensity = img[i, j]
            out[i, j] = np.round(255 * cmf[intensity])  #(l-1)*(sum of pdf)=cmf

    return out.astype(np.uint8)

def gauss_dist(miu,sigma):
    
    g = np.zeros(256,dtype=np.float32)
    variance = sigma*sigma
    constant = 1/(np.sqrt(2*3.1416*variance))
    
    for i in range (256):
        g[i]=np.exp(-((i-miu)**2)/(2*variance))*constant
    
    return g

def get_pdf_from_hist(freq,size):
    pdf = np.zeros(256,dtype=np.float32) 
    pdf=freq/size
    return pdf

def get_cdf_from_hist(pdf):
    cdf = np.zeros(256,dtype=np.float32) 
    cdf[0]=pdf[0]
    for i in range (1,256):
        cdf[i]=cdf[i-1]+pdf[i]
    return cdf

def histogram_matching(input_img, target_hist):
    input_pmf = get_pdf(input_img)
    input_cdf = get_cdf(input_pmf)

    size=sum(target_hist)
    target_pdf = get_pdf_from_hist(target_hist,size)
    target_cdf = get_cdf(target_pdf)

    output_img = np.zeros_like(input_img)

    for i in range(256):
        # Find the intensity value in the target histogram that matches the current intensity value in the input image
        match_intensity = np.argmin(np.abs(target_cdf - input_cdf[i]))

        # Assign the matched intensity value to the output image
        output_img[input_img == i] = match_intensity

    return output_img

img = cv2.imread("histogram.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("input",img)

img_ht=img.shape[0]
img_wt= img.shape[1]
size= img_ht*img_wt

plt.figure(1)
plt.title("input image histogram")
plt.hist(img.ravel(),256,[0,255])
plt.show()
#initialize
cdf=np.zeros((256),dtype=np.float32)
pdf=np.zeros((256),dtype=np.float32)

# ---------input image---------
# Plot PDF 
pdf=get_pdf(img)
plt.plot(pdf, color='red')
plt.title('Probability Density Function (PDF) of input')
plt.xlabel('Intensity')
plt.ylabel('Probability')
plt.show()
# Plot CMF
cdf=get_cdf(pdf)
plt.plot(cdf, color='blue')
plt.title('Cumulative Distribution Function (CDF) of input image')
plt.xlabel('Intensity')
plt.ylabel('Cumulative Probability')
plt.show()

###target histogram
'''u1=30,sigma1=8
u2=165,sigma2=20'''

gauss1=gauss_dist(30, 8)
gauss2= gauss_dist(165, 20)
target_hist = gauss1+gauss2

plt.figure(2)
plt.title("gaussian function")
plt.plot(target_hist, color='blue')
plt.fill_between(np.arange(len(target_hist)), target_hist, color='skyblue', alpha=1.0)
plt.xlabel('Intensity')
plt.ylabel('gaussian')
plt.show()

##  --------------    now doing histogram matching -----------

final_output_img = np.zeros_like(img)
final_output_img=histogram_matching(img,target_hist)

cv2.imshow("after matching",final_output_img)
plt.figure(3)
plt.title("Output histogram(after matching)")
plt.hist(final_output_img.ravel(),256,[0,255])
plt.show()

#initialize
cdf_final=np.zeros((256),dtype=np.float32)
pdf_final=np.zeros((256),dtype=np.float32)

# Plot PDF  of final
pdf_final=get_pdf(final_output_img)
plt.plot(pdf_final, color='red')
plt.title('Probability Density Function (PDF) of image after histogram matching')
plt.xlabel('Intensity')
plt.ylabel('Probability')
plt.show()
# Plot Cdf of final
cdf_final=get_cdf(pdf_final)
plt.plot(cdf_final, color='blue')
plt.title('Cumulative Distribution Function (CDF) of image after histogram matching')
plt.xlabel('Intensity')
plt.ylabel('Cumulative Probability')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

