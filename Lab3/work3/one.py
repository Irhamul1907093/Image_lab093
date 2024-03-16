import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("histogram.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("input",img)

img_ht=img.shape[0]
img_wt= img.shape[1]
size= img_ht*img_wt


plt.figure(1)
plt.title("input channel histogram")
plt.hist(img.ravel(),256,[0,255])
plt.show()

cmf=np.zeros((256),dtype=np.float32)
pmf=np.zeros((256),dtype=np.float32)
freq=np.zeros((256),dtype=np.int32)

out = np.zeros((img_ht,img_wt),np.uint8)
# getting frequency and PMF for grayscale image
for i in range(img_ht):
    for j in range(img_wt):
        intensity = img[i, j]
        freq[intensity] += 1

pmf = freq / size

# getting CMF for grayscale image
cmf[0] = pmf[0]
for i in range(1, 256):
    cmf[i] = cmf[i - 1] + pmf[i]

# Applying equalization
for i in range(img_ht):
    for j in range(img_wt):
        intensity = img[i, j]
        out[i, j] = np.round(255 * cmf[intensity]) #(l-1)*(sum of pdf)=cmf


out = out.astype(np.uint8)

# Display t
cv2.imshow('Equalized Image', out)
plt.figure(2)
plt.title("Equalized image histogram")
plt.hist(out.ravel(),256,[0,255])
plt.show()

# Plot PDF
plt.plot(pmf, color='red')
plt.title('Probability Density Function (PDF) of input')
plt.xlabel('Intensity')
plt.ylabel('Probability')
plt.show()

# Plot CMF of input image

plt.plot(cmf, color='blue')
plt.title('Cumulative Distribution Function (CDF) of input image')
plt.xlabel('Intensity')
plt.ylabel('Cumulative Probability')
plt.show()
#####-------so far input image------#

#again getting pmf,cmf for output image
cmf_out=np.zeros((256),dtype=np.float32)
pmf_out=np.zeros((256),dtype=np.float32)
freq_out=np.zeros((256),dtype=np.int32)

#getting frequency and PMF for equalized image
for i in range(img_ht):
    for j in range(img_wt):
        intensity = out[i, j]
        freq_out[intensity] += 1

pmf_out = freq_out / size

# getting CMF for equalized image
cmf_out[0] = pmf_out[0]
for i in range(1, 256):
    cmf_out[i] = cmf_out[i - 1] + pmf_out[i]


# Plot PDF
plt.plot(pmf_out, color='red')
plt.title('Probability Density Function (PDF) of equalized image')
plt.xlabel('Intensity')
plt.ylabel('Probability')
plt.show()

# Plot CMF of input image

plt.plot(cmf_out, color='blue')
plt.title('Cumulative Distribution Function (CDF) of equalized image')
plt.xlabel('Intensity')
plt.ylabel('Cumulative Probability')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

