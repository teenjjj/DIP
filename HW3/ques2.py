import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lenna.tif", 0)  # gray scale image

f = np.fft.fft2(img)  # 2-dimensional discrete Fourier Transform
# zero-frequency component to the center of the spectrum.
fshift = np.fft.fftshift(f)
fimg = 20*np.log(np.abs(fshift))  # spectrum image


# mag_spectrum = 20*np.log(np.abs(fshift))
mag_img = np.fft.ifft2(fimg)
mag_img = np.abs(mag_img)


phase_spectrum = np.angle(fshift)
phase_img = np.fft.ifft2(phase_spectrum)
phase_img = np.abs(phase_img)

plt.gray()
plt.subplot(221).set_title('origin')
plt.axis('off')
plt.imshow(img)

plt.subplot(222).set_title('2D-FFT')
plt.axis('off')
plt.imshow(fimg)

plt.subplot(223).set_title('magnitude-only image')
plt.axis('off')
plt.imshow(mag_img)

plt.subplot(224).set_title('phase-only image')
plt.axis('off')
plt.imshow(phase_img)


plt.show()
