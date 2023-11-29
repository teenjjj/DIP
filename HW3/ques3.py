import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageChops

img = cv2.imread("DIP_image.tif", 0)

# step 1
# step1 = np.fft.fftshift(img)
N, M = np.shape(img)
i, j = np.meshgrid(np.arange(M), np.arange(N))
mult_factor = np.power(np.ones((N, M)) * -1, i + j)
step1 = img * mult_factor

plt.gray()
plt.subplot(321).set_title('step 1')
plt.axis('off')
plt.imshow(step1)


# step2

dft = cv2.dft(np.float32(step1), flags=cv2.DFT_COMPLEX_OUTPUT)
step2 = 20 * np.log(cv2.magnitude(dft[:, :, 0], dft[:, :, 1]))

plt.subplot(322).set_title('step2')
plt.axis('off')
plt.imshow(step2)

# step3
dft_con = dft.conj()
step3 = 20 * np.log(cv2.magnitude(dft_con[:, :, 0], dft_con[:, :, 1]))

plt.subplot(323).set_title('step3')
plt.axis('off')
plt.imshow(step3)

# step4

# f_ishift = np.fft.ifftshift(dft_con)
img_back = cv2.idft(dft_con)
step4 = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

plt.subplot(324).set_title('step4')
plt.axis('off')
plt.imshow(step4)

# step5
# step5 = np.abs(
#     (mult_factor * img_back[:, :, 0].real) + (1j * img_back[:, :, 1].imag))

f_ishift = np.fft.ifftshift(dft_con)
img_back = cv2.idft(f_ishift)
step5 = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

plt.subplot(325).set_title('step5')
plt.axis('off')
plt.imshow(step5)


plt.tight_layout()
plt.show()
