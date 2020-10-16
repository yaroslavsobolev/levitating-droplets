import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageDraw, ImageFont

fps = 2102
nframes = 1200

plt.style.use('dark_background')
for i in range(nframes):
    numpy_image = np.zeros(shape=(70, 200, 3), dtype=np.uint8)
    img = Image.fromarray(numpy_image.astype('uint8'), 'RGB')
    # img = img.rotate(90)
    d = ImageDraw.Draw(img)
    w, h = img.size
    fontsize = 40
    font = ImageFont.truetype("arial.ttf", fontsize)
    t = i/fps*1000
    d.text((10, 10), "{0:05.1f} ms".format(t), fill=(255, 255, 255), font=font)
    img.save('timecode/{0:07d}.png'.format(i))