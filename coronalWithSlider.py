import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import sys
import glob
from scipy import ndimage

np.set_printoptions(threshold=sys.maxsize)


# load the DICOM files
files = []

dossier = "./patients/dossier_1/*"

print('glob: {}'.format(dossier))
for fname in glob.glob(dossier, recursive=False):
    files.append(pydicom.dcmread(fname, force=True))


print("file count: {}".format(len(files)))

# skip files with no SliceLocation (eg scout views)
slices = []
skipcount = 0
for f in files:
    if hasattr(f, 'SliceLocation'):
        slices.append(f)
    else:
        skipcount = skipcount + 1

print("skipped, no SliceLocation: {}".format(skipcount))


# ensure they are in the correct order
slices = sorted(slices, key=lambda s: s.SliceLocation)

# create 3D array
img_shape = list(slices[0].pixel_array.shape)
img_shape.append(len(slices))

img3d = np.zeros(img_shape)

# fill 3D array with the images from the files
def img3DWithRotation(angle):
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        img3d[:, :, i] = ndimage.rotate(img2d, angle, reshape=False)


def get_pixels_hu(image):
    #image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

# plot 3 orthogonal slices
def coronalHuDepth(depth):
    coronal = img3d[depth, :, :].T
    coronal_hu = get_pixels_hu(coronal)
    return coronal_hu[::-1]

img3DWithRotation(333)
coronal_hu = coronalHuDepth(img_shape[0]//2)

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, right=0.9, top=0.83, bottom=0.1)


axcolor = 'lightgoldenrodyellow'
axAngle = plt.axes([0.25, 0.85, 0.65, 0.03], facecolor=axcolor)
axDepth = plt.axes([0.25, 0.95, 0.65, 0.03], facecolor=axcolor)

sAngle = Slider(axAngle, 'Angle', 0, 360, valinit=333, valstep=1)
sDepth = Slider(axDepth, 'Depth', 230, 280, valinit=img_shape[0]//2, valstep=1)

def updateAngle(val):
    angleV = sAngle.val
    angleV = sAngle.val
    depthV = sDepth.val

    img3DWithRotation(angleV)
    ax.imshow(coronalHuDepth(depthV), cmap=plt.cm.gray, vmin=-120, vmax=1000)

def update(val):
    depthV = sDepth.val
    ax.imshow(coronalHuDepth(depthV), cmap=plt.cm.gray, vmin=-120, vmax=1000)

sAngle.on_changed(updateAngle)
sDepth.on_changed(update)

ax.imshow(coronal_hu, cmap=plt.cm.gray, vmin=-120, vmax=1000)
plt.show()


#https://stackoverflow.com/questions/58791377/medical-image-quality-problem-with-dicom-files