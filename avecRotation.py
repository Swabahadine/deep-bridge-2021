import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import sys
import glob
from scipy import ndimage
import math
import tkinter 
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

np.set_printoptions(threshold=sys.maxsize)

ROTATION_X = 332

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

def rotateX(originX, originY, x, y, angleD):
    angle = angleD * math.pi / 180
    if (originY <= y):
        cos = math.cos(angle) * (y - originY)
        sin = math.sin(angle) * (y - originY)
        return cos + originY, sin + originX
    else:
        cos = math.cos(angle + math.pi) * (originY - y)
        sin = math.sin(angle + math.pi) * (originY - y)
        return cos + originY, sin + originX

# fill 3D array with the images from the files
def img3DWithRotation(angle, depth):
    depthY = ROTATION_X
    imgFinal = []
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        rowArray = []
        for j in range(len(img2d)):
            cos, sin = rotateX(depth, depthY, depth, j, angle)
            if (math.ceil(sin) >= 512 or math.ceil(cos) >=512):
                rowArray.append(0)
            else:
                rowArray.append(img2d[math.ceil(sin)][math.ceil(cos)])
        imgFinal.append(rowArray)
    return imgFinal

def get_pixels_hu(image):
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
def coronalHuDepth(angle, depth):
    imgFinal = img3DWithRotation(angle, depth)
    coronal = np.array(imgFinal) #img3d[depth, :, :].T
    coronal_hu = get_pixels_hu(coronal)
    return coronal_hu[::-1]



# TKINTER 
root = tkinter.Tk()
root.title('Deep Bridge')



## PLOT
lines = []

fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
canvas.draw()
canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)


plt.subplots_adjust(left=0.1, right=0.9, top=0.7, bottom=0.1)


## INIT PLOT

coronal_hu = coronalHuDepth(0, img_shape[0]//2)
ax.imshow(coronal_hu, cmap=plt.cm.gray, vmin=0, vmax=1000)
lines = ax.plot([ROTATION_X,ROTATION_X], [0,570], "c")


axcolor = 'lightgoldenrodyellow'


axCenter = plt.axes([0.25, 0.75, 0.65, 0.03], facecolor=axcolor)
axMin = plt.axes([0.25, 0.8, 0.65, 0.03], facecolor=axcolor)
axMax = plt.axes([0.25, 0.85, 0.65, 0.03], facecolor=axcolor)
axAngle = plt.axes([0.25, 0.9, 0.65, 0.03], facecolor=axcolor)
axDepth = plt.axes([0.25, 0.95, 0.65, 0.03], facecolor=axcolor)


sCenter = Slider(axCenter, 'Center', 0, 512, valinit=ROTATION_X, valstep=1)
sMin = Slider(axMin, 'Min', -1000, 1000, valinit=0, valstep=10)
sMax = Slider(axMax, 'Max', -1000, 1000, valinit=1000, valstep=10)
sAngle = Slider(axAngle, 'Angle', 0, 360, valinit=0, valstep=1)
sDepth = Slider(axDepth, 'Depth', 150, 350, valinit=img_shape[0]//2, valstep=1)

def update(val):
    global lines
    angleV = sAngle.val
    depthV = sDepth.val
    minV = sMin.val
    maxV = sMax.val
    ROTATION_X = sCenter.val
    lines.pop(0).remove()
    ax.imshow(coronalHuDepth(angleV, depthV), cmap=plt.cm.gray, vmin=minV, vmax=maxV)
    lines = ax.plot([ROTATION_X,ROTATION_X], [0,570], "c")




sCenter.on_changed(update)
sAngle.on_changed(update)
sDepth.on_changed(update)
sMin.on_changed(update)
sMax.on_changed(update)




# ADD TOOLBAR TO TKINTER
toolbar = NavigationToolbar2Tk(canvas, root)
toolbar.update()
canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)


def on_key_press(event):
    print("you pressed {}".format(event.key))
    key_press_handler(event, canvas, toolbar)


canvas.mpl_connect("key_press_event", on_key_press)



bottomframe = tkinter.Frame(root)
bottomframe.pack( side = tkinter.BOTTOM )
e1 = tkinter.Entry()
e1.pack(side = tkinter.BOTTOM)


def save_rotation_center():
    global ROTATION_X
    ROTATION_X = int(e1.get())
    print("CHANGED : ", ROTATION_X)

saveButton = tkinter.Button(bottomframe, command = save_rotation_center, text="Save", fg="black")
saveButton.pack( side = tkinter.BOTTOM)

tkinter.mainloop()

#https://stackoverflow.com/questions/58791377/medical-image-quality-problem-with-dicom-files
