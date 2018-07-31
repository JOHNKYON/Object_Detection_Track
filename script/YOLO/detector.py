from __future__ import division

import time
from script.YOLO.util import *
import argparse
import os
from script.YOLO.darknet import Darknet
import pickle as pkl
import pandas as pd
import random
import csv


def arg_parse():
    """
    Parse arguements to the detect module

    :return:
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest='images', help=
                        "Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)
    parser.add_argument("--out", dest='out', help=
                        "Image / Directory to store detections to",
                        default="out", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help=
                        "Config file", default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help=
                        "weightsfile", default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)

    return parser.parse_args()


args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
cuda = torch.cuda.is_available()

num_classes = 80    #For COCO
classes = load_classes("data/coco.names")

# Prepare the outputfile
file = open("out/submission.txt", 'w')
file.write("ImageId,PredictionString")

# Prepare label
name_label = {}
with open("class-descriptions-boxable.csv") as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        name_label[row[1].lower()] = row[0]



# Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

# If there's a GPU availible, put the model on GPU
if cuda:
    model.cuda()

# Set the model in evaluation mode
model.eval()


read_dir = time.time()
# Detection phase
print("File loading begins.")
try:
    imlist = [os.path.join(os.path.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = []
    imlist.append(os.path.join(os.path.realpath('.'), images))
except FileNotFoundError:
    print("No file or directory with the name {}".format(images))
    exit()
print("File loading finished.")

if not os.path.exists(args.out):
    os.makedirs(args.out)

load_batch = time.time()
loaded_ims = [cv2.imread(x) for x in imlist]

# pyTorch Variables for images
im_batches = list(map(pre_image, loaded_ims, [inp_dim for x in range(len(imlist))]))

# List containing dimensions of original images
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

if cuda:
    im_dim_list = im_dim_list.cuda()

letfover = 0
if len(im_dim_list) % batch_size:
    leftover = 1

if batch_size != 1:
    num_batches = len(imlist) // batch_size + leftover
    im_dim_list = [torch.cat((im_batches[i*batch_size : min((i+1)*batch_size,
                                                            len(im_batches))])) for i in range(num_batches)]

write = 0

print("Start detection loop")

start_det_loop = time.time()
for i, batch in enumerate(im_batches):
    # Load the image
    start = time.time()
    if cuda:
        batch = batch.cuda()
    with torch.no_grad():
        prediction = model(Variable(batch), cuda)

    prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thesh)

    end = time.time()

    if type(prediction) == int:

        for im_num, image in enumerate(imlist[i*batch_size: min((i+1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            print("{0:20s} predictied in {1:6.3f} seconds".format(image.split("/")[-1], (end-start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
        continue

    prediction[:,0] += i*batch_size     # Transform the attribute from index in batch to index in imlist

    if not write:
        output = prediction
        write = 1
    else:
        output = torch.cat((output, prediction))

    for im_num, image in enumerate(imlist[i*batch_size: min((i+1)*batch_size, len(imlist))]):
        im_id = i*batch_size + im_num
        obj = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(obj)))
        print("----------------------------------------------------------")

    if cuda:
        torch.cuda.synchronize()

try:
    output
except NameError:
    print("No detection were made")
    exit()

im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

scaling_factor = torch.min(inp_dim/im_dim_list, 1)[0].view(-1,1)

output[:, [1, 3]] -= (inp_dim - scaling_factor*im_dim_list[:, 0].view(-1, 1))/2
output[:, [2, 4]] -= (inp_dim - scaling_factor*im_dim_list[:, 1].view(-1, 1))/2

output[:, 1:5] /= scaling_factor

for i in range(output.shape[0]):
    output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
    output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])

output_recast = time.time()
class_load = time.time()
colors = pkl.load(open("pallete", "rb"))

draw = time.time()


def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,225,225], 1)

    # Output prediction String to output file
    if label in name_label.keys():
        file.write("{label} {confidence} {x1} {x2} {y1} {y2}".format(label=name_label[label], confidence=x[5],
                                                                     x1=c1[0], x2=c1[1], y1=c2[0], y2=c2[1]))
    return img


temp = -1
for x in output:
    if temp != int(x[0]):
        temp = int(x[0])
        file.write("\n"+imlist[temp].split("/")[-1].split(".")[0] + ",")
    write(x, loaded_ims)

# list(map(lambda x: write(x, loaded_ims), output))
det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.out, x.split("/")[-1]))

list(map(cv2.imwrite, det_names, loaded_ims))
end = time.time()

print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) + " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
print("----------------------------------------------------------")

torch.cuda.empty_cache()


