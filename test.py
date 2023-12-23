# Import package & set environment
import sys, os
import numpy as np

# set according to your device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# set according to your path
sys.path.append("/home/chentao/BerDiff/codes")

# Create diffusion model
from scripts.script_util import create_model_and_diffusion
model, diffusion = create_model_and_diffusion(
    image_size=128,
    img_channels=1,
    num_channels=128,
    num_res_blocks=2,
    num_heads=1,
    num_heads_upsample=-1,
    attention_resolutions="16",
    dropout=0.0,
    diffusion_steps=1000,
    timestep_respacing="ddimuni10",
    noise_schedule="cosine2",
    ltype="mix",
    mean_type="epsilon",
    rescale_timesteps=False,
    use_checkpoint="",
    use_scale_shift_norm=False,
)

# Load model from checkpoint
from scripts.script_util import load_state_dict, dev

# set according to your path
model_path = "/data/chentao/diff/BerDiff_epsilon.pt"

model.load_state_dict(load_state_dict(model_path, map_location="cpu"))
model.to(dev())
model.eval()

# Define LIDC dataset class
import torch
import torch.nn
import numpy as np
import os
import os.path
from glob import glob
import matplotlib.image as img

class LIDCDatasetTest(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        self.database = []
        
        # set according to your path
        data_path = "/data/chentao/DATA/LIDC/test/"
        
        patient_dirs = glob(os.path.join(data_path, 'images', '*'))
        for i in range(len(patient_dirs)):
            img_paths = glob(os.path.join(patient_dirs[i], '*'))
            for j in range(len(img_paths)):
                datapoint = dict()
                img_path = img_paths[j]
                datapoint["image"] = img_path
                # get the corresponding ground truth labels
                gt_base_path = img_path.replace('images', 'gt')
                for l in range(4):
                    gt_path = gt_base_path.replace('.png', '_l{}.png'.format(l))
                    datapoint["gt_{}".format(l)] = gt_path
                self.database.append(datapoint) 

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        nib_img = img.imread(filedict["image"])
        out.append(torch.tensor(nib_img))
        
        for l in range(4):
            nib_img = img.imread(filedict["gt_{}".format(l)])
            out.append(torch.tensor(nib_img))
        out = torch.stack(out)
        
        image = out[:1, ...]
        label = out[1:, ...]
        image = image[..., 26:-26, 26:-26]      #crop to a size of (224, 224)
        label = label[..., 26:-26, 26:-26]
        label=torch.where(label > 0, 1, 0).float()  #merge all tumor classes into one
        return (image, label, filedict["image"])

    def __len__(self):
        return len(self.database)
    
ds = LIDCDatasetTest()
datal = torch.utils.data.DataLoader(ds, batch_size=90, shuffle=False)
data = iter(datal)

# Define sampling function
sample_fn = diffusion.ddim_sample_loop
def sample_n(image, n):
    example = []
    for i in range(n):
        sample = sample_fn(model, (90,1,128,128), model_kwargs={"img": image.cuda()}, progress=True)
        example.append(sample[:, 0].detach().cpu().numpy())
    return np.stack(example)

# Sampling
ds = LIDCDatasetTest()
datal = torch.utils.data.DataLoader(ds, batch_size=90, shuffle=False)
data = iter(datal)
n_data = len(data)
i = 0
samples = []
while i < n_data:
    print(i)
    image, label, _ = next(data)
    i += 1
    samples.append(sample_n(image, 16))
samples = np.concatenate(samples, axis=1)

# Define evalution metric
## GED2
def iou_distance(src, tgt):
    intersection = np.logical_and(src, tgt)
    union = np.logical_or(src, tgt)
    if union.sum() == 0:  # avoid divide by 0
        return 0.0
    return 1 - intersection.sum() / union.sum()

def get_ged2(label, sample):
    dss = 0
    for i in range(sample.shape[0]):
        for j in range(sample.shape[0]):
            dss += iou_distance(sample[i], sample[j])
    dss = dss / (sample.shape[0]*sample.shape[0])
    
    dll = 0
    for i in range(label.shape[0]):
        for j in range(label.shape[0]):
            dll += iou_distance(label[i], label[j])
    dll = dll / (label.shape[0]*label.shape[0])
    
    dls = 0
    for i in range(label.shape[0]):
        for j in range(sample.shape[0]):
            dls += iou_distance(label[i], sample[j])
    dls = dls / (label.shape[0] * sample.shape[0])
    
    return dls * 2 - dll - dss

## HM-IoU
from scipy.optimize import linear_sum_assignment

def get_hiou(label, segmentation):
    label = np.repeat(label,4,0)
    iou_d = np.zeros([16,16])
    for i in range(16):
        for j in range(16):
            iou_d[i, j] = iou_distance(label[i], segmentation[j])
    row_ind,col_ind=linear_sum_assignment(iou_d)
    hiou = (1-iou_d)[row_ind, col_ind].mean()
    return hiou

## Soft-Dice
def get_dice_threshold(output, mask, threshold):
    smooth = 1e-6

    zero = np.zeros(output.shape)
    one = np.ones(output.shape)
    output = np.where(output > threshold, one, zero)
    mask = np.where(mask > threshold, one, zero)
    output = output.reshape(-1)
    mask = mask.reshape(-1)
    intersection = (output * mask).sum()
    dice = (2. * intersection + smooth) / (output.sum() + mask.sum() + smooth)

    return dice

def get_soft_dice(output, mask):
    """
    :param outputs: H W
    :param masks: H W
    :return: average dice of B items
    """
    dice_item_thres_list = []
    for thres in [0.1, 0.3, 0.5, 0.7, 0.9]:
        dice_item_thres = get_dice_threshold(output, mask, thres)
        dice_item_thres_list.append(dice_item_thres)
    dice_item_thres_mean = np.mean(dice_item_thres_list)

    return dice_item_thres_mean

# Evalution
## GED2
ds = LIDCDatasetTest()
datal = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
data = iter(datal)
n_data = len(data)
i = 0
ged2 = []
while i < n_data:
    image, label, _ = next(data)
    ged2.append(get_ged2(label[0], samples[:, i]>0.5))
    i += 1
print(f"GED2: {np.mean(ged2)}")

## HM-IoU 
ds = LIDCDatasetTest()
datal = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
data = iter(datal)
n_data = len(data)
i = 0
hiou = []
while i < n_data and i <180:
    image, label,_ = next(data)
    hiou.append(get_hiou(label[0], samples[:, i]>0.5))
    i += 1
print(f"HM-IoU: {np.mean(hiou)}")

## Soft-Dice
ds = LIDCDatasetTest()
datal = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
data = iter(datal)
n_data = len(data)
i = 0
soft_dice = []
while i < n_data and i<180:
    image, mask, _ = next(data)
    mask_major_vote = (mask[0][0]+mask[0][1]+mask[0][2]+mask[0][3])/4.0
    soft_dice.append(get_soft_dice(output=samples[:, i].mean(axis=0), mask=mask_major_vote))
    i += 1
print(f"Soft-Dice: {np.mean(soft_dice)}")