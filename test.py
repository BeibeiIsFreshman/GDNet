import torch
import sys

sys.path.append('./models')
import os
import cv2
from torch.utils.data import DataLoader
from util.PL_dataset import PL_dataset

# from Final.Teacher3 import Teacher
# from model_others.BBS.BBSNet_model import BBSNet



# from Final.Teacher_nofusion import Teacher
# from Distill_shunt.normal import Teacher
# from Distill_shunt.Mine0331_shuntBase import Teacher
# from Distill_shunt.Teacher_bb import Teacher
# path = "/home/xug/PycharmProjects/TLD/model_others/HRTransNet/hrt_base.yaml"
# config = yaml.load(open(path, "r"),yaml.SafeLoader)['MODEL']['HRT']
# from Distill_shunt.normal import Teacher
# from model_others.TBINet.model import Net
# from model_others.MobileSal.model import MobileSal
# from Final.Teacher import Teacher
# from model_others.BBS.BBSNet_model import BBSNet
# from model_others.BTSNet.BTSNet import BTSNet
# from model_others.EGANet.network import Segment
# from model_others.MobileSal.model import MobileSal
# from model_others.LSNet.LSNet import UNet
# from model_others.CIRNet.CIRNet_Res50 import CIRNet_R50
# from model_others.RGBTScribble.pvtmodel import PvtNet
# from Final.Student2 import Student
# from model_others.TBINet.model import Net
# from Contrastive.student_segformerB0 import SNet
# from Contrastive.Teacher_P2TLarge import Teacher
from Contrastive.Student_P2T import Student
model = Student()
from config import opt
# set device for test
# os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU:', opt.gpu_id)
device = torch.device('cuda:0')
def get_files_list(raw_dir):
    files_list = []
    for filepath, dirnames, filenames in os.walk(raw_dir):
        for filename in filenames:
            files_list.append(filepath+'/'+filename)
    return files_list

test_dataset_path = opt.test_path
image_root = get_files_list(test_dataset_path + '/vl')
ti_root = get_files_list(test_dataset_path + '/ir')
gt_root = get_files_list(test_dataset_path + '/gt')
test_dataset = PL_dataset(image_root, ti_root, gt_root, is_train=False)

test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=opt.test_batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )


print('test_dataset', len(test_dataset))
print('test_loader_size', len(test_loader))
# load the model
# model = UTA(cfg="train")
# Large epoch size may not generalize well. You can choose a good model to load according to the log file and pth files saved in ('./BBSNet_cpts/') when training.
# model.load_state_dict(torch.load('/media/wby/F426D6F026D6B2BA/best_epoch.pth'))  ##/media/maps/shuju/osrn999et/

model.load_state_dict(torch.load('/media/xug/shuju/datasets/Two202394/Student_P2T2/best_138_epoch.pth', map_location='cuda:0'))   #163nei 100  153

model.to(device)
model.eval()
test_datasets = ['1']  # 'VT5000'
# test
test_mae = []
for dataset in test_datasets:
    mae_sum = 0
    save_path = '/media/xug/shuju/datasets/Two202394/Student_P2T2/result' + '/'
    # print('这个地方为什么进不去')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # print('test_loader', len(test_loader))
    # print('test_loader', test_loader)
    # for ti, (input_vl, input_ir, labels) in enumerate(test_loader):
    for n_iter, batch_data in enumerate(test_loader):
        with torch.no_grad():
            image, ti, labels, name = batch_data
            image = image.to(device)
            # print(image.filename)
            ti = ti.to(device)
            labels = labels.to(device)
            # print('labels', labels.shape)
            res = model(image)[0]
            # res = model(image, ti)[0]
            # print('res', res.shape)
            name = str(name).replace('\'', "").replace('[','').replace(']','')
            predict = torch.sigmoid(res)
            predict = (predict - predict.min()) / (predict.max() - predict.min() + 1e-8)
            mae = torch.sum(torch.abs(predict - labels)) * 1.0 / torch.numel(labels)
            # mae_sum += torch.sum(torch.abs(predict - labels)) * 1.0 / torch.numel(labels)
            mae_sum = mae.item() + mae_sum
        predict = predict.data.cpu().numpy().squeeze()
        # print(predict.shape)
        print('save img to: ', save_path + name)
        cv2.imwrite(save_path + name, predict * 256)
    test_mae.append(mae_sum / len(test_loader))
print('Test Done!', 'MAE{}'.format(test_mae))