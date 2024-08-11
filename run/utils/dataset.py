import json
import os
from os import path as osp
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils import data
from torchvision import transforms

class VITONDataset(data.Dataset):
    def __init__(self, args,type):
        super(VITONDataset, self).__init__()
        self.img_height = args.img_height
        self.img_width = args.img_width
        self.semantic_nc = args.semantic_nc
        self.data_path = osp.join(args.dataset_dir,type)
        self.transform = transforms.Compose([transforms.ToTensor(),
        ])
        
        img_names = [] # model image
        garm_names = []  # cloth image
        
        with open(osp.join(args.dataset_dir,f"{type}_pairs.txt"), 'r') as f:
            for line in f.readlines():
                img_name, garm_name = line.strip().split()
                img_names.append(img_name)
                garm_names.append(garm_name)

        self.img_names = img_names    
        self.garm_names = dict()
        
        # paired model img and cloth img have the same name
        self.garm_names['paired'] = img_names

    def get_img_agnostic(self, img, parse, pose_data):
        parse_array = np.array(parse)
        parse_head = ((parse_array == 4).astype(np.float32) +
                      (parse_array == 13).astype(np.float32))
        parse_lower = ((parse_array == 9).astype(np.float32) +
                       (parse_array == 12).astype(np.float32) +
                       (parse_array == 16).astype(np.float32) +
                       (parse_array == 17).astype(np.float32) +
                       (parse_array == 18).astype(np.float32) +
                       (parse_array == 19).astype(np.float32))

        
        agnostic = img.copy()
        agnostic_draw = ImageDraw.Draw(agnostic)

        length_a = np.linalg.norm(pose_data[5] - pose_data[2])
        length_b = np.linalg.norm(pose_data[12] - pose_data[9])
        point = (pose_data[9] + pose_data[12]) / 2
        pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
        pose_data[12] = point + (pose_data[12] - point) / length_b * length_a
        
        r = int(length_a / 16) + 1

        # mask torso
        for i in [9, 12]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r * 3, pointy - r * 6, pointx + r * 3, pointy + r * 6), "gray", "gray")
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], "gray", width=r * 6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], "gray", width=r * 6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], "gray", width=r * 12)
        agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], "gray", "gray")

        # mask neck
        pointx, pointy = pose_data[1]
        agnostic_draw.rectangle((pointx - r * 5, pointy - r * 9, pointx + r * 5, pointy), "gray", "gray")

        # mask arms
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], "gray", width=r * 12)
        for i in [2, 5]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r * 5, pointy - r * 6, pointx + r * 5, pointy + r * 6), "gray", "gray")
        for i in [3, 4, 6, 7]:
            if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], "gray", width=r * 10)
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r * 5, pointy - r * 5, pointx + r * 5, pointy + r * 5), "gray", "gray")

        for parse_id, pose_ids in [(14, [5, 6, 7]), (15, [2, 3, 4])]:
            # mask_arm = Image.new('L', (self.fine_width, self.fine_height), 'white')
            mask_arm = Image.new("L", (768, 1024), "white")
            mask_arm_draw = ImageDraw.Draw(mask_arm)
            pointx, pointy = pose_data[pose_ids[0]]
            mask_arm_draw.ellipse((pointx - r * 5, pointy - r * 6, pointx + r * 5, pointy + r * 6), "black", "black")
            for i in pose_ids[1:]:
                if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                    continue
                mask_arm_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], "black", width=r * 10)
                pointx, pointy = pose_data[i]
                if i != pose_ids[-1]:
                    mask_arm_draw.ellipse((pointx - r * 5, pointy - r * 5, pointx + r * 5, pointy + r * 5), "black", "black")
            mask_arm_draw.ellipse((pointx - r * 4, pointy - r * 4, pointx + r * 4, pointy + r * 4), "black", "black")

            parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
            agnostic.paste(img, None, Image.fromarray(np.uint8(parse_arm * 255), "L"))

        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), "L"))
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), "L"))
        return agnostic

    def __getitem__(self, index):
        img_name = self.img_names[index]
        garm_name = {}
        garm = {}
        garm_mask = {}
        
        # load model garm garm_mask image
        for key in self.garm_names:
            garm_name[key] = self.garm_names[key][index]
            garm[key] = Image.open(osp.join(self.data_path, 'cloth', garm_name[key])).convert('RGB')
            garm[key] = transforms.Resize(self.img_width, interpolation=2)(garm[key])
            garm_mask[key] = Image.open(osp.join(self.data_path, 'cloth-mask', garm_name[key]))
            garm_mask[key] = transforms.Resize(self.img_width, interpolation=0)(garm_mask[key])

            garm[key] = self.transform(garm[key])  # [-1,1]
            garm_mask_array = np.array(garm_mask[key])
            garm_mask_array = (garm_mask_array >= 128).astype(np.float32)
            garm_mask[key] = torch.from_numpy(garm_mask_array)  # [0,1]
            garm_mask[key].unsqueeze_(0)

        # load pose image
        pose_name = img_name.replace('.jpg', '_rendered.png')
        pose_rgb = Image.open(osp.join(self.data_path, 'openpose_img', pose_name))
        pose_rgb = transforms.Resize(self.img_width, interpolation=2)(pose_rgb)
        pose_rgb = self.transform(pose_rgb)  # [-1,1]

        # load pose data
        pose_name = img_name.replace('.jpg', '_keypoints.json')
        with open(osp.join(self.data_path, 'openpose_json', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints_2d']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))[:, :2]

        # load parsing image
        parse_name = img_name.replace('.jpg', '.png')
        parse_big = Image.open(osp.join(self.data_path, 'image-parse-v3', parse_name))

        # load person image
        img_pil_big = Image.open(osp.join(self.data_path, 'image', img_name))
        img_pil = transforms.Resize(self.img_width, interpolation=2)(img_pil_big)
        img = self.transform(img_pil)      
        
        # load masked vton image
        img_agnostic = self.get_img_agnostic(img_pil_big, parse_big, pose_data)
        img_agnostic = transforms.Resize(self.img_width, interpolation=2)(img_agnostic)
        img_agnostic = self.transform(img_agnostic) 
        
        # load caption
        caption_name = garm_name["paired"].replace(".jpg", ".txt")
        caption_path = osp.join(self.data_path, 'cloth_caption', caption_name)
        if os.path.exists(caption_path):
            with open(caption_path, 'r') as file:
                caption = file.read()
        else:
            print("File does not exist. ", caption_name)
            caption = "A cloth"

        result = {
            'img_name': img_name,
            'garm_name': garm_name["paired"],
            'img_ori': img,
            'img_vton': img_agnostic,
            'img_garm': garm["paired"],
            "prompt":caption
        }
        return result

    def __len__(self):
        return len(self.img_names)

class VITONDataLoader:
    def __init__(self, args, dataset):
        super(VITONDataLoader, self).__init__()
        train_sampler = data.sampler.RandomSampler(dataset)
        self.data_loader = data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
        self.batch_size = args.batch_size
        
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()
        return batch
    
class DressCodeDataset(data.Dataset):
    def __init__(self, args,type):
        super(DressCodeDataset, self).__init__()
        self.img_height = args.img_height
        self.img_width = args.img_width
        self.semantic_nc = args.semantic_nc
        self.data_path = osp.join(args.dataset_dir)
        self.transform = transforms.Compose([transforms.ToTensor(),
        ])

        # load data list
        self.data_list=[]
        with open(osp.join(args.dataset_dir,f"{type}_pairs_paired.txt"), 'r') as f:
            for line in f.readlines():
                img_name, garm_name,label = line.strip().split()
                self.data_list.append((img_name,garm_name,label))
        self.img_names = [x[0] for x in self.data_list]
        self.garm_names = [x[1] for x in self.data_list]
        self.labels = [x[2] for x in self.data_list] 
        self.label2idx = {0:'upper_body',1:'lower_body',2:'dresses'}
        self.label2category = {0:'upperbody',1:'lowerbody',2:'fullbody'}
        self.label_map={
            "background": 0,
            "hat": 1,
            "hair": 2,
            "sunglasses": 3,
            "upper_clothes": 4,
            "skirt": 5,
            "pants": 6,
            "dress": 7,
            "belt": 8,
            "left_shoe": 9,
            "right_shoe": 10,
            "head": 11,
            "left_leg": 12,
            "right_leg": 13,
            "left_arm": 14,
            "right_arm": 15,
            "bag": 16,
            "scarf": 17,
        }
        self.keypoints_map={
            0.0: "nose",
            1.0: "neck",
            2.0: "right shoulder",
            3.0: "right elbow",
            4.0: "right wrist",
            5.0: "left shoulder",
            6.0: "left elbow",
            7.0: "left wrist",
            8.0: "right hip",
            9.0: "right knee",
            10.0: "right ankle",
            11.0: "left hip",
            12.0: "left knee",
            13.0: "left ankle",
            14.0: "right eye",
            15.0: "left eye",
            16.0: "right ear",
            17.0: "left ear"
        }
            
    def get_parse_agnostic(self, parse, pose_data):
        parse_array = np.array(parse)
        parse_upper = ((parse_array == 5).astype(np.float32) +
                       (parse_array == 6).astype(np.float32) +
                       (parse_array == 7).astype(np.float32))
        parse_neck = (parse_array == 10).astype(np.float32)

        r = 10
        agnostic = parse.copy()

        # mask arms
        for parse_id, pose_ids in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
            mask_arm = Image.new('L', (self.img_width, self.img_height), 'black')
            mask_arm_draw = ImageDraw.Draw(mask_arm)
            i_prev = pose_ids[0]
            for i in pose_ids[1:]:
                if (pose_data[i_prev, 0] == 0.0 and pose_data[i_prev, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                    continue
                mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r*10)
                pointx, pointy = pose_data[i]
                radius = r*4 if i == pose_ids[-1] else r*15
                mask_arm_draw.ellipse((pointx-radius, pointy-radius, pointx+radius, pointy+radius), 'white', 'white')
                i_prev = i
            parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
            agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

        # mask torso & neck
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))

        return agnostic
            
    def __getitem__(self, index):
        img_name = self.img_names[index]
        garm_name = self.garm_names[index]
        label = int(self.labels[index])
        garm = Image.open(osp.join(self.data_path, self.label2idx[label], "images",garm_name)).convert('RGB')
        garm = transforms.Resize(self.img_width, interpolation=2)(garm)
        garm = self.transform(garm) 
    
        # load pose image
        pose_name = img_name.replace('0.jpg', '5.jpg')
        pose_rgb = Image.open(osp.join(self.data_path,self.label2idx[label] ,'skeletons', pose_name))
        pose_rgb = transforms.Resize(self.img_width, interpolation=2)(pose_rgb)
        pose_rgb = self.transform(pose_rgb)  # [-1,1]

        # load pose data
        pose_name = img_name.replace('0.jpg', '2.json')
        with open(osp.join(self.data_path,self.label2idx[label] ,'keypoints', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data[:, :2]

        # load parsing image
        parse_name = img_name.replace('0.jpg', '4.png')
        parse_big = Image.open(osp.join(self.data_path,self.label2idx[label] ,'label_maps', parse_name))
        
        # load person image  
        img_pil_big = Image.open(osp.join(self.data_path, self.label2idx[label],'images', img_name))
        img_pil = transforms.Resize(self.img_width, interpolation=2)(img_pil_big)
        img = self.transform(img_pil)      
        
        # get masked vton image
        mask, mask_gray = get_mask_location("dc", self.label2idx[label], parse_big, pose_data)    
        masked_vton_img = Image.composite(mask_gray,img_pil,mask)
        masked_vton_img = self.transform(masked_vton_img)
        
        result = {
            'img_name': img_name,
            'garm_name': garm_name,
            'img_ori': img,
            'img_vton': masked_vton_img,
            'pose': pose_rgb,
            'img_garm': garm,
            'prompt': self.label2category[label],
            "pixel_values": img
        }
        return result

    def __len__(self):
        return len(self.img_names)

class DressCodeDataLoader:
    def __init__(self, args, dataset):
        super(DressCodeDataLoader, self).__init__()
        train_sampler = data.sampler.RandomSampler(dataset)
        self.data_loader = data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
        self.batch_size = args.batch_size
        
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch
