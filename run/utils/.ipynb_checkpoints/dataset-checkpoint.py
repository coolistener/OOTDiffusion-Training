import json
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
        self.data_path = osp.join(args.dataset_dir, type)
        self.transform = transforms.Compose([transforms.ToTensor(),
        ])

        # load data list
        img_names = []     #模特图片
        c_names = []       #服装图片
        with open(osp.join(args.dataset_dir,f"{type}_pairs.txt"), 'r') as f:
            for line in f.readlines():
                img_name, c_name = line.strip().split()
                img_names.append(img_name)
                c_names.append(c_name)

        self.img_names = img_names    
        self.c_names = dict()
        # self.c_names['paired'] = c_names
        ###img跟cloth名称相同，在不同文件夹下
        self.c_names['paired'] = img_names

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

        r = 20
        agnostic = img.copy()
        agnostic_draw = ImageDraw.Draw(agnostic)

        length_a = np.linalg.norm(pose_data[5] - pose_data[2])
        length_b = np.linalg.norm(pose_data[12] - pose_data[9])
        point = (pose_data[9] + pose_data[12]) / 2
        pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
        pose_data[12] = point + (pose_data[12] - point) / length_b * length_a

        # mask arms
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*10)
        for i in [2, 5]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
        for i in [3, 4, 6, 7]:
            if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')

        # mask torso
        for i in [9, 12]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r*6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r*6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r*12)
        agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

        # mask neck
        pointx, pointy = pose_data[1]
        agnostic_draw.rectangle((pointx-r*7, pointy-r*7, pointx+r*7, pointy+r*7), 'gray', 'gray')
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))

        return agnostic

    def __getitem__(self, index):
        img_name = self.img_names[index]
        c_name = {}
        c = {}
        cm = {}
        # load model cloth cloth_mask image
        for key in self.c_names:
            c_name[key] = self.c_names[key][index]
            c[key] = Image.open(osp.join(self.data_path, 'cloth', c_name[key])).convert('RGB')
            c[key] = transforms.Resize(self.img_width, interpolation=2)(c[key])
            cm[key] = Image.open(osp.join(self.data_path, 'cloth-mask', c_name[key]))
            cm[key] = transforms.Resize(self.img_width, interpolation=0)(cm[key])

            c[key] = self.transform(c[key])  # [-1,1]
            cm_array = np.array(cm[key])
            cm_array = (cm_array >= 128).astype(np.float32)
            cm[key] = torch.from_numpy(cm_array)  # [0,1]
            cm[key].unsqueeze_(0)

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
        parse = Image.open(osp.join(self.data_path, 'image-parse-v3', parse_name))
        parse = transforms.Resize(self.img_width, interpolation=0)(parse)
        parse_agnostic = self.get_parse_agnostic(parse, pose_data)
        parse_agnostic = torch.from_numpy(np.array(parse_agnostic)[None]).long()

        labels = {
            0: ['background', [0, 10]],
            1: ['hair', [1, 2]],
            2: ['face', [4, 13]],
            3: ['upper', [5, 6, 7]],
            4: ['bottom', [9, 12]],
            5: ['left_arm', [14]],
            6: ['right_arm', [15]],
            7: ['left_leg', [16]],
            8: ['right_leg', [17]],
            9: ['left_shoe', [18]],
            10: ['right_shoe', [19]],
            11: ['socks', [8]],
            12: ['noise', [3, 11]]
        }
        parse_agnostic_map = torch.zeros(20, self.img_height, self.img_width, dtype=torch.float)
        parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
        new_parse_agnostic_map = torch.zeros(self.semantic_nc, self.img_height, self.img_width, dtype=torch.float)
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]

        # load person image
        img = Image.open(osp.join(self.data_path, 'image', img_name))
        img = transforms.Resize(self.img_width, interpolation=2)(img)
        img_agnostic = self.get_img_agnostic(img, parse, pose_data)
        img = self.transform(img)
        img_agnostic = self.transform(img_agnostic)  # [-1,1]

        result = {
            'img_name': img_name,
            'c_name': c_name,
            'img': img,
            'img_agnostic': img_agnostic,
            'parse_agnostic': new_parse_agnostic_map,
            'pose': pose_rgb,
            'cloth': c,
            'cloth_mask': cm,
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

# def make_train_dataset(args, tokenizer, accelerator):
#     # Get the datasets: you can either provide your own training and evaluation files (see below)
#     # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

#     # In distributed training, the load_dataset function guarantees that only one local process can concurrently
#     # download the dataset.
#     if args.dataset_name is not None:
#         # Downloading and loading a dataset from the hub.
#         dataset = load_dataset(
#             args.dataset_name,
#             args.dataset_config_name,
#             cache_dir=args.cache_dir,
#         )
#     else:
#         if args.train_data_dir is not None:
#             dataset = load_dataset(
#                 args.train_data_dir,
#                 cache_dir=args.cache_dir,
#             )
#         # See more about loading custom images at
#         # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

#     # Preprocessing the datasets.
#     # We need to tokenize inputs and targets.
#     column_names = dataset["train"].column_names

#     # 6. Get the column names for input/target.
#     if args.image_column is None:
#         image_column = column_names[0]
#         logger.info(f"image column defaulting to {image_column}")
#     else:
#         image_column = args.image_column
#         if image_column not in column_names:
#             raise ValueError(
#                 f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
#             )

#     if args.caption_column is None:
#         caption_column = column_names[1]
#         logger.info(f"caption column defaulting to {caption_column}")
#     else:
#         caption_column = args.caption_column
#         if caption_column not in column_names:
#             raise ValueError(
#                 f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
#             )

#     if args.conditioning_image_column is None:
#         conditioning_image_column = column_names[2]
#         logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
#     else:
#         conditioning_image_column = args.conditioning_image_column
#         if conditioning_image_column not in column_names:
#             raise ValueError(
#                 f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
#             )

#     def tokenize_captions(examples, is_train=True):
#         captions = []
#         for caption in examples[caption_column]:
#             if random.random() < args.proportion_empty_prompts:
#                 captions.append("")
#             elif isinstance(caption, str):
#                 captions.append(caption)
#             elif isinstance(caption, (list, np.ndarray)):
#                 # take a random caption if there are multiple
#                 captions.append(random.choice(caption) if is_train else caption[0])
#             else:
#                 raise ValueError(
#                     f"Caption column `{caption_column}` should contain either strings or lists of strings."
#                 )
#         inputs = tokenizer(
#             captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
#         )
#         return inputs.input_ids

#     image_transforms = transforms.Compose(
#         [
#             transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
#             transforms.CenterCrop(args.resolution),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5], [0.5]),
#         ]
#     )

#     conditioning_image_transforms = transforms.Compose(
#         [
#             transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
#             transforms.CenterCrop(args.resolution),
#             transforms.ToTensor(),
#         ]
#     )

#     def preprocess_train(examples):
#         images = [image.convert("RGB") for image in examples[image_column]]
#         images = [image_transforms(image) for image in images]

#         conditioning_images = [image.convert("RGB") for image in examples[conditioning_image_column]]
#         conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]

#         examples["pixel_values"] = images
#         examples["conditioning_pixel_values"] = conditioning_images
#         examples["input_ids"] = tokenize_captions(examples)

#         return examples

#     with accelerator.main_process_first():
#         if args.max_train_samples is not None:
#             dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
#         # Set the training transforms
#         train_dataset = dataset["train"].with_transform(preprocess_train)

#     return train_dataset