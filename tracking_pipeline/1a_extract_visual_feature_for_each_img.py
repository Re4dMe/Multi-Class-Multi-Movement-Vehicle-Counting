import torch, sys, os, re
from torch.utils.data import DataLoader, Dataset
from PIL              import Image

sys.path.append("..")

from model import build_transforms, build_reid_model
from config     import cfg

INPUT_DIR     = cfg.PATH.INPUT_PATH
DEVICE        = cfg.DEVICE.TYPE

if DEVICE == "cuda":
    torch.cuda.set_device(cfg.DEVICE.GPU)

NUM_WORKERS   = 8
IMS_PER_BATCH = 64

class ImageDataset(Dataset):

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def read_image(self, img_path):
        got_img = False
        if not os.path.exists(img_path):
            raise IOError("{} does not exist".format(img_path))
        while not got_img:
            try:
                img = Image.open(img_path).convert('RGB')
                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                exit()
        return img

    def __getitem__(self, index):
        img_path = self.dataset[index]
        img = self.read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, img_path

def collate_fn(batch):
    imgs, path = zip(*batch)
    return torch.stack(imgs, dim=0), path


def _process_data():
    imgs = list()
    
    for scene_dir in os.listdir(INPUT_DIR):
        if not scene_dir.startswith("S0"):
            continue
        for camera_dir in os.listdir(os.path.join(INPUT_DIR, scene_dir)):
            if not camera_dir.startswith("c0"):
                continue
            feature_file = os.path.join(INPUT_DIR, scene_dir, camera_dir, "deep_features.txt")
            if os.path.exists(feature_file):
                os.remove(feature_file)

            data_dir = os.path.join(INPUT_DIR, scene_dir, camera_dir, "cropped_images")
            img_list = os.listdir(data_dir)
            imgs.extend([os.path.join(data_dir, img) for img in img_list])

    return imgs

def _inference(model, data_loader):
    model.eval()
    with torch.no_grad():
        for data, paths in data_loader:
            data = data.cuda()
            feat = model(data)
            for i,p in enumerate(paths):
                scene_dir = re.search(r"S([0-9]){2}", p).group(0)
                camera_dir = re.search(r"c([0-9]){3}", p).group(0)
                path = os.path.join(INPUT_DIR, scene_dir, camera_dir)
                with open(os.path.join(path, 'deep_features.txt'), 'a+') as f:
                    line = p.split('/')[-1]+ ' '
                    print(line)
                    feature = list(feat[i].cpu().numpy())
                    for fea in feature:
                        line = line + str(fea) + ' '
                    f.write(line.strip()+'\n')
    
if __name__ == "__main__":
    imgs = _process_data()
    transforms = build_transforms(cfg)
    model = build_reid_model(cfg)
    model = model.to(DEVICE)
    dataset = ImageDataset(imgs, transforms)
    dataloader = DataLoader(dataset, batch_size=IMS_PER_BATCH, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    _inference(model, dataloader)
