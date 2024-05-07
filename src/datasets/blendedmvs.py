import cv2, re, os
import numpy as np
import torch
from torch.utils.data import Dataset
import kornia as K
from kornia.geometry.transform import warp_perspective
import kornia.augmentation as KA
import albumentations as A

from src.utils.dataset import cv_imread


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


class GeometricSequential:
    def __init__(self, *transforms, align_corners=True) -> None:
        self.transforms = transforms
        self.align_corners = align_corners

    def __call__(self, x, mode="bilinear"):
        b, c, h, w = x.shape
        M = torch.eye(3, device=x.device)[None].expand(b, 3, 3)
        for t in self.transforms:
            if np.random.rand() < t.p:
                M = M.matmul(
                    t.compute_transformation(x, t.generate_parameters((b, c, h, w)), flags=None)
                )
        return (
            warp_perspective(
                x, M, dsize=(h, w), mode=mode, align_corners=self.align_corners
            ),
            M,
        )

    def apply_transform(self, x, M, mode="bilinear"):
        b, c, h, w = x.shape
        return warp_perspective(
            x, M, dsize=(h, w), align_corners=self.align_corners, mode=mode
        )


class BlendedMVSDataset(Dataset):
    def __init__(self, datapath, scene_list, mode, augment_fn=None, geo_aug=True):
        super(BlendedMVSDataset, self).__init__()
        self.datapath = datapath
        self.scene_list = scene_list
        self.mode = mode
        self.augment_fn = augment_fn if mode == 'train' else None
        self.geo_aug = GeometricSequential(KA.RandomAffine(degrees=90, p=0.5)) if geo_aug and (mode == "train") else None

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        # with open(self.listfile) as f:
        #     scans = f.readlines()
        #     scans = [line.rstrip() for line in scans]
        scans = self.scene_list

        # scans
        for scan in scans:
            pair_file = "{}/cams/pair.txt".format(scan)
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]

                    # filter by no src view and fill to nviews
                    if len(src_views) > 0:
                        # if len(src_views) < self.nviews:
                        #     print("{}< num_views:{}".format(len(src_views), self.nviews))
                        #     src_views += [src_views[0]] * (self.nviews - len(src_views))
                        # src_views = src_views[:(self.nviews-1)]
                        metas.append((scan, ref_view, src_views, scan))

        # self.interval_scale = interval_scale_dict
        # print("dataset ", self.mode, "metas: ", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

        intrinsics[0, 2] -= 48.0
        intrinsics[1, 2] -= 32.0

        return torch.tensor(intrinsics, dtype=torch.float32), torch.tensor(extrinsics, dtype=torch.float32)

    def prepare_img(self, img):
        h, w = img.shape[:2]
        target_h, target_w = 512, 672 # 576, 768 #512, 640
        start_h, start_w = (h - target_h)//2, (w - target_w)//2
        img_crop = img[start_h: start_h + target_h, start_w: start_w + target_w]
        return img_crop

    def read_img(self, filename):
        img = cv_imread(filename, augment_fn=self.augment_fn)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        np_img = self.prepare_img(np_img)

        return torch.tensor(np_img, dtype=torch.float32)

    def read_depth(self, filename):
        # read pfm depth file
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)
        depth = self.prepare_img(depth)

        return torch.tensor(depth, dtype=torch.float32)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views, scene_name = meta
        # use only the reference view and first nviews-1 source views
        if self.mode == 'train':
            src_view = np.random.choice(src_views)
        else:
            src_view = src_views[3]

        image0 = self.read_img(os.path.join(self.datapath, '{}/blended_images/{:0>8}.jpg'.format(scan, ref_view)))
        K0, T0 = self.read_cam_file(os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, ref_view)))
        depth0 = self.read_depth(os.path.join(self.datapath, '{}/rendered_depth_maps/{:0>8}.pfm'.format(scan, ref_view)))
        P0 = K0 @ T0[:3, :4]

        image1 = self.read_img(os.path.join(self.datapath, '{}/blended_images/{:0>8}.jpg'.format(scan, src_view)))
        K1, T1 = self.read_cam_file(os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, src_view)))
        depth1 = self.read_depth(os.path.join(self.datapath, '{}/rendered_depth_maps/{:0>8}.pfm'.format(scan, src_view)))

        if self.geo_aug:
            image1, H_mat = self.geo_aug(image1.permute(2, 0, 1).unsqueeze(0))
            image1 = image1.squeeze(0).permute(1, 2, 0)
            depth1 = self.geo_aug.apply_transform(depth1[None, None], H_mat)
            depth1 = depth1.squeeze(0).squeeze(0)
            K1 = H_mat[0] @ K1

        P1 = K1 @ T1[:3, :4]

        T_0to1 = T1 @ torch.inverse(T0)  # (4, 4)
        T_1to0 = torch.inverse(T_0to1)


        data = {
            'image0': image0.permute(2, 0, 1),  # (1, h, w)
            'depth0': depth0,  # (h, w)
            'image1': image1.permute(2, 0, 1),
            'depth1': depth1,
            'T_0to1': T_0to1,  # (4, 4)
            'T_1to0': T_1to0,
            'K0': K0,  # (3, 3)
            'K1': K1,
            'proj_mat0': P0, 'proj_mat1': P1,
            'dataset_name': 'BlendedMVS',
            'scene_id': scan,
            'pair_id': idx,
            'pair_names': (f"{scan}_{ref_view}", f"{scan}_{src_view}"),
        }

        return data