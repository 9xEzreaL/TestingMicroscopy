import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy import linalg

class FID3DCalculator:
    def __init__(self, device='cpu', batch_size=16, force_resize=False):
        """
        Args:
            device: 'cpu' or 'cuda'
            batch_size:  batch size
            force_resize: force resize to 299x299 or not
        """
        self.device = torch.device(device)
        self.batch_size = batch_size

        # preload Inception v3，remove FC layer
        self.inception = inception_v3(pretrained=True, aux_logits=True)
        self.inception.aux_logits = False
        self.inception.AuxLogits = None
        self.inception.fc = nn.Identity()
        self.inception.eval().to(self.device)

        # preprocess：resize → 3-channel → normalize
        transform_list = []
        if force_resize:
            transform_list.append(transforms.Resize((299, 299)))
        transform_list += [
            transforms.Lambda(lambda img: img.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
        self.transform = transforms.Compose(transform_list)

    def _get_activations(self, volumes):
        """
        將 volumes (N, D, H, W)→(N*D, 2048) 的 Inception feature
        """
        # N, D, H, W = volumes.shape
        # # flatten 成 (N*D, H, W)
        # slices = volumes.reshape(-1, H, W)
        print(volumes.shape)
        slices = volumes
        # 轉成 tensor 並做前處理
        tensor_slices = []
        for sl in slices:
            t = torch.from_numpy(sl).float().to(self.device)
            t = t.unsqueeze(0)               # (1, H, W)
            t = self.transform(t)            # (3,299,299)
            tensor_slices.append(t)
        tensor_slices = torch.stack(tensor_slices, dim=0)  # (N*D,3,299,299)

        # 分 batch 處理
        acts = []
        with torch.no_grad():
            for i in range(0, tensor_slices.size(0), self.batch_size):
                batch = tensor_slices[i:i+self.batch_size]
                feat = self.inception(batch)                   # (b,2048)
                acts.append(feat.cpu().numpy())
        acts = np.concatenate(acts, axis=0)  # (N*D,2048)
        return acts

    def calculate_fid(self, real_vols, fake_vols):
        """
        計算 Fréchet Inception Distance
        Args:
            real_vols, fake_vols: numpy arrays of shape (N, D, H, W)
        Returns:
            fid value (float)
        """
        real_vols = self._normailze(real_vols)
        fake_vols = self._normailze(fake_vols)

        act1 = self._get_activations(real_vols)
        act2 = self._get_activations(fake_vols)

        mu1, mu2 = act1.mean(axis=0), act2.mean(axis=0)
        sigma1 = np.cov(act1, rowvar=False)
        sigma2 = np.cov(act2, rowvar=False)

        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
        return float(fid)

    def calculate_is(self, fake_vols, splits=10):
        """
        計算 Inception Score
        Args:
            fake_vols: numpy array (N, D, H, W)
            splits: 切分數
        Returns:
            mean_IS, std_IS
        """
        # 先用 Inception 原本的分類 head
        inc = inception_v3(pretrained=True, aux_logits=False).to(self.device)
        inc.eval()
        # 前處理同上
        N, D, H, W = fake_vols.shape
        slices = fake_vols.reshape(-1, H, W)
        tensor_slices = []
        for sl in slices:
            t = torch.from_numpy(sl).float().to(self.device)
            t = t.unsqueeze(0)
            t = self.transform(t)
            tensor_slices.append(t)
        tensor_slices = torch.stack(tensor_slices, dim=0)

        # 取得所有 slice 的 class probability
        preds = []
        with torch.no_grad():
            for i in range(0, tensor_slices.size(0), self.batch_size):
                logits = inc(tensor_slices[i:i+self.batch_size])
                p = F.softmax(logits, dim=1).cpu().numpy()
                preds.append(p)
        preds = np.concatenate(preds, axis=0)  # (N*D,1000)

        # 平均到每個 volume (每 D 片共享一組 p)：reshape→(N, D, 1000)→沿 D 平均
        preds = preds.reshape(N, D, -1).mean(axis=1)  # (N,1000)

        # 計算 IS
        split_scores = []
        for k in range(splits):
            part = preds[k * (N//splits):(k+1) * (N//splits), :]
            py = part.mean(axis=0, keepdims=True)
            kl = part * (np.log(part+1e-6) - np.log(py+1e-6))
            kl = kl.sum(axis=1)
            split_scores.append(np.exp(kl.mean()))
        return float(np.mean(split_scores)), float(np.std(split_scores))

    def _normailze(self, image):
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if image.max() > 1 or image.min() < 0:
            image = (image - image.min()) / (image.max() - image.min())
        return image


