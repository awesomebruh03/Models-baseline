import torch
import random
from torch.utils.data import Dataset


class loaddataset(Dataset):
    def __init__(self, clean_path, noise_path, train=True, train_split=0.8):
        self.clean_bank = torch.load(clean_path, map_location='cpu')
        self.noise_bank = torch.load(noise_path, map_location='cpu')
        n = len(self.clean_bank)
        n_train = int(n * train_split)
        if train:
            self.indices = list(range(n_train))
        else:
            self.indices = list(range(n_train, n))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        clean = self.clean_bank[self.indices[idx]].float()
        noise = random.choice(self.noise_bank).float()

        if len(noise) > len(clean):
            start = random.randint(0, len(noise) - len(clean))
            noise = noise[start:start + len(clean)]
        elif len(noise) < len(clean):
            repeats = (len(clean) + len(noise) - 1) // len(noise)
            noise = noise.repeat(repeats)[:len(clean)]

        clean = self.spl_cal(clean, 65)

        snr = random.uniform(0.0, 5.0)
        rms_clean = self.rms_energy(clean)
        rms_noise = self.rms_energy(noise)
        noise_scaled = 10 ** ((rms_clean - snr - rms_noise) / 20) * noise
        noisy = clean + noise_scaled

        noisy = self.spl_cal(noisy, 65)

        return {'noisy': noisy, 'clean': clean}

    @staticmethod
    def spl_cal(x, target_spl):
        spl_before = 20 * torch.log10(torch.sqrt(torch.mean(x ** 2)) / (20e-6) + 1e-12)
        scale = 10 ** ((target_spl - spl_before) / 20)
        return x * scale

    @staticmethod
    def rms_energy(x):
        return 10 * torch.log10((1e-12 + torch.sum(x ** 2)) / len(x))
