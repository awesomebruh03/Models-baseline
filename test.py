import torch, os, sys, argparse
import soundfile as sf
from pathlib import Path
from glob import glob
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
from librosa import stft, istft, griffinlim
from pystoi import stoi
from pesq import pesq
from auraloss.time import SISDRLoss, SNRLoss
from objective_metrics import *
from torch import nn
from torch.utils.data import DataLoader
from dataloader import loaddataset
from Network import *


class RMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(pred, actual))


class test_model:
    def __init__(self, modelname, modelfile, evalfile, Loss_function):
        super(test_model, self).__init__()
        self.evalfile = evalfile
        self.modelname = modelname
        self.modelfile = modelfile
        self.Loss_function = Loss_function
        # self.si_sdr = SISDRLoss()
        self.rmse = RMSE()
        self.SNRLoss = SNRLoss()

    def mapmodule(self, state_dict, keyword='model'):
        model_dict = []
        for key in state_dict.keys():
            if keyword in key:
                new_key = key.replace(keyword + '.', '')
                model_dict.append((new_key, state_dict[key]))
        new_state_dict = OrderedDict(model_dict)
        return new_state_dict

    def loadmodel(self):
        if self.modelname == 'CFTNet':
            from Network import CFTNet
            bestmodel = glob(os.getcwd() + '/Saved_Models/CFTNet/' + self.modelfile)[0]
            model = CFTNet()
        elif self.modelname == 'DCCTN':
            from Network import DCCTN
            bestmodel = glob(os.getcwd() + '/Saved_Models/DCCTN/' + self.modelfile)[0]
            model = DCCTN()
        elif self.modelname == 'DATCFTNET':
            from Network import DATCFTNET
            bestmodel = glob(os.getcwd() + '/Saved_Models/DATCFTNET/' + self.modelfile)[0]
            model = DATCFTNET()
        elif self.modelname == 'DATCFTNET_DSC':
            from Network import DATCFTNET_DSC
            bestmodel = glob(os.getcwd() + '/Saved_Models/DATCFTNET_DSC/' + self.modelfile)[0]
            model = DATCFTNET_DSC()

        print('Initializing model :  ' + self.modelname)
        print('Loading weights    :  ' + bestmodel)
        saved_model = torch.load(bestmodel, map_location='cpu')
        saved_model['state_dict'] = self.mapmodule(saved_model['state_dict'])
        model.load_state_dict(saved_model['state_dict'])
        model = model.to('cuda')
        model.eval()
        return model

    def main(self):
        model = self.loadmodel()
        eval_files = open(self.evalfile).readlines()
        eval_files = [i.strip().split(' ') for i in eval_files]
        STOI_C_N = np.zeros(len(eval_files))
        STOI_C_E = np.zeros(len(eval_files))
        PESQ_C_N = np.zeros(len(eval_files))
        PESQ_C_E = np.zeros(len(eval_files))
        LSD_C_N = np.zeros(len(eval_files))
        LSD_C_E = np.zeros(len(eval_files))
        SNRloss_C_N = np.zeros(len(eval_files));
        SNRloss_C_E = np.zeros(len(eval_files))
        SISDR_C_N = np.zeros(len(eval_files))
        SISDR_C_E = np.zeros(len(eval_files))
        RMSE_C_N = np.zeros(len(eval_files));
        RMSE_C_E = np.zeros(len(eval_files))

        Path(os.path.dirname(os.getcwd() + '/Result/')).mkdir(parents=True, exist_ok=True)
        j = 0
        for audiofile in tqdm(eval_files, desc='Decoidng Eval Files '):
            noisy = self.norm(sf.read(audiofile[0])[0])
            clean = self.norm(sf.read(audiofile[1])[0])
            clean = self.SPL_cal(clean, 65);
            noisy = self.SPL_cal(noisy, 65)
            # enh_audio = (model(torch.FloatTensor(noisy).unsqueeze(0).to('cuda')).detach().cpu().numpy()).squeeze(0)
            enh_audio = model(torch.FloatTensor(noisy).unsqueeze(0).to('cuda')).detach().cpu().numpy()
            # enh_audio=enh_audio.squeeze(0)
            enh_audio = self.norm(self.match_dims(noisy, enh_audio))
            enh_audio = self.SPL_cal(enh_audio, 65)
            desname = audiofile[2]
            sf.write(desname, enh_audio, 16000)
            Path(os.path.dirname(desname)).mkdir(parents=True, exist_ok=True)
            si_sdr_loss = SISDRLoss()
            ref_t = torch.from_numpy(clean).unsqueeze(0).unsqueeze(0)
            noisy_t = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0)
            enh_t = torch.from_numpy(enh_audio).unsqueeze(0).unsqueeze(0)
            SISDR_C_N[j] = -si_sdr_loss(ref_t, noisy_t).item()
            SISDR_C_E[j] = -si_sdr_loss(ref_t, enh_t).item()
            RMSE_C_N[j], RMSE_C_E[j] = self.rmse(torch.from_numpy(clean), torch.from_numpy(noisy)), self.rmse(
                torch.from_numpy(clean), torch.from_numpy(enh_audio))
            PESQ_C_N[j], PESQ_C_E[j] = pesq(16000, clean, noisy, 'wb'), pesq(16000, clean, enh_audio, 'wb')
            STOI_C_N[j], STOI_C_E[j] = stoi(clean, noisy, 16000, extended=False), stoi(clean, enh_audio, 16000,
                                                                                       extended=False)
            LSD_C_N[j], LSD_C_E[j] = lsd(clean, noisy), lsd(clean, enh_audio)
            SNRloss_C_N[j], SNRloss_C_E[j] = self.SNRLoss(torch.from_numpy(clean),
                                                          torch.from_numpy(noisy)), self.SNRLoss(
                torch.from_numpy(clean), torch.from_numpy(enh_audio))

            print('Sample: ' + str(j) + ' The Noisy PESQ score: ' + str(
                (PESQ_C_N[j])) + '\nThe Enhanced PESQ score: ' + str((PESQ_C_E[j])))
            print('The Noisy STOI score: ' + str((STOI_C_N[j])) + '\nThe Enhanced STOI score: ' + str((STOI_C_E[j])))
            print('The Noisy LSD score: ' + str((LSD_C_N[j])) + '\nThe Enhanced LSD score: ' + str((LSD_C_E[j])))
            j += 1
        print('The mean Noisy STOI score: ' + str(np.mean(STOI_C_N)) + '\nThe mean Enhanced STOI score: ' + str(
            np.mean(STOI_C_E)))
        print('The mean Noisy PESQ score: ' + str(np.mean(PESQ_C_N)) + '\nThe mean Enhanced PESQ score: ' + str(
            np.mean(PESQ_C_E)))
        print('The mean Noisy SISDR score: ' + str(np.mean(SISDR_C_N)) + '\nThe mean Enhanced SISDR score: ' + str(
            np.mean(SISDR_C_E)))
        print('The mean Noisy LSD score: ' + str(np.mean(LSD_C_N)) + '\nThe mean Enhanced LSD score: ' + str(
            np.mean(LSD_C_E)))
        print(
            'The mean Noisy SNRLoss score: ' + str(np.mean(SNRloss_C_N)) + '\nThe mean Enhanced SNRLoss score: ' + str(
                np.mean(SNRloss_C_E)))
        print('The mean Noisy RMSE score: ' + str(np.mean(RMSE_C_N)) + '\nThe mean Enhanced RMSE score: ' + str(
            np.mean(RMSE_C_E)))
        np.savez(os.getcwd() + '/Result/IEEE_Objective_score_' + self.modelname + '_' + self.Loss_function, PESQ_C_N,
                 PESQ_C_E, STOI_C_N, STOI_C_E, LSD_C_N, LSD_C_E, SISDR_C_N, SISDR_C_E, RMSE_C_N, RMSE_C_E, SNRloss_C_N,
                 SNRloss_C_E)

    def match_dims(self, rev, enh):
        output = np.zeros_like(rev)
        if len(enh) >= len(rev):
            output = enh[:len(rev)]
        if len(enh) < len(rev):
            output[:len(enh)] = enh
        return output

    def norm(self, x):
        return x / (np.max(np.abs(x)) + 1e-10)

    def SPL_cal(self, x, SPL):
        SPL_before = 20 * np.log10(np.sqrt(np.mean(x ** 2)) / (20 * 1e-6))
        y = x * 10 ** ((SPL - SPL_before) / 20)
        return y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Speech Enhancement Model')
    parser.add_argument('--model', type=str, default='DATCFTNET')
    parser.add_argument('--modelfile', type=str, default='*.ckpt',
                        help='Glob pattern for checkpoint (uses latest match)')
    parser.add_argument('--clean_path', type=str, required=True,
                        help='Path to clean_bank.pt')
    parser.add_argument('--noise_path', type=str, required=True,
                        help='Path to noise_bank.pt')
    parser.add_argument('--loss', type=str, default='SISDR+FreqLoss')
    parser.add_argument('--b', type=int, default=1, help='Batch size')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--full-metrics', action='store_true',
                        help='Compute PESQ and STOI (slower)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    ckpt_dir = os.getcwd() + '/Saved_Models/' + args.model + '/'
    modelfiles = sorted(glob(ckpt_dir + args.modelfile))
    if not modelfiles:
        print(f'No checkpoint found in {ckpt_dir}{args.modelfile}')
        sys.exit(1)
    ckpt_path = modelfiles[-1]
    print(f'Checkpoint: {ckpt_path}')

    test_data = loaddataset(args.clean_path, args.noise_path, train=False)
    test_loader = DataLoader(test_data, batch_size=args.b, shuffle=False, num_workers=2)
    print(f'Test samples: {len(test_data)}')

    model_map = {
        'CFTNet': lambda: __import__('Network', fromlist=['CFTNet']).CFTNet(),
        'DCCTN': lambda: __import__('Network', fromlist=['DCCTN']).DCCTN(),
        'DATCFTNET': lambda: __import__('Network', fromlist=['DATCFTNET']).DATCFTNET(),
        'DATCFTNET_DSC': lambda: __import__('Network', fromlist=['DATCFTNET_DSC']).DATCFTNET_DSC(),
    }
    if args.model not in model_map:
        print(f'Unknown model: {args.model}'); sys.exit(1)

    model = model_map[args.model]().to(device)
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state.get('state_dict', state)
    fixed_sd = OrderedDict()
    for k, v in state_dict.items():
        fixed_sd[k.replace('model.', '', 1) if k.startswith('model.') else k] = v

    # Filter out keys whose tensor shapes don't match the current model.
    # DPAtBlock.Inter_LN / Intra_LN are rebuilt dynamically on every forward()
    # pass, so their __init__ weights are never actually used – it's safe to
    # skip them when the checkpoint was trained with a different sequence length.
    model_state = model.state_dict()
    skipped = [k for k, v in fixed_sd.items()
               if k in model_state and v.shape != model_state[k].shape]
    if skipped:
        print(f'Skipping {len(skipped)} mismatched param(s) (shape mismatch): {skipped}')
    filtered_sd = {k: v for k, v in fixed_sd.items()
                   if k not in skipped}
    model.load_state_dict(filtered_sd, strict=False)
    model.eval()
    print('Model loaded successfully')

    si_sdr_loss = SISDRLoss()
    snr_loss = SNRLoss()
    rmse_fn = RMSE()

    sisdr_c_n, sisdr_c_e = [], []
    snrloss_c_n, snrloss_c_e = [], []
    rmse_c_n, rmse_c_e = [], []
    lsd_c_n, lsd_c_e = [], []
    if args.full_metrics:
        pesq_c_n, pesq_c_e = [], []
        stoi_c_n, stoi_c_e = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)
            enh = model(noisy)

            for i in range(len(noisy)):
                n = noisy[i].cpu().numpy()
                c = clean[i].cpu().numpy()
                e = enh[i].cpu().numpy()

                if len(e) >= len(c):
                    e = e[:len(c)]
                else:
                    e = np.pad(e, (0, len(c) - len(e)))

                norm_fn = lambda x: x / (np.max(np.abs(x)) + 1e-10)
                n, c, e = norm_fn(n), norm_fn(c), norm_fn(e)

                spl_cal = lambda x, spl: x * 10 ** ((spl - 20 * np.log10(np.sqrt(np.mean(x ** 2)) / (20 * 1e-6))) / 20)
                c = spl_cal(c, 65)
                n = spl_cal(n, 65)
                e = spl_cal(e, 65)

                c_t = torch.from_numpy(c).float().unsqueeze(0).unsqueeze(0)
                n_t = torch.from_numpy(n).float().unsqueeze(0).unsqueeze(0)
                e_t = torch.from_numpy(e).float().unsqueeze(0).unsqueeze(0)

                sisdr_c_n.append(-si_sdr_loss(c_t, n_t).item())
                sisdr_c_e.append(-si_sdr_loss(c_t, e_t).item())
                snrloss_c_n.append(-snr_loss(c_t, n_t).item())
                snrloss_c_e.append(-snr_loss(c_t, e_t).item())

                c_f = torch.from_numpy(c).float()
                n_f = torch.from_numpy(n).float()
                e_f = torch.from_numpy(e).float()
                rmse_c_n.append(rmse_fn(c_f, n_f).item())
                rmse_c_e.append(rmse_fn(c_f, e_f).item())

                lsd_c_n.append(lsd(c, n))
                lsd_c_e.append(lsd(c, e))

                if args.full_metrics:
                    pesq_c_n.append(pesq(16000, c, n, 'wb'))
                    pesq_c_e.append(pesq(16000, c, e, 'wb'))
                    stoi_c_n.append(stoi(c, n, 16000, extended=False))
                    stoi_c_e.append(stoi(c, e, 16000, extended=False))

    print('\n========== RESULTS ==========')
    print(f'SI-SDR:    Noisy={np.mean(sisdr_c_n):.2f}   Enhanced={np.mean(sisdr_c_e):.2f}')
    print(f'SNR Loss:  Noisy={np.mean(snrloss_c_n):.2f}   Enhanced={np.mean(snrloss_c_e):.2f}')
    print(f'RMSE:      Noisy={np.mean(rmse_c_n):.4f}   Enhanced={np.mean(rmse_c_e):.4f}')
    print(f'LSD:       Noisy={np.mean(lsd_c_n):.2f}   Enhanced={np.mean(lsd_c_e):.2f}')
    if args.full_metrics:
        print(f'PESQ:      Noisy={np.mean(pesq_c_n):.2f}   Enhanced={np.mean(pesq_c_e):.2f}')
        print(f'STOI:      Noisy={np.mean(stoi_c_n):.2f}   Enhanced={np.mean(stoi_c_e):.2f}')

    save_dir = os.getcwd() + '/Result/'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    np.savez(save_dir + 'Test_' + args.model + '_' + args.loss,
             sisdr_c_n=np.array(sisdr_c_n), sisdr_c_e=np.array(sisdr_c_e),
             snrloss_c_n=np.array(snrloss_c_n), snrloss_c_e=np.array(snrloss_c_e),
             rmse_c_n=np.array(rmse_c_n), rmse_c_e=np.array(rmse_c_e),
             lsd_c_n=np.array(lsd_c_n), lsd_c_e=np.array(lsd_c_e),
             **(dict(pesq_c_n=np.array(pesq_c_n), pesq_c_e=np.array(pesq_c_e),
                     stoi_c_n=np.array(stoi_c_n), stoi_c_e=np.array(stoi_c_e))
                if args.full_metrics else {}))
