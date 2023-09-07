from speechtokenizer import SpeechTokenizer
from tqdm import tqdm
import os
import numpy as np
import torchaudio
from pathlib import Path
from einops import rearrange


src_root = '/remote-home/share/SpeechPretrain/LibriSpeech/LibriSpeech/'
target_root = '/remote-home/share/SpeechPretrain/spt_tokens/LibriSpeech/LibriSpeech/'
train_split = ['train-clean-360', 'train-clean-100', 'train-other-500']
exts = ['flac']
train_file_list = []
for split in train_split:
    path = Path(f'{src_root}/{split}')
    train_file_list += [str(file) for ext in exts for file in path.glob(f'**/*.{ext}')]
with open('./train_file_list_wav.txt', 'w+') as f:
    f.write('\n'.join(train_file_list))
valid_split = ['dev-clean']
valid_file_list = []
for split in valid_split:
    path = Path(f'{src_root}/{split}')
    valid_file_list += [str(file) for ext in exts for file in path.glob(f'**/*.{ext}')]
with open('./valid_file_list_wav.txt', 'w+') as f:
    f.write('\n'.join(valid_file_list))
file_list = train_file_list + valid_file_list

tokens_file_dict = {'train':[],
                    'valid': []}    
st_cfg = '/remote-home/xzhang/Speech/SpeechTokenizer/ckpt/config.json'
st_ckpt = '/remote-home/xzhang/Speech/SpeechTokenizer/ckpt/SpeechTokenizer.pt'  
tokenizer = SpeechTokenizer.load_from_checkpoint(st_cfg, st_ckpt).cuda()
for file in tqdm(file_list):
    split, spk, chapter, filename = file.split('/')[-4:]
    target_dir = f'{target_root}/{split}/{chapter}'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    wav, sr = torchaudio.load(file)
    if wav.size(0) > 1:
        wav = wav.mean(axis=0)
        wav = wav.unsqueeze(0)
    if sr != tokenizer.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, tokenizer.sample_rate)
    wav = wav.cuda()
    tokens = tokenizer.encode(wav.unsqueeze(0))
    tokens = rearrange(tokens, 'q b n -> b n q')
    target_file = f"{target_dir}/{filename.split('.')[0]}.spt"
    np.save(target_file, tokens.squeeze(0).cpu().detach().numpy())
    target_file = target_file + '.npy'
    if split in train_split:
        tokens_file_dict['train'].append(target_file)
    else:
        tokens_file_dict['valid'].append(target_file)
for split, filelist in tokens_file_dict.items():
    with open(f'./{split}_file_list_tokens.txt', 'w+') as f:
        f.write('\n'.join(filelist))
    
    