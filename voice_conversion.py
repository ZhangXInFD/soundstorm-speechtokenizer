from soundstorm import SoundStorm, SoundStormDataset, ConformerWrapper
import torch
from speechtokenizer import SpeechTokenizer
import torchaudio
from einops import rearrange
import os
import random
from tqdm import tqdm

class VoiceConversion:
    
    def __init__(self, 
                 tokenizer: SpeechTokenizer, 
                 soundstorm: SoundStorm, 
                 device='cpu'):
        self.tokenizer = tokenizer.to(device)
        self.tokenizer.eval()
        self.soundstorm = soundstorm.to(device)
        self.soundstorm.eval()
        self.device = device
    
    @torch.no_grad()    
    def encode(self, wav_file):
        wav, sr = torchaudio.load(wav_file)
        if sr != self.tokenizer.sample_rate:
            wav = torchaudio.functional.resample(wav, sr , self.tokenizer.sample_rate)
        tokens = self.tokenizer.encode(wav.unsqueeze(0).to(self.device))
        return rearrange(tokens, 'q b n -> b n q').squeeze(0)
    
    @torch.no_grad()
    def decode(self, file, tokens):
        wav = self.tokenizer.decode(rearrange(tokens, 'n q -> q 1 n'))
        torchaudio.save(file, wav.squeeze(0).cpu().detach(), 16000)
        
    @torch.no_grad()    
    def generate(self, prompt_file, src_file, tgt_dir, max_prompt_token_length=150, steps=[8], greedy=True):
        if not os.path.exists(tgt_dir):
            os.makedirs(tgt_dir)
        src_tokens = self.encode(src_file).unsqueeze(0)
        self.decode(f'{tgt_dir}/raw.wav', src_tokens.squeeze(0))
        prompt_tokens = self.encode(prompt_file).unsqueeze(0)[:, :max_prompt_token_length]
        self.decode(f'{tgt_dir}/prompt.wav', prompt_tokens.squeeze(0))
        semantic_tokens = src_tokens[:, :, 0]
        for step in steps:
            generated = self.soundstorm.genenrate(semantic_tokens=semantic_tokens,
                                                steps=step,
                                                greedy=greedy)
            self.decode(f'{tgt_dir}/unconditonal_{step}.wav', generated.squeeze(0))
            generated = self.soundstorm.genenrate(semantic_tokens=semantic_tokens,
                                                prompt_tokens=prompt_tokens,
                                                steps=step,
                                                greedy=greedy)
            self.decode(f'{tgt_dir}/gererate_{step}.wav', generated.squeeze(0))
        
        
    

if __name__ == '__main__':
    st_cfg = '/remote-home/xzhang/Speech/SpeechTokenizer/ckpt/config.json'
    st_ckpt = '/remote-home/xzhang/Speech/SpeechTokenizer/ckpt/SpeechTokenizer.pt'  
    tokenizer = SpeechTokenizer.load_from_checkpoint(st_cfg, st_ckpt)

    conformer = ConformerWrapper(codebook_size=1024,
                                num_quantizers=7,
                                conformer={'dim':1024, 'depth': 12, 'heads':8, 'dim_head': 128, 'attn_flash': False},
                                )

    soundstorm = SoundStorm(net=conformer,
                            num_semantic_token_ids=1024,
                            semantic_pad_id=1024,
                            pad_id=1024,
                            schedule = 'cosine')
    soundstorm.load('/remote-home/xzhang/Speech/soundstorm-speechtokenizer/ls_result_wav_3090/SoundStorm_best_dev.pt')
    device = 'cuda:1'
    vc = VoiceConversion(tokenizer=tokenizer,
                         soundstorm=soundstorm,
                         device=device)
    root_dir = '/remote-home/share/SpeechPretrain/LibriSpeech/LibriSpeech/dev-clean'
    speakers = [folder for folder in os.listdir(root_dir) if '.txt' not in folder]
    file_dict = {speaker:[f'{chapter}/{file}' for chapter in os.listdir(f'{root_dir}/{speaker}') for file in os.listdir(f'{root_dir}/{speaker}/{chapter}') if '.txt' not in file] for speaker in speakers}
    tgt_dir = './vc_ls_dev_clean'
    k = 30
    random.seed(0)
    prompt_speakers = random.sample(speakers, k)
    src_speakers = random.sample(speakers, k)
    for prompt_speaker, src_speaker in tqdm(zip(prompt_speakers, src_speakers)):
        prompt_file = random.choice(file_dict[prompt_speaker])
        while src_speaker == prompt_speaker:
            src_speaker = random.choice(speakers)
        src_file = random.choice(file_dict[src_speaker])
        vc.generate(prompt_file=f'{root_dir}/{prompt_speaker}/{prompt_file}',
                    src_file=f'{root_dir}/{src_speaker}/{src_file}',
                    tgt_dir=tgt_dir + '/' + '-'.join(src_file.split('-')[:2]) + '_' + '-'.join(prompt_file.split('-')[:2]),
                    steps=[4, 8, 16]
        )