from soundstorm_speechtokenizer import SoundStorm, SoundStormTrainer, ConformerWrapper
import torch
from speechtokenizer import SpeechTokenizer

if __name__ == '__main__':
    
    
    conformer = ConformerWrapper(codebook_size=1024,
                                num_quantizers=7,
                                conformer={'dim':1024, 'depth': 12, 'heads':8, 'dim_head': 128, 'attn_flash': False},
                                )

    soundstorm = SoundStorm(net=conformer,
                            num_semantic_token_ids=1024,
                            semantic_pad_id=1024,
                            pad_id=1024,
                            schedule = 'cosine')

    # Initial parameters with codebooks of SpeechTokenizer
    sp_params = '/remote-home/xzhang/Speech/SpeechTokenizer/ckpt/SpeechTokenizer.pt'
    sp_params = torch.load(sp_params, map_location='cpu')
    soundstorm.semantic_token_emb.weight = torch.nn.Parameter(sp_params['quantizer.vq.layers.0._codebook.embed'])
    acoustic_embeds = []
    for i in range(1, 8):
        acoustic_embed = torch.cat([sp_params[f'quantizer.vq.layers.{i}._codebook.embed'], torch.zeros(1,1024)], axis=0)
        acoustic_embeds.append(acoustic_embed)
    acoustic_embeds = torch.cat(acoustic_embeds, axis=0)
    soundstorm.net.code_embeds.weight = torch.nn.Parameter(acoustic_embeds)


    # Tokens file
    train_file_list = '/path/train_file_list.txt'
    with open(train_file_list, 'r') as f:
        train_file_list = f.readlines()
    valid_file_list = '/path/valid_file_list.txt'
    with open(valid_file_list, 'r') as f:
        valid_file_list = f.readlines()

    result_folder = './Log/result'

    # Initial tokenizer
    # st_cfg = '/path/config.json'
    # st_ckpt = '/path/SpeechTokenizer.pt'  
    # tokenizer = SpeechTokenizer.load_from_checkpoint(st_cfg, st_ckpt)
    # tokenizer.eval()
    tokenizer = None

    trainer = SoundStormTrainer(model=soundstorm,
                                num_warmup_steps=4000,
                                batch_size=8,
                                epochs=50,
                                train_file_list=train_file_list,
                                valid_file_list=valid_file_list,
                                is_raw_wav=False,
                                is_tokens=True,
                                max_sequence=750,
                                tokenizer=tokenizer,
                                lr=6e-4,
                                initial_lr=3e-5,
                                # lr=3e-4,
                                # initial_lr=3e-5,
                                grad_accum_every=2,
                                log_steps=10,
                                save_model_steps=5000,
                                results_folder=result_folder,
                                accelerate_kwargs={'log_with':"tensorboard", 'project_dir':f'{result_folder}'},
                                num_workers=8)
    trainer.train()