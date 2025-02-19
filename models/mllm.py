from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch.nn as nn
import torch
from models.training_utils import *
import numpy as np
import models.vqvae as vqvae

class MotionLLM(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.llm_backbone)
        self.llm = AutoModelForCausalLM.from_pretrained(self.args.llm_backbone)
        self.nb_text_tokens = len(self.tokenizer)
        self.mean = np.load('checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy')
        self.std = np.load('checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy')
        self.device = args.device

        self.lora_config_t2m = LoraConfig(
            r=self.args.lora_r_t2m,
            lora_alpha=self.args.lora_alpha_t2m,
            target_modules=['o_proj', 'q_proj', 'up_proj', 'v_proj', 'k_proj', 'down_proj', 'gate_proj'],
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        ) 
        self.lora_config_m2t = LoraConfig(
            r=self.args.lora_r_m2t,
            lora_alpha=self.args.lora_alpha_m2t,
            target_modules=['o_proj', 'q_proj', 'up_proj', 'v_proj', 'k_proj', 'down_proj', 'gate_proj'],
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.llm = get_peft_model(self.llm, self.lora_config_t2m, adapter_name='t2m')
        self.llm.add_adapter('m2t', self.lora_config_m2t)

        self.args.nb_joints = 22
        self.args.dataname = 't2m'
        self.args.vq_path = "ckpt/vqvae.pth"
        self.net = vqvae.HumanVQVAE(self.args, ## use args to define different parameters in different quantizers
                           self.args.nb_code,
                           self.args.code_dim,
                           self.args.output_emb_width,
                           self.args.down_t,
                           self.args.stride_t,
                           self.args.width,
                           self.args.depth,
                           self.args.dilation_growth_rate,
                           self.args.vq_act,
                           self.args.vq_norm)
        print ('loading vqvae from {}'.format(self.args.vq_path))
        ckpt = torch.load(self.args.vq_path, map_location='cpu')
        self.net.load_state_dict(ckpt['net'], strict=True)
        self.net.eval()
        self.net.to(self.device)

        self.tokenizer.add_tokens(['<Motion>', '</Motion>'])
        self.motion_token_indices = np.arange(self.args.nb_code) 
        self.motion_token_indices = len(self.tokenizer) + self.motion_token_indices
        for i in range(self.args.nb_code):
            self.tokenizer.add_tokens([f'<Motion_{i}>'])
        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.llm.to(self.device)
        self.llm.eval()

        # print(self.llm)
    
    def forward(self, caption, motion):

        inputs_ids, targets, attention_mask = process_batch(tokenizer=self.tokenizer, 
                                                            caption=caption, 
                                                            max_tgt_len=200, 
                                                            batch_of_motions=motion)
        
        # print(inputs_ids.shape)
        # print(targets.shape)
        # print(attention_mask.shape)
        # print(tokenizer.decode(inputs_ids[0]))
        inputs_ids = inputs_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        targets = targets.to(self.device)

        outputs = self.llm(
            input_ids=inputs_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            labels=targets,
        )

        loss = outputs.loss
        # print(outputs.logits.shape)
        # calculate the token accuracy
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]  # [B, S-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)  # [B*S]
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask  # [B*S]
        gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item() + 1.0)
        return loss, gen_acc, chosen_tokens, labels
    
    def generate(self, caption):
        self.llm.set_adapter('t2m')
        self.llm.eval()
        prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        instruction = "### Instruction:\nGenerate a motion matching the following input human motion description\n\n"
        input_text = '### Input:\n' + caption + '\n\nResponse: <Motion>'
        input = prompt + instruction + input_text
        input_ids = self.tokenizer.encode(input, return_tensors="pt").to(self.device)
        outputs = self.llm.generate(
            input_ids, 
            max_length=200, 
            num_beams=2, 
            early_stopping=True, 
            return_dict_in_generate=True, 
            output_scores=True
        )

        scores = torch.stack(outputs.scores)  # [num_generated_tokens, num_beams, vocab_size]
        # print(scores.shape)
        # Take only the best beam (beam 0)
        best_beam_scores = scores[:, 0, :]  # [num_generated_tokens, vocab_size]
        motion_logits = best_beam_scores[:, -(self.args.nb_code+2):]
        # print(motion_logits.shape)
        motion_tokens = torch.argmax(motion_logits, dim=-1)  # [num_generated_tokens]
        # print(motion_tokens)
        # Remove end_of_motion token (index=1) if present
        if 1 in motion_tokens:
            motion_tokens = motion_tokens[:motion_tokens.tolist().index(1)]
        # Ensure tokens don't go below 0 when adjusting for special tokens
        motion_tokens = torch.clamp(motion_tokens - 2, min=0)  # remove the first two special tokens while preventing negative values
        
        # print(motion_tokens)
        return motion_tokens
    
    def caption(self, motion):
        self.llm.set_adapter('m2t')
        self.llm.eval()
        motion = self.normalize(motion)
        motion = torch.from_numpy(motion).float().to(self.device).unsqueeze(0)
        motion_tokens = self.net.encode(motion).squeeze(0)
        motion_tokens = motion_tokens + self.nb_text_tokens + 2 # reindex the motion tokens
        # print(motion_tokens)

        prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        instruction = "### Instruction:\nGenerate a caption matching the following input human motion token sequence.\n\n"
        input_text = '### Input:\n' + "<Motion>" + self.tokenizer.decode(motion_tokens) + '</Motion>' + '\n\nResponse: '
        input_texts = prompt + instruction + input_text
        # print(input_texts)
        input_ids = self.tokenizer.encode(input_texts, return_tensors="pt").to(self.device)
        pred = self.llm.generate(
            input_ids, 
            max_length=200, 
            num_beams=2
        )
        pred = pred[0, len(input_ids[0]):]
        pred = self.tokenizer.decode(pred)
        caption = pred.split('<eos>')[0]

        return caption
    
    def save_model(self, path):
        # only save the lora weights of the model
        save_dict = {}
        for name, param in self.llm.named_parameters():
            if 'lora' in name:
                save_dict[name] = param

        # save the additional token embeddings
        embeddings = self.llm.get_input_embeddings().weight[self.nb_text_tokens:]
        save_dict['embeddings'] = embeddings

        # save the lm_head of the additional tokens
        lm_head = self.llm.lm_head.weight[self.nb_text_tokens:]
        save_dict['lm_head'] = lm_head

        torch.save(save_dict, path)

    def load_model(self, path):
        print(f"Loading model from {path}")
        save_dict = torch.load(path, map_location=self.device)
        for name, param in self.llm.named_parameters():
            # print(name)
            if name in save_dict:
                param.data = save_dict[name]
        self.llm.get_input_embeddings().weight.data[self.nb_text_tokens:] = save_dict['embeddings']
        self.llm.lm_head.weight.data[self.nb_text_tokens:] = save_dict['lm_head']

    def denormalize(self, motion):
        return self.mean + motion * self.std

    def normalize(self, motion):
        return (motion - self.mean) / self.std
    
