from os.path import join as pjoin
from torch.utils.data import Dataset, DataLoader
from dataset.interhuman import InterHumanDataset
# from models import *
import copy
from dataset.evaluator_models import InterCLIP
from tqdm import tqdm
import torch
import random
import numpy as np
import torch.nn as nn

def generate_one_motion_IG_rvq(vq_model, gemma_model, res_model, tokenizer, input, codebook_emb):
    
    def find_closest_tokens(net, res_model, input, embeddings):
        converted_embeddings = []
        for i in range(embeddings.shape[0]):
            loss = []
            for j in range(codebook_emb.shape[0]):
                loss.append(nn.functional.mse_loss(embeddings[i:i+1, :], codebook_emb[j:j+1, :]))
            loss = torch.stack(loss)
            converted_embeddings.append(torch.argmin(loss).item())
        mids = torch.tensor(converted_embeddings).cuda()
        mids = mids.unsqueeze(0)
        length = torch.LongTensor([mids.shape[-1]]).cuda()
        mids = res_model.generate(mids, [input], length, temperature=1, cond_scale=5)
        return net.forward_decoder(mids)

    prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
    instruction = "### Instruction:\nGenerate motions of two people interacting matching the following input human motions description. Two people's motion should be divided using <Divide>.\n\n"
    input_text = '### Input:\n' + input + '\n\nResponse: <Motion>'
    input_texts = prompt + instruction + input_text
    input_ids = tokenizer.encode(input_texts, return_tensors="pt").cuda()
    motion = gemma_model.generate(input_ids, max_length=300, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    # print(motion.shape)
    motion = motion[0, len(input_ids[0]):]
    print(tokenizer.decode(motion))
    div_id = tokenizer.encode('<Divide>', return_tensors="pt", add_special_tokens=False).item()
    eom_id = tokenizer.encode('</Motion>', return_tensors="pt", add_special_tokens=False).item()
    if eom_id in motion.tolist():
        motion = motion[:motion.tolist().index(eom_id)]
    if div_id in motion.tolist():
        motion1 = motion[:motion.tolist().index(div_id)]
        motion2 = motion[motion.tolist().index(div_id)+1:]
    else:
        motion1 = motion[:len(motion)//2]
        motion2 = motion[len(motion)//2:]
    if len(motion2) == 0:
        motion2 = motion1[:len(motion1)//2]
        motion1 = motion1[len(motion1)//2:]
    if len(motion1) == 0:
        motion1 = motion2[:len(motion2)//2]
        motion2 = motion2[len(motion2)//2:]
    if len(motion1) < len(motion2):
        motion2 = motion2[:len(motion1)]
    elif len(motion1) > len(motion2):
        motion1 = motion1[:len(motion2)]
    assert len(motion1) == len(motion2)
    
    motion1 = gemma_model.model.model.embed_tokens(motion1)
    motion2 = gemma_model.model.model.embed_tokens(motion2)
    motion1 = find_closest_tokens(vq_model, res_model, input, motion1)
    motion2 = find_closest_tokens(vq_model, res_model, input, motion2)

    return motion1, motion2

def generate_one_motion_IG_rvq_v1_1(vq_model, gemma_model, res_model, tokenizer, input, codebook_emb):
    
    def find_closest_tokens(net, res_model, input, embeddings):
        converted_embeddings = []
        for i in range(embeddings.shape[0]):
            loss = []
            for j in range(codebook_emb.shape[0]):
                loss.append(nn.functional.mse_loss(embeddings[i:i+1, :], codebook_emb[j:j+1, :]))
            loss = torch.stack(loss)
            converted_embeddings.append(torch.argmin(loss).item())
        mids = torch.tensor(converted_embeddings).cuda()
        mids = mids.unsqueeze(0)
        length = torch.LongTensor([mids.shape[-1]]).cuda()
        mids = res_model.generate(mids, [input], length, temperature=1, cond_scale=5)
        return net.forward_decoder(mids)

    prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
    instruction = "### Instruction:\nGenerate motions of two interacting humans matching the following input human motions description. Two people's motion should be divided using a comma.\n\n"
    input_text = '### Input:\n' + input + '\n\nResponse: '
    input_texts = prompt + instruction + input_text
    input_ids = tokenizer.encode(input_texts, return_tensors="pt").cuda()
    motion = gemma_model.generate(input_ids, max_length=300, num_beams=5)
    # print(motion.shape)
    motion = motion[0, len(input_ids[0]):]
    # print(tokenizer.decode(motion))
    div_id = tokenizer.encode(',', return_tensors="pt", add_special_tokens=False).item()
    eos_id = tokenizer.eos_token_id
    if eos_id in motion.tolist():
        motion = motion[:motion.tolist().index(eos_id)]
    if div_id in motion.tolist():
        motion1 = motion[:motion.tolist().index(div_id)]
        motion2 = motion[motion.tolist().index(div_id)+1:]
    else:
        motion1 = motion[:len(motion)//2]
        motion2 = motion[len(motion)//2:]
    if len(motion2) == 0:
        motion2 = motion1[:len(motion1)//2]
        motion1 = motion1[len(motion1)//2:]
    if len(motion1) == 0:
        motion1 = motion2[:len(motion2)//2]
        motion2 = motion2[len(motion2)//2:]
    if len(motion1) < len(motion2):
        motion2 = motion2[:len(motion1)]
    elif len(motion1) > len(motion2):
        motion1 = motion1[:len(motion2)]
    assert len(motion1) == len(motion2)
    
    motion1 = gemma_model.model.model.embed_tokens(motion1)
    motion2 = gemma_model.model.model.embed_tokens(motion2)
    motion1 = find_closest_tokens(vq_model, res_model, input, motion1)
    motion2 = find_closest_tokens(vq_model, res_model, input, motion2)

    return motion1, motion2

def generate_one_motion_IG_rvq_v3(vq_model, gemma_model, res_model, tokenizer, input, codebook_emb):
    
    def find_closest_tokens(net, res_model, input, embeddings):
        converted_embeddings = []
        for i in range(embeddings.shape[0]):
            loss = []
            for j in range(codebook_emb.shape[0]):
                loss.append(nn.functional.mse_loss(embeddings[i:i+1, :], codebook_emb[j:j+1, :]))
            loss = torch.stack(loss)
            converted_embeddings.append(torch.argmin(loss).item())
        mids = torch.tensor(converted_embeddings).cuda()
        mids = mids.unsqueeze(0)
        length = torch.LongTensor([mids.shape[-1]]).cuda()
        mids = res_model.generate(mids, [input], length, temperature=1, cond_scale=5)
        return net.forward_decoder(mids)

    prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
    instruction = "### Instruction:\nGenerate a motion of the first person matching the following input two humans motion description. \n\n"
    input_text = '### Input:\n' + "Description: " + input + '\n\nResponse: <Motion>'
    input_texts = prompt + instruction + input_text
    input_ids = tokenizer.encode(input_texts, return_tensors="pt").cuda()
    motion1 = gemma_model.generate(input_ids, max_length=300, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    motion1 = motion1[0, len(input_ids[0]):]
    print(tokenizer.decode(motion1))
    eom_id = tokenizer.encode('</Motion>', return_tensors="pt", add_special_tokens=False).item()
    if eom_id in motion1.tolist():
        motion1 = motion1[:motion1.tolist().index(eom_id)]

    instruction = "### Instruction:\nGenerate a motion of the second person matching the following input two humans motion description and the motion of the first person. \n\n"
    input_text = '### Input:\n' + "Description: " + input + "\nMotion of the first person: <Motion>" + tokenizer.decode(motion1) + '</Motion>\n\nResponse: <Motion>'
    input_texts = prompt + instruction + input_text
    print(input_texts)
    input_ids = tokenizer.encode(input_texts, return_tensors="pt").cuda()
    motion2 = gemma_model.generate(input_ids, max_length=300, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    motion2 = motion2[0, len(input_ids[0]):]
    print(tokenizer.decode(motion2))
    eom_id = tokenizer.encode('</Motion>', return_tensors="pt", add_special_tokens=False).item()
    if eom_id in motion2.tolist():
        motion2 = motion2[:motion2.tolist().index(eom_id)]

    if len(motion2) == 0:
        motion2 = motion1
    if len(motion1) < len(motion2):
        motion2 = motion2[:len(motion1)]
    elif len(motion1) > len(motion2):
        motion1 = motion1[:len(motion2)]
    assert len(motion1) == len(motion2)
    
    motion1 = gemma_model.model.model.embed_tokens(motion1)
    motion2 = gemma_model.model.model.embed_tokens(motion2)
    motion1 = find_closest_tokens(vq_model, res_model, input, motion1)
    motion2 = find_closest_tokens(vq_model, res_model, input, motion2)

    return motion1, motion2

class EvaluationDataset(Dataset):

    def __init__(self, model, dataset, device, mm_num_samples, mm_num_repeats, res_model, gemma_model, tokenizer, token_trans):
        # self.normalizer = MotionNormalizer()
        self.model = model.to(device)
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)
        self.max_length = dataset.max_length

        idxs = list(range(len(dataset)))
        random.shuffle(idxs)
        mm_idxs = idxs[:mm_num_samples]

        generated_motions = []
        mm_generated_motions = []
        # Pre-process all target captions
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                name, text, motion1, motion2, motion_lens = data
                batch = {}
                if i in mm_idxs:
                    batch["text"] = list(text) * mm_num_repeats
                else:
                    batch["text"] = list(text)
                batch["motion_lens"] = motion_lens

                # print(batch)
                    # batch = self.model.forward_test(motion1, motion2, motion_lens)
                # if i in mm_idxs:
                #     batches = []
                #     for j in range(mm_num_repeats):
                #         batches.append(self.model.forward_test(motion1, motion2, motion_lens)['output'].squeeze(0))
                #     batches = torch.stack(batches)
                # else:
                # batch = self.model.forward_test(motion1, motion2, motion_lens)
                # batches = batch['output']
                max_len = motion1.shape[1]
                codebook_emb = gemma_model.model.model.embed_tokens(torch.tensor(token_trans).cuda())
                motion1, motion2 = generate_one_motion_IG_rvq(self.model, gemma_model, res_model, tokenizer, text[0], codebook_emb)
                # print(motion1.shape)
                if motion1.shape[1] > max_len:
                    motion1 = motion1[:,:max_len]
                    motion2 = motion2[:,:max_len]
                # print(motion1.shape)
                # batch = {}
                batches = torch.cat([motion1, motion2], dim=-1)
                # print(batch["output"].shape)
                # print(batches.shape)
                motions_output = batches.reshape(batches.shape[0], batches.shape[1], 2, -1).cpu()
                # motions_output = self.normalizer.backward(motions_output.cpu().detach().numpy())

                # motions_output[..., :22 * 3] = filters.gaussian_filter1d(motions_output[..., :22 * 3], 1, axis=0, mode='nearest')
                # motions_output[..., 22 * 3:22 * 6] = filters.gaussian_filter1d(motions_output[..., 22 * 3:22 * 6], 0.1, axis=0, mode='nearest')
                # motions_output[..., 22 * 6:22 * 6 + 21 * 6] = filters.gaussian_filter1d(motions_output[..., 22 * 6:22 * 6 + 21 * 6], 0.5, axis=0, mode='nearest')

                B,T = motions_output.shape[0], motions_output.shape[1]
                if T < self.max_length:
                    padding_len = self.max_length - T
                    D = motions_output.shape[-1]
                    # padding_zeros = np.zeros((B, padding_len, 2, D))
                    # motions_output = np.concatenate((motions_output, padding_zeros), axis=1)
                    padding_zeros = torch.zeros((B, padding_len, 2, D))
                    motions_output = torch.cat([motions_output, padding_zeros], dim=1)
                assert motions_output.shape[1] == self.max_length


                sub_dict = {'motion1': motions_output[0, :,0],
                            'motion2': motions_output[0, :,1],
                            'motion_lens': motion_lens[0],
                            'text': text[0]}
                generated_motions.append(sub_dict)
                if i in mm_idxs:
                    mm_sub_dict = {'mm_motions': motions_output,
                                   'motion_lens': motion_lens[0],
                                    'text': text[0]}
                    mm_generated_motions.append(mm_sub_dict)


        self.generated_motions = generated_motions
        self.mm_generated_motions = mm_generated_motions

    def __len__(self):
        return len(self.generated_motions)

    def __getitem__(self, item):
        data = self.generated_motions[item]
        motion1, motion2, motion_lens, text = data['motion1'], data['motion2'], data['motion_lens'], data['text']
        return "generated", text, motion1, motion2, motion_lens


class MMGeneratedDataset(Dataset):
    def __init__(self, motion_dataset):
        self.dataset = motion_dataset.mm_generated_motions

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        mm_motions = data['mm_motions']
        motion_lens = data['motion_lens']
        mm_motions1 = mm_motions[:,:,0]
        mm_motions2 = mm_motions[:,:,1]
        text = data['text']
        motion_lens = np.array([motion_lens]*mm_motions1.shape[0])
        return "mm_generated", text, mm_motions1, mm_motions2, motion_lens


def get_dataset_motion_loader(opt, batch_size):
    opt = copy.deepcopy(opt)
    # Configurations of T2M dataset and KIT dataset is almost the same
    if opt.NAME == 'interhuman':
        print('Loading dataset %s ...' % opt.NAME)

        dataset = InterHumanDataset(opt)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=True)
    else:
        raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset




def get_motion_loader(batch_size, model, ground_truth_dataset, device, mm_num_samples, mm_num_repeats, res_model, gemma_model, tokenizer, token_trans):
    # Currently the configurations of two datasets are almost the same
    dataset = EvaluationDataset(model, ground_truth_dataset, device, mm_num_samples=mm_num_samples, \
                                mm_num_repeats=mm_num_repeats, res_model=res_model, gemma_model=gemma_model, 
                                tokenizer=tokenizer, token_trans=token_trans)
    mm_dataset = MMGeneratedDataset(dataset)

    motion_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=0, shuffle=True)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=0)

    print('Generated Dataset Loading Completed!!!')

    return motion_loader, mm_motion_loader




def build_models(cfg):
    model = InterCLIP(cfg)

    checkpoint = torch.load(pjoin('../T2M-GPT/eval_model/interclip.ckpt'),map_location="cpu")
    # checkpoint = torch.load(pjoin('checkpoints/interclip/model/5.ckpt'),map_location="cpu")
    for k in list(checkpoint["state_dict"].keys()):
        if "model" in k:
            checkpoint["state_dict"][k.replace("model.", "")] = checkpoint["state_dict"].pop(k)
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    # print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return model


class EvaluatorModelWrapper(object):

    def __init__(self, cfg, device):

        self.model = build_models(cfg)
        self.cfg = cfg
        self.device = device

        self.model = self.model.to(device)
        self.model.eval()


    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, batch_data):
        with torch.no_grad():
            name, text, motion1, motion2, motion_lens = batch_data
            motion1 = motion1.detach().float()  # .to(self.device)
            motion2 = motion2.detach().float()  # .to(self.device)
            motions = torch.cat([motion1, motion2], dim=-1)
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(motion_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            motion_lens = motion_lens[align_idx]
            text = list(text)

            B, T = motions.shape[:2]
            cur_len = torch.LongTensor([min(T, m_len) for m_len in motion_lens]).to(self.device)
            padded_len = cur_len.max()

            batch = {}
            batch["text"] = text
            batch["motions"] = motions.reshape(B, T, -1)[:, :padded_len]
            batch["motion_lens"] = motion_lens

            '''Motion Encoding'''
            motion_embedding = self.model.encode_motion(batch)['motion_emb']

            '''Text Encoding'''
            text_embedding = self.model.encode_text(batch)['text_emb'][align_idx]

        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, batch_data):
        with torch.no_grad():
            name, text, motion1, motion2, motion_lens = batch_data
            motion1 = motion1.detach().float()  # .to(self.device)
            motion2 = motion2.detach().float()  # .to(self.device)
            motions = torch.cat([motion1, motion2], dim=-1)
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(motion_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            motion_lens = motion_lens[align_idx]
            text = list(text)

            B, T = motions.shape[:2]
            cur_len = torch.LongTensor([min(T, m_len) for m_len in motion_lens]).to(self.device)
            padded_len = cur_len.max()

            batch = {}
            batch["text"] = text
            batch["motions"] = motions.reshape(B, T, -1)[:, :padded_len]
            batch["motion_lens"] = motion_lens

            '''Motion Encoding'''
            motion_embedding = self.model.encode_motion(batch)['motion_emb']

        return motion_embedding
