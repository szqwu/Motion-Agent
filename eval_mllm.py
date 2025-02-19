from options.option_llm import get_args_parser
from models.mllm import MotionLLM
import torch
from utils.evaluation import evaluation_test
from dataset import dataset_TM_eval
from utils.word_vectorizer import WordVectorizer
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
import numpy as np
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def eval_t2m():
    args = get_args_parser()
    args.device = torch.device('cuda:0')
    model = MotionLLM(args)
    model.load_model('ckpt/motionllm.pth')

    w_vectorizer = WordVectorizer('../LLM-MotionGen/glove', 'our_vab')
    args.dataname = 't2m'
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    wrapper_opt = get_opt(dataset_opt_path, args.device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    test_loader = dataset_TM_eval.DATALoader(args.dataname, "test", 32, w_vectorizer, unit_length=2**args.down_t) # batch size should be fixed to 32

    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    matching = []
    multi = []
    repeat_time = 20
    for i in range(repeat_time):
        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_multi = evaluation_test(args.out_dir, test_loader, model, eval_wrapper=eval_wrapper, draw=False, savenpy=False)
        fid.append(best_fid)
        div.append(best_div)
        top1.append(best_top1)
        top2.append(best_top2)
        top3.append(best_top3)
        matching.append(best_matching)
        multi.append(best_multi)

    print('final result:')
    print('fid: ', sum(fid)/repeat_time)
    print('div: ', sum(div)/repeat_time)
    print('top1: ', sum(top1)/repeat_time)
    print('top2: ', sum(top2)/repeat_time)
    print('top3: ', sum(top3)/repeat_time)
    print('matching: ', sum(matching)/repeat_time)
    print('multi: ', sum(multi)/repeat_time)

    fid = np.array(fid)
    div = np.array(div)
    top1 = np.array(top1)
    top2 = np.array(top2)
    top3 = np.array(top3)
    matching = np.array(matching)
    multi = np.array(multi)
    msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}, Multi. {np.mean(multi):.3f}, conf. {np.std(multi)*1.96/np.sqrt(repeat_time):.3f}"
    print(msg_final)

if __name__ == "__main__":
    eval_t2m()