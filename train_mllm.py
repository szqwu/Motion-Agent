import torch
import numpy as np
from dataset import dataset_TM_eval
from utils.evaluation import evaluation_test
import os
from dataset import dataset_TM_eval
from utils.word_vectorizer import WordVectorizer
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
from models.mllm import MotionLLM
from options.option_train import get_args_parser
import logging
import json
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_logger(out_dir):
    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger

if __name__ == "__main__":
    args = get_args_parser()

    model = MotionLLM(args)
    model.train()

    # set up logging
    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
    os.makedirs(args.out_dir, exist_ok = True)
    logger = get_logger(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    # load dataset for evaluation
    w_vectorizer = WordVectorizer('../LLM-MotionGen/glove', 'our_vab')
    args.dataname = 't2m'
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    wrapper_opt = get_opt(dataset_opt_path, args.device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    val_loader = dataset_TM_eval.DATALoader(args.dataname, "val", 32, w_vectorizer, unit_length=2**args.down_t) # batch size should be fixed to 32
    # logger.info(f'Training on {args.dataname}')

    # load dataset for training
    train_loader = dataset_TM_eval.DATALoader(args.dataname, "train", args.batch_size, w_vectorizer, unit_length=2**args.down_t)

    # load optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.training_task == 't2m':
        # load the latest model
        # model.load_model(os.path.join(args.out_dir, f'motionllm_t2m_latest.pth'))

        # training loop
        model.training_task = 't2m'
        best_fid = 1000
        for epoch in range(args.epochs_t2m):
            batch_losses = []
            batch_accs = []
            for batch in train_loader:
                word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name = batch
                motion_tokens = []

                # encode the motion tokens
                for i in range(motion.shape[0]):
                    tokens = model.net.encode(motion[i:i+1, :m_length[i], :].to(args.device)).squeeze(0)
                    for j in range(tokens.shape[0]):
                        tokens[j] = model.motion_token_indices[tokens[j]]
                    motion_tokens.append(tokens)

                optimizer.zero_grad()
                loss, gen_acc, output, labels = model.forward(caption, motion_tokens)
                loss.backward()
                optimizer.step()
        
                batch_losses.append(loss.item())
                batch_accs.append(gen_acc)
                
            # print the loss and accuracy
            logger.info(f'Epoch {epoch}, Loss: {np.mean(batch_losses)}, Accuracy: {np.mean(batch_accs)}')

            # save the model
            model.save_model(os.path.join(args.out_dir, f'motionllm_t2m_latest.pth'))

            # validation
            if epoch > args.epochs_start_val and epoch % args.epochs_val_interval == 0:
                model.eval()
                fid, div, top1, top2, top3, matching, multi = evaluation_test(args.out_dir, val_loader, model, eval_wrapper=eval_wrapper, draw=False, savenpy=False)
                model.train()
                logger.info(f'Epoch [{epoch}/{args.epochs_t2m}], FID: {fid}, Div: {div}, Top1: {top1}, Top2: {top2}, Top3: {top3}, Matching: {matching}, Multi: {multi}')
                if fid < best_fid:
                    best_fid = fid
                    model.save_model(os.path.join(args.out_dir, f'motionllm_t2m_best.pth'))
                    logger.info(f'Best FID: {best_fid}')

    elif args.training_task == 'm2t':
        # IMPORTANT NOTICE: If you want two LoRAs to coexist in the final model, you need to make sure the embedding layer and lm_head are shared.
        # This can be done by loading the weights before the second LoRA is trained.
        model.load_model(os.path.join(args.out_dir, f'motionllm_t2m_best.pth'))

        # training loop
        model.training_task = 'm2t'
        for epoch in range(args.epochs_m2t):
            batch_losses = []
            batch_accs = []
            for batch in train_loader:
                word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name = batch
                motion_tokens = []

                # encode the motion tokens
                for i in range(motion.shape[0]):
                    tokens = model.net.encode(motion[i:i+1, :m_length[i], :].to(args.device)).squeeze(0)
                    for j in range(tokens.shape[0]):
                        tokens[j] = model.motion_token_indices[tokens[j]]
                    motion_tokens.append(tokens)

                optimizer.zero_grad()
                loss, gen_acc, output, labels = model.forward(caption, motion_tokens)
                loss.backward()
                optimizer.step()
        
                batch_losses.append(loss.item())
                batch_accs.append(gen_acc)
                
            # print the loss and accuracy
            logger.info(f'Epoch {epoch}, Loss: {np.mean(batch_losses)}, Accuracy: {np.mean(batch_accs)}')

            # save the model
            model.save_model(os.path.join(args.out_dir, f'motionllm.pth'))
