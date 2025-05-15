import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ## device
    parser.add_argument('--device', type=str, default='cuda:0', help='device')

    ## MotionLLM training
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--batch-size', type=int, default=6, help='batch size')
    parser.add_argument('--epochs-t2m', type=int, default=500, help='number of epochs for t2m')
    parser.add_argument('--epochs-m2t', type=int, default=10, help='number of epochs for m2t')
    parser.add_argument('--training-task', type=str, default='t2m', help='training task, t2m or m2t')
    parser.add_argument('--epochs-start-val', type=int, default=70, help='number of epochs to start validation')
    parser.add_argument('--epochs-val-interval', type=int, default=3, help='number of epochs between validation')

    ## LLM 
    parser.add_argument('--llm-backbone', type=str, default='google/gemma-2-2b-it', help='name of huggingface model backbone')
    parser.add_argument('--lora-r-t2m', type=int, default=64, help='lora_r for t2m')
    parser.add_argument('--lora-alpha-t2m', type=int, default=64, help='lora_alpha for t2m')
    parser.add_argument('--lora-r-m2t', type=int, default=32, help='lora_r for m2t')
    parser.add_argument('--lora-alpha-m2t', type=int, default=32, help='lora_alpha for m2t')
    parser.add_argument('--lora-dropout', type=float, default=0.1, help='lora_dropout')

    ## dataloader  
    parser.add_argument('--dataname', type=str, default='t2m', help='dataset directory')
    
    ## vqvae arch
    parser.add_argument("--code-dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=512, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')
    parser.add_argument('--vq-norm', type=str, default=None, help='dataset directory')
    
    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    parser.add_argument('--beta', type=float, default=1.0, help='commitment loss in standard VQ')
    
    
    ## output directory 
    parser.add_argument('--out-dir', type=str, default='experiments', help='output directory')
    parser.add_argument('--exp-name', type=str, default='test', help='name of the experiment, will create a file inside out-dir')
    
    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=1000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')
    
    
    return parser.parse_args()