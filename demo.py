from openai import AzureOpenAI
from models.motion_agent import MotionAgent
from models.mllm import MotionLLM
from options.option_llm import get_args_parser
from utils.motion_utils import recover_from_ric, plot_3d_motion
from utils.paramUtil import t2m_kinematic_chain
import torch

def motion_agent_demo():
    # Initialize the client
    client = AzureOpenAI(
        api_key="********", # your api key
        api_version="2024-10-21",
        azure_endpoint="********" # your azure endpoint
    )

    args = get_args_parser()
    args.save_dir = "./demo"
    args.device = 'cuda:0'

    motion_agent = MotionAgent(args, client)
    motion_agent.chat()

def motionllm_demo():
    model = MotionLLM(get_args_parser())
    model.load_model('ckpt/motionllm.pth')
    model.llm.eval()
    model.llm.cuda()
    
    caption = 'A man is doing cartwheels.'
    motion = model.generate(caption)
    motion = model.net.forward_decoder(motion)

    motion = model.denormalize(motion.detach().cpu().numpy())
    motion = recover_from_ric(torch.from_numpy(motion).float().cuda(), 22)
    print(motion.shape)
    plot_3d_motion(f"motionllm_demo.mp4", t2m_kinematic_chain, motion.squeeze().detach().cpu().numpy(), title=caption, fps=20, radius=4)

if __name__ == "__main__":
    motion_agent_demo()
