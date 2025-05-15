from models.mllm import MotionLLM
import json
import torch
from utils.motion_utils import recover_from_ric, plot_3d_motion
from utils.paramUtil import t2m_kinematic_chain
import time
import os
import numpy as np


class MotionAgent:
    def __init__(self, args, client):
        self.args = args
        self.device = args.device
        self.model = MotionLLM(self.args)
        self.model.load_model('ckpt/motionllm.pth')
        self.model.eval()
        self.model.to(self.device)
        self.client = client
        self.save_dir = args.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # self.if_replace_each_turn = args.if_replace_each_turn
        self.context = []
        self.motion_history = {}

        print("Loading example prompt from example_prompt.txt, feel free to use your own prompt")
        self.prompt = open("example_prompt.txt", "r").read() # loading the example prompt, feel free to use your own prompt
        self.context.append({"role": "system", "content": self.prompt})

    def process_motion_dialogue(self, message):
        # if the message contains 'npy', it means the user wants to reason on a motion
        if 'npy' in message:
            motion_file = message.split(' ')[-1]
            # print(motion_file)
            assert motion_file.endswith('.npy'), "The file must be a npy file and should be the last word of your message"
            message = message.replace(motion_file, '<motion_file>') # replace the motion file with a placeholder
            motion_input = np.load(motion_file)
            # print(message)

        # Update context with the new message
        self.context.append({"role": "user", "content": message})
        
        # Create chat completion request
        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=self.context
        )
        
        # Extract and store the assistant's response
        assistant_response = response.choices[0].message.content
        # print(assistant_response)
        self.context.append({"role": "assistant", "content": assistant_response})

        # parse the assistant's response to get the plan
        try:
            plan = json.loads(assistant_response)["plan"]
        except:
            plan = None

        try:
            reasoning = json.loads(assistant_response)["reasoning"]
        except:
            reasoning = None

        # if the plan is not None, it means the user wants to generate a motion or reason on a motion
        if plan is not None:
            if "generate" in plan:
                motion_tokens_to_generate = [] # list of motion tokens to generate

                descriptions = plan.split(";")
                for description in descriptions:
                    description = description.strip()
                    if description:
                        description = description.split("MotionLLM.generate('")[1].rstrip("');")
                        # print(description)
                        if description not in self.motion_history:
                            motion_tokens = self.model.generate(description)
                            self.motion_history[description] = motion_tokens
                            motion_tokens_to_generate.append(motion_tokens)
                        else:
                            motion_tokens_to_generate.append(self.motion_history[description])

                # print(self.motion_history)
                motion_tokens = torch.cat(motion_tokens_to_generate)
                motion = self.model.net.forward_decoder(motion_tokens)
                motion = self.model.denormalize(motion.detach().cpu().numpy())
                motion = recover_from_ric(torch.from_numpy(motion).float().to(self.device), 22)
                filename = f"{self.save_dir}/motion_{int(time.time())}.mp4"
                print('Plotting motion...')
                plot_3d_motion(filename, t2m_kinematic_chain, motion.squeeze().detach().cpu().numpy(), title=message, fps=20, radius=4)
                np.save(f"{self.save_dir}/motion_{int(time.time())}.npy", motion.squeeze().detach().cpu().numpy())
                print(f"Motion saved to {filename}")

            elif 'caption' in plan:
                caption = self.model.caption(motion_input)
                # caption = 'A person is walking.' # TODO: remove this
                new_message = f"MotionLLM: '{caption}'"
                self.process_motion_dialogue(new_message)

            else:
                raise ValueError(f"Invalid format of the assistant's response: {assistant_response}")

        # if the reasoning is not None, it means the model is reasoning on a motion
        elif reasoning is not None:
            print(reasoning)

        else:
            raise ValueError(f"Invalid format of the assistant's response: {assistant_response}")
        
    def clean(self):
        self.context = []
        self.motion_history = {}
        self.context.append({"role": "system", "content": self.prompt})
        print("Cleaned up the context and motion history.")

    def chat(self):
        print("Welcome to Motion-Agent! Type 'exit' to quit.")
        print("Generate a motion: directly type your prompt.")
        print("Reason on a motion: type the file name of the npy motion file you want to reason on after your question.")
        print("Clean the context and motion history: type 'clean'.")
        while True:
            message = input("User: ")
            if message == "exit":
                break
            if message == "clean":
                self.clean()
                continue
            try:
                self.process_motion_dialogue(message)
            except Exception as e:
                print(f"Error: {e}")
        