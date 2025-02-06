# Motion-Agent: A Conversational Framework for Human Motion Generation with LLMs

<p align="center">
    <a href="https://arxiv.org/abs/2405.17013"><img alt='arXiv' src="https://img.shields.io/badge/arXiv-2405.17013-b31b1b.svg"></a>
    <a href="https://knoxzhao.github.io/Motion-Agent/"><img alt='page' src="https://img.shields.io/badge/Project-Page-orange"></a>
  </p>

## Overview

While previous approaches to 3D human motion generation have achieved notable success, they often rely on extensive training and are limited to specific tasks. To address these challenges, we introduce **Motion-Agent**, an efficient conversational framework designed for general human motion generation, editing, and understanding. Motion-Agent employs an open-source pre-trained language model to develop a generative agent, **MotionLLM**, that bridges the gap between motion and text. This is accomplished by encoding and quantizing motions into discrete tokens that align with the language model's vocabulary. With only 1-3\% of the model's parameters fine-tuned using adapters, MotionLLM delivers performance on par with diffusion models and other transformer-based methods trained from scratch. By integrating MotionLLM with GPT-4 without additional training, Motion-Agent is able to generate highly complex motion sequences through multi-turn conversations, a capability that previous models have struggled to achieve. Motion-Agent supports a wide range of motion-language tasks, offering versatile capabilities for generating and customizing human motion through interactive conversational exchanges.

## Updates

- [2025/02/06] Motion-Agent is accepted to ICLR 2025.
- [2024/10/08] Motion-Agent paper is available.
- [2024/05/28] Original version MotionLLM paper is available.

## Citation
If you find our work useful, please cite us. The BibTeX is as follows.
```
@inproceedings{
wu2025motionagent,
title={Motion-Agent: A Conversational Framework for Human Motion Generation with {LLM}s},
author={Qi Wu and Yubo Zhao and Yifan Wang and Xinhang Liu and Yu-Wing Tai and Chi-Keung Tang},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=AvOhBgsE5R}
}


