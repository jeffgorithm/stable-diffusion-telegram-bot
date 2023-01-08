# Stable Diffusion Telegram Bot
<p align="center">
    <img src="demo.gif" alt="demo"/>
</p>

Generate images from text prompts on the go with your own [Telegram Bot](https://github.com/python-telegram-bot/python-telegram-bot) and [Stable Diffusion](https://github.com/Stability-AI/stablediffusion) hosted on your own deep learning rig!

# Main Features
1. Telegram Bot that listens and replies to text prompts
2. Stable Diffusion for generating images from text prompts

# Getting Started

## Prerequisites

* [Anaconda](https://www.anaconda.com/)
* Compatible version of [CUDA](https://developer.nvidia.com/cuda-toolkit)

## Installation
1. Install dependencies with [Anaconda](https://www.anaconda.com/)
    ```
    conda env create -f environment.yml
    ```
2. Optional step: Follow the [official xFormers installation guide](https://github.com/facebookresearch/xformers#installing-xformers) for efficient GPU memory utilisation and inference speedup
3. Setup custom Telegram bot using [BotFather](https://t.me/botfather)
4. Create .env file in this project directory and paste your Telegram API Token into TELEGRAM_TOKEN
    ```
    TELEGRAM_TOKEN=INSERT_TELEGRAM_TOKEN_HERE
    ```
5. Download [Stable Diffusion 2.1 Base model weights](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) from Hugging Face
6. Start the bot
    ```
    python telegram_bot.py --config <config_path> --ckpt <model_path>

    # Example
    python telegram_bot.py --config configs/stable-diffusion/v2-inference.yaml --ckpt v2-1_512-ema-pruned.ckpt
    ```
7. Start generating images by sending a text prompt to your own Telegram bot!
8. Optional step: Your Telegram bot is public and searchable, to ensure that only you can issue text prompts and generate images, find out the Chat ID and add it to the .env file
    ```
    TELEGRAM_CHAT_ID=INSERT_TELEGRAM_CHAT_ID_HERE
    ```

# Acknowledgements
* [Official Implementation of Stable Diffusion by Stability AI](https://github.com/Stability-AI/stablediffusion)
* [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)
