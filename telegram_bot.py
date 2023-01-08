import os
import argparse
from dotenv import load_dotenv
from telegram import Bot
from telegram.ext import Updater, MessageHandler, Filters
import logging
import io
import time

from stable_diffusion import StableDiffusion

load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
    )

logger = logging.getLogger(__name__)

class TelegramBot():
    def __init__(self, token, filter_chat_id):
        self.token = token
        self.bot = Bot(token=token)
        self.filter_chat_id = filter_chat_id
        self.updater = Updater(token=token, use_context=True)
        self.dispatcher = self.updater.dispatcher

    def process_message(self, update, context):
        if self.filter_chat_id:
            if str(update.message.chat.id) == TELEGRAM_CHAT_ID:
                update.message.reply_text('Working on it right now!')

                t1 = time.time()
                img = self.generate(prompt=update.message.text)
                t2 = time.time() - t1

                # Send image and reply
                self.bot.send_photo(
                chat_id=update.message.chat.id,
                photo=img
                )
                update.message.reply_text('Image generation took {} seconds!'.format(str(int(t2))))
            else:
                update.message.reply_text('You are not authorised!')
        else:
            update.message.reply_text('Working on it right now!')
            
            t1 = time.time()
            img = self.generate(prompt=update.message.text)
            t2 = time.time() - t1

            # Send image and reply
            self.bot.send_photo(
            chat_id=update.message.chat.id,
            photo=img
            )
            update.message.reply_text('Image generation took {} seconds!'.format(str(int(t2))))
    
    def generate(self, prompt):
        # Generate prompt from text
        samples = generator.generate(prompt=prompt)
        grid = generator.save_grid(samples)

        # Convert PIL Image to Bytes object
        img_byte_arr = io.BytesIO()
        grid.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        return img_byte_arr

    def error(self, update, context):
        logger.warning('Update "%s" caused error "%s"', update, context.error)

    def start_bot(self):
        self.dispatcher.add_handler(MessageHandler(Filters.text, self.process_message))
        self.dispatcher.add_error_handler(self.error)
        self.updater.start_polling()
        self.updater.idle()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v2-inference.yaml",
        help="path to config which constructs model",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model",
    )

    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    opt = parse_args()

    # Initialise Telegram Bot
    token = TELEGRAM_TOKEN

    filter_chat_id = False
    if TELEGRAM_CHAT_ID is not None:
        filter_chat_id = True

    bot = TelegramBot(
        token=token, 
        filter_chat_id=filter_chat_id
        )

    # Initialise Stable Diffusion Image Generator
    generator = StableDiffusion(
        config=opt.config,
        ckpt=opt.ckpt
    )

    bot.start_bot()
    