"""
Telegram bot interface for the ME engine.

Uses python-telegram-bot v21 async Application API to provide
a Telegram interface to the existing Engine in me.py.

Usage:
1. Install dependencies: pip install -r requirements.txt
2. Set up environment:
   - Copy .env.example to .env
   - Set TELEGRAM_TOKEN=your_bot_token_from_BotFather
3. Run the bot: python merobot.py

For Railway deployment:
- Set TELEGRAM_TOKEN environment variable in Railway dashboard
- Deploy this repository directly to Railway
- The bot will automatically start via python merobot.py
"""

import asyncio
import logging
import os

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from me import Engine
from me_predict import predict_next

# Configure minimal logging for Railway
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class MeBot:
    """Telegram bot wrapper for the ME engine."""
    
    def __init__(self):
        """Initialize the bot with the ME engine."""
        self.engine = Engine()
        self.predictions: dict[int, asyncio.Task] = {}
        logger.info("ME engine initialized")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /start command."""
        await update.message.reply_text(
            "Hello! I'm the ME bot. Send me any message and I'll respond using the ME engine."
        )
        logger.info(f"Start command from user {update.effective_user.id}")
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /help command."""
        help_text = (
            "ME Bot - Method Engine Interface\n\n"
            "Send me any text message and I'll generate a response using the ME engine.\n"
            "Commands:\n"
            "/start - Start the bot\n"
            "/help - Show this help message"
        )
        await update.message.reply_text(help_text)
        logger.info(f"Help command from user {update.effective_user.id}")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming text messages."""
        if not update.message or not update.message.text:
            return
        
        user_message = update.message.text
        user_id = update.effective_user.id
        
        logger.info(f"Message from user {user_id}: {user_message[:50]}...")

        # Check for a previous prediction
        previous_task = self.predictions.get(user_id)
        if previous_task:
            if previous_task.done():
                try:
                    predicted_text = previous_task.result()
                    logger.info(
                        f"Previous prediction for user {user_id}: {predicted_text[:50]}..."
                    )
                    if predicted_text.strip().lower() == user_message.strip().lower():
                        logger.info(f"Prediction matched actual message for user {user_id}")
                except Exception as e:
                    logger.warning(f"Previous prediction failed: {e}")
                finally:
                    self.predictions.pop(user_id, None)
            else:
                logger.info(f"Prediction for user {user_id} not ready yet")

        try:
            # Generate reply using the ME engine
            reply = self.engine.reply(user_message)

            # Send the reply
            await update.message.reply_text(reply)
            logger.info(f"Replied to user {user_id}: {reply[:50]}...")

            # Generate next message prediction in the background
            task = asyncio.create_task(asyncio.to_thread(predict_next, user_message, reply))
            self.predictions[user_id] = task

        except Exception as e:
            logger.error(f"Error processing message from user {user_id}: {e}")
            await update.message.reply_text("Sorry, I encountered an error processing your message.")
    
    async def handle_non_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle non-text messages (photos, stickers, etc.)."""
        if update.message:
            user_id = update.effective_user.id
            logger.info(f"Non-text message from user {user_id}, ignoring")
            await update.message.reply_text("I can only respond to text messages.")

def create_application(token: str) -> Application:
    """Create and configure the Telegram application."""
    app = Application.builder().token(token).build()
    
    bot = MeBot()
    
    # Add handlers
    app.add_handler(CommandHandler("start", bot.start_command))
    app.add_handler(CommandHandler("help", bot.help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))
    app.add_handler(MessageHandler(~filters.TEXT, bot.handle_non_text))
    
    logger.info("Telegram application configured")
    return app

def main():
    """Main entry point for the Telegram bot."""
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Get Telegram token from environment
    token = os.getenv('TELEGRAM_TOKEN')
    if not token:
        logger.error("TELEGRAM_TOKEN environment variable is required")
        print("Error: TELEGRAM_TOKEN environment variable must be set")
        print("Create a .env file with TELEGRAM_TOKEN=your_bot_token")
        exit(1)
    
    logger.info("Starting ME Telegram bot...")
    
    # Create and run the application
    app = create_application(token)
    
    try:
        logger.info("Bot is running. Press Ctrl+C to stop.")
        app.run_polling(allowed_updates=Update.ALL_TYPES)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot error: {e}")
        exit(1)

if __name__ == '__main__':
    main()
