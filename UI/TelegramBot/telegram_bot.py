import asyncio
import logging
import os

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import Command
from pydantic import ValidationError

from dotenv import load_dotenv

from API.Routes.get_answer_route import (
    QueryPayload,
    input_guardrail,
    rag_service,
)

# Load environment variables
load_dotenv()

# ---------------- LOGGING SETUP ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ---------------- BOT SETUP ----------------
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

bot = Bot(token=BOT_TOKEN) if BOT_TOKEN else None
dp = Dispatcher()

# ---------------- HANDLERS ----------------


@dp.message(Command("start"))
async def start_handler(message: Message):
    logger.info(
        f"/start command received from "
        f"user_id={message.from_user.id}, "
        f"username={message.from_user.username}"
    )

    await message.answer("Hello! I am your minimal Telegram bot.")


@dp.message(F.text)
async def echo_handler(message: Message):
    logger.info(
        f"Message received from "
        f"user_id={message.from_user.id}, "
        f"text='{message.text}'"
    )

    try:
        q_payload = QueryPayload(query=message.text)
        clean_query, guardrail_triggered = input_guardrail.run_input_guardrails(
            q_payload.query
        )

        if guardrail_triggered:
            await message.answer(clean_query)
            return

        answer_payload = await rag_service.get_answer(
            query=clean_query,
            top_k=5,
        )

        if answer_payload.get("cache_hit"):
            logger.info(
                "Telegram query returned cached answer | user_id=%s | distance=%s",
                message.from_user.id,
                answer_payload.get("cache_distance"),
            )
        else:
            logger.info(
                "Telegram query generated fresh RAG answer | user_id=%s | chunks=%s",
                message.from_user.id,
                answer_payload.get("retrieved_chunks"),
            )

        await message.answer(
            answer_payload.get(
                "answer",
                "Sorry, I could not generate an answer.",
            )
        )

    except ValidationError as exc:
        logger.warning(
            "Validation failed for Telegram message: %s",
            exc,
        )
        await message.answer(
            "Please ask a longer question or provide more details."
        )

    except Exception:
        logger.exception("Unexpected error while processing Telegram message")
        await message.answer(
            "Sorry, I couldn't process your request right now."
        )


# ---------------- MAIN ----------------
async def start_telegram_bot():
    if not BOT_TOKEN:
        logger.warning(
            "TELEGRAM_BOT_TOKEN environment variable not set; skipping Telegram bot startup"
        )
        return None

    if bot is None:
        logger.error("Telegram bot instance is not initialized")
        return None

    logger.info("Starting Telegram bot polling in background...")
    task = asyncio.create_task(dp.start_polling(bot, skip_updates=True))
    return task


async def stop_telegram_bot(task):
    if task is None:
        return

    logger.info("Stopping Telegram bot polling...")
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        logger.info("Telegram polling task cancelled")
    except Exception:
        logger.exception("Error while stopping Telegram polling task")

    if bot is not None:
        await bot.session.close()


async def main():
    if not BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN environment variable not set")
        raise SystemExit("Set TELEGRAM_BOT_TOKEN environment variable")

    logger.info("Starting Telegram bot polling...")
    await dp.start_polling(bot, skip_updates=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Bot stopped manually")
    except Exception as e:
        logger.exception(f"Unexpected error occurred: {e}")
