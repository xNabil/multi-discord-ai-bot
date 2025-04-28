# Discord Chat Bot by Nabil
# GitHub: https://github.com/xNabil
# Enhanced Version with Memory, Prioritized Replies, Slow Mode Fix, and Channel ID from .env

import os
import aiohttp
import random
import time
import asyncio
import google.generativeai as genai
import json
from dotenv import load_dotenv
from langdetect import detect
from dateutil.parser import isoparse
from colorama import init, Fore, Style
import logging

# Initialize colorama and logging
init()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load .env file
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHANNEL_ID = os.getenv("CHANNEL_ID")
SLOW_MODE = os.getenv("SLOW_MODE")

# Validate environment variables
if not DISCORD_TOKEN:
    logger.error("DISCORD_TOKEN is missing or empty in .env file.")
    exit(1)
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY is missing or empty in .env file.")
    exit(1)

# Define HEADERS for Discord API requests
HEADERS = {
    "Authorization": DISCORD_TOKEN,
    "Content-Type": "application/json"
}

# Gemini AI Configuration
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Global variables
CHANNEL_SLOW_MODES = {}
BOT_USER_ID = None
BOT_USERNAME = None
PERSONAL_INFO = {}
CONVERSATION_MEMORY = []

# Load personal info from myinfo.txt
def load_personal_info():
    try:
        with open("myinfo.txt", "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
            info = {}
            key = None
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#"):
                    if ": " in line:
                        key, value = line.split(": ", 1)
                        info[key] = value
                    elif key == "Bio" and line:
                        info[key] = (info.get(key, "") + " " + line).strip()
            return info
    except FileNotFoundError:
        logger.error("myinfo.txt not found.")
        return {}
    except Exception as e:
        logger.error(f"Error reading myinfo.txt: {e}")
        return {}

# Load personal info at startup
PERSONAL_INFO = load_personal_info()

# Memory Management
def load_memory():
    global CONVERSATION_MEMORY
    try:
        with open("memory.txt", "r", encoding="utf-8") as f:
            CONVERSATION_MEMORY = json.load(f)
    except FileNotFoundError:
        CONVERSATION_MEMORY = []
    except Exception as e:
        logger.error(f"Error loading memory.txt: {e}")

def save_memory():
    try:
        with open("memory.txt", "w", encoding="utf-8") as f:
            json.dump(CONVERSATION_MEMORY[-50:], f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving memory.txt: {e}")

# Add interaction to memory
def add_to_memory(message_id, author, content, bot_response=None):
    memory_entry = {
        "timestamp": time.time(),
        "message_id": message_id,
        "author": author,
        "content": content,
        "bot_response": bot_response
    }
    CONVERSATION_MEMORY.append(memory_entry)
    save_memory()

# Get recent memory context
def get_memory_context(channel_id):
    recent_memory = [m for m in CONVERSATION_MEMORY[-10:] if m.get("content")]
    if recent_memory:
        return "\n".join([f"{m['author']}: {m['content']}" for m in recent_memory])
    return ""

# Language Detection
def detect_language(text):
    try:
        lang = detect(text)
        supported_languages = ['en', 'hi', 'es', 'fr', 'de']
        return lang if lang in supported_languages else 'en'
    except:
        return 'en'

# Emoji Handling
def get_random_emojis(count=1, sentiment='happy'):
    emoji_map = {
        'happy': ['😎', '🔥', '🙌', '😄', '🎉'],
        'thinking': ['🤔', '🧠', '🕵️'],
        'helpful': ['👍', '💡', '🚀'],
        'sympathetic': ['🤗', '🙏', '🥺'],
        'sarcastic': ['🙄', '😏', '🤷']
    }
    emojis = emoji_map.get(sentiment, emoji_map['happy'])
    return ''.join(random.choice(emojis) for _ in range(count)) if random.random() < 0.3 else ''

# Rate Limiter
class RateLimiter:
    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    def can_make_request(self):
        now = time.time()
        self.requests = [req_time for req_time in self.requests if now - req_time < self.time_window]
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False

ai_rate_limiter = RateLimiter(max_requests=30, time_window=60)

# Mood System
def get_bot_mood():
    moods = ['excited', 'chill', 'sarcastic']
    return random.choice(moods)

# Question Templates
def get_random_question():
    questions = [
        "yo what’s everyone vibing with rn?",
        "anybody got game recs? i’m bored af",
        "what’s the wildest thing you did this week?",
        "y’all watching any dope shows? spill the tea",
        "what’s your go-to snack? i need inspo",
        "who’s got the best playlist? drop it",
        "what’s the vibe today? hit me with it",
        "anybody into memes? share the gold"
    ]
    return random.choice(questions)

# Random Message Templates
def get_random_message():
    messages = [
        "just chilling, what’s good?",
        "vibes only, let’s gooo",
        "yo this server’s lit fr",
        "kinda hungry, who’s got food pics?",
        "bet we all need coffee rn",
        "skrrt, what’s poppin?",
        "lowkey loving this chat",
        "who’s up for some chaos?"
    ]
    return random.choice(messages)

# Gemini Response by Nabil
# GitHub: https://github.com/xNabil
def get_gemini_response(prompt, detected_lang, message_type='general', mood='chill'):
    try:
        if not ai_rate_limiter.can_make_request():
            return "rate limit hit, give me a sec!"

        lang_instructions = {
            'hi': 'reply in hindi with 1-2 chill sentences.',
            'en': 'reply in english with 1-2 chill sentences.',
            'es': 'reply in spanish with 1-2 chill sentences.',
            'fr': 'reply in french with 1-2 chill sentences.',
            'de': 'reply in german with 1-2 chill sentences.'
        }
        
        mood_templates = {
            'excited': [
                'talk like a hyped gen z teen, full energy and slang. keep it short, one capital letter.',
                'reply like you’re stoked af, texting your squad. one capital only.',
                'chat like a teen on discord, mad pumped and fun. one capital letter.'
            ],
            'chill': [
                'talk like a gen z teen, super chill and slangy. keep it short, one capital letter.',
                'reply like you’re texting your homie, no formal vibes. one capital only.',
                'chat like a teen on discord, lowkey and fun. one capital letter.'
            ],
            'sarcastic': [
                'talk like a gen z teen, dripping with sarcasm and shade. keep it short, one capital letter.',
                'reply like you’re side-eyeing your homie, no serious vibes. one capital only.',
                'chat like a teen on discord, sassy and quick. one capital letter.'
            ]
        }
        
        templates = {
            'general': mood_templates[mood],
            'question': [
                'answer like a gen z explaining to their squad, simple and chill. one capital.',
                'help out like you’re texting a friend, no big words. one capital only.',
                'drop a quick answer, teen vibe, keep it real. one capital letter.'
            ],
            'help': [
                'offer help like a cool teen, mad chill and supportive. one capital.',
                'give advice like you’re hyping up your homie, no formal stuff. one capital.',
                'help out with gen z energy, keep it light. one capital letter.'
            ],
            'reply': [
                'keep the convo going like a teen texting, super casual. one capital.',
                'reply like you’re vibing with your squad, no stiff vibes. one capital.',
                'chat back like a gen z on discord, chill and quick. one capital letter.'
            ],
            'ask_question': [
                'ask a fun question like a gen z teen, curious and chill. one capital.',
                'drop a question like you’re texting your homie, no formal vibes. one capital.',
                'spark convo like a teen on discord, lowkey and engaging. one capital letter.'
            ],
            'random': [
                'say something random like a gen z teen, super chill and fun. one capital.',
                'drop a casual vibe like you’re texting your squad, no deep stuff. one capital.',
                'chat like a teen on discord, light and spontaneous. one capital letter.'
            ]
        }
        
        # 10% chance for a short, Gen Z reply
        if random.random() < 0.1 and message_type in ['general', 'reply', 'random']:
            short_responses = ['yo', 'lit', 'vibes', 'lol', 'bet', 'fr', 'skrrt', 'mood']
            return random.choice(short_responses)
        
        template = random.choice(templates.get(message_type, templates['general']))
        
        # Add personal info for relevant prompts
        if any(keyword in prompt.lower() for keyword in ['who', 'you', 'your', 'where', 'live', 'hobby', 'job', 'occupation', 'favorite', 'bio']):
            personal_context = f"use this info if it fits: name: {PERSONAL_INFO.get('Name', '')}, " \
                              f"age: {PERSONAL_INFO.get('Age', '')}, " \
                              f"hobby: {PERSONAL_INFO.get('Hobby', '')}, " \
                              f"location: {PERSONAL_INFO.get('Where I live', '')}, " \
                              f"occupation: {PERSONAL_INFO.get('Occupation', '')}, " \
                              f"favorite thing: {PERSONAL_INFO.get('Favorite Thing', '')}, " \
                              f"bio: {PERSONAL_INFO.get('Bio', '')}. "
        else:
            personal_context = ""
        
        # Add conversation memory context
        memory_context = get_memory_context(None)
        if memory_context:
            memory_context = f"recent chat history for context (don’t repeat names): {memory_context}\n"
        
        full_prompt = f"{lang_instructions.get(detected_lang, 'reply with 1-2 chill sentences.')}\n" \
                     f"{template}\n" \
                     f"{personal_context}\n" \
                     f"{memory_context}\n" \
                     f"don’t use usernames. use simple words, no extra commas. lowercase except first letter.\n\n{prompt}"
        
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.9,
                top_p=0.9,
                max_output_tokens=100
            )
        )
        response_text = response.text.strip()
        
        # Capitalize only the first letter
        if response_text:
            response_text = response_text[0].upper() + response_text[1:].lower()
        
        emoji = get_random_emojis(count=1, sentiment=mood if mood != 'chill' else message_type)
        return f"{response_text} {emoji}".strip()
    except Exception as e:
        logger.error(f"AI Error: {e}")
        return "yo, something broke, my bad"

# Discord API Request by Nabil
# GitHub: https://github.com/xNabil
async def make_discord_request(url, method="GET", json_data=None):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, headers=HEADERS, json=json_data) as response:
                if response.status == 429:
                    retry_after = float((await response.json()).get("retry_after", 1))
                    logger.warning(f"Rate limited, retrying after {retry_after} seconds...")
                    await asyncio.sleep(retry_after)
                    return await make_discord_request(url, method, json_data)
                response.raise_for_status()
                return await response.json()
    except Exception as e:
        logger.error(f"Error in Discord API request: {e}")
        raise

# Fetch Messages
async def fetch_channel_messages(channel_id, limit=50):
    try:
        url = f"https://discord.com/api/v9/channels/{channel_id}/messages?limit={limit}"
        return await make_discord_request(url)
    except Exception as e:
        logger.error(f"Error fetching messages: {e}")
        return []

# Countdown Display
async def print_countdown(seconds):
    if seconds < 1:
        return
    print(f"\r{Fore.CYAN}Sending reply in {int(seconds)} seconds...{Style.RESET_ALL}", end='')
    for i in range(int(seconds), 0, -1):
        print(f"\r{Fore.CYAN}Sending reply in {i} seconds...{Style.RESET_ALL}", end='')
        await asyncio.sleep(1)
    print(f"\r{' ' * 50}\r", end='')

# Send Reply
async def send_reply(channel_id, message, delay_range, message_id=None):
    try:
        delay = random.uniform(delay_range[0], delay_range[1])
        await print_countdown(delay)
        data = {"content": message}
        if message_id:
            data["message_reference"] = {"message_id": message_id}
        await make_discord_request(
            f"https://discord.com/api/v9/channels/{channel_id}/messages",
            method="POST",
            json_data=data
        )
    except aiohttp.ClientResponseError as e:
        if e.status == 400:
            logger.error(f"Bad Request (400) when sending message: '{message}'. Skipping to next message.")
            return
        else:
            raise

# Get User ID and Username
async def get_bot_user_id():
    try:
        user_data = await make_discord_request("https://discord.com/api/v9/users/@me")
        global BOT_USERNAME
        BOT_USERNAME = user_data.get("username")
        return user_data.get("id")
    except Exception as e:
        logger.error(f"Error fetching user ID: {e}")
        return None

# Check if Message is Old
def is_message_old(timestamp_str):
    try:
        message_time = isoparse(timestamp_str)
        message_timestamp = message_time.timestamp()
        return time.time() - message_timestamp > 300
    except Exception as e:
        logger.error(f"Error parsing timestamp: {e}")
        return True

# Terminal UI
def print_header(text):
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*50}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{Style.BRIGHT}{text.center(50)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'='*50}{Style.RESET_ALL}\n")

def print_status(text, status_type='info'):
    colors = {
        'success': Fore.GREEN,
        'error': Fore.RED,
        'info': Fore.BLUE,
        'warning': Fore.YELLOW
    }
    color = colors.get(status_type, Fore.WHITE)
    print(f"{color}{Style.BRIGHT}{text}{Style.RESET_ALL}")

# Main Function by Nabil
# GitHub: https://github.com/xNabil
async def selfbot():
    global BOT_USER_ID
    print_header("Discord Chat Bot by Nabil")
    print_status("Bot is starting...", 'info')
    
    # Load memory
    load_memory()
    
    # Fetch user ID and username
    BOT_USER_ID = await get_bot_user_id()
    if not BOT_USER_ID:
        print_status("Failed to fetch user ID. Check DISCORD_TOKEN in .env file.", 'error')
        return
    
    # Get channel ID from .env or prompt
    channel_id = CHANNEL_ID
    if not channel_id or not channel_id.isdigit():
        print_status("Invalid or missing CHANNEL_ID in .env file.", 'warning')
        while True:
            channel_id = input(f"{Fore.CYAN}👉 Enter channel ID: {Style.RESET_ALL}").strip()
            if not channel_id or not channel_id.isdigit():
                print_status("Channel ID must be a numeric value.", 'error')
                continue
            break

    # Get slow mode from .env
    default_slow_mode = (5, 5)
    if channel_id not in CHANNEL_SLOW_MODES:
        if SLOW_MODE:
            try:
                slow_mode_input = SLOW_MODE0 = SLOW_MODE.replace('s', '').strip()
                if ',' in slow_mode_input:
                    min_delay, max_delay = map(int, slow_mode_input.split(','))
                    if min_delay < 0 or max_delay < min_delay:
                        print_status("Invalid SLOW_MODE in .env: min must be >= 0 and <= max.", 'error')
                        slow_mode = default_slow_mode
                    else:
                        slow_mode = (min_delay, max_delay)
                else:
                    delay = int(slow_mode_input)
                    if delay < 0:
                        print_status("Invalid SLOW_MODE in .env: must be non-negative.", 'error')
                        slow_mode = default_slow_mode
                    else:
                        slow_mode = (delay, delay)
            except ValueError:
                print_status("Invalid SLOW_MODE in .env: must be a number or range (e.g., 60 or 60,65).", 'error')
                slow_mode = default_slow_mode
        else:
            print_status("SLOW_MODE missing in .env, using default 5s.", 'warning')
            slow_mode = default_slow_mode
        CHANNEL_SLOW_MODES[channel_id] = slow_mode
    
    print_status(f"✅ Bot successfully initialized! Channel ID: {channel_id}, Slow Mode: {CHANNEL_SLOW_MODES[channel_id]}", 'success')
    
    try:
        while True:
            # Initialize wait time for this cycle
            wait_time = random.uniform(CHANNEL_SLOW_MODES[channel_id][0], CHANNEL_SLOW_MODES[channel_id][1])
            elapsed_time = 0
            bot_relevant_messages = []  # Store all replies/mentions to bot
            action = None
            target_message = None
            remaining_delay = (0, 0)  # Default to no delay after monitoring
            
            print_status(f"\n⏳ Monitoring for {wait_time:.1f}s...", 'info')
            
            # Monitor messages during wait time
            while elapsed_time < wait_time:
                messages = await fetch_channel_messages(channel_id, 50)
                if not messages:
                    print_status("No messages found or error occurred.", 'error')
                    await asyncio.sleep(2)
                    elapsed_time += 2
                    continue
                
                print_header("Recent Messages")
                for i, msg in enumerate(messages[:50], 1):
                    author = msg.get("author", {}).get("username", "Unknown")
                    content = msg.get("content", "")
                    if content:
                        truncated_content = f"{content[:50]}..." if len(content) > 50 else content
                        print(f"{Fore.GREEN}{i}.{Style.RESET_ALL} {Fore.BLUE}{author}{Style.RESET_ALL}: {Fore.WHITE}{truncated_content}{Style.RESET_ALL}")
                
                # Check for replies to bot's messages or mentions
                for msg in messages[:50]:
                    author = msg.get("author", {}).get("username", "Unknown")
                    content = msg.get("content", "")
                    if not content:
                        continue
                    
                    if (author == "Unknown" or
                        is_message_old(msg.get("timestamp", "")) or
                        msg.get("author", {}).get("id") == BOT_USER_ID):
                        continue
                    
                    is_reply_to_bot = msg.get("referenced_message") is not None and \
                                     msg.get("referenced_message", {}).get("author", {}).get("id") == BOT_USER_ID
                    is_mention = f"@{BOT_USERNAME}" in content or f"<@{BOT_USER_ID}>" in content
                    
                    if is_reply_to_bot or is_mention:
                        bot_relevant_messages.append(msg)
                
                await asyncio.sleep(2)
                elapsed_time += 2
                print(f"\r{Fore.CYAN}⏳ Checking messages, {wait_time - elapsed_time:.1f}s left...{Style.RESET_ALL}", end='')
            
            print(f"\r{' ' * 50}\r", end='')
            
            # Decide action
            if bot_relevant_messages:
                # Randomly select one of the relevant messages to reply to
                target_message = random.choice(bot_relevant_messages)
                action = 'reply'
                # Calculate remaining delay for this reply
                remaining_time = max(0, wait_time - elapsed_time)
                remaining_delay = (remaining_time, remaining_time)
            else:
                messages = await fetch_channel_messages(channel_id, 50)
                valid_messages = [msg for msg in messages if msg.get("content") and
                                not is_message_old(msg.get("timestamp", "")) and
                                msg.get("author", {}).get("id") != BOT_USER_ID]
                
                action_weights = {'reply': 0.5, 'ask_question': 0.25, 'random': 0.25}
                action = random.choices(list(action_weights.keys()), weights=list(action_weights.values()), k=1)[0]
                
                if action == 'reply' and valid_messages:
                    target_message = random.choice(valid_messages)
                elif action == 'ask_question':
                    target_message = None
                elif action == 'random':
                    target_message = None
            
            # Execute action
            if action:
                mood = get_bot_mood()
                detected_lang = detect_language(target_message.get("content", "") if target_message else "en")
                
                if action == 'reply' and target_message:
                    context = f"this is a reply to you. they said: {target_message.get('content')}"
                    message_type = 'reply'
                elif action == 'ask_question':
                    context = get_random_question()
                    message_type = 'ask_question'
                elif action == 'random':
                    context = get_random_message()
                    message_type = 'random'
                else:
                    context = f"someone mentioned you: {target_message.get('content')}"
                    message_type = 'helpful'
                
                max_retries = 3
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        prompt = f"you’re a chill discord user. reply naturally:\n{context}"
                        ai_response = get_gemini_response(prompt, detected_lang, message_type, mood)
                        
                        if ai_response and not ai_response.startswith(("AI Error", "rate limit")):
                            # Log the original message and bot's response
                            original_content = target_message.get('content', context) if target_message else context
                            truncated_original = f"{original_content[:50]}..." if len(original_content) > 50 else original_content
                            logger.info(
                                f"Responding to {Fore.BLUE}{target_message.get('author', {}).get('username', 'Unknown') if target_message else 'channel'}{Style.RESET_ALL} "
                                f"in {detected_lang}: {Fore.YELLOW}Original: {truncated_original}{Style.RESET_ALL} "
                                f"{Fore.GREEN}Response: {ai_response}{Style.RESET_ALL}"
                            )
                            await send_reply(channel_id, ai_response, remaining_delay, target_message.get("id") if target_message else None)
                            print_status("Reply sent!", 'success')
                            
                            # Add to memory
                            add_to_memory(
                                target_message.get("id") if target_message else str(time.time()),
                                target_message.get("author", {}).get("username", "Unknown") if target_message else "Channel",
                                original_content,
                                ai_response
                            )
                            break
                        elif ai_response.startswith("rate limit"):
                            logger.warning("Rate limit reached, waiting before retry...")
                            await asyncio.sleep(5)
                        else:
                            logger.error(f"AI Error: {ai_response}")
                            await asyncio.sleep(2)
                        retry_count += 1
                    except Exception as e:
                        logger.error(f"Error in response generation: {e}")
                        await asyncio.sleep(2)
                        retry_count += 1
                
                if retry_count == max_retries:
                    print_status("Max retries reached for this message.", 'error')
            
            print_status("\n⏳ Monitoring for new messages...", 'info')
            
            for i in range(4, 0, -1):
                print(f"\r{Fore.CYAN}⏳ Refreshing in {i} seconds...{Style.RESET_ALL}", end='')
                await asyncio.sleep(1)
            print()
    except KeyboardInterrupt:
        print_status("🛑 Bot stopped by user.", 'warning')
        return
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(selfbot())