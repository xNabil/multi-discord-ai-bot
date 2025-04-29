# Discord Chat Bot by Nabil
# GitHub: https://github.com/xNabil
# Version: Enhanced with Priority Replies, Smart AI, and Modern Features

import os
import aiohttp
import random
import time
import asyncio
import google.generativeai as genai
import json
from dotenv import load_dotenv
from dateutil.parser import isoparse
from colorama import init, Fore, Style
import logging
from textblob import TextBlob
import re

# Initialize colorama for colored terminal output and set up logging
init()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

print(f"{Fore.CYAN}Starting Discord Chat Bot...{Style.RESET_ALL}")

# Load environment variables from .env file
try:
    load_dotenv()
    print(f"{Fore.GREEN}Loaded .env file successfully{Style.RESET_ALL}")
except Exception as e:
    print(f"{Fore.RED}Error loading .env: {e}{Style.RESET_ALL}")
    exit(1)

# Retrieve environment variables
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHANNEL_ID = os.getenv("CHANNEL_ID")
SLOW_MODE = os.getenv("SLOW_MODE")

# Validate environment variables
if not all([DISCORD_TOKEN, GEMINI_API_KEY]):
    print(f"{Fore.RED}Missing required environment variables!{Style.RESET_ALL}")
    exit(1)
print(f"{Fore.GREEN}Environment variables validated{Style.RESET_ALL}")

# Set up Discord API headers
HEADERS = {
    "Authorization": DISCORD_TOKEN,
    "Content-Type": "application/json"
}

# Configure Gemini AI
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    print(f"{Fore.GREEN}Gemini AI configured successfully{Style.RESET_ALL}")
except Exception as e:
    print(f"{Fore.RED}Error configuring Gemini AI: {e}{Style.RESET_ALL}")
    exit(1)

# Global variables for bot state
CHANNEL_SLOW_MODES = {}
BOT_USER_ID = None
BOT_USERNAME = None
PERSONAL_INFO = {}
CONVERSATION_MEMORY = []
USER_PROFILES = {}

# Topic keywords for detection
TOPICS = {
    'gaming': ['game', 'play', 'console', 'pc', 'xbox', 'playstation', 'controller', 'gamer'],
    'music': ['song', 'album', 'artist', 'band', 'concert', 'beats', 'tune'],
    'movies': ['film', 'movie', 'cinema', 'actor', 'director', 'scene', 'plot'],
    'tech': ['code', 'tech', 'software', 'hardware', 'gadget', 'app', 'update'],
    'food': ['food', 'eat', 'cook', 'recipe', 'snack', 'meal', 'yummy']
}

# Regex patterns for user profiling
FAVORITE_GAME_PATTERN = re.compile(r"my favorite game is (\w+)", re.IGNORECASE)
FAVORITE_FOOD_PATTERN = re.compile(r"my favorite food is (\w+)", re.IGNORECASE)

# Load personal info from myinfo.txt
def load_personal_info():
    """Load bot's personal info from myinfo.txt for context in responses."""
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
            print(f"{Fore.GREEN}Personal info loaded: {info}{Style.RESET_ALL}")
            return info
    except FileNotFoundError:
        print(f"{Fore.YELLOW}myinfo.txt not found, using defaults{Style.RESET_ALL}")
        return {}
    except Exception as e:
        print(f"{Fore.RED}Error loading myinfo.txt: {e}{Style.RESET_ALL}")
        return {}

PERSONAL_INFO = load_personal_info()

# Memory management functions
def load_memory():
    """Load conversation memory from memory.txt."""
    global CONVERSATION_MEMORY
    try:
        with open("memory.txt", "r", encoding="utf-8") as f:
            CONVERSATION_MEMORY = json.load(f)
        print(f"{Fore.GREEN}Loaded conversation memory{Style.RESET_ALL}")
    except FileNotFoundError:
        CONVERSATION_MEMORY = []
        print(f"{Fore.YELLOW}memory.txt not found, starting fresh{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error loading memory: {e}{Style.RESET_ALL}")

def save_memory():
    """Save the last 50 conversation entries to memory.txt."""
    try:
        with open("memory.txt", "w", encoding="utf-8") as f:
            json.dump(CONVERSATION_MEMORY[-50:], f, ensure_ascii=False, indent=2)
        print(f"{Fore.GREEN}Saved conversation memory{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error saving memory: {e}{Style.RESET_ALL}")

def add_to_memory(message_id, author, content, bot_response=None):
    """Add a message and bot response to conversation memory."""
    entry = {
        "timestamp": time.time(),
        "message_id": message_id,
        "author": author,
        "content": content,
        "bot_response": bot_response
    }
    CONVERSATION_MEMORY.append(entry)
    save_memory()

def get_memory_context():
    """Get recent conversation context for AI prompts."""
    recent = CONVERSATION_MEMORY[-10:]
    if recent:
        return "\n".join([f"{m['author']}: {m['content']}" for m in recent])
    return ""

# User profile management
def load_user_profiles():
    """Load user profiles from user_profiles.json."""
    global USER_PROFILES
    try:
        with open("user_profiles.json", "r", encoding="utf-8") as f:
            USER_PROFILES = json.load(f)
        print(f"{Fore.GREEN}Loaded user profiles{Style.RESET_ALL}")
    except FileNotFoundError:
        USER_PROFILES = {}
        print(f"{Fore.YELLOW}user_profiles.json not found, starting fresh{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error loading user profiles: {e}{Style.RESET_ALL}")

def save_user_profiles():
    """Save user profiles to user_profiles.json."""
    try:
        with open("user_profiles.json", "w", encoding="utf-8") as f:
            json.dump(USER_PROFILES, f, ensure_ascii=False, indent=2)
        print(f"{Fore.GREEN}Saved user profiles{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error saving user profiles: {e}{Style.RESET_ALL}")

def update_user_profile(user_id, key, value):
    """Update a user's profile with a new key-value pair."""
    if user_id not in USER_PROFILES:
        USER_PROFILES[user_id] = {}
    USER_PROFILES[user_id][key] = value
    save_user_profiles()

def get_user_profile(user_id):
    """Retrieve a user's profile."""
    return USER_PROFILES.get(user_id, {})

# Emoji generation
def get_random_emojis(count=1, mood='chill'):
    """Return random emojis based on the bot's mood."""
    emoji_map = {
        'excited': ['🤩', '🥳', '💥', '🔥', '🎉'],
        'chill': ['😌', '🍃', '🛋️', '✌️', '😎'],
        'sarcastic': ['🙄', '😏', '🤷', '😒', '👀'],
        'joking': ['😂', '🤣', '😜', '😝', '🤡'],
        'sympathetic': ['🤗', '🙏', '🥺', '❤️', '😢']
    }
    emojis = emoji_map.get(mood, emoji_map['chill'])
    return ''.join(random.choice(emojis) for _ in range(count)) if random.random() < 0.4 else ''

# Rate limiter for AI requests
class RateLimiter:
    """Limit the number of AI requests per time window."""
    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []

    def can_make_request(self):
        """Check if a request can be made within the rate limit."""
        now = time.time()
        self.requests = [t for t in self.requests if now - t < self.time_window]
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False

ai_rate_limiter = RateLimiter(max_requests=30, time_window=60)

# Mood system
def get_bot_mood(sentiment='neutral'):
    """Determine the bot's mood based on message sentiment."""
    mood_map = {
        'positive': ['excited', 'chill', 'joking'],
        'negative': ['sympathetic', 'chill', 'sarcastic'],
        'neutral': ['chill', 'joking', 'excited']
    }
    return random.choice(mood_map.get(sentiment, ['chill']))

# Sentiment analysis
def get_sentiment(text):
    """Analyze the sentiment of a message."""
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'
    except Exception as e:
        print(f"{Fore.RED}Sentiment analysis error: {e}{Style.RESET_ALL}")
        return 'neutral'

# Topic detection
def detect_topic(text):
    """Detect the topic of a message based on keywords."""
    text = text.lower()
    for topic, keywords in TOPICS.items():
        if any(keyword in text for keyword in keywords):
            return topic
    return 'general'

# Profile extraction
def extract_user_preferences(text):
    """Extract preferences like favorite game or food from text."""
    preferences = {}
    game_match = FAVORITE_GAME_PATTERN.search(text)
    if game_match:
        preferences['favorite_game'] = game_match.group(1)
    food_match = FAVORITE_FOOD_PATTERN.search(text)
    if food_match:
        preferences['favorite_food'] = food_match.group(1)
    return preferences

# Extensive response templates
def get_random_question():
    """Return a random question to spark conversation."""
    questions = [
        "what’s your fave game rn?",
        "anyone got a killer playlist to share?",
        "what’s the best movie you’ve seen lately?",
        "y’all into tech? what’s your go-to gadget?",
        "what’s your comfort food? i need ideas",
        "who’s your dream collab artist?",
        "what’s the wildest thing you’ve done this month?",
        "anybody got a meme to drop?",
        "what’s your vibe today? spill it",
        "if you could live anywhere, where’d it be?"
    ]
    return random.choice(questions)

def get_random_message():
    """Return a random casual message."""
    messages = [
        "just vibin, what’s up?",
        "this chat’s fire, keep it going",
        "who’s got the snacks? i’m starving",
        "bet we’re all legends here",
        "lowkey loving this server",
        "let’s make some chaos, fam",
        "i’m here for the vibes, no cap",
        "who’s got the tea today?",
        "this place is my vibe rn",
        "let’s turn it up a notch"
    ]
    return random.choice(messages)

def get_greeting():
    """Return a casual greeting."""
    greetings = [
        "yo what’s good?",
        "sup fam, how’s it hangin?",
        "hey legends, what’s poppin?",
        "what’s up, my people?",
        "yo, who’s ready to chat?",
        "hey, let’s get this vibe going",
        "sup, how’s the crew?",
        "what’s crackin, squad?",
        "yo, let’s make it lit",
        "hey, fam’s here, let’s roll"
    ]
    return random.choice(greetings)

def get_farewell():
    """Return a casual farewell."""
    farewells = [
        "catch y’all later",
        "peace out, squad",
        "i’m out, stay cool",
        "gtg, vibes forever",
        "later, homies",
        "time to dip, peace",
        "i’m ghostin, but i’ll be back",
        "see ya on the flip side",
        "outtie, keep it real",
        "peace and love, fam"
    ]
    return random.choice(farewells)

def get_joke():
    """Return a random joke."""
    jokes = [
        "why don’t skeletons fight? no guts",
        "what’s a ghost’s fave food? boo-ritos",
        "why’d the tomato blush? saw the sauce",
        "what do you call a lazy kangaroo? pouch potato",
        "why don’t eggs laugh? they’d crack up",
        "what’s a pirate’s fave letter? R matey",
        "why’d the cat sit alone? he’s purrfect",
        "what do you call a fake noodle? impasta",
        "why don’t programmers sleep? bugs keep em up",
        "what’s a bear with no teeth? gummy bear"
    ]
    return random.choice(jokes)

# AI response generation
def get_gemini_response(prompt, message_type='reply', mood='chill', user_profile=None, sentiment='neutral', topic='general'):
    """Generate a smart, context-aware response using Gemini AI."""
    try:
        if not ai_rate_limiter.can_make_request():
            return "chill, i’m hittin the rate limit rn"

        # Mood-based templates
        mood_templates = {
            'excited': [
                'talk like a hyped gen z, all caps energy but one capital. short and slangy.',
                'reply like you’re stoked af, teen vibe. one capital only.',
                'chat like a discord hypebeast, mad pumped. one capital.'
            ],
            'chill': [
                'talk like a laid-back gen z, super chill. one capital letter.',
                'reply like you’re texting your homie, no rush. one capital.',
                'chat like a discord regular, lowkey af. one capital.'
            ],
            'sarcastic': [
                'talk like a gen z with mad shade, sarcastic af. one capital.',
                'reply like you’re roasting but chill, teen style. one capital.',
                'chat like a sassy discord user, quick wit. one capital.'
            ],
            'joking': [
                'talk like a gen z jokester, playful af. one capital.',
                'reply like you’re cracking up your squad, fun vibe. one capital.',
                'chat like a discord memer, light and funny. one capital.'
            ],
            'sympathetic': [
                'talk like a gen z with feels, supportive af. one capital.',
                'reply like you’re there for your homie, real talk. one capital.',
                'chat like a discord friend, caring vibe. one capital.'
            ]
        }

        # Message type templates
        templates = {
            'reply': [
                'keep it flowing like a gen z reply, casual af. one capital.',
                'chat back like you’re texting your squad, no formal bs. one capital.',
                'respond like a discord regular, chill and quick. one capital.'
            ],
            'question': [
                'answer like a gen z explainer, simple af. one capital.',
                'drop a quick answer, teen style, no big words. one capital.',
                'help out like a discord homie, real talk. one capital.'
            ],
            'random': [
                'say something chill like a gen z, random af. one capital.',
                'drop a vibe like you’re texting your crew, no deep stuff. one capital.',
                'chat like a discord wildcard, spontaneous af. one capital.'
            ],
            'greeting': [
                'greet like a gen z legend, super chill. one capital.',
                'say hi like you’re hittin up your squad, no formal vibes. one capital.',
                'welcome like a discord pro, quick and cool. one capital.'
            ],
            'farewell': [
                'say bye like a gen z, casual af. one capital.',
                'peace out like you’re dippin from your homies, chill. one capital.',
                'leave like a discord ghost, smooth exit. one capital.'
            ],
            'joke': [
                'tell a joke like a gen z memer, funny af. one capital.',
                'drop a punchline like you’re texting your squad, quick. one capital.',
                'crack up the discord crew, light vibe. one capital.'
            ]
        }

        template = random.choice(templates.get(message_type, mood_templates[mood]))

        # Personal context
        personal_context = ""
        if any(kw in prompt.lower() for kw in ['who', 'you', 'your', 'hobby', 'live']):
            personal_context = f"use this if it fits: name: {PERSONAL_INFO.get('Name', 'unknown')}, " \
                              f"hobby: {PERSONAL_INFO.get('Hobby', 'none')}, " \
                              f"location: {PERSONAL_INFO.get('Where I live', 'somewhere')}, " \
                              f"bio: {PERSONAL_INFO.get('Bio', 'just a chill bot')}"
        
        # Memory context
        memory_context = get_memory_context()
        if memory_context:
            memory_context = f"recent chat for context:\n{memory_context}\n"

        # User profile context
        user_context = ""
        if user_profile:
            if topic == 'gaming' and 'favorite_game' in user_profile:
                user_context += f"they like {user_profile['favorite_game']} "
            if topic == 'food' and 'favorite_food' in user_profile:
                user_context += f"their fave food is {user_profile['favorite_food']} "

        # Construct full prompt
        full_prompt = f"reply in english with 1-2 chill sentences. mood is {mood}. sentiment is {sentiment}. topic is {topic}. {template}\n" \
                     f"{personal_context}\n{memory_context}\n{user_context}\n" \
                     f"no usernames. simple words. lowercase except first letter.\n\n{prompt}"

        # Adjust response length based on input
        target_length = len(prompt)
        max_tokens = 50 if target_length < 50 else 100 if target_length < 100 else 150

        # Generate response
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.9,
                top_p=0.9,
                max_output_tokens=max_tokens
            )
        )
        response_text = response.text.strip()
        if response_text:
            response_text = response_text[0].upper() + response_text[1:].lower()

        # Add emoji based on mood
        emoji = get_random_emojis(count=1, mood=mood)
        return f"{response_text} {emoji}".strip()
    except Exception as e:
        print(f"{Fore.RED}AI generation error: {e}{Style.RESET_ALL}")
        return "oops, something crashed, my bad"

# Discord API utilities
async def make_discord_request(url, method="GET", json_data=None):
    """Make a request to the Discord API with rate limit handling."""
    async with aiohttp.ClientSession() as session:
        async with session.request(method, url, headers=HEADERS, json=json_data) as response:
            if response.status == 429:
                retry_after = float((await response.json()).get("retry_after", 1))
                print(f"{Fore.YELLOW}Rate limited, waiting {retry_after}s{Style.RESET_ALL}")
                await asyncio.sleep(retry_after)
                return await make_discord_request(url, method, json_data)
            response.raise_for_status()
            return await response.json()

async def fetch_channel_messages(channel_id, limit=50):
    """Fetch recent messages from a Discord channel."""
    try:
        url = f"https://discord.com/api/v9/channels/{channel_id}/messages?limit={limit}"
        return await make_discord_request(url)
    except Exception as e:
        print(f"{Fore.RED}Error fetching messages: {e}{Style.RESET_ALL}")
        return []

async def send_reply(channel_id, message, delay_range, message_id=None):
    """Send a reply to a Discord channel with optional message reference."""
    delay = random.uniform(delay_range[0], delay_range[1])
    await print_countdown(delay, "Sending reply in")
    data = {"content": message}
    if message_id:
        data["message_reference"] = {"message_id": message_id}
    await make_discord_request(
        f"https://discord.com/api/v9/channels/{channel_id}/messages",
        method="POST",
        json_data=data
    )
    print(f"{Fore.GREEN}Sent: {message}{Style.RESET_ALL}")

async def get_bot_user_id():
    """Fetch the bot's user ID and username from Discord."""
    user_data = await make_discord_request("https://discord.com/api/v9/users/@me")
    global BOT_USERNAME
    BOT_USERNAME = user_data.get("username")
    print(f"{Fore.GREEN}Bot ID: {user_data.get('id')}, Username: {BOT_USERNAME}{Style.RESET_ALL}")
    return user_data.get("id")

def is_message_old(timestamp_str):
    """Check if a message is older than 5 minutes."""
    try:
        message_time = isoparse(timestamp_str).timestamp()
        return time.time() - message_time > 300
    except Exception:
        return True

# Terminal UI utilities
def print_header():
    """Display a fancy header in the terminal."""
    header = """
    ╔════════════════════════════════════════════════════╗
    ║           Discord Chat Bot by Nabil                ║
    ║         Smart, Modern, and Vibin'                  ║
    ╚════════════════════════════════════════════════════╝
    """
    print(f"{Fore.CYAN}{Style.BRIGHT}{header}{Style.RESET_ALL}")

def print_status(message, status_type='info'):
    """Print a status message with color coding."""
    colors = {
        'info': Fore.CYAN,
        'success': Fore.GREEN,
        'error': Fore.RED,
        'warning': Fore.YELLOW
    }
    color = colors.get(status_type, Fore.WHITE)
    print(f"{color}{Style.BRIGHT}{message}{Style.RESET_ALL}")

async def print_countdown(seconds, message="Waiting"):
    """Display a visible countdown in the terminal."""
    if seconds < 1:
        return
    for i in range(int(seconds), -1, -1):
        print(f"\r{Fore.CYAN}{message} {i} seconds...{Style.RESET_ALL}", end='')
        await asyncio.sleep(1)
    print(f"\r{' ' * 50}\r", end='')

# Main bot logic
async def selfbot():
    global BOT_USER_ID
    print_header()
    print_status("Initializing bot...", 'info')

    # Load memory and profiles
    load_memory()
    load_user_profiles()

    # Get bot user ID
    BOT_USER_ID = await get_bot_user_id()
    if not BOT_USER_ID:
        print_status("Failed to get bot ID, check DISCORD_TOKEN", 'error')
        return

    channel_id = CHANNEL_ID
    if not channel_id or not channel_id.isdigit():
        print_status("Invalid CHANNEL_ID", 'warning')
        channel_id = input(f"{Fore.CYAN}Enter channel ID: {Style.RESET_ALL}").strip()
        if not channel_id.isdigit():
            print_status("Channel ID must be numeric", 'error')
            return

    # Set slow mode
    default_slow_mode = (5, 10)
    if SLOW_MODE:
        try:
            min_delay, max_delay = map(int, SLOW_MODE.split(','))
            CHANNEL_SLOW_MODES[channel_id] = (min_delay, max_delay)
        except:
            CHANNEL_SLOW_MODES[channel_id] = default_slow_mode
            print_status("Invalid SLOW_MODE, using default 5-10s", 'warning')
    else:
        CHANNEL_SLOW_MODES[channel_id] = default_slow_mode
        print_status("SLOW_MODE not set, using default 5-10s", 'warning')

    print_status(f"Bot running on channel {channel_id} with slow mode {CHANNEL_SLOW_MODES[channel_id]}s", 'success')

    while True:
        try:
            wait_time = random.uniform(*CHANNEL_SLOW_MODES[channel_id])
            await print_countdown(wait_time, "Monitoring for")

            # Fetch messages
            messages = await fetch_channel_messages(channel_id, 50)
            if not messages:
                print_status("No messages fetched, retrying...", 'warning')
                continue

            # Update user profiles
            for msg in messages:
                prefs = extract_user_preferences(msg.get('content', ''))
                for key, value in prefs.items():
                    update_user_profile(msg['author']['id'], key, value)

            # Categorize messages
            replies_to_bot = [
                msg for msg in messages
                if msg.get("referenced_message", {}).get("author", {}).get("id") == BOT_USER_ID
                and not is_message_old(msg.get("timestamp", ""))
            ]
            other_messages = [
                msg for msg in messages
                if msg.get("content")
                and not is_message_old(msg.get("timestamp", ""))
                and msg.get("author", {}).get("id") != BOT_USER_ID
                and msg not in replies_to_bot
            ]

            # Check for inactivity and initiate conversation
            last_message_time = max([isoparse(msg['timestamp']).timestamp() for msg in messages])
            if time.time() - last_message_time > 300 and random.random() < 0.2:
                print_status("Chat’s been quiet, starting a convo...", 'info')
                context = get_random_question() if random.random() < 0.5 else get_random_message()
                response = get_gemini_response(context, 'random', 'chill')
                await send_reply(channel_id, response, (0, 2))
                add_to_memory(None, BOT_USERNAME, context, response)
                continue

            # Decide target message (prioritize replies to bot)
            if replies_to_bot:
                target_message = random.choice(replies_to_bot)
            elif other_messages:
                target_message = random.choice(other_messages)
            else:
                print_status("No valid messages to reply to", 'info')
                continue

            # Analyze message
            sentiment = get_sentiment(target_message['content'])
            topic = detect_topic(target_message['content'])
            mood = get_bot_mood(sentiment)
            user_profile = get_user_profile(target_message['author']['id'])

            # Generate response
            context = f"they said: {target_message['content']}"
            prompt = f"you’re a chill discord user replying to this:\n{context}"
            response = get_gemini_response(prompt, 'reply', mood, user_profile, sentiment, topic)

            # Display in terminal
            print_status(f"Responding to {Fore.BLUE}{target_message['author']['username']}{Style.RESET_ALL}: {Fore.YELLOW}{target_message['content']}{Style.RESET_ALL}", 'info')

            # Send response
            await send_reply(channel_id, response, (0, 2), target_message.get("id"))
            add_to_memory(target_message['id'], target_message['author']['username'], target_message['content'], response)

           

        except Exception as e:
            print_status(f"Error in main loop: {e}", 'error')
            await asyncio.sleep(5)

if __name__ == "__main__":
    print_status("Starting bot...", 'info')
    asyncio.run(selfbot())
