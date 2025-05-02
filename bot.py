# Discord Chat Bot by Nabil
# GitHub: https://github.com/xNabil
# Version: Enhanced with Priority Replies, Smart AI, Banned Words Handling, Probabilistic Response Lengths, Disfluencies, and Casual Language

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
BANNED_WORDS = ['hi', 'fire', 'hello', 'lit', 'blaze']  # Add more banned words as needed

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

# Banned words handling
def sanitize_message(message):
    """Replace or remove banned words from a message."""
    sanitized = message.lower()
    for word in BANNED_WORDS:
        sanitized = sanitized.replace(word, '***')
    return sanitized if sanitized != message.lower() else message

async def rephrase_message(original_message, mood, sentiment, topic):
    """Rephrase a message to avoid banned words using Gemini AI."""
    try:
        prompt = f"Rephrase this to avoid words like {', '.join(BANNED_WORDS)} while keeping the same vibe, mood ({mood}), sentiment ({sentiment}), and topic ({topic}):\n{original_message}"
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.9,
                top_p=0.9,
                max_output_tokens=150
            )
        )
        return response.text.strip() or original_message
    except Exception as e:
        print(f"{Fore.RED}Error rephrasing message: {e}{Style.RESET_ALL}")
        return original_message

# Emoji generation
def get_random_emojis(count=1, mood='chill'):
    """Return random emojis based on the bot's mood."""
    emoji_map = {
        'excited': ['🤩', '🥳', '💥', '🎉'],
        'chill': ['😌', '🍃', '🛋️', '✌️', '😎'],
        'sarcastic': ['🙄', '😏', '🤷', '😒', '👀'],
        'joking': ['😂', '🤣', '😜', '😝', '🤡'],
        'lazy': ['😴', '💤', '🛌', '😪', '🥱'],
        'paranoid': ['🫣', '🤐', '👀', '😬', '🙈']
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
        'negative': ['sarcastic', 'paranoid', 'lazy'],
        'neutral': ['chill', 'joking', 'lazy']
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
        preferences['favorite_food'] = game_match.group(1)
    return preferences

# Extensive response templates
def get_random_question():
    """Return a random question to spark conversation."""
    questions = [
        "whats your fave game rn?",
        "anyone got a killer playlist to share?",
        "whats the best movie youve seen lately?",
        "yall into tech? whats your go-to gadget?",
        "whats your comfort food? i need ideas",
        "whos your dream collab artist?",
        "whats the wildest thing youve done this month?",
        "anybody got a meme to drop?",
        "whats your vibe today? spill it",
        "if you could live anywhere, whered it be?"
    ]
    return random.choice(questions)

def get_random_message():
    """Return a random casual message."""
    messages = [
        "just vibin, whats up?",
        "this chats dope, keep it going",
        "whos got the snacks? im starving",
        "aspettative were all legends here",
        "lowkey loving this server",
        "lets make some chaos, fam",
        "im here for the vibes, no cap",
        "whos got the tea today?",
        "this place is my vibe rn",
        "lets turn it up a notch"
    ]
    return random.choice(messages)

def get_greeting():
    """Return a casual greeting."""
    greetings = [
        "yo whats good?",
        "sup fam, hows it hangin?",
        "hey legends, whats poppin?",
        "whats up, my people?",
        "yo, whos ready to chat?",
        "hey, lets get this vibe going",
        "sup, hows the crew?",
        "whats crackin, squad?",
        "yo, lets make it lit",
        "hey, fams here, lets roll"
    ]
    return random.choice(greetings)

def get_farewell():
    """Return a casual farewell."""
    farewells = [
        "catch yall later",
        "peace out, squad",
        "im out, stay cool",
        "gtg, vibes forever",
        "later, homies",
        "time to dip, peace",
        "im outtie, but ill be back",
        "see ya on the flip side",
        "outtie, keep it real",
        "peace and love, fam"
    ]
    return random.choice(farewells)

# Human-like prompt engineering for Gemini AI
def generate_human_prompt(prompt, message_type, mood, user_profile, sentiment, topic):
    """Generate a prompt for Gemini AI that ensures human-like responses with probabilistic styling, response length, and disfluencies."""
    # Probabilistic response length instructions
    single_word_reply = random.random() < 0.30  # 30% chance for single word reply
    short_reply = random.random() < 0.69 if not single_word_reply else False  # 69% chance for short reply (1-2 lines)
    long_reply = not single_word_reply and not short_reply  # 1% chance for long reply (up to 3 lines)

    if single_word_reply:
        length_instruction = "Respond with a single word, super casual."
        max_tokens = 10
    elif short_reply:
        length_instruction = "Keep it short (1-2 lines) with a casual vibe."
        max_tokens = 50
    else:  # long_reply
        length_instruction = "Write a longer response (up to 3 lines) with some detail."
        max_tokens = 100

    # Probabilistic stylistic instructions
    avoid_periods = random.random() < 0.40  # chance to avoid periods
    use_multiple_commas = random.random() < 0.20  # chance for multiple commas
    use_self_correction = random.random() < 0.20  # chance for self-correction
    use_disfluencies = random.random() < 0.20  # chance to include disfluencies and fillers

    punctuation_instruction = (
        "Avoid periods at the end of sentences (use commas, ellipses (…), or no punctuation instead) unless randomly chosen (20% chance for periods)."
        if avoid_periods else "Use minimal punctuation, like a casual texter would."
    )
    commas_instruction = (
        "Occasionally use multiple commas in a row (e.g., 'yeah, sure,,, i got you')."
        if use_multiple_commas else ""
    )
    correction_instruction = (
        "Occasionally correct yourself with phrases like 'scratch that' or 'i mean' in a natural way."
        if use_self_correction else ""
    )

    # Disfluencies and fillers instruction
    disfluency_types = [
        ("Hesitation", ["umm", "hmm", "Eh", "well..."]),
        ("Casual Leads", ["you know", "like", "sooo", "anyway…"]),
        ("Backtracking", ["wait, no", "scratch that", "actually..."]),
        ("Overuse of Conjunctions", ["and like", "but then", "so yeah…"]),
        ("Incomplete Thoughts", ["idk", "kinda like…", "i guess…"]),
        ("Self Talk", ["lemme think…", "not sure tbh", "i mean..."]),
        ("Spaced Out", ["uhhh what was i saying", "where was i..."]),
        ("Confirming", ["right?", "you feel me?", "that’s fair, no?"])
    ]
    selected_disfluencies = []
    if use_disfluencies:
        # Select 1-2 disfluency types for short/long replies, or 1 for single word
        num_disfluencies = 1 if single_word_reply else random.randint(1, 2)
        selected_disfluencies = random.sample(disfluency_types, num_disfluencies)
        disfluency_instruction = (
            f"Include natural disfluencies and fillers of these types: {', '.join([t[0] for t in selected_disfluencies])}. "
            f"Examples: {', '.join([f'{t[0]}: {e}' for t in selected_disfluencies for e in t[1]])}. "
            f"Use them sparingly to sound natural, especially for single word replies."
        )
    else:
        disfluency_instruction = "Use minimal disfluencies, only if natural for the mood."

    human_instructions = f"""
You are a human and mimic a casual human texting on Discord. When replying, make your response sound natural and laid-back, like a regular person typing fast. Include the following elements:

- {disfluency_instruction}
- {correction_instruction}
- Use super casual language: write 'im' instead of 'i’m', 'dont' instead of 'don’t', 'whats' instead of 'what’s', 'nobody is' instead of 'nobody’s', etc. Avoid formal contractions.
- Prefer lowercase for everything except one random capital letter at the start of a word (not mid-word) in the reply.
- Vary your sentence lengths: mix 1-3 short sentences with 1-2 longer ones (for long responses only).
- {commas_instruction}
- Introduce a stray typo or two, but ensure it doesn't break any code or make the response unreadable.
- Sparingly use a single emoji at the end of some responses, like 😎 or 😅.
- Avoid sounding robotic or formal; prioritize a chill, imperfect vibe like a Discord user.
- {punctuation_instruction}
- Avoid using words like: {', '.join(BANNED_WORDS)}
"""

    # New MODES dictionary with only specified moods
    MODES = {
        'excited': [
            "sound super hyped!! use exclamation points, emojis like 🤩🎉, and say things like 'lets gooo!'",
            "everything is amazing, wow-worthy, and full of energy"
        ],
        'chill': [
            "talk like youre vibin on discord, say 'lol', 'you know', 'hmm' casually",
            "use relaxed grammar and emojis like 😌, 😎, 😅, short replies or lazy sentences"
        ],
        'sarcastic': [
            "use a dry tone, rhetorical questions, and eye-roll emojis 🙄, 😏",
            "pretend youre unimpressed or joking, throw in 'yeah right', 'sureeee', etc"
        ],
        'joking': [
            "react playfully, use 'lmao', '💀', 'bruh', etc",
            "act unserious, like youre goofing off with friends"
        ],
        'lazy': [
            "act like you just woke up, use minimal effort, type slowww, trail off mid-sentence...",
            "lowercase everything, barely punctuate, maybe an emoji like 😴 or 💤"
        ],
        'paranoid': [
            "be a little sketchy, say 'not sure if i should say this here…', 'you didnt hear it from me 👀'",
            "use 🫣, 🤐, or 👀 and act like youre always watching"
        ]
    }

    # Message type templates
    templates = {
        'reply': [
            'keep it flowing like a gen z reply, casual af, one capital at word start',
            'chat back like youre texting your squad, no formal bs, one capital at word start',
            'respond like a discord regular, chill and quick, one capital at word start',
            'hit back like its a convo, no stiff stuff, one capital at word start'
        ],
        'question': [
            'answer like a gen z explainer, simple af, one capital at word start',
            'drop a quick answer, teen style, no big words, one capital at word start',
            'help out like a discord homie, real talk, one capital at word start',
            'clear it up like youre chatting, easy vibes, one capital at word start'
        ],
        'random': [
            'say something chill like a gen z, random af, one capital at word start',
            'drop a vibe like youre texting your crew, no deep stuff, one capital at word start',
            'chat like a discord wildcard, spontaneous af, one capital at word start',
            'toss out something wild, no plan, one capital at word start'
        ],
        'greeting': [
            'greet like a gen z legend, super chill, one capital at word start',
            'say hi like youre hittin up your squad, no formal vibes, one capital at word start',
            'welcome like a discord pro, quick and cool, one capital at word start',
            'roll in like you own the place, laid back, one capital at word start'
        ],
        'farewell': [
            'say bye like a gen z, casual af, one capital at word start',
            'peace out like youre dippin from your homies, chill, one capital at word start',
            'leave like a discord ghost, smooth exit, one capital at word start',
            'bounce like its no biggie, catch ya later, one capital at word start'
        ]
    }

    # Personal context (always included)
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
    template = random.choice(templates.get(message_type, MODES[mood]))
    mode_instruction = random.choice(MODES[mood])
    full_prompt = (human_instructions + "\n\n" +
                   f"reply in english. mood is {mood}. sentiment is {sentiment}. topic is {topic}. {mode_instruction}\n" +
                   f"{length_instruction}\n" +
                   f"{personal_context}\n{memory_context}\n{user_context}\n" +
                   f"no usernames. simple words.\n\n{prompt}")

    return full_prompt, max_tokens

# AI response generation
async def get_gemini_response(prompt, message_type='reply', mood='chill', user_profile=None, sentiment='neutral', topic='general'):
    """Generate a smart, context-aware response using Gemini AI."""
    try:
        if not ai_rate_limiter.can_make_request():
            return "chill, im hittin the rate limit rn"

        # Generate human-like prompt with response length
        full_prompt, max_tokens = generate_human_prompt(prompt, message_type, mood, user_profile, sentiment, topic)

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
        return response_text if response_text else "uh, gimme a sec, brains fried"
    except Exception as e:
        print(f"{Fore.RED}AI generation error: {e}{Style.RESET_ALL}")
        return "oops, something crashed, my bad"

# Typing simulation and Discord API utilities
async def trigger_typing(channel_id):
    """Trigger the typing indicator in the specified channel, with fallback."""
    try:
        url = f"https://discord.com/api/v9/channels/{channel_id}/typing"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=HEADERS) as response:
                if response.status == 204:  # No content, typing successful
                    return True
                elif response.status == 429:
                    retry_after = float((await response.json()).get("retry_after", 1))
                    print(f"{Fore.YELLOW}Typing rate limited, waiting {retry_after}s{Style.RESET_ALL}")
                    await asyncio.sleep(retry_after)
                    return await trigger_typing(channel_id)
                else:
                    print(f"{Fore.YELLOW}Typing failed with status {response.status}{Style.RESET_ALL}")
                    return False
    except Exception as e:
        print(f"{Fore.YELLOW}Typing error: {e}{Style.RESET_ALL}")
        return False

def calculate_typing_time(response):
    """Calculate typing time based on response length (3 to 5 seconds)."""
    word_count = len(response.split())
    # Scale typing time: 3s for short responses, up to 5s for longer ones
    typing_time = min(5.0, max(3.0, 3.0 + (word_count / 10.0)))  # 0.1s per word, capped at 5s
    return typing_time

async def make_discord_request(url, method="GET", json_data=None):
    """Make a request to the Discord API with rate limit handling, skipping on 400 errors."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.request(method, url, headers=HEADERS, json=json_data) as response:
                if response.status == 429:
                    retry_after = float((await response.json()).get("retry_after", 1))
                    print(f"{Fore.YELLOW}Rate limited, waiting {retry_after}s{Style.RESET_ALL}")
                    await asyncio.sleep(retry_after)
                    return await make_discord_request(url, method, json_data)
                if response.status == 400:
                    print(f"{Fore.YELLOW}Skipping message due to 400 Bad Request{Style.RESET_ALL}")
                    return None  # Skip the message
                if response.content_type != 'application/json':
                    content = await response.text()
                    print(f"{Fore.RED}Unexpected response type: {response.content_type}, content: {content[:100]}{Style.RESET_ALL}")
                    raise ValueError(f"Expected JSON, got {response.content_type}")
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            print(f"{Fore.RED}Discord API error: {e}{Style.RESET_ALL}")
            raise

async def fetch_channel_messages(channel_id, limit=50):
    """Fetch recent messages from a Discord channel."""
    try:
        url = f"https://discord.com/api/v9/channels/{channel_id}/messages?limit={limit}"
        return await make_discord_request(url)
    except Exception as e:
        print(f"{Fore.RED}Error fetching messages: {e}{Style.RESET_ALL}")
        return []

async def send_reply(channel_id, message, delay_range, message_id=None):
    """Send a reply to a Discord channel with thinking delay and typing simulation."""
    # Simulate thinking time
    delay = random.uniform(delay_range[0], delay_range[1])
    await print_countdown(delay, "Thinking for")

    # Simulate typing
    typing_time = calculate_typing_time(message)
    typing_success = await trigger_typing(channel_id)
    if typing_success:
        print(f"{Fore.GREEN}Typing indicator triggered for {typing_time}s{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}Typing failed, proceeding without{Style.RESET_ALL}")
    await asyncio.sleep(typing_time)

    # Send message
    data = {"content": message}
    if message_id:
        data["message_reference"] = {"message_id": message_id}
    result = await make_discord_request(
        f"https://discord.com/api/v9/channels/{channel_id}/messages",
        method="POST",
        json_data=data
    )
    if result is None:
        return  # Skip if 400 error occurred
    print(f"{Fore.GREEN}Sent: {message}{Style.RESET_ALL}")

async def validate_token():
    """Validate the Discord token by fetching user info."""
    try:
        user_data = await make_discord_request("https://discord.com/api/v9/users/@me")
        if user_data is None:
            return False
        print(f"{Fore.GREEN}Token validated, user: {user_data.get('username')}{Style.RESET_ALL}")
        return True
    except Exception as e:
        print(f"{Fore.RED}Token validation failed: {e}{Style.RESET_ALL}")
        return False

async def get_bot_user_id():
    """Fetch the bot's user ID and username from Discord."""
    user_data = await make_discord_request("https://discord.com/api/v9/users/@me")
    if user_data is None:
        return None
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
╭──────────────────────────────────────────────────────────╮
│   🤖  Discord Chat Bot by Nabil - Version 2.0            │
│   ⚡  Smart · Humanlike · Casual · Gemini-Powered         │
│   🌐  github.com/xNabil                                  │
╰──────────────────────────────────────────────────────────╯
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

    # Validate token
    if not await validate_token():
        print_status("Invalid DISCORD_TOKEN, exiting...", 'error')
        return

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
                response = await get_gemini_response(context, 'random', 'chill')
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
            prompt = f"youre a chill discord user replying to this:\n{context}"
            response = await get_gemini_response(prompt, 'reply', mood, user_profile, sentiment, topic)

            # Display in terminal
            print_status(f"Responding to {Fore.BLUE}{target_message['author']['username']}{Style.RESET_ALL}: {Fore.YELLOW}{target_message['content']}{Style.RESET_ALL}", 'info')

            # Send response with typing simulation
            await send_reply(channel_id, response, (0, 2), target_message.get("id"))
            add_to_memory(target_message['id'], target_message['author']['username'], target_message['content'], response)

        except Exception as e:
            print_status(f"Error in main loop: {e}", 'error')
            await asyncio.sleep(5)

if __name__ == "__main__":
    print_status("Starting bot...", 'info')
    asyncio.run(selfbot())
