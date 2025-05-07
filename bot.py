# Discord Chat Bot by Nabil
# GitHub: https://github.com/xNabil
# Version: Enhanced with Priority Replies, Smart AI, Banned Words Handling, Probabilistic Response Lengths, Disfluencies, and Casual Professional Style

import os
import aiohttp
import random
import time
import asyncio
import google.generativeai as genai
import json
import sqlite3
from dotenv import load_dotenv
from dateutil.parser import isoparse
from colorama import init, Fore, Style
import logging
from textblob import TextBlob
import re
from collections import deque

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
USER_PROFILES = {}
BANNED_WORDS = ['hi', 'fire', 'hello', 'lit', 'blaze']  # Add more banned words as needed
RESPONDED_MESSAGES = deque(maxlen=100)  # In-memory cache for recent responses
RESPONSE_COUNTER = 0  # Track responses for pattern enforcement
RESPONSE_TYPES = []  # Track response types for 15-message cycle

# Topic keywords for detection
TOPICS = {
    'gaming': ['game', 'play', 'console', 'pc', 'xbox', 'playstation', 'controller', 'gamer'],
    'music': ['song', 'album', 'artist', 'band', 'concert', 'beats', 'tune'],
    'movies': ['film', 'movie', 'cinema', 'actor', 'director', 'scene', 'plot'],
    'tech': ['code', 'tech', 'software', 'hardware', 'gadget', 'app', 'update'],
    'food': ['food', 'eat', 'cook', 'recipe', 'snack', 'meal', 'yummy'],
    'anime': ['anime', 'manga', 'episode', 'series', 'character', 'aot', 'naruto', 'one piece']
}

# Regex patterns for user profiling
FAVORITE_GAME_PATTERN = re.compile(r"my favorite game is (\w+)", re.IGNORECASE)
FAVORITE_FOOD_PATTERN = re.compile(r"my favorite food is (\w+)", re.IGNORECASE)
FAVORITE_ANIME_PATTERN = re.compile(r"my favorite anime is (\w+)", re.IGNORECASE)

# SQLite database initialization
def init_db():
    """Initialize SQLite database for conversation memory and user profiles."""
    try:
        conn = sqlite3.connect("conversations.db")
        c = conn.cursor()
        # Create memory table
        c.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT UNIQUE,
                author TEXT,
                content TEXT,
                bot_response TEXT,
                timestamp REAL,
                topic TEXT
            )
        """)
        # Create user profiles table
        c.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                profile_data TEXT
            )
        """)
        conn.commit()
        print(f"{Fore.GREEN}SQLite database initialized{Style.RESET_ALL}")
        return conn
    except Exception as e:
        print(f"{Fore.RED}Error initializing database: {e}{Style.RESET_ALL}")
        return None

DB_CONN = init_db()

# Load personal info from myinfo.txt
def load_personal_info():
    """Load bot's personal info from myinfo.txt into a dictionary."""
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
                        info[key.lower()] = value
                    elif key == "bio" and line:
                        info[key.lower()] = (info.get(key.lower(), "") + " " + line).strip()
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
def ensure_db_connection():
    """Ensure the database connection is open, reinitialize if closed."""
    global DB_CONN
    if DB_CONN is None or (hasattr(DB_CONN, 'cursor') and DB_CONN.cursor() is None):
        print(f"{Fore.YELLOW}Database connection closed, reinitializing...{Style.RESET_ALL}")
        DB_CONN = init_db()
    return DB_CONN

def add_to_memory(message_id, author, content, bot_response=None, topic='general'):
    """Add a message and bot response to SQLite memory."""
    try:
        conn = ensure_db_connection()
        if conn is None:
            raise Exception("Failed to initialize database connection")
        c = conn.cursor()
        c.execute("""
            INSERT OR IGNORE INTO memory (message_id, author, content, bot_response, timestamp, topic)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (message_id, author, content, bot_response, time.time(), topic))
        # Purge entries older than 24 hours
        c.execute("DELETE FROM memory WHERE timestamp < ?", (time.time() - 24 * 3600,))
        conn.commit()
        print(f"{Fore.GREEN}Added to memory{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error adding to memory: {e}{Style.RESET_ALL}")

def get_memory_context():
    """Get recent conversation context from SQLite for AI prompts."""
    try:
        conn = ensure_db_connection()
        if conn is None:
            raise Exception("Failed to initialize database connection")
        c = conn.cursor()
        c.execute("SELECT author, content FROM memory ORDER BY timestamp DESC LIMIT 10")
        recent = c.fetchall()
        if recent:
            return "\n".join([f"{author}: {content}" for author, content in recent])
        return ""
    except Exception as e:
        print(f"{Fore.RED}Error retrieving memory: {e}{Style.RESET_ALL}")
        return ""

def has_responded(message_id):
    """Check if the bot has already responded to a message."""
    try:
        conn = ensure_db_connection()
        if conn is None:
            raise Exception("Failed to initialize database connection")
        c = conn.cursor()
        c.execute("SELECT bot_response FROM memory WHERE message_id = ?", (message_id,))
        result = c.fetchone()
        return result is not None and result[0] is not None
    except Exception as e:
        print(f"{Fore.RED}Error checking response status: {e}{Style.RESET_ALL}")
        return False

# User profile management
def load_user_profiles():
    """Load user profiles from SQLite database."""
    global USER_PROFILES
    try:
        conn = ensure_db_connection()
        if conn is None:
            raise Exception("Failed to initialize database connection")
        c = conn.cursor()
        c.execute("SELECT user_id, profile_data FROM user_profiles")
        for user_id, profile_data in c.fetchall():
            USER_PROFILES[user_id] = json.loads(profile_data)
        print(f"{Fore.GREEN}Loaded user profiles{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error loading user profiles: {e}{Style.RESET_ALL}")
        USER_PROFILES = {}

def save_user_profiles():
    """Save user profiles to SQLite database."""
    try:
        conn = ensure_db_connection()
        if conn is None:
            raise Exception("Failed to initialize database connection")
        c = conn.cursor()
        for user_id, profile in USER_PROFILES.items():
            c.execute("""
                INSERT OR REPLACE INTO user_profiles (user_id, profile_data)
                VALUES (?, ?)
            """, (user_id, json.dumps(profile)))
        conn.commit()
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
        sanitized = re.sub(rf'\b{word}\b', '***', sanitized, flags=re.IGNORECASE)
    return sanitized if sanitized != message.lower() else message

async def rephrase_message(original_message, mood, sentiment, topic):
    """Rephrase a message to avoid banned words using Gemini AI."""
    try:
        prompt = f"rephrase this to avoid words like {', '.join(BANNED_WORDS)} while keeping the same vibe, mood ({mood}), sentiment ({sentiment}), and topic ({topic}), use plain text, no markdown:\n{original_message}"
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
    """Return random emojis based on the bot's mood, used sparingly."""
    emoji_map = {
        'excited': ['🤩', '🥳', '💥', '🎉'],
        'chill': ['😌', '🍃', '🛋️', '✌️', '😎'],
        'sarcastic': ['🙄', '😏', '🤷', '😒', '👀'],
        'joking': ['😂', '🤣', '😜', '😝', '🤡'],
        'lazy': ['😴', '💤', '🛌', '😪', '🥱'],
        'paranoid': ['🫣', '🤐', '👀', '😬', '🙈']
    }
    emojis = emoji_map.get(mood, emoji_map['chill'])
    if random.random() < 0.067:  # ~1 in 15 chance
        return ''.join(random.choice(emojis) for _ in range(count))
    return ''

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
    """Extract preferences like favorite game, food, or anime from text."""
    preferences = {}
    game_match = FAVORITE_GAME_PATTERN.search(text)
    if game_match:
        preferences['favorite_game'] = game_match.group(1)
    food_match = FAVORITE_FOOD_PATTERN.search(text)
    if food_match:
        preferences['favorite_food'] = game_match.group(1)
    anime_match = FAVORITE_ANIME_PATTERN.search(text)
    if anime_match:
        preferences['favorite_anime'] = anime_match.group(1)
    return preferences

# Response templates with casual professional style
def get_random_question():
    """Return a random question to spark conversation."""
    questions = [
        "what game you into these days",
        "whos making the best music rn",
        "seen any good movies lately",
        "what tech you messing with",
        "whats your favorite thing to eat",
        "what anime you watching rn",
        "whos an artist youd love to collab with",
        "whats the wildest thing you done lately",
        "got something cool to share drop it",
        "hows your day going fr"
    ]
    return random.choice(questions)

def get_random_message():
    """Return a random casual message."""
    messages = [
        "just chilling whats up",
        "this place wild huh",
        "anyone craving a snack rn",
        "we keeping it real in here",
        "this servers got some energy",
        "lets mix things up you in",
        "just here for the good chats",
        "whats the latest thing going on",
        "this groups pretty solid",
        "ready to make things interesting"
    ]
    return random.choice(messages)

def get_greeting():
    """Return a casual greeting."""
    greetings = [
        "yo whats good",
        "hey hows the crew doing",
        "whats up everyone",
        "whos running things around here",
        "ready to jump in or what",
        "whats the vibe today",
        "sup hows it going",
        "is this the place to be",
        "lets get things rolling",
        "yo good to see you all"
    ]
    return random.choice(greetings)

def get_farewell():
    """Return a casual farewell."""
    farewells = [
        "alright im out see you",
        "catch you later stay cool",
        "gotta go keep it real",
        "im off take care",
        "see you around folks",
        "heading out later",
        "alright im done peace",
        "back soon hold it down",
        "time to dip stay good",
        "later everyone be easy"
    ]
    return random.choice(farewells)

# Human-like prompt engineering for Gemini AI
def generate_human_prompt(prompt, message_type, mood, user_profile, sentiment, topic):
    """Generate a prompt for Gemini AI with human-like, casual responses."""
    global RESPONSE_COUNTER, RESPONSE_TYPES

    # Increment response counter and reset every 15 messages
    RESPONSE_COUNTER = (RESPONSE_COUNTER + 1) % 15
    if RESPONSE_COUNTER == 0:
        RESPONSE_TYPES = []

    # Determine response type based on pattern (over 15 messages)
    single_word_count = RESPONSE_TYPES.count('single_word')
    short_reply_count = RESPONSE_TYPES.count('short')
    long_reply_count = RESPONSE_TYPES.count('long')

    if long_reply_count == 0 and random.random() < 0.067:  # 1 long reply per 15
        response_type = 'long'
        length_instruction = "write 2 to 4 sentences max with personality no more"
        max_tokens = 100
    elif single_word_count < 3 and random.random() < 0.2:  # 3 or fewer single-word replies
        response_type = 'single_word'
        length_instruction = "respond with exactly one word like yo nah fr cool"
        max_tokens = 10
    else:  # Aim for 11 short replies
        response_type = 'short'
        length_instruction = "exactly one sentence short and chill no extra lines"
        max_tokens = 30

    RESPONSE_TYPES.append(response_type)

    # Stylistic instructions
    use_multiple_commas = random.random() < 0.30  # 30% chance for multiple commas
    use_self_correction = random.random() < 0.20  # 20% chance for self-correction
    use_disfluencies = random.random() < 0.20  # 20% chance for disfluencies

    punctuation_instruction = "no quotes apostrophes periods semicolons colons dashes underscores just commas or nothing"
    commas_instruction = (
        "use multiple commas sometimes like yo,,, sounds good"
        if use_multiple_commas else "use single commas if needed"
    )
    correction_instruction = (
        "correct yourself sometimes like wait i mean or my bad"
        if use_self_correction else ""
    )

    # Disfluencies and fillers
    disfluency_types = [
        ("Hesitation", ["well", ]),
        ("Casual Leads", ["so like", "honestly", "basically"]),
        ("Backtracking", ["wait i mean", "my bad", "hold on"]),
        ("Overuse of Conjunctions", ["and uh", "but like", "so yeah"]),
        ("Incomplete Thoughts", ["not sure", "kinda", "maybe"]),
        ("Self Talk", ["lets see", "i guess", "thinking"]),
        ("Spaced Out", ["what was i saying", "got distracted"]),
        ("Confirming", ["right", "you follow", "cool"])
    ]
    selected_disfluencies = []
    if use_disfluencies:
        num_disfluencies = 1 if max_tokens <= 10 else random.randint(1, 2)
        selected_disfluencies = random.sample(disfluency_types, num_disfluencies)
        disfluency_instruction = (
            f"use natural disfluencies like {', '.join([t[0] for t in selected_disfluencies])}. "
            f"examples: {', '.join([f'{t[0]}: {e}' for t in selected_disfluencies for e in t[1]])}"
        )
    else:
        disfluency_instruction = "skip disfluencies unless it fits"

    human_instructions = f"""
youre a chill discord user not a bot, keep it super casual and humanlike like texting friends. heres the vibe:

- {disfluency_instruction}
- {correction_instruction}
- all lowercase no caps unless its a name
- use im not i am, whats not what is, gonna not going to
- one sentence max for short replies no extra lines
- {commas_instruction}
- minor typos ok but keep it readable
- no emoji spam add one only if i say so
- stay witty friendly never robotic think discord group chat energy
- {punctuation_instruction}
- plain text no markdown no ** * _ etc
- avoid words like {', '.join(BANNED_WORDS)}
- stick to the topic and sentiment
- use myinfo.txt for preferences like favorite anime game food, if asked about something specific like is aot your fav anime check myinfo.txt and respond based on it, if no info say you dont have a fave but suggest something
- no multi line responses unless i say two to four sentences
- dont repeat previous responses keep it fresh
"""

    # MODES dictionary with casual style
    MODES = {
        'excited': [
            "sound hyped use exclamations say stuff like thats sick",
            "act like its a big deal high energy but clean"
        ],
        'chill': [
            "stay relaxed say cool nice hey like you’re vibing",
            "keep it simple short and easy"
        ],
        'sarcastic': [
            "use a sharp tone say oh really or sure thing",
            "throw light shade but stay friendly"
        ],
        'joking': [
            "be playful",
            "keep it fun like teasing a friend"
        ],
        'lazy': [
            "type like you’re tired say eh not sure or too much work",
            "low effort maybe a sleepy vibe"
        ],
        'paranoid': [
            "act cautious say is this fr or better watch out",
            "sound like you’re double-checking stuff"
        ]
    }

    # Message type templates
    templates = {
        'reply': [
            'respond like a chill user friendly witty plain text',
            'chat like you’re in a group convo plain text',
            'answer like you’re keeping the vibe going short sharp plain text',
            'hit back like its a casual talk engaging plain text'
        ],
        'question': [
            'answer like explaining to a friend simple clear plain text',
            'drop a quick response casual no fancy words plain text',
            'break it down like you’re chatting keep it real plain text',
            'clear it up like you’re helping out no fluff plain text'
        ],
        'random': [
            'say something fun like a chill user random but clean plain text',
            'drop a casual comment like texting friends plain text',
            'talk like you’re sparking a convo spontaneous witty plain text',
            'throw out something light no script plain text'
        ],
        'greeting': [
            'greet like a friendly user welcoming clean plain text',
            'say yo like you’re joining a group no weak vibes plain text',
            'welcome like you’re happy to be here quick cool plain text',
            'jump in like you’re part of the crew keep it light plain text'
        ],
        'farewell': [
            'leave like a chill user casual friendly plain text',
            'dip out like saying bye to friends smooth exit plain text',
            'sign off like its no big deal stay cool plain text',
            'head out like you’ll be back keep it easy plain text'
        ]
    }

    # Personal context
    personal_context = (
    f"use this from myinfo.txt to answer preference questions: "
    f"name: {PERSONAL_INFO.get('name', 'unknown')}, "
    f"age: {PERSONAL_INFO.get('age', 'unknown')}, "
    f"interests: {PERSONAL_INFO.get('interests', 'none')}, "
    f"location: {PERSONAL_INFO.get('location', 'somewhere')}, "
    f"favorite_anime: {PERSONAL_INFO.get('favorite_anime', 'none')}, "
    f"favorite_game: {PERSONAL_INFO.get('favorite_game', 'none')}, "
    f"favorite_food: {PERSONAL_INFO.get('favorite_food', 'none')}, "
    f"discord_level: {PERSONAL_INFO.get('discord_level', 'none')}, "
    f"tvl: {PERSONAL_INFO.get('tvl', 'none')}, "
    f"trading_volume: {PERSONAL_INFO.get('trading_volume', 'none')}, "
    f"bio: {PERSONAL_INFO.get('bio', 'just a chill bot')}"
)

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
            user_context += f"they’re into {user_profile['favorite_food']} "
        if topic == 'anime' and 'favorite_anime' in user_profile:
            user_context += f"they’re into {user_profile['favorite_anime']} "

    # Construct full prompt
    template = random.choice(templates.get(message_type, MODES[mood]))
    mode_instruction = random.choice(MODES[mood])
    full_prompt = (
        human_instructions + "\n\n" +
        f"reply in english, mood is {mood}, sentiment is {sentiment}, topic is {topic}, {mode_instruction}\n" +
        f"{length_instruction}\n" +
        f"{personal_context}\n{memory_context}\n{user_context}\n" +
        f"no usernames, simple words, plain text no markdown\n\n{prompt}"
    )

    return full_prompt, max_tokens

# AI response generation
async def get_gemini_response(prompt, message_type='reply', mood='chill', user_profile=None, sentiment='neutral', topic='general'):
    """Generate a smart, context-aware response using Gemini AI."""
    try:
        if not ai_rate_limiter.can_make_request():
            return "yo im maxed out rn give me a sec"

        # Generate human-like prompt
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
        response_text = response.text.strip().lower()

        # Strip any markdown and unwanted punctuation
        response_text = re.sub(r'[\*\_\~\`\#\'\"\;\:\-\_]+', '', response_text)
        response_text = response_text.replace("'", "").replace('"', "")

        # Enforce length constraints
        sentences = [s.strip() for s in response_text.split('.') if s.strip()]
        if max_tokens <= 10:  # 1-word reply
            response_text = response_text.split()[0]
        elif max_tokens <= 30:  # 1-sentence reply
            response_text = sentences[0] if sentences else response_text
        else:  # 2-4 sentence reply
            response_text = ' '.join(sentences[:4]) if sentences else response_text

        # Add emojis sparingly
        response_text += get_random_emojis(1, mood)

        return response_text if response_text else "uh my brains blank give me a sec"
    except Exception as e:
        print(f"{Fore.RED}AI generation error: {e}{Style.RESET_ALL}")
        return "oops something broke my bad"

# Typing simulation and Discord API utilities
async def trigger_typing(channel_id):
    """Trigger the typing indicator in the specified channel, with fallback."""
    try:
        url = f"https://discord.com/api/v9/channels/{channel_id}/typing"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=HEADERS) as response:
                if response.status == 204:
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
    typing_time = min(5.0, max(3.0, 3.0 + (word_count / 10.0)))
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
                    return None
                if response.content_type != 'application/json':
                    content = await response.text()
                    print(f"{Fore.RED}Unknown response type: {response.content_type}, content: {content[:100]}{Style.RESET_ALL}")
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
    # Truncate message if too long
    max_length = 2000
    if len(message) > max_length:
        message = message[:max_length - 3] + "..."
        print(f"{Fore.YELLOW}Message truncated to fit Discord limit{Style.RESET_ALL}")

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
        return
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
    return user_data.get('id')

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
│   🤖  Discord Chat Bot by Nabil - Version 3.0            │
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
    print_status("Getting bot ready...", 'info')

    # Validate token
    if not await validate_token():
        print_status("DISCORD_TOKEN is invalid, shutting down...", 'error')
        return

    # Load profiles
    load_user_profiles()

    # Get bot user ID
    BOT_USER_ID = await get_bot_user_id()
    if not BOT_USER_ID:
        print_status("Can’t grab bot ID, check DISCORD_TOKEN", 'error')
        return

    channel_id = CHANNEL_ID
    if not channel_id or not channel_id.isdigit():
        print_status("CHANNEL_ID is invalid", 'warning')
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
            print_status("SLOW_MODE invalid, using 5-10s default", 'warning')
    else:
        CHANNEL_SLOW_MODES[channel_id] = default_slow_mode
        print_status("No SLOW_MODE set, using 5-10s", 'warning')

    print_status(f"Bot’s live on channel {channel_id} with slow mode {CHANNEL_SLOW_MODES[channel_id]}s", 'success')

    try:
        while True:
            try:
                wait_time = random.uniform(*CHANNEL_SLOW_MODES[channel_id])
                await print_countdown(wait_time, "Checking for")

                # Fetch messages
                messages = await fetch_channel_messages(channel_id, 50)
                if not messages:
                    print_status("No messages found, trying again...", 'warning')
                    continue

                # Update user profiles
                for msg in messages:
                    prefs = extract_user_preferences(msg.get('content', ''))
                    for key, value in prefs.items():
                        update_user_profile(msg['author']['id'], key, value)

                # Categorize messages, exclude already responded
                replies_to_bot = [
                    msg for msg in messages
                    if msg.get("referenced_message", {}).get("author", {}).get("id") == BOT_USER_ID
                    and not is_message_old(msg.get("timestamp", ""))
                    and not has_responded(msg.get("id"))
                    and msg.get("id") not in RESPONDED_MESSAGES
                ]
                other_messages = [
                    msg for msg in messages
                    if msg.get("content")
                    and not is_message_old(msg.get("timestamp", ""))
                    and msg.get("author", {}).get("id") != BOT_USER_ID
                    and not has_responded(msg.get("id"))
                    and msg.get("id") not in RESPONDED_MESSAGES
                    and msg not in replies_to_bot
                ]

                # Check for inactivity and initiate conversation
                last_message_time = max([isoparse(msg['timestamp']).timestamp() for msg in messages])
                if time.time() - last_message_time > 300 and random.random() < 0.2:
                    print_status("Chat’s quiet, starting something...", 'info')
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
                    print_status("Nothing to respond to", 'info')
                    continue

                # Analyze message
                sentiment = get_sentiment(target_message['content'])
                topic = detect_topic(target_message['content'])
                mood = get_bot_mood(sentiment)
                user_profile = get_user_profile(target_message['author']['id'])

                # Generate response
                context = f"they said: {target_message['content']}"
                prompt = f"you’re a chill discord user responding to this:\n{context}"
                response = await get_gemini_response(prompt, 'reply', mood, user_profile, sentiment, topic)

                # Display in terminal
                print_status(
                    f"Replying to {Fore.BLUE}{target_message['author']['username']}{Style.RESET_ALL}: "
                    f"{Fore.YELLOW}{target_message['content']}{Style.RESET_ALL}", 'info'
                )

                # Send response with typing simulation
                await send_reply(channel_id, response, (0, 2), target_message.get("id"))
                add_to_memory(target_message['id'], target_message['author']['username'], target_message['content'], response, topic)
                RESPONDED_MESSAGES.append(target_message['id'])

            except Exception as e:
                print_status(f"Main loop error: {e}", 'error')
                await asyncio.sleep(5)

    except KeyboardInterrupt:
        print_status("Shutting down bot...", 'info')
    finally:
        if DB_CONN is not None:
            DB_CONN.close()
            print_status("Database connection closed", 'info')

if __name__ == "__main__":
    print_status("Starting bot...", 'info')
    asyncio.run(selfbot())