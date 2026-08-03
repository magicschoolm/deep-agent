from dotenv import load_dotenv

load_dotenv()

# Models
DEFAULT_SUMMARIZATION_MODEL = "groq:llama-3.3-70b-versatile"
DEFAULT_RESEARCH_MODEL = "anthropic:claude-haiku-4-5-20251001"
DEFAULT_COMPRESSION_MODEL = "anthropic:claude-haiku-4-5-20251001"
DEFAULT_FINAL_REPORT_MODEL = "anthropic:claude-haiku-4-5-20251001"

# Limits
MAX_RESEARCHER_ITERATIONS = 6
MAX_CONCURRENT_RESEARCHERS = 3