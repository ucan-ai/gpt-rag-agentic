from enum import Enum

# strategies
class Strategy(Enum):
    CLASSIC_RAG = 'classic_rag'
    MULTIMODAL_RAG = 'multimodal_rag'
    NL2SQL = 'nl2sql'
    NL2SQL_FEWSHOT = 'nl2sql_fewshot'
    CHAT_WITH_FABRIC = 'chat_with_fabric'
    FINE_TUNED_SLM = 'fine_tuned_slm'
    FINE_TUNED_SLM_MULTIMODAL = 'fine_tuned_slm_multimodal'

# orchestrator output types
class OutputFormat(Enum):
    TEXT = "text"
    TEXT_TTS = "text_tts"    
    JSON = "json"
    
class OutputMode(Enum):
    STREAMING = "streaming"
    REQUEST_RESPONSE = "request_response"