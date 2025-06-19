import base64
import json
import logging
import os
import re
from typing import Annotated, Sequence

from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
from autogen_agentchat.base._chat_agent import Response
from autogen_agentchat.messages import (
    ChatMessage,
    MultiModalMessage,
    TextMessage,
    ToolCallSummaryMessage,
)
from autogen_core import CancellationToken, Image
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.tools import FunctionTool

from connectors import BlobClient
from tools import get_time, get_today_date, multimodal_vector_index_retrieve
from tools.ragindex.types import MultimodalVectorIndexRetrievalResult

from .base_agent_strategy import BaseAgentStrategy
from ..constants import Strategy


class MultimodalContextFormatterAgent(BaseChatAgent):
    """
    A custom agent that formats RAG retrieval results into the CONTEXT/QUESTION format
    with both text and images for the fine-tuned synthesis SLM.
    """
    def __init__(self, name: str, model_context: BufferedChatCompletionContext):
        super().__init__(
            name=name,
            description="An agent that formats RAG retrieval results with text and images for the synthesis SLM."
        )
        self._model_context = model_context

    @property
    def produced_message_types(self):
        """Return the message types this agent can produce."""
        return (MultiModalMessage,)

    async def on_messages(
        self, 
        messages: Sequence[ChatMessage], 
        cancellation_token: CancellationToken
    ) -> Response:
        """
        Formats the RAG results with text and images into CONTEXT/QUESTION format.
        """
        # Find the original user question
        original_question = None
        for msg in messages:
            if msg.source == "user":
                original_question = msg.content
                break

        # Find the RAG retrieval results
        retrieval_data = None
        for msg in reversed(messages):
            if isinstance(msg, ToolCallSummaryMessage):
                try:
                    msg_content = msg.content
                    parsed_content = json.loads(msg_content)
                    if "texts" in parsed_content or "images" in parsed_content:
                        retrieval_data = parsed_content
                        break
                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse message content as JSON: {e}")
                    continue

        if not retrieval_data or not original_question:
            fallback_msg = TextMessage(
                content="Unable to find RAG retrieval data or original question.",
                source=self.name
            )
            return Response(chat_message=fallback_msg)

        # Extract text chunks and image data from retrieval data
        texts = retrieval_data.get("texts", [])
        image_urls_list = retrieval_data.get("images", [])
        captions_lists = retrieval_data.get("captions", [])
        
        # Format context sections
        context_sections = []
        for i, text in enumerate(texts):
            context_sections.append(f"---\n{text}\n---")

        # Format the complete text message for the synthesis SLM
        formatted_context = "\n".join(context_sections)
        formatted_text = f"CONTEXT:\n{formatted_context}\n\nQUESTION:\n{original_question}"

        # Send text-only message to synthesis agent (compatible with fine-tuned Phi-4)
        text_msg = TextMessage(
            content=formatted_text,
            source=self.name
        )
        return Response(chat_message=text_msg)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset the agent state."""
        pass


class FinalMultimodalResponseAgent(BaseChatAgent):
    """
    A custom agent that combines the synthesis agent's text response with the downloaded images
    to create the final multimodal response for the user (similar to original multimodal strategy).
    """
    def __init__(self, name: str, system_prompt: str, model_context: BufferedChatCompletionContext):
        super().__init__(
            name=name,
            description="An agent that combines text response with images for final multimodal output."
        )
        self._model_context = model_context
        self.system_prompt = system_prompt + "\n\n"

    @property
    def produced_message_types(self):
        """Return the message types this agent can produce."""
        return (MultiModalMessage,)

    async def on_messages(
        self, 
        messages: Sequence[ChatMessage], 
        cancellation_token: CancellationToken
    ) -> Response:
        """
        Combines the synthesis agent's text response with the downloaded images.
        """
        # Find the synthesis agent's response
        synthesis_response = None
        for msg in reversed(messages):
            if msg.source == "synthesis_agent":
                synthesis_response = msg.content
                break

        # Find the RAG retrieval results to get images
        retrieval_data = None
        for msg in reversed(messages):
            if isinstance(msg, ToolCallSummaryMessage):
                try:
                    msg_content = msg.content
                    parsed_content = json.loads(msg_content)
                    if "texts" in parsed_content or "images" in parsed_content:
                        retrieval_data = parsed_content
                        break
                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse message content as JSON: {e}")
                    continue

        if not synthesis_response:
            fallback_msg = TextMessage(
                content="No synthesis response found.",
                source=self.name
            )
            return Response(chat_message=fallback_msg)

        # Use synthesis response as the main content (following original multimodal pattern)
        combined_text = self.system_prompt + synthesis_response if synthesis_response else "No synthesis response"
        logging.debug(f"[fine_tuned_slm_multimodal_strategy] combined_text: {combined_text}")

        # Download and process images if available
        image_objects = []
        if retrieval_data:
            image_urls_list = retrieval_data.get("images", [])
            captions_lists = retrieval_data.get("captions", [])
            
            max_images = 50  # maximum number of images to process
            document_count = 0
            
            for image_urls_list_item in image_urls_list:
                image_count = 0
                for url in image_urls_list_item:
                    if len(image_objects) >= max_images:
                        logging.info(f"[fine_tuned_slm_multimodal_strategy] Reached the maximum image limit of {max_images}.")
                        break
                    try:
                        # Initialize BlobClient with the blob URL
                        blob_client = BlobClient(blob_url=url)
                        logging.debug(f"[fine_tuned_slm_multimodal_strategy] Initialized BlobClient for URL: {url}")
                        
                        # Download the blob data as bytes
                        blob_data = blob_client.download_blob()
                        logging.debug(f"[fine_tuned_slm_multimodal_strategy] Downloaded blob data for URL: {url}")
                        
                        # Open the image using PIL
                        base64_str = base64.b64encode(blob_data).decode('utf-8')
                        pil_img = Image.from_base64(base64_str)
                        logging.debug(f"[fine_tuned_slm_multimodal_strategy] Opened image from URL: {url}")
                        uri = re.sub(r'https://[^/]+\.blob\.core\.windows\.net', '', url)
                        pil_img.filepath = uri
                        logging.debug(f"[fine_tuned_slm_multimodal_strategy] Filepath (uri): {uri}")
                        pil_img.caption = captions_lists[document_count][image_count] if document_count < len(captions_lists) and image_count < len(captions_lists[document_count]) else None                 

                        # Append the PIL Image object to your list
                        image_objects.append(pil_img)
                        image_count += 1
                        logging.info(f"[fine_tuned_slm_multimodal_strategy] Successfully loaded image from {url}")

                    except Exception as e:
                        logging.error(f"[fine_tuned_slm_multimodal_strategy] Could not load image from {url}: {e}")
                    
                    if len(image_objects) >= max_images:
                        break
                        
                document_count += 1
                if len(image_objects) >= max_images:
                    break

        # Construct and return the MultiModalMessage response (like original multimodal strategy)
        multimodal_msg = MultiModalMessage(
            content=[combined_text, *image_objects],
            source=self.name
        )
        return Response(chat_message=multimodal_msg)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset the agent state."""
        pass


class FineTunedSLMMultimodalStrategy(BaseAgentStrategy):
    
    def __init__(self):
        super().__init__()
        self.strategy_type = Strategy.FINE_TUNED_SLM_MULTIMODAL
        
        # Fine-tuned SLM model configuration (Azure AI Inference)
        self.slm_endpoint = os.environ.get('FINE_TUNED_SLM_ENDPOINT', 'https://phi4-finetuned-model-300i.eastus2.models.ai.azure.com')
        self.slm_api_key = os.environ.get('AZURE_INFERENCE_CREDENTIAL', 'uPm3YhuUVHSmqaqJQdWSBr8ksy2gzSwF')
        self.slm_max_tokens = int(os.environ.get('FINE_TUNED_SLM_MAX_TOKENS', 10000))
        self.slm_temperature = float(os.environ.get('FINE_TUNED_SLM_TEMPERATURE', 0.7))
        
    def set_slm_model_config(self, endpoint, api_key, max_tokens=10000, temperature=0.7):
        """Set the fine-tuned SLM model configuration"""
        self.slm_endpoint = endpoint
        self.slm_api_key = api_key
        self.slm_max_tokens = max_tokens
        self.slm_temperature = temperature
        
    def _get_slm_model_client(self):
        """Get model client for the fine-tuned SLM using Azure AI Inference endpoint"""
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        
        # Use OpenAI client with custom endpoint and API key for Azure AI Inference
        return OpenAIChatCompletionClient(
            model="gpt-4o-mini",  # Use a known model name to avoid model_info lookup issues
            api_key=self.slm_api_key,
            base_url=self.slm_endpoint,  # Use endpoint as-is (Azure AI Inference format)
            temperature=self.slm_temperature,
            max_tokens=self.slm_max_tokens
        )

    async def create_agents(self, history, client_principal=None, access_token=None, output_mode=None, output_format=None):
        """
        Fine-tuned SLM Multimodal strategy that implements a SLM workflow with image support:
        1. Query Formulation SLM - transforms user query into precise RAG query
        2. RAG Retrieval - uses vector search to find relevant content (text + images)
        3. Multimodal Context Formatting - formats text and images for synthesis
        4. Synthesis SLM - synthesizes retrieved content into final answer (with image support)
        
        Parameters:
        - history: The conversation history, which will be summarized to provide context for the assistant's responses.
        
        Returns:
        - agent_configuration: A dictionary that includes the agents team, default model client, termination conditions and selector function.
        """

        # Model Context
        shared_context = await self._get_model_context(history) 

        # Wrapper Functions for Tools

        async def vector_index_retrieve_wrapper(
            input: Annotated[str, "An optimized query string based on the user's ask and conversation history, when available"]
        ) -> MultimodalVectorIndexRetrievalResult:
            return await multimodal_vector_index_retrieve(input, self._generate_security_ids(client_principal))

        vector_index_retrieve_tool = FunctionTool(
            vector_index_retrieve_wrapper, 
            name="vector_index_retrieve", 
            description="Performs a vector search using Azure AI Search fetching text and related images to get relevant sources for answering the user's query."
        )

        # Agents

        ## Query Formulation Agent (Fine-tuned SLM - Step 1)
        query_formulation_prompt = await self._read_prompt("query_formulation_agent")
        query_formulation_agent = AssistantAgent(
            name="query_formulation_agent",
            system_message=query_formulation_prompt,
            model_client=self._get_slm_model_client(),  # Use fine-tuned SLM
            tools=[],  # No tools needed for query formulation
            reflect_on_tool_use=False,
            model_context=shared_context
        )

        ## RAG Retrieval Agent  
        rag_retrieval_prompt = await self._read_prompt("rag_retrieval_agent")
        rag_retrieval_agent = AssistantAgent(
            name="rag_retrieval_agent",
            system_message=rag_retrieval_prompt,
            model_client=self._get_model_client(),  # Use regular model
            tools=[vector_index_retrieve_tool],
            reflect_on_tool_use=False,
            model_context=shared_context
        )

        ## Multimodal Context Formatter Agent
        multimodal_context_formatter = MultimodalContextFormatterAgent(
            name="multimodal_context_formatter", 
            model_context=shared_context
        )

        ## Synthesis Agent (Fine-tuned SLM - Step 2) - Text only
        synthesis_prompt = await self._read_prompt("synthesis_agent")
        synthesis_agent = AssistantAgent(
            name="synthesis_agent",
            system_message=synthesis_prompt,
            model_client=self._get_slm_model_client(),  # Use fine-tuned SLM
            tools=[],  # No tools needed for synthesis
            reflect_on_tool_use=True,
            model_context=shared_context
        )

        ## Final Multimodal Response Agent - Combines text response with images for display
        multimodal_rag_message_prompt = await self._read_prompt("multimodal_rag_message")
        final_multimodal_response = FinalMultimodalResponseAgent(
            name="final_multimodal_response", 
            system_prompt=multimodal_rag_message_prompt,
            model_context=shared_context
        )

        ## Chat Closure Agent
        chat_closure = await self._create_chat_closure_agent(output_format, output_mode)

        # Agent Configuration

        def custom_selector_func(messages):
            """
            Selects the next agent based on the source of the last message.
            
            Transition Rules:
                user -> query_formulation_agent (Step 1 - Query Formulation)
                query_formulation_agent -> rag_retrieval_agent (Step 2 - RAG Retrieval)
                rag_retrieval_agent (ToolCallSummaryMessage) -> multimodal_context_formatter (Step 3 - Text Context Formatting)
                multimodal_context_formatter -> synthesis_agent (Step 4 - Text Synthesis)
                synthesis_agent -> final_multimodal_response (Step 5 - Combine text with images)
                final_multimodal_response -> chat_closure (Final output)
            """            
            last_msg = messages[-1]
            logging.debug(f"[fine_tuned_slm_multimodal_strategy] last message: {last_msg}")

            agent_selection = {
                "user": "query_formulation_agent",
                "query_formulation_agent": "rag_retrieval_agent",
                "rag_retrieval_agent": "multimodal_context_formatter" if isinstance(last_msg, ToolCallSummaryMessage) else None,
                "multimodal_context_formatter": "synthesis_agent",
                "synthesis_agent": "final_multimodal_response",
                "final_multimodal_response": "chat_closure",
            }

            selected_agent = agent_selection.get(last_msg.source)

            # Fallback for rag_retrieval_agent without tool call
            if selected_agent is None and last_msg.source == "rag_retrieval_agent":
                selected_agent = "chat_closure"

            if selected_agent:
                logging.debug(f"[fine_tuned_slm_multimodal_strategy] selected {selected_agent} agent")
                return selected_agent

            logging.debug("[fine_tuned_slm_multimodal_strategy] selected None")
            return None
        
        self.selector_func = custom_selector_func

        self.agents = [query_formulation_agent, rag_retrieval_agent, multimodal_context_formatter, synthesis_agent, final_multimodal_response, chat_closure]
        
        return self._get_agents_configuration() 