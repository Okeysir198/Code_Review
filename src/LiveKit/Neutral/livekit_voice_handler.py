"""
LiveKit Voice Handler with LangGraph Integration
Complete migration from FastRTC to LiveKit
"""

import asyncio
import logging
from typing import Optional, Dict, Any, AsyncIterator
from dataclasses import dataclass
import numpy as np

from livekit import rtc, api
from livekit.agents import (
    VoiceAssistant,
    JobContext,
    WorkerOptions,
    cli,
    llm,
    stt,
    tts,
    vad,
    transcription
)
from livekit.agents.voice_assistant import AssistantCallContext
from livekit.agents.pipeline import VoicePipelineAgent
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class AudioConfig:
    """Audio configuration settings"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 20
    vad_sensitivity: float = 0.7
    noise_cancellation: bool = True
    echo_cancellation: bool = True

@dataclass
class ConversationState:
    """State for LangGraph conversation flow"""
    user_input: str = ""
    assistant_response: str = ""
    context: Dict[str, Any] = None
    turn_count: int = 0
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}

class LiveKitVoiceHandler:
    """Main voice handler using LiveKit with LangGraph integration"""
    
    def __init__(
        self,
        room_name: str,
        participant_identity: str,
        audio_config: Optional[AudioConfig] = None,
        langgraph_workflow: Optional[CompiledStateGraph] = None
    ):
        self.room_name = room_name
        self.participant_identity = participant_identity
        self.audio_config = audio_config or AudioConfig()
        self.langgraph_workflow = langgraph_workflow
        
        # LiveKit components
        self.room: Optional[rtc.Room] = None
        self.voice_assistant: Optional[VoiceAssistant] = None
        self.local_audio_track: Optional[rtc.LocalAudioTrack] = None
        
        # Audio processing
        self.audio_buffer = []
        self.is_speaking = False
        
        # Initialize VAD with LiveKit's built-in solution
        self.vad = vad.SileroVAD(
            min_speech_duration=0.1,
            min_silence_duration=0.3,
            padding_duration=0.1,
            sample_rate=self.audio_config.sample_rate,
            activation_threshold=self.audio_config.vad_sensitivity
        )
        
    async def connect(self, livekit_url: str, api_key: str, api_secret: str):
        """Connect to LiveKit room"""
        try:
            # Create room instance
            self.room = rtc.Room()
            
            # Configure audio processing
            options = rtc.RoomOptions(
                auto_subscribe=True,
                adaptive_stream=True,
                dynacast=True,
            )
            
            # Generate token
            token = self._generate_token(api_key, api_secret)
            
            # Connect to room
            await self.room.connect(livekit_url, token, options)
            
            # Set up audio track with noise cancellation
            audio_source = rtc.AudioSource(
                sample_rate=self.audio_config.sample_rate,
                num_channels=self.audio_config.channels,
                noise_suppression=self.audio_config.noise_cancellation,
                echo_cancellation=self.audio_config.echo_cancellation,
                auto_gain_control=True
            )
            
            self.local_audio_track = rtc.LocalAudioTrack.create_audio_track(
                "voice", audio_source
            )
            
            # Publish audio track
            await self.room.local_participant.publish_track(
                self.local_audio_track,
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            )
            
            # Set up voice assistant
            await self._setup_voice_assistant()
            
            logger.info(f"Connected to LiveKit room: {self.room_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to LiveKit: {e}")
            raise
    
    def _generate_token(self, api_key: str, api_secret: str) -> str:
        """Generate LiveKit access token"""
        token = api.AccessToken(api_key, api_secret)
        token.with_identity(self.participant_identity)
        token.with_name(self.participant_identity)
        token.add_grant(
            api.VideoGrants(
                room_join=True,
                room=self.room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True
            )
        )
        return token.to_jwt()
    
    async def _setup_voice_assistant(self):
        """Initialize voice assistant with STT/TTS providers"""
        # Configure STT (Speech-to-Text)
        stt_provider = stt.DeepgramSTT(
            model="nova-2",
            language="en-US",
            punctuate=True,
            interim_results=True
        )
        
        # Configure TTS (Text-to-Speech)
        tts_provider = tts.ElevenLabsTTS(
            voice_id="EXAVITQu4vr4xnSDxMaL",  # Default voice
            model_id="eleven_turbo_v2",
            stability=0.5,
            similarity_boost=0.75
        )
        
        # Configure LLM with LangGraph integration
        if self.langgraph_workflow:
            llm_provider = LangGraphLLMAdapter(self.langgraph_workflow)
        else:
            # Fallback to standard LLM
            llm_provider = llm.OpenAILLM(
                model="gpt-4-turbo",
                temperature=0.7
            )
        
        # Create voice assistant
        self.voice_assistant = VoiceAssistant(
            vad=self.vad,
            stt=stt_provider,
            llm=llm_provider,
            tts=tts_provider,
            allow_interruptions=True,
            interrupt_speech_duration=0.5,
            min_endpointing_delay=0.5,
            transcription=transcription.StreamAdapter(
                stt=stt_provider,
                vad=self.vad
            )
        )
        
        # Set up event handlers
        self.voice_assistant.on("user_speech_started", self._on_user_speech_started)
        self.voice_assistant.on("user_speech_ended", self._on_user_speech_ended)
        self.voice_assistant.on("agent_speech_started", self._on_agent_speech_started)
        self.voice_assistant.on("agent_speech_ended", self._on_agent_speech_ended)
        
        # Start the assistant
        self.voice_assistant.start(self.room)
    
    async def _on_user_speech_started(self, ctx: AssistantCallContext):
        """Handle user speech start"""
        logger.debug("User started speaking")
        self.is_speaking = True
        
    async def _on_user_speech_ended(self, ctx: AssistantCallContext):
        """Handle user speech end"""
        logger.debug("User stopped speaking")
        self.is_speaking = False
        
    async def _on_agent_speech_started(self, ctx: AssistantCallContext):
        """Handle agent speech start"""
        logger.debug("Agent started speaking")
        
    async def _on_agent_speech_ended(self, ctx: AssistantCallContext):
        """Handle agent speech end"""
        logger.debug("Agent stopped speaking")
    
    async def process_audio_stream(self, audio_iterator: AsyncIterator[bytes]):
        """Process incoming audio stream"""
        try:
            async for audio_chunk in audio_iterator:
                if self.local_audio_track and self.local_audio_track.source:
                    # Convert bytes to numpy array
                    audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                    
                    # Capture audio frame
                    frame = rtc.AudioFrame.create(
                        sample_rate=self.audio_config.sample_rate,
                        num_channels=self.audio_config.channels,
                        samples_per_channel=len(audio_data) // self.audio_config.channels
                    )
                    
                    # Copy audio data to frame
                    frame_data = np.frombuffer(frame.data, dtype=np.int16)
                    np.copyto(frame_data, audio_data)
                    
                    # Send to LiveKit
                    await self.local_audio_track.source.capture_frame(frame)
                    
        except Exception as e:
            logger.error(f"Error processing audio stream: {e}")
    
    async def switch_stt_provider(self, provider: str, **kwargs):
        """Switch STT provider dynamically"""
        if not self.voice_assistant:
            return
            
        # Map provider names to implementations
        stt_providers = {
            "deepgram": lambda: stt.DeepgramSTT(**kwargs),
            "google": lambda: stt.GoogleSTT(**kwargs),
            "azure": lambda: stt.AzureSTT(**kwargs),
            "whisper": lambda: stt.WhisperSTT(**kwargs),
        }
        
        if provider in stt_providers:
            new_stt = stt_providers[provider]()
            await self.voice_assistant.update_stt(new_stt)
            logger.info(f"Switched to {provider} STT provider")
    
    async def switch_tts_provider(self, provider: str, **kwargs):
        """Switch TTS provider dynamically"""
        if not self.voice_assistant:
            return
            
        # Map provider names to implementations
        tts_providers = {
            "elevenlabs": lambda: tts.ElevenLabsTTS(**kwargs),
            "google": lambda: tts.GoogleTTS(**kwargs),
            "azure": lambda: tts.AzureTTS(**kwargs),
            "cartesia": lambda: tts.CartesiaTTS(**kwargs),
        }
        
        if provider in tts_providers:
            new_tts = tts_providers[provider]()
            await self.voice_assistant.update_tts(new_tts)
            logger.info(f"Switched to {provider} TTS provider")
    
    async def update_audio_settings(self, **kwargs):
        """Update audio processing settings"""
        if "noise_cancellation" in kwargs:
            self.audio_config.noise_cancellation = kwargs["noise_cancellation"]
        
        if "echo_cancellation" in kwargs:
            self.audio_config.echo_cancellation = kwargs["echo_cancellation"]
            
        if "vad_sensitivity" in kwargs:
            self.audio_config.vad_sensitivity = kwargs["vad_sensitivity"]
            # Update VAD threshold
            if hasattr(self.vad, 'activation_threshold'):
                self.vad.activation_threshold = kwargs["vad_sensitivity"]
    
    async def disconnect(self):
        """Disconnect from LiveKit room"""
        try:
            if self.voice_assistant:
                await self.voice_assistant.stop()
                
            if self.room:
                await self.room.disconnect()
                
            logger.info("Disconnected from LiveKit room")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")


class LangGraphLLMAdapter:
    """Adapter to integrate LangGraph workflows with LiveKit LLM interface"""
    
    def __init__(self, workflow: CompiledStateGraph):
        self.workflow = workflow
        self.conversation_state = ConversationState()
    
    async def chat(self, messages: list, **kwargs) -> str:
        """Process chat through LangGraph workflow"""
        # Extract latest user message
        user_message = messages[-1]["content"] if messages else ""
        
        # Update state
        self.conversation_state.user_input = user_message
        self.conversation_state.turn_count += 1
        
        # Run through LangGraph workflow
        result = await self.workflow.ainvoke({
            "messages": messages,
            "user_input": user_message,
            "context": self.conversation_state.context,
            "turn_count": self.conversation_state.turn_count
        })
        
        # Extract response
        response = result.get("response", "I'm sorry, I didn't understand that.")
        self.conversation_state.assistant_response = response
        
        # Update context for next turn
        if "context" in result:
            self.conversation_state.context.update(result["context"])
        
        return response


# Example LangGraph workflow setup
def create_langgraph_workflow() -> CompiledStateGraph:
    """Create a sample LangGraph workflow for voice conversations"""
    from typing import TypedDict
    
    class ConversationState(TypedDict):
        messages: list
        user_input: str
        context: dict
        response: str
        turn_count: int
    
    # Create workflow
    workflow = StateGraph(ConversationState)
    
    # Define nodes
    async def process_input(state: ConversationState) -> ConversationState:
        """Process user input"""
        # Add your custom logic here
        return state
    
    async def generate_response(state: ConversationState) -> ConversationState:
        """Generate assistant response"""
        # Add your LLM logic here
        state["response"] = f"You said: {state['user_input']}"
        return state
    
    async def update_context(state: ConversationState) -> ConversationState:
        """Update conversation context"""
        # Add context management logic
        return state
    
    # Add nodes to workflow
    workflow.add_node("process_input", process_input)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("update_context", update_context)
    
    # Define edges
    workflow.add_edge(START, "process_input")
    workflow.add_edge("process_input", "generate_response")
    workflow.add_edge("generate_response", "update_context")
    workflow.add_edge("update_context", END)
    
    return workflow.compile()


# For standalone testing
if __name__ == "__main__":
    async def main():
        # Create workflow
        workflow = create_langgraph_workflow()
        
        # Create handler
        handler = LiveKitVoiceHandler(
            room_name="test-room",
            participant_identity="test-user",
            langgraph_workflow=workflow
        )
        
        # Connect
        await handler.connect(
            livekit_url="ws://localhost:7880",
            api_key="your-api-key",
            api_secret="your-api-secret"
        )
        
        # Keep running
        await asyncio.Event().wait()
    
    asyncio.run(main())
