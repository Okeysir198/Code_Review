"""
Hybrid Voice Handler: FastRTC + LiveKit Audio Processing
Maintains FastRTC as core framework while integrating LiveKit's advanced audio features
"""

import asyncio
import logging
import numpy as np
from typing import Optional, Dict, Any, AsyncIterator, Callable
from dataclasses import dataclass
import json

# FastRTC imports (existing)
from fastrtc import RTCPeerConnection, RTCSessionDescription, AudioStreamTrack
from fastrtc.contrib.media import MediaPlayer, MediaRecorder

# LiveKit imports (for audio processing only)
from livekit.agents import vad
from livekit.rtc import AudioFrame
import noisereduce as nr

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class HybridAudioConfig:
    """Audio configuration for hybrid approach"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 20
    
    # LiveKit audio processing features
    use_livekit_vad: bool = True
    vad_sensitivity: float = 0.7
    use_noise_cancellation: bool = True
    noise_reduction_strength: float = 0.8
    use_echo_cancellation: bool = True
    
    # FastRTC settings
    rtc_ice_servers: list = None
    
    def __post_init__(self):
        if self.rtc_ice_servers is None:
            self.rtc_ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]


class LiveKitAudioProcessor:
    """LiveKit audio processing pipeline for FastRTC integration"""
    
    def __init__(self, config: HybridAudioConfig):
        self.config = config
        
        # Initialize LiveKit VAD
        if config.use_livekit_vad:
            self.vad = vad.SileroVAD(
                min_speech_duration=0.1,
                min_silence_duration=0.3,
                padding_duration=0.1,
                sample_rate=config.sample_rate,
                activation_threshold=config.vad_sensitivity
            )
        else:
            self.vad = None
        
        # Audio state tracking
        self.is_speaking = False
        self.speech_start_callback: Optional[Callable] = None
        self.speech_end_callback: Optional[Callable] = None
        
    async def process_audio_chunk(self, audio_data: np.ndarray) -> np.ndarray:
        """Process audio chunk through LiveKit pipeline"""
        processed_audio = audio_data.copy()
        
        # Apply noise cancellation
        if self.config.use_noise_cancellation:
            processed_audio = self._apply_noise_cancellation(processed_audio)
        
        # Apply echo cancellation (simplified version)
        if self.config.use_echo_cancellation:
            processed_audio = self._apply_echo_cancellation(processed_audio)
        
        # Run VAD if enabled
        if self.vad and self.config.use_livekit_vad:
            await self._run_vad(audio_data)
        
        return processed_audio
    
    def _apply_noise_cancellation(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply LiveKit-style noise cancellation"""
        try:
            # Use spectral gating for noise reduction
            reduced_noise = nr.reduce_noise(
                y=audio_data,
                sr=self.config.sample_rate,
                stationary=True,
                prop_decrease=self.config.noise_reduction_strength
            )
            return reduced_noise
        except Exception as e:
            logger.warning(f"Noise cancellation failed: {e}")
            return audio_data
    
    def _apply_echo_cancellation(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply simplified echo cancellation"""
        # This is a placeholder - in production, you'd use a proper AEC algorithm
        # LiveKit uses WebRTC's AEC internally
        return audio_data
    
    async def _run_vad(self, audio_data: np.ndarray):
        """Run LiveKit VAD on audio chunk"""
        # Create AudioFrame for VAD processing
        frame = AudioFrame.create(
            sample_rate=self.config.sample_rate,
            num_channels=self.config.channels,
            samples_per_channel=len(audio_data)
        )
        
        # Check for speech
        speech_probability = await self.vad.detect(frame)
        
        # Handle state transitions
        if speech_probability > self.config.vad_sensitivity:
            if not self.is_speaking:
                self.is_speaking = True
                if self.speech_start_callback:
                    await self.speech_start_callback()
        else:
            if self.is_speaking:
                self.is_speaking = False
                if self.speech_end_callback:
                    await self.speech_end_callback()
    
    def set_speech_callbacks(self, on_start: Callable, on_end: Callable):
        """Set callbacks for speech events"""
        self.speech_start_callback = on_start
        self.speech_end_callback = on_end


class HybridVoiceHandler:
    """Hybrid voice handler combining FastRTC transport with LiveKit audio processing"""
    
    def __init__(
        self,
        config: Optional[HybridAudioConfig] = None,
        langgraph_workflow: Optional[CompiledStateGraph] = None
    ):
        self.config = config or HybridAudioConfig()
        self.langgraph_workflow = langgraph_workflow
        
        # FastRTC components
        self.pc: Optional[RTCPeerConnection] = None
        self.audio_track: Optional[AudioStreamTrack] = None
        self.audio_player: Optional[MediaPlayer] = None
        
        # LiveKit audio processor
        self.audio_processor = LiveKitAudioProcessor(self.config)
        
        # Audio buffering
        self.audio_buffer = []
        self.buffer_lock = asyncio.Lock()
        
        # STT/TTS providers (modular)
        self.stt_provider = None
        self.tts_provider = None
        
        # Conversation state
        self.conversation_active = False
        self.current_transcript = ""
        
    async def initialize(self):
        """Initialize the hybrid voice handler"""
        # Create peer connection
        self.pc = RTCPeerConnection(
            configuration={"iceServers": self.config.rtc_ice_servers}
        )
        
        # Set up event handlers
        self.pc.on("track", self._on_track)
        self.pc.on("connectionstatechange", self._on_connection_state_change)
        
        # Set up speech callbacks
        self.audio_processor.set_speech_callbacks(
            on_start=self._on_speech_start,
            on_end=self._on_speech_end
        )
        
        # Initialize default STT/TTS
        await self._setup_default_providers()
        
        logger.info("Hybrid voice handler initialized")
    
    async def _setup_default_providers(self):
        """Set up default STT/TTS providers"""
        # Import providers dynamically to avoid hard dependencies
        try:
            from deepgram import Deepgram
            self.stt_provider = DeepgramSTTAdapter(api_key="your-key")
        except ImportError:
            logger.warning("Deepgram not available, using mock STT")
            self.stt_provider = MockSTTProvider()
        
        try:
            from elevenlabs import ElevenLabs
            self.tts_provider = ElevenLabsTTSAdapter(api_key="your-key")
        except ImportError:
            logger.warning("ElevenLabs not available, using mock TTS")
            self.tts_provider = MockTTSProvider()
    
    def _on_track(self, track):
        """Handle incoming audio track from FastRTC"""
        if track.kind == "audio":
            logger.info("Received audio track")
            asyncio.create_task(self._process_incoming_audio(track))
    
    async def _on_connection_state_change(self):
        """Handle connection state changes"""
        state = self.pc.connectionState
        logger.info(f"Connection state: {state}")
        
        if state == "connected":
            self.conversation_active = True
        elif state in ["failed", "closed"]:
            self.conversation_active = False
    
    async def _process_incoming_audio(self, track):
        """Process incoming audio through hybrid pipeline"""
        while self.conversation_active:
            try:
                # Receive audio frame from FastRTC
                frame = await track.recv()
                
                # Convert to numpy array
                audio_data = np.frombuffer(frame.to_bytes(), dtype=np.int16)
                
                # Process through LiveKit audio pipeline
                processed_audio = await self.audio_processor.process_audio_chunk(audio_data)
                
                # Buffer processed audio
                async with self.buffer_lock:
                    self.audio_buffer.append(processed_audio)
                
                # If we have enough audio, process through STT
                if len(self.audio_buffer) >= 50:  # ~1 second at 20ms chunks
                    await self._process_buffered_audio()
                    
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                break
    
    async def _process_buffered_audio(self):
        """Process buffered audio through STT"""
        async with self.buffer_lock:
            if not self.audio_buffer:
                return
                
            # Concatenate audio chunks
            audio_data = np.concatenate(self.audio_buffer)
            self.audio_buffer.clear()
        
        # Send to STT
        if self.stt_provider:
            transcript = await self.stt_provider.transcribe(audio_data)
            if transcript:
                self.current_transcript = transcript
                await self._process_transcript(transcript)
    
    async def _process_transcript(self, transcript: str):
        """Process transcript through LangGraph workflow"""
        if not self.langgraph_workflow:
            # Fallback to simple echo
            response = f"You said: {transcript}"
        else:
            # Run through LangGraph
            result = await self.langgraph_workflow.ainvoke({
                "user_input": transcript,
                "context": {}
            })
            response = result.get("response", "I didn't understand that.")
        
        # Generate TTS response
        await self._generate_tts_response(response)
    
    async def _generate_tts_response(self, text: str):
        """Generate and send TTS response"""
        if not self.tts_provider:
            return
            
        # Generate audio
        audio_data = await self.tts_provider.synthesize(text)
        
        # Send through FastRTC
        if self.audio_track:
            # Convert audio to FastRTC format and send
            await self._send_audio_through_rtc(audio_data)
    
    async def _send_audio_through_rtc(self, audio_data: np.ndarray):
        """Send audio data through FastRTC connection"""
        # This would integrate with your existing FastRTC audio sending logic
        pass
    
    async def _on_speech_start(self):
        """Handle speech start event from LiveKit VAD"""
        logger.debug("Speech started (LiveKit VAD)")
        # Could interrupt TTS playback here
    
    async def _on_speech_end(self):
        """Handle speech end event from LiveKit VAD"""
        logger.debug("Speech ended (LiveKit VAD)")
        # Trigger final STT processing
        await self._process_buffered_audio()
    
    async def create_offer(self) -> Dict[str, Any]:
        """Create WebRTC offer"""
        # Add audio transceiver
        self.audio_track = AudioStreamTrack()
        self.pc.addTrack(self.audio_track)
        
        # Create offer
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        
        return {
            "sdp": self.pc.localDescription.sdp,
            "type": self.pc.localDescription.type
        }
    
    async def handle_answer(self, answer: Dict[str, Any]):
        """Handle WebRTC answer"""
        await self.pc.setRemoteDescription(
            RTCSessionDescription(
                sdp=answer["sdp"],
                type=answer["type"]
            )
        )
    
    async def update_audio_settings(self, **kwargs):
        """Update audio processing settings"""
        # Update config
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Update VAD sensitivity if changed
        if "vad_sensitivity" in kwargs and self.audio_processor.vad:
            self.audio_processor.vad.activation_threshold = kwargs["vad_sensitivity"]
        
        # Recreate audio processor if major settings changed
        if any(key in kwargs for key in ["use_livekit_vad", "use_noise_cancellation"]):
            self.audio_processor = LiveKitAudioProcessor(self.config)
            self.audio_processor.set_speech_callbacks(
                on_start=self._on_speech_start,
                on_end=self._on_speech_end
            )
    
    async def switch_stt_provider(self, provider: str, **kwargs):
        """Switch STT provider dynamically"""
        provider_map = {
            "deepgram": lambda: DeepgramSTTAdapter(**kwargs),
            "google": lambda: GoogleSTTAdapter(**kwargs),
            "whisper": lambda: WhisperSTTAdapter(**kwargs),
        }
        
        if provider in provider_map:
            self.stt_provider = provider_map[provider]()
            logger.info(f"Switched to {provider} STT provider")
    
    async def switch_tts_provider(self, provider: str, **kwargs):
        """Switch TTS provider dynamically"""
        provider_map = {
            "elevenlabs": lambda: ElevenLabsTTSAdapter(**kwargs),
            "google": lambda: GoogleTTSAdapter(**kwargs),
            "azure": lambda: AzureTTSAdapter(**kwargs),
        }
        
        if provider in provider_map:
            self.tts_provider = provider_map[provider]()
            logger.info(f"Switched to {provider} TTS provider")
    
    async def close(self):
        """Close the connection"""
        self.conversation_active = False
        if self.pc:
            await self.pc.close()


# STT/TTS Adapter base classes
class STTAdapter:
    """Base class for STT adapters"""
    async def transcribe(self, audio_data: np.ndarray) -> Optional[str]:
        raise NotImplementedError

class TTSAdapter:
    """Base class for TTS adapters"""
    async def synthesize(self, text: str) -> np.ndarray:
        raise NotImplementedError


# Mock providers for testing
class MockSTTProvider(STTAdapter):
    async def transcribe(self, audio_data: np.ndarray) -> Optional[str]:
        return "Mock transcription"

class MockTTSProvider(TTSAdapter):
    async def synthesize(self, text: str) -> np.ndarray:
        # Return silent audio
        duration = len(text) * 0.05  # Rough estimate
        samples = int(16000 * duration)
        return np.zeros(samples, dtype=np.int16)


# Real provider adapters (implement based on your needs)
class DeepgramSTTAdapter(STTAdapter):
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Initialize Deepgram client
    
    async def transcribe(self, audio_data: np.ndarray) -> Optional[str]:
        # Implement Deepgram transcription
        pass

class ElevenLabsTTSAdapter(TTSAdapter):
    def __init__(self, api_key: str, voice_id: str = "default"):
        self.api_key = api_key
        self.voice_id = voice_id
    
    async def synthesize(self, text: str) -> np.ndarray:
        # Implement ElevenLabs synthesis
        pass


# Example usage
if __name__ == "__main__":
    async def main():
        # Create handler
        handler = HybridVoiceHandler()
        await handler.initialize()
        
        # Create offer
        offer = await handler.create_offer()
        print(f"Created offer: {offer['type']}")
        
        # In real usage, you'd exchange SDP with the client
        # and handle the answer
        
        # Keep running
        await asyncio.sleep(60)
        await handler.close()
    
    asyncio.run(main())
