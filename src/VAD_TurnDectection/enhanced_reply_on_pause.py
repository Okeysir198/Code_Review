"""
ReplyOnPauseWithNoiseReduction - Lean noise cancellation for FastRTC

Extends ReplyOnPause with efficient real-time noise reduction.
"""

import logging
import numpy as np
from typing import Optional, Any, Literal, Callable

from fastrtc.reply_on_pause import ReplyOnPause, ReplyFnGenerator, AlgoOptions, AppState
from fastrtc.pause_detection import ModelOptions, PauseDetectionModel

logger = logging.getLogger(__name__)


class ReplyOnPauseWithNoiseReduction(ReplyOnPause):
    """
    ReplyOnPause with real-time noise cancellation.
    
    Applies noise reduction to audio chunks before VAD processing,
    improving both start/pause detection and providing clean audio to AI.
    
    Args:
        fn: Response generator function
        noise_model: Model name ("deepfilternet3", "mossformer2_se_48k", etc.)
        enable_noise_reduction: Enable/disable noise reduction
        **kwargs: Standard ReplyOnPause parameters
        
    Example:
        ```python
        def response(audio):
            yield audio  # Echo clean audio
        
        handler = ReplyOnPauseWithNoiseReduction(response)
        stream = Stream(handler=handler)
        ```
    """
    
    def __init__(
        self,
        fn: ReplyFnGenerator,
        noise_model: str = "deepfilternet3",
        enable_noise_reduction: bool = True,
        **kwargs
    ):
        super().__init__(fn, **kwargs)
        self.noise_model = noise_model
        self.enable_noise_reduction = enable_noise_reduction
        self._noise_reducer = None  # Lazy init
    
    @property
    def noise_reducer(self):
        """Lazy load noise reduction model"""
        if self._noise_reducer is None and self.enable_noise_reduction:
            try:
                from src.NoiseCancelation.fastrtc_noise_reduction import FastRTCNoiseReduction
                self._noise_reducer = FastRTCNoiseReduction(self.noise_model)
                logger.debug(f"Loaded noise model: {self.noise_model}")
            except Exception as e:
                logger.warning(f"Noise reduction failed to load: {e}")
                self.enable_noise_reduction = False
        return self._noise_reducer
    
    def process_audio(self, audio: tuple[int, np.ndarray], state: AppState) -> None:
        """Apply noise reduction then continue with normal processing"""
        frame_rate, array = audio
        array = np.squeeze(array)
        
        # Apply noise reduction to chunk before VAD processing
        if self.enable_noise_reduction and self.noise_reducer:
            try:
                array = self.noise_reducer.process_audio_chunk(array, frame_rate)
                array = array.astype(np.int16)  # Ensure correct dtype
            except Exception as e:
                logger.debug(f"Noise reduction error: {e}")
                # Continue with original audio
        
        # Continue with ReplyOnPause logic using clean audio
        super().process_audio((frame_rate, array), state)
    
    def copy(self):
        """Create a copy with same configuration"""
        return ReplyOnPauseWithNoiseReduction(
            fn=self.fn,
            noise_model=self.noise_model,
            enable_noise_reduction=self.enable_noise_reduction,
            startup_fn=self.startup_fn,
            algo_options=self.algo_options,
            model_options=self.model_options,
            can_interrupt=self.can_interrupt,
            expected_layout=self.expected_layout,
            output_sample_rate=self.output_sample_rate,
            input_sample_rate=self.input_sample_rate,
            model=self.model,
            needs_args=self._needs_additional_inputs
        )


# Quick constructor function
def NoiseReducedReply(
    fn: ReplyFnGenerator,
    model: str = "deepfilternet3",
    **kwargs
) -> ReplyOnPauseWithNoiseReduction:
    """
    Quick constructor for noise-reduced ReplyOnPause.
    
    Args:
        fn: Response function
        model: Noise reduction model
        **kwargs: ReplyOnPause parameters
    """
    return ReplyOnPauseWithNoiseReduction(
        fn=fn,
        noise_model=model,
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    def echo(audio):
        """Echo the clean audio back"""
        yield audio
    
    # Simple usage
    handler = ReplyOnPauseWithNoiseReduction(echo)
    
    # Or with quick constructor
    handler2 = NoiseReducedReply(echo, model="mossformer2_se_48k")
    
    print(f"Handler: {handler}")
    print(f"Noise enabled: {handler.enable_noise_reduction}")