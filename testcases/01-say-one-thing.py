import argparse
import os
import asyncio

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.services.daily import DailyParams

load_dotenv(override=True)

# Get FastRTC TURN credentials for smooth connection
async def get_turn_config():
    """Get TURN configuration using FastRTC's built-in helpers."""
    try:
        # Import FastRTC credential helpers
        from fastrtc import get_cloudflare_turn_credentials_async
        
        # Try Hugging Face token first (10GB free per month)
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            logger.info("Using Cloudflare TURN with HF token")
            return await get_cloudflare_turn_credentials_async(hf_token=hf_token)
        
        # Try Cloudflare API credentials
        turn_key_id = os.getenv("TURN_KEY_ID") 
        turn_api_token = os.getenv("TURN_KEY_API_TOKEN")
        if turn_key_id and turn_api_token:
            logger.info("Using Cloudflare TURN with API credentials")
            return await get_cloudflare_turn_credentials_async(
                turn_key_id=turn_key_id,
                turn_api_token=turn_api_token
            )
            
    except ImportError:
        logger.warning("FastRTC not available, install with: pip install fastrtc")
    except Exception as e:
        logger.warning(f"Failed to get TURN credentials: {e}")
    
    # Fallback to basic STUN (may not work through all firewalls)
    logger.info("Using basic STUN configuration")
    return {
        "iceServers": [
            {"urls": "stun:stun.l.google.com:19302"},
            {"urls": "stun:stun1.l.google.com:19302"},
        ]
    }

# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(audio_out_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_out_enabled=True),
    "webrtc": lambda: TransportParams(
        audio_out_enabled=True,
        # Use async function to get TURN config
        rtc_config=asyncio.run(get_turn_config())
    ),
}


async def run_example(transport: BaseTransport, _: argparse.Namespace, handle_sigint: bool):
    logger.info(f"Starting bot with transport: {type(transport).__name__}")
    
    # Log RTC configuration for debugging
    if hasattr(transport, '_params') and hasattr(transport._params, 'rtc_config'):
        rtc_config = transport._params.rtc_config
        if rtc_config and 'iceServers' in rtc_config:
            ice_servers = rtc_config['iceServers']
            logger.info(f"✅ Using {len(ice_servers)} ICE servers:")
            for server in ice_servers:
                urls = server['urls']
                if 'turn:' in urls:
                    logger.info(f"  🔄 TURN: {urls}")
                else:
                    logger.info(f"  📡 STUN: {urls}")
        else:
            logger.warning("⚠️ No RTC configuration found!")

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    task = PipelineTask(Pipeline([tts, transport.output()]))

    # Register an event handler so we can play the audio when the client joins
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"🎉 Client connected: {client}")
        await task.queue_frames([TTSSpeakFrame(f"Hello there! Connection successful!"), EndFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"👋 Client disconnected: {client}")

    runner = PipelineRunner(handle_sigint=handle_sigint)

    logger.info("🚀 Pipecat server ready!")
    logger.info("🌐 Navigate to the WebRTC endpoint to test")

    await runner.run(task)


if __name__ == "__main__":
    from pipecat.examples.run import main

    main(run_example, transport_params=transport_params)

# #
# # Copyright (c) 2024–2025, Daily
# #
# # SPDX-License-Identifier: BSD 2-Clause License
# #

# import argparse
# import os

# from dotenv import load_dotenv
# from loguru import logger

# from pipecat.frames.frames import EndFrame, TTSSpeakFrame
# from pipecat.pipeline.pipeline import Pipeline
# from pipecat.pipeline.runner import PipelineRunner
# from pipecat.pipeline.task import PipelineTask
# from pipecat.services.cartesia.tts import CartesiaTTSService
# from pipecat.transports.base_transport import BaseTransport, TransportParams
# from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
# from pipecat.transports.services.daily import DailyParams

# load_dotenv(override=True)
# print(os.environ.get("HF_TOKEN"))

# # # We store functions so objects (e.g. SileroVADAnalyzer) don't get
# # # instantiated. The function will be called when the desired transport gets
# # # selected.
# # transport_params = {
# #     "daily": lambda: DailyParams(audio_out_enabled=True),
# #     "twilio": lambda: FastAPIWebsocketParams(audio_out_enabled=True),
# #     "webrtc": lambda: TransportParams(audio_out_enabled=True),
# # }


# # async def run_example(transport: BaseTransport, _: argparse.Namespace, handle_sigint: bool):
# #     logger.info(f"Starting bot")

# #     tts = CartesiaTTSService(
# #         api_key=os.getenv("CARTESIA_API_KEY"),
# #         voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
# #     )

# #     task = PipelineTask(Pipeline([tts, transport.output()]))

# #     # Register an event handler so we can play the audio when the client joins
# #     @transport.event_handler("on_client_connected")
# #     async def on_client_connected(transport, client):
# #         await task.queue_frames([TTSSpeakFrame(f"Hello there!"), EndFrame()])

# #     runner = PipelineRunner(handle_sigint=handle_sigint)

# #     await runner.run(task)


# # if __name__ == "__main__":
# #     from pipecat.examples.run import main

# #     main(run_example, transport_params=transport_params)
