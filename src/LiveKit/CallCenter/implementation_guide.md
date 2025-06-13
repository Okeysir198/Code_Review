# LiveKit Integration Implementation Guide

## Overview

This guide covers the setup and deployment of both LiveKit integration approaches:
- **Version 1**: Full LiveKit migration (self-hosted, no API keys)
- **Version 2**: Hybrid approach (FastRTC + LiveKit audio processing)

## Prerequisites

### Common Requirements
```bash
# Python 3.8+
pip install gradio>=4.0.0
pip install numpy
pip install langchain langchain-core langgraph
pip install asyncio
pip install pyaudio

# For Version 2 (Hybrid)
pip install noisereduce
pip install scipy
```

### Version 1 Requirements
```bash
# LiveKit Python SDK
pip install livekit livekit-agents

# Self-hosted LiveKit server (using Docker)
docker run -d \
  -p 7880:7880 \
  -p 7881:7881 \
  -p 7882:7882/udp \
  -e LIVEKIT_KEYS="devkey: secret" \
  -e LIVEKIT_CONFIG="
port: 7880
rtc:
  tcp_port: 7881
  udp_port: 7882
  use_external_ip: false
" \
  livekit/livekit-server:latest
```

### Version 2 Requirements
```bash
# Existing FastRTC setup (no changes needed)
# Additional audio processing libraries
pip install livekit-plugins-silero  # For LiveKit VAD only
```

## Version 1: Full LiveKit Setup

### 1. Server Setup

```bash
# Start LiveKit server (self-hosted)
docker-compose.yml:
```
```yaml
version: '3.8'
services:
  livekit:
    image: livekit/livekit-server:latest
    ports:
      - "7880:7880"
      - "7881:7881"
      - "7882:7882/udp"
    environment:
      - LIVEKIT_KEYS=devkey:secret
      - LIVEKIT_WEBHOOK_URLS=
      - LIVEKIT_CONFIG=
          port: 7880
          bind_addresses:
            - "0.0.0.0"
          rtc:
            tcp_port: 7881
            udp_port: 7882
            use_external_ip: false
    volumes:
      - ./livekit-data:/data
```

### 2. Code Integration

```python
# Update imports in your main application
from src.VoiceHandler.voice_handler import LiveKitVoiceInteractionHandler as VoiceInteractionHandler
```

### 3. Client Connection

The Gradio interface handles client connections automatically. For custom clients:

```javascript
// Client-side LiveKit connection
import { Room, RoomEvent, Track } from 'livekit-client';

const room = new Room({
  adaptiveStream: true,
  dynacast: true,
  publishDefaults: {
    audioPreset: {
      maxBitrate: 32000,
    },
  },
});

// Connect to self-hosted server
await room.connect('ws://localhost:7880', token, {
  autoSubscribe: true,
});
```

### 4. Running Version 1

```bash
# Start LiveKit server
docker-compose up -d

# Run the application
python voice_chat_test_app.py

# Access at http://localhost:7860
```

## Version 2: Hybrid Setup

### 1. Update FastRTC Installation

Replace the default `reply_on_pause.py` with the enhanced version:

```bash
# Backup original
cp backend/fastrtc/reply_on_pause.py backend/fastrtc/reply_on_pause.py.bak

# Copy enhanced version
cp hybrid_reply_on_pause.py backend/fastrtc/reply_on_pause.py
```

### 2. Code Integration

No changes needed to existing imports. The enhanced `ReplyOnPause` is drop-in compatible:

```python
# Existing code continues to work
from fastrtc import ReplyOnPause, AlgoOptions

# New LiveKit features available through configuration
from backend.fastrtc.reply_on_pause import LiveKitAudioConfig
```

### 3. Configuration

```python
# Enable LiveKit features in AlgoOptions
algo_options = AlgoOptions(
    audio_chunk_duration=0.6,
    started_talking_threshold=0.3,
    speech_threshold=0.2,
    livekit_config=LiveKitAudioConfig(
        use_livekit_vad=True,
        use_noise_cancellation=True,
        noise_reduction_strength=0.8,
        use_echo_cancellation=True
    )
)
```

### 4. Running Version 2

```bash
# No additional servers needed
python voice_chat_test_app.py

# Access at http://localhost:7860
```

## Integration with LangGraph Workflow

Both versions maintain full compatibility with your `call_center_agent_graph`:

```python
# Both versions work identically with your workflow
def workflow_factory(client_data):
    return create_call_center_agent(
        model=llm,
        client_data=client_data,
        agent_name="Qwen",
        config=config,
        verbose=True
    )

# Initialize voice handler (same for both versions)
voice_handler = VoiceInteractionHandler(config, workflow_factory)
```

## Performance Tuning

### Version 1 (LiveKit)
```python
# Optimize for low latency
config['audio'] = {
    'sample_rate': 16000,  # Lower for faster processing
    'channels': 1,
    'noise_suppression': True,
    'echo_cancellation': True,
    'auto_gain_control': True
}

# VAD settings for natural conversation
vad = silero.VAD.load(
    min_speech_duration=0.1,    # Quick response
    min_silence_duration=0.3,   # Natural pauses
    activation_threshold=0.6    # Balanced sensitivity
)
```

### Version 2 (Hybrid)
```python
# Fine-tune LiveKit audio processing
livekit_config = LiveKitAudioConfig(
    use_livekit_vad=True,
    vad_activation_threshold=0.6,  # Adjust for environment
    noise_reduction_strength=0.8,   # 0.5-1.0 range
    vad_min_silence_duration=0.3    # Natural conversation flow
)
```

## Migration Strategy

### Phase 1: Test Version 2 (Week 1)
1. Deploy hybrid version alongside existing system
2. A/B test with select users
3. Monitor performance metrics
4. Gather feedback on audio quality

### Phase 2: Evaluate Results (Week 2)
1. Compare metrics between FastRTC and hybrid
2. Assess noise cancellation effectiveness
3. Review turn detection accuracy
4. Decision point: Continue with hybrid or move to Version 1

### Phase 3A: Stay with Hybrid
- Optimize LiveKit settings
- Fine-tune for your environment
- Consider partial Version 1 features

### Phase 3B: Migrate to Version 1 (Weeks 3-4)
1. Set up LiveKit infrastructure
2. Migrate STT/TTS adapters
3. Update client connections
4. Gradual rollout with fallback

## Troubleshooting

### Version 1 Issues

**Connection Failed**
```bash
# Check LiveKit server
docker logs livekit-server

# Verify ports are open
netstat -an | grep 7880

# Test with LiveKit CLI
livekit-cli room list --url ws://localhost:7880 --api-key devkey --api-secret secret
```

**Audio Quality Issues**
```python
# Adjust audio constraints
audio_constraints = {
    "echoCancellation": {"exact": True},
    "noiseSuppression": {"exact": True},
    "sampleRate": {"ideal": 16000},  # Try 8000 for lower bandwidth
}
```

### Version 2 Issues

**LiveKit Features Not Working**
```python
# Verify LiveKit plugins installed
try:
    from livekit.plugins import silero
    print("LiveKit plugins available")
except ImportError:
    print("Install with: pip install livekit-plugins-silero")
```

**High CPU Usage**
```python
# Reduce processing load
livekit_config = LiveKitAudioConfig(
    use_livekit_vad=False,  # Use only Silero
    noise_reduction_strength=0.5,  # Lower strength
)
```

## Monitoring & Metrics

### Key Metrics to Track

1. **Audio Quality**
   - Signal-to-noise ratio
   - Echo return loss
   - Speech clarity score

2. **Performance**
   - Processing latency
   - CPU usage
   - Memory consumption

3. **User Experience**
   - Turn detection accuracy
   - False positive/negative rates
   - User satisfaction scores

### Dashboard Integration

```python
# Add to your monitoring system
metrics = {
    "audio_quality": voice_handler.get_audio_quality_metrics(),
    "performance": voice_handler.get_performance_metrics(),
    "livekit_stats": voice_handler.get_livekit_statistics()
}
```

## Security Considerations

### Version 1 (Self-Hosted)
- No external API keys needed
- All data stays on-premise
- Configure firewall for LiveKit ports
- Use proper JWT tokens in production

### Version 2 (Hybrid)
- Maintains existing FastRTC security
- Audio processing is local only
- No external service dependencies

## Cost Analysis

### Version 1
- **Infrastructure**: LiveKit server (self-hosted)
- **Bandwidth**: ~32kbps per connection
- **CPU**: Medium (single framework)
- **Maintenance**: Low (unified system)

### Version 2
- **Infrastructure**: Existing FastRTC
- **Bandwidth**: Same as current
- **CPU**: Higher (dual processing)
- **Maintenance**: Medium (two systems)

## Next Steps

1. **Choose Your Path**
   - Quick improvement: Start with Version 2
   - Long-term solution: Plan for Version 1

2. **Set Up Testing**
   - Deploy in staging environment
   - Create A/B test groups
   - Define success metrics

3. **Monitor & Iterate**
   - Track performance daily
   - Gather user feedback
   - Optimize settings

4. **Scale Gradually**
   - Start with 10% of users
   - Increase based on metrics
   - Full rollout when stable

## Support & Resources

- LiveKit Documentation: https://docs.livekit.io
- FastRTC Documentation: Your existing docs
- Community Support: LiveKit Discord
- Custom Support: Your team

Remember: The hybrid approach (Version 2) provides immediate benefits with minimal risk, while the full migration (Version 1) offers the best long-term solution.