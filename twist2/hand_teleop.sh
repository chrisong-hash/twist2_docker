#!/bin/bash
# Hand-only teleop with safety features
# Controls Inspire hands from Pico without leg tracking

# Configuration
LEFT_HAND_IP="192.168.123.210"
RIGHT_HAND_IP="192.168.123.211"

# Voice model path (required for voice commands)
VOICE_MODEL="/workspace/vosk-model-small-en-us-0.15"

cd "$(dirname "$0")/deploy_real"

# Force PulseAudio to avoid ALSA assertion failures
export SDL_AUDIODRIVER=pulseaudio
export AUDIODEV=pulse

echo "=============================================="
echo "🤖 HAND TELEOP - Safety-Enabled"
echo "=============================================="
echo ""
echo "Safety features:"
echo "  👏 CLAP → Pause/Resume"
echo "  🎮 A Button TAP → Pause/Resume"  
echo "  🎮 A Button HOLD (500ms) → Exit"
echo "  🎤 Voice (via Redis) → Pause/Resume"
echo "  ⌨️  Ctrl+C → Exit"
echo ""
echo "Make sure:"
echo "  1. XRoboToolkit is running on PC"
echo "  2. Pico is connected to PC's hotspot"
echo "  3. (Optional) Inspire hands are powered on"
echo ""

# Parse arguments
ENABLE_HANDS=""
ENABLE_VOICE=""  # Voice disabled by default (ALSA bug in Docker)
SKIP_CALIB=""

for arg in "$@"; do
    case $arg in
        --hands)
            ENABLE_HANDS="--enable_hands"
            echo "✓ Inspire hands ENABLED"
            ;;
        --voice)
            ENABLE_VOICE="--enable_voice"
            echo "✓ Voice commands ENABLED (experimental - may crash)"
            ;;
        --skip-calibration|-s)
            SKIP_CALIB="--skip_calibration"
            echo "✗ Calibration SKIPPED (using defaults)"
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --hands              Enable Inspire hand control"
            echo "  --voice              Enable voice commands (experimental)"
            echo "  --skip-calibration   Skip hand calibration"
            echo "  --help               Show this help"
            exit 0
            ;;
    esac
done

# Add voice model path if voice is enabled
if [ -n "$ENABLE_VOICE" ] && [ -n "$VOICE_MODEL" ]; then
    ENABLE_VOICE="$ENABLE_VOICE --voice_model $VOICE_MODEL"
fi

if [ -z "$ENABLE_HANDS" ]; then
    echo "ℹ  Running in SIMULATION mode (hands not connected)"
    echo "   Use --hands flag to enable real hand control"
fi

if [ -z "$ENABLE_VOICE" ]; then
    echo "ℹ  Voice DISABLED (use --voice, then run on host: ./scripts/voice_service.sh)"
fi

echo ""
echo "Starting..."
echo ""

python hand_teleop.py \
    $ENABLE_HANDS \
    $ENABLE_VOICE \
    $SKIP_CALIB \
    --left_hand_ip "$LEFT_HAND_IP" \
    --right_hand_ip "$RIGHT_HAND_IP"

