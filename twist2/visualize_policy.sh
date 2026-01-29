#!/bin/bash
#
# Policy Visualization Scripts
# ============================
#
# Visualize trained policies in Isaac Gym simulation.
#
# Usage:
#   ./visualize_policy.sh play <policy_name>     # Visualize single policy
#   ./visualize_policy.sh record <policy_name>   # Record video of policy
#   ./visualize_policy.sh list                   # List available policies
#

SCRIPT_DIR=$(dirname $(realpath $0))
cd ${SCRIPT_DIR}/legged_gym/legged_gym/scripts

echo "======================================"
echo "  TWIST2 Policy Visualizer"
echo "======================================"
echo ""

case "$1" in
    play)
        # Play single policy in Isaac Gym
        POLICY_NAME=${2:-"default"}
        
        case "$POLICY_NAME" in
            default|20k)
                echo "▶ Playing: Default 20k Policy"
                echo "  Path: logs/g1_priv_mimic/my_experiment/model_20000.pt"
                echo ""
                # Use --num_envs 1 to minimize GPU memory
                python play.py --task g1_priv_mimic --proj_name g1_priv_mimic --exptid my_experiment --checkpoint 20000 --num_envs 1
                ;;
            v1)
                echo "▶ Playing: V1 (Original Scales)"
                echo "  Path: logs/h1/backward_walking_v1/model_2000.pt"
                echo ""
                python play.py --task g1_priv_mimic --proj_name h1 --exptid backward_walking_v1 --checkpoint 2000
                ;;
            v2)
                # Find latest V2 checkpoint
                V2_DIR="${SCRIPT_DIR}/legged_gym/logs/h1/backward_walking_v2"
                if [ -d "$V2_DIR" ]; then
                    LATEST=$(ls -t ${V2_DIR}/model_*.pt 2>/dev/null | head -1 | grep -oP 'model_\K[0-9]+')
                    if [ -n "$LATEST" ]; then
                        echo "▶ Playing: V2 (Reduce Joint Tracking)"
                        echo "  Path: logs/h1/backward_walking_v2/model_${LATEST}.pt"
                        echo ""
                        python play.py --task g1_priv_mimic --proj_name h1 --exptid backward_walking_v2 --checkpoint $LATEST
                    else
                        echo "❌ No V2 checkpoints found!"
                    fi
                else
                    echo "❌ V2 directory not found: $V2_DIR"
                fi
                ;;
            *)
                echo "❌ Unknown policy: $POLICY_NAME"
                echo ""
                echo "Available policies:"
                echo "  default, 20k  - Original twist2_1017_20k teacher"
                echo "  v1            - backward_walking_v1 (original scales)"
                echo "  v2            - backward_walking_v2 (new scales)"
                exit 1
                ;;
        esac
        ;;
        
    record)
        # Record video of policy
        POLICY_NAME=${2:-"default"}
        
        case "$POLICY_NAME" in
            default|20k)
                echo "📹 Recording: Default 20k Policy"
                python play.py --task g1_priv_mimic --proj_name g1_priv_mimic --exptid my_experiment --checkpoint 20000 --record_video
                ;;
            v1)
                echo "📹 Recording: V1"
                python play.py --task g1_priv_mimic --proj_name h1 --exptid backward_walking_v1 --checkpoint 2000 --record_video
                ;;
            v2)
                V2_DIR="${SCRIPT_DIR}/legged_gym/logs/h1/backward_walking_v2"
                LATEST=$(ls -t ${V2_DIR}/model_*.pt 2>/dev/null | head -1 | grep -oP 'model_\K[0-9]+')
                if [ -n "$LATEST" ]; then
                    echo "📹 Recording: V2 (checkpoint $LATEST)"
                    python play.py --task g1_priv_mimic --proj_name h1 --exptid backward_walking_v2 --checkpoint $LATEST --record_video
                fi
                ;;
        esac
        ;;
        
    list)
        echo "Available Policies:"
        echo "==================="
        echo ""
        
        # Default
        if [ -f "${SCRIPT_DIR}/legged_gym/logs/g1_priv_mimic/my_experiment/model_20000.pt" ]; then
            echo "✅ default (20k) - logs/g1_priv_mimic/my_experiment/model_20000.pt"
        else
            echo "❌ default - NOT FOUND"
        fi
        
        # V1
        if [ -f "${SCRIPT_DIR}/legged_gym/logs/h1/backward_walking_v1/model_2000.pt" ]; then
            echo "✅ v1           - logs/h1/backward_walking_v1/model_2000.pt"
        else
            echo "❌ v1 - NOT FOUND"
        fi
        
        # V2
        V2_DIR="${SCRIPT_DIR}/legged_gym/logs/h1/backward_walking_v2"
        if [ -d "$V2_DIR" ]; then
            CKPTS=$(ls ${V2_DIR}/model_*.pt 2>/dev/null | wc -l)
            LATEST=$(ls -t ${V2_DIR}/model_*.pt 2>/dev/null | head -1 | grep -oP 'model_\K[0-9]+')
            echo "✅ v2           - logs/h1/backward_walking_v2/ ($CKPTS checkpoints, latest: $LATEST)"
        else
            echo "❌ v2 - NOT FOUND"
        fi
        ;;
        
    *)
        echo "Usage:"
        echo "  $0 play [policy]     - Visualize policy in Isaac Gym"
        echo "  $0 record [policy]   - Record video of policy"  
        echo "  $0 list              - List available policies"
        echo ""
        echo "Policies: default, v1, v2"
        echo ""
        echo "Examples:"
        echo "  $0 play default      # Play original 20k policy"
        echo "  $0 play v2           # Play V2 (still training)"
        echo "  $0 record v2         # Record V2 video"
        echo "  $0 list              # Show all available policies"
        ;;
esac

