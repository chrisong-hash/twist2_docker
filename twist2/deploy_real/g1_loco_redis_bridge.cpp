/**
 * G1 LocoClient Redis Bridge
 * 
 * Reads controller data from Redis and controls G1 locomotion via LocoClient.
 * Features:
 * - Joystick velocity control (vx, vy, vyaw)
 * - Sprint mode (hold B button for 2x speed)
 * - A button to stop/exit
 * - Automatic deadzone handling
 */

#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <memory>
#include <utility>

#include <unitree/robot/g1/loco/g1_loco_api.hpp>
#include <unitree/robot/g1/loco/g1_loco_client.hpp>
#include <atomic>
#include <cmath>
#include <sstream>
#include <map>
#include <hiredis/hiredis.h>

// Global flag for clean shutdown
std::atomic<bool> g_running{true};

// Simple sleep function using std::this_thread
void sleep_ms(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

// Simple JSON value extractor (minimal parser for our specific use case)
float extractFloat(const std::string& json, const std::string& key) {
    size_t pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return 0.0f;
    
    pos = json.find(":", pos);
    if (pos == std::string::npos) return 0.0f;
    
    // Skip whitespace and find number
    pos++;
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    
    // Extract number
    size_t end = pos;
    while (end < json.length() && (isdigit(json[end]) || json[end] == '.' || json[end] == '-')) end++;
    
    if (end > pos) {
        return std::stof(json.substr(pos, end - pos));
    }
    return 0.0f;
}

// Extract axis array [x, y] from JSON - returns pair of floats
std::pair<float, float> extractAxis(const std::string& json, const std::string& key) {
    size_t pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return {0.0f, 0.0f};
    
    // Find the opening bracket
    pos = json.find("[", pos);
    if (pos == std::string::npos) return {0.0f, 0.0f};
    pos++;
    
    // Skip whitespace
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    
    // Extract first number (x)
    size_t end = pos;
    while (end < json.length() && (isdigit(json[end]) || json[end] == '.' || json[end] == '-' || json[end] == 'e' || json[end] == 'E' || json[end] == '+')) end++;
    float x = 0.0f;
    if (end > pos) {
        try {
            x = std::stof(json.substr(pos, end - pos));
        } catch (...) {
            x = 0.0f;
        }
    }
    
    // Find comma and second number
    pos = json.find(",", end);
    if (pos == std::string::npos) return {x, 0.0f};
    pos++;
    
    // Skip whitespace
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    
    // Extract second number (y)
    end = pos;
    while (end < json.length() && (isdigit(json[end]) || json[end] == '.' || json[end] == '-' || json[end] == 'e' || json[end] == 'E' || json[end] == '+')) end++;
    float y = 0.0f;
    if (end > pos) {
        try {
            y = std::stof(json.substr(pos, end - pos));
        } catch (...) {
            y = 0.0f;
        }
    }
    
    return {x, y};
}

bool extractBool(const std::string& json, const std::string& key) {
    size_t pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return false;
    
    pos = json.find(":", pos);
    if (pos == std::string::npos) return false;
    
    // Look for true/false
    if (json.find("true", pos) != std::string::npos && 
        json.find("true", pos) < json.find(",", pos)) {
        return true;
    }
    return false;
}

// Apply deadzone to joystick input
float applyDeadzone(float value, float deadzone = 0.15f) {
    if (std::fabs(value) < deadzone) {
        return 0.0f;
    }
    // Rescale so deadzone->1.0 maps to 0.0->1.0
    float sign = value > 0 ? 1.0f : -1.0f;
    return sign * (std::fabs(value) - deadzone) / (1.0f - deadzone);
}

class LocoRedisBridge {
public:
    LocoRedisBridge(const std::string& redis_ip = "localhost", 
                     int redis_port = 6379,
                     const std::string& network_interface = "enp4s0")
        : redis_ip_(redis_ip)
        , redis_port_(redis_port)
        , network_interface_(network_interface)
        , redis_context_(nullptr)
        , control_dt_(0.02)  // 50 Hz
        , max_forward_speed_(0.8f)
        , max_lateral_speed_(0.5f)
        , max_turn_speed_(0.8f)
        , sprint_multiplier_(2.0f)
        , is_sprinting_(false)
        , b_button_prev_(false)
        , a_button_prev_(false)
        , locomotion_started_(false)
        , standing_up_(false)
        , awaiting_confirm_(false)
        , vx_(0.0f)
        , vy_(0.0f)
        , vyaw_(0.0f)
    {
    }

    ~LocoRedisBridge() {
        cleanup();
    }

    bool init() {
        // Initialize Redis connection
        std::cout << "Connecting to Redis at " << redis_ip_ << ":" << redis_port_ << std::endl;
        redis_context_ = redisConnect(redis_ip_.c_str(), redis_port_);
        
        if (redis_context_ == nullptr || redis_context_->err) {
            if (redis_context_) {
                std::cerr << "Redis connection error: " << redis_context_->errstr << std::endl;
                redisFree(redis_context_);
            } else {
                std::cerr << "Redis connection error: can't allocate redis context" << std::endl;
            }
            return false;
        }
        std::cout << "Connected to Redis successfully!" << std::endl;

        // Initialize Unitree SDK - MUST be done before creating LocoClient
        std::cout << "Initializing Unitree SDK with interface: " << network_interface_ << std::endl;
        unitree::robot::ChannelFactory::Instance()->Init(0, network_interface_);
        
        // Create and initialize LocoClient AFTER ChannelFactory::Init()
        std::cout << "Creating LocoClient..." << std::endl;
        loco_client_ = std::make_unique<unitree::robot::g1::LocoClient>();
        
        std::cout << "Initializing LocoClient..." << std::endl;
        loco_client_->Init();
        loco_client_->SetTimeout(10.0f);
        
        std::cout << "LocoClient initialized successfully!" << std::endl;
        std::cout << "\n=== G1 Locomotion Control Ready ===" << std::endl;
        std::cout << "Startup sequence:" << std::endl;
        std::cout << "  1. Move joystick -> Robot stands up" << std::endl;
        std::cout << "  2. Press A button -> Confirm & start locomotion" << std::endl;
        std::cout << "\nControls (after locomotion starts):" << std::endl;
        std::cout << "  Left Joystick:  Forward/Backward & Left/Right" << std::endl;
        std::cout << "  Right Joystick: Turn Left/Right" << std::endl;
        std::cout << "  B Button (hold): Sprint (2x speed)" << std::endl;
        std::cout << "  A Button:        Stop and Exit" << std::endl;
        std::cout << "===================================\n" << std::endl;
        
        return true;
    }

    void run() {
        while (g_running) {
            auto loop_start = std::chrono::high_resolution_clock::now();
            
            // Read controller data from Redis
            std::string controller_json = getRedisKey("controller_data");
            
            if (controller_json.empty()) {
                // No data yet, wait and retry
                sleep_ms(10);
                continue;
            }
            
            // Process controller input
            processControllerInput(controller_json);
            
            // Send velocity commands
            sendVelocityCommand();
            
            // Sleep to maintain control rate
            auto loop_end = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end - loop_start);
            auto sleep_time = std::chrono::milliseconds(static_cast<int>(control_dt_ * 1000)) - elapsed;
            
            if (sleep_time.count() > 0) {
                sleep_ms(sleep_time.count());
            }
        }
        
        // Clean shutdown
        std::cout << "\nStopping locomotion..." << std::endl;
        if (loco_client_) {
            loco_client_->StopMove();
        }
        sleep_ms(500);
    }

private:
    std::string getRedisKey(const std::string& key) {
        redisReply* reply = (redisReply*)redisCommand(redis_context_, "GET %s", key.c_str());
        
        if (reply == nullptr) {
            std::cerr << "Redis GET failed for key: " << key << std::endl;
            return "";
        }
        
        std::string value;
        if (reply->type == REDIS_REPLY_STRING) {
            value = std::string(reply->str, reply->len);
        }
        
        freeReplyObject(reply);
        return value;
    }

    void processControllerInput(const std::string& controller_json) {
        // Extract joystick values
        // Format: {"LeftController": {"axis": [x, y], "key_one": false, ...}, 
        //          "RightController": {"axis": [x, y], "key_two": false, ...}}
        
        // Left joystick (forward/backward and left/right)
        size_t left_pos = controller_json.find("\"LeftController\"");
        float lx = 0.0f, ly = 0.0f;
        bool a_button = false;
        
        if (left_pos != std::string::npos) {
            // Find the end of LeftController object (handle nested braces)
            size_t brace_count = 0;
            size_t left_start = controller_json.find("{", left_pos);
            size_t left_end = left_start;
            for (size_t i = left_start; i < controller_json.length(); i++) {
                if (controller_json[i] == '{') brace_count++;
                else if (controller_json[i] == '}') {
                    brace_count--;
                    if (brace_count == 0) {
                        left_end = i;
                        break;
                    }
                }
            }
            std::string left_json = controller_json.substr(left_pos, left_end - left_pos + 1);
            
            auto left_axis = extractAxis(left_json, "axis");
            lx = left_axis.first;   // Lateral (left/right)
            ly = left_axis.second;  // Forward/backward
            a_button = extractBool(left_json, "key_one");  // A button
        }
        
        // Right joystick (turning) and B button
        size_t right_pos = controller_json.find("\"RightController\"");
        float rx = 0.0f;
        bool b_button = false;
        
        if (right_pos != std::string::npos) {
            // Find the end of RightController object
            size_t brace_count = 0;
            size_t right_start = controller_json.find("{", right_pos);
            size_t right_end = right_start;
            for (size_t i = right_start; i < controller_json.length(); i++) {
                if (controller_json[i] == '{') brace_count++;
                else if (controller_json[i] == '}') {
                    brace_count--;
                    if (brace_count == 0) {
                        right_end = i;
                        break;
                    }
                }
            }
            std::string right_json = controller_json.substr(right_pos, right_end - right_pos + 1);
            
            auto right_axis = extractAxis(right_json, "axis");
            rx = right_axis.first;  // Turn left/right (use X axis for turning)
            b_button = extractBool(right_json, "key_two");  // B button
        }
        
        // Handle A button (exit) - only when locomotion is active
        if (locomotion_started_ && a_button && !a_button_prev_) {
            std::cout << "A button pressed - initiating shutdown..." << std::endl;
            g_running = false;
            a_button_prev_ = a_button;
            return;
        }
        // Note: a_button_prev_ is updated later in the state machine for confirmation
        
        // Handle B button (sprint mode)
        if (b_button && !b_button_prev_) {
            is_sprinting_ = true;
            loco_client_->SetSpeedMode(2);  // High speed mode
            std::cout << ">>> SPRINT MODE ON <<<" << std::endl;
        } else if (!b_button && b_button_prev_) {
            is_sprinting_ = false;
            loco_client_->SetSpeedMode(0);  // Normal speed mode
            std::cout << ">>> Normal speed <<<" << std::endl;
        }
        b_button_prev_ = b_button;
        
        // Apply deadzone and calculate velocities
        vx_ = applyDeadzone(ly) * max_forward_speed_;   // Forward/backward
        vy_ = applyDeadzone(lx) * max_lateral_speed_;   // Left/right
        vyaw_ = applyDeadzone(rx) * max_turn_speed_;    // Turn
        
        // Apply sprint multiplier to forward speed
        if (is_sprinting_) {
            vx_ *= sprint_multiplier_;
        }
        
        // Debug output for joystick values (every 50 loops = ~1 second)
        static int debug_counter = 0;
        if (debug_counter++ % 50 == 0 && !locomotion_started_) {
            std::cout << "[Debug] Raw joystick: lx=" << lx << " ly=" << ly << " rx=" << rx 
                      << " | After deadzone: vx=" << vx_ << " vy=" << vy_ << " vyaw=" << vyaw_ << std::endl;
            // Print first 200 chars of raw JSON for debugging
            if (debug_counter % 150 == 1) {
                std::cout << "[Debug] Raw JSON (first 200 chars): " 
                          << controller_json.substr(0, std::min((size_t)200, controller_json.length())) << std::endl;
            }
        }
        
        // State machine for safe startup
        if (!standing_up_ && !awaiting_confirm_ && !locomotion_started_) {
            // Waiting for first joystick input to trigger stand up
            if (std::fabs(vx_) > 0.01f || std::fabs(vy_) > 0.01f || std::fabs(vyaw_) > 0.01f) {
                std::cout << "\n>>> Standing up... Please wait <<<" << std::endl;
                
                // First check current FSM state
                int fsm_id = -1;
                int ret = loco_client_->GetFsmId(fsm_id);
                std::cout << "Current FSM ID: " << fsm_id << " (ret=" << ret << ")" << std::endl;
                
                // Try to stand up
                ret = loco_client_->StandUp();
                if (ret != 0) {
                    std::cout << "WARNING: StandUp() returned error code: " << ret << std::endl;
                    std::cout << "Robot may need to be in damping mode first." << std::endl;
                    std::cout << "Try: Using Unitree app to put robot in damping mode" << std::endl;
                } else {
                    std::cout << "StandUp() command sent successfully" << std::endl;
                }
                
                standing_up_ = true;
                standup_time_ = std::chrono::high_resolution_clock::now();
            }
        }
        else if (standing_up_ && !awaiting_confirm_) {
            // Wait 3 seconds for robot to stand up
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - standup_time_);
            if (elapsed.count() > 3000) {
                standing_up_ = false;
                awaiting_confirm_ = true;
                std::cout << "\n========================================" << std::endl;
                std::cout << ">>> Robot is standing. <<<" << std::endl;
                std::cout << ">>> Press A button to START locomotion <<<" << std::endl;
                std::cout << "========================================\n" << std::endl;
            }
        }
        else if (awaiting_confirm_) {
            // Wait for A button press to confirm and start locomotion
            if (a_button && !a_button_prev_) {
                std::cout << ">>> A button pressed - Starting locomotion! <<<" << std::endl;
                loco_client_->Start();  // FSM ID 500
                sleep_ms(500);
                awaiting_confirm_ = false;
                locomotion_started_ = true;
                std::cout << "Locomotion started! Joystick control active." << std::endl;
                std::cout << "(Press A again to STOP and exit)\n" << std::endl;
            }
        }
        
        // Update A button previous state
        a_button_prev_ = a_button;
    }

    void sendVelocityCommand() {
        if (!locomotion_started_) {
            return;
        }
        
        // Send velocity command to LocoClient
        // Duration is control timestep
        loco_client_->SetVelocity(vx_, vy_, vyaw_, control_dt_);
        
        // Print status occasionally
        static int counter = 0;
        if (counter++ % 25 == 0) {  // Every 0.5 seconds at 50Hz
            if (std::fabs(vx_) > 0.01f || std::fabs(vy_) > 0.01f || std::fabs(vyaw_) > 0.01f) {
                std::cout << "Vel: [vx=" << vx_ << ", vy=" << vy_ << ", vyaw=" << vyaw_ << "]";
                if (is_sprinting_) {
                    std::cout << " [SPRINTING]";
                }
                std::cout << std::endl;
            }
        }
    }

    void cleanup() {
        if (redis_context_) {
            redisFree(redis_context_);
            redis_context_ = nullptr;
        }
    }

    // Configuration
    std::string redis_ip_;
    int redis_port_;
    std::string network_interface_;
    
    // Redis
    redisContext* redis_context_;
    
    // Unitree SDK - use pointer to defer construction until after ChannelFactory::Init()
    std::unique_ptr<unitree::robot::g1::LocoClient> loco_client_;
    
    // Control parameters
    float control_dt_;
    float max_forward_speed_;
    float max_lateral_speed_;
    float max_turn_speed_;
    float sprint_multiplier_;
    
    // State
    float vx_, vy_, vyaw_;
    bool is_sprinting_;
    bool b_button_prev_;
    bool a_button_prev_;
    bool locomotion_started_;
    bool standing_up_;
    bool awaiting_confirm_;
    std::chrono::high_resolution_clock::time_point standup_time_;
};

int main(int argc, char* argv[]) {
    // Parse arguments
    std::string redis_ip = "localhost";
    int redis_port = 6379;
    std::string network_interface = "enp4s0";
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--redis_ip" && i + 1 < argc) {
            redis_ip = argv[++i];
        } else if (arg == "--redis_port" && i + 1 < argc) {
            redis_port = std::stoi(argv[++i]);
        } else if (arg == "--network_interface" && i + 1 < argc) {
            network_interface = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --redis_ip <ip>              Redis server IP (default: localhost)" << std::endl;
            std::cout << "  --redis_port <port>          Redis server port (default: 6379)" << std::endl;
            std::cout << "  --network_interface <iface>  Network interface (default: enp4s0)" << std::endl;
            std::cout << "  --help, -h                   Show this help message" << std::endl;
            return 0;
        }
    }
    
    // Create and run bridge
    std::cout << "Note: Press A button on controller to exit" << std::endl;
    LocoRedisBridge bridge(redis_ip, redis_port, network_interface);
    
    if (!bridge.init()) {
        std::cerr << "Failed to initialize bridge!" << std::endl;
        return 1;
    }
    
    std::cout << "Bridge initialized successfully. Starting control loop..." << std::endl;
    bridge.run();
    
    std::cout << "Bridge shutdown complete." << std::endl;
    return 0;
}

