#pragma once
#include <string>
#include <optional>
#include <memory>

#include "util/Window.hpp"
#include "vulkan/VulkanContext.hpp"

class MinecraftClient {
public:
    MinecraftClient();
    void run();

    void setCurrentServerName(const std::string& name);
    void setSinglePlayer(bool sp);

    std::string getWindowTitle() const;

private:
    std::string gameVersion;
    std::optional<std::string> currentServerName;
    bool singlePlayer;
    int fps;

    std::unique_ptr<Window> window;
    std::unique_ptr<VulkanContext> vulkan;
};
