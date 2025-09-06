#pragma once
#include <string>
#include <optional>
#include <memory>

#include "util/Window.hpp"
#include "util/Camera.hpp"
#include "vulkan/VulkanContext.hpp"
#include "vulkan/Renderer.hpp"

class MinecraftClient {
public:
    MinecraftClient();
    void run();

    void setCurrentServerName(const std::string& name);
    void setSinglePlayer(bool sp);

    std::string getWindowTitle() const;
    Camera* getCamera() { return camera.get(); }

private:
    void processInput(float deltaTime);

    std::string gameVersion;
    std::optional<std::string> currentServerName;
    bool singlePlayer;
    int fps;

    std::unique_ptr<Window> window;
    std::unique_ptr<VulkanContext> vulkan;
    std::unique_ptr<Renderer> renderer;
    std::unique_ptr<Camera> camera;
};
