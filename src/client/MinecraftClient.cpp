#include "client/MinecraftClient.hpp"

#include <sstream>
#include <thread>
#include <chrono>

#include "Logger.hpp"
#include "resource/language/I18n.hpp"

MinecraftClient::MinecraftClient()
    : gameVersion("1.21.8"),
      singlePlayer(true),
      fps(60) {}

void MinecraftClient::run() {
    Logger::init();

    window = std::make_unique<Window>(this);
    window->create(854, 480);

    vulkan = std::make_unique<VulkanContext>(window.get());
    vulkan->init();

    renderer = std::make_unique<Renderer>(vulkan.get(), window.get());
    renderer->init();

    // Initialize camera at origin
    camera = std::make_unique<Camera>(glm::vec3(0.0f, 0.0f, 0.0f));
    renderer->setCamera(camera.get());

    auto lastTime = std::chrono::high_resolution_clock::now();
    
    while (!window->shouldClose()) {
        auto currentTime = std::chrono::high_resolution_clock::now();
        float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
        lastTime = currentTime;
        
        window->pollEvents();
        processInput(deltaTime);

        // Render frame
        try {
            renderer->render();
        } catch (const std::exception& e) {
            Logger::error("Renderer", e.what());
        }

        fps = 60;

        window->updateTitle();
    }

    Logger::info("Main thread", "Stopping!");
}

void MinecraftClient::setCurrentServerName(const std::string& name) {
    currentServerName = name;
}

void MinecraftClient::setSinglePlayer(bool sp) {
    singlePlayer = sp;
}

std::string MinecraftClient::getWindowTitle() const {
    std::ostringstream title;

    title << "Minecraft";

    title << " " << gameVersion;

    if (currentServerName.has_value()) {
        title << " - " << currentServerName.value();
    } else {
        title << " - " << I18n::translate("title.singleplayer");
    }

    return title.str();
}

void MinecraftClient::processInput(float deltaTime) {
    // Use the new camera interface that handles all input internally
    camera->processKeyboard(window->getHandle(), deltaTime);
    
    // Process mouse movement
    double xDelta, yDelta;
    window->getMouseDelta(xDelta, yDelta);
    if (xDelta != 0.0 || yDelta != 0.0) {
        camera->processMouseMovement(xDelta, yDelta);
    }
}
