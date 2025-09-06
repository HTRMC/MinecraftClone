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

    while (!window->shouldClose()) {
        window->pollEvents();

        std::this_thread::sleep_for(std::chrono::milliseconds(16));

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
