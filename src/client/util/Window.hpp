#pragma once
#include <vector>
#include <GLFW/glfw3.h>

class MinecraftClient;

class Window {
public:
    explicit Window(MinecraftClient* client);
    ~Window();

    void create(int width, int height);
    void updateTitle();

    bool shouldClose() const;
    void pollEvents();

private:
    void loadIcons();

    MinecraftClient* client;
    GLFWwindow* glfwWindow;
    int width;
    int height;
    std::vector<std::vector<uint8_t>> iconBuffers;
    VkSurfaceKHR createSurface(VkInstance instance);
};
