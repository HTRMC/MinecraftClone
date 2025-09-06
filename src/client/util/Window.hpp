#pragma once
#include <vector>
#include <set>
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
    
    GLFWwindow* getHandle() const { return glfwWindow; }
    
    bool isKeyPressed(int key) const;
    void getMouseDelta(double& xDelta, double& yDelta);

private:
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void mouseCallback(GLFWwindow* window, double xpos, double ypos);
    void loadIcons();

    MinecraftClient* client;
    GLFWwindow* glfwWindow;
    int width;
    int height;
    std::vector<std::vector<uint8_t>> iconBuffers;
    
    std::set<int> pressedKeys;
    double lastMouseX = 0.0, lastMouseY = 0.0;
    double mouseDeltaX = 0.0, mouseDeltaY = 0.0;
    bool firstMouse = true;
    
    VkSurfaceKHR createSurface(VkInstance instance);
};
