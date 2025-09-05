#include "client/util/Window.hpp"

#include <spng.h>
#include <fstream>
#include <iostream>

#include "client/MinecraftClient.hpp"

static std::vector<uint8_t> loadPNG(const std::string &path, int &width, int &height) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open PNG: " + path);

    std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)),
                               std::istreambuf_iterator<char>());

    spng_ctx *ctx = spng_ctx_new(0);
    spng_set_png_buffer(ctx, data.data(), data.size());

    spng_ihdr ihdr;
    spng_get_ihdr(ctx, &ihdr);
    width = ihdr.width;
    height = ihdr.height;

    size_t out_size;
    spng_decoded_image_size(ctx, SPNG_FMT_RGBA8, &out_size);
    std::vector<uint8_t> out(out_size);
    spng_decode_image(ctx, out.data(), out.size(), SPNG_FMT_RGBA8, 0);
    spng_ctx_free(ctx);

    return out;
}

Window::Window(MinecraftClient* client)
    : client(client), glfwWindow(nullptr), width(854), height(480) {}

Window::~Window() {
    if (glfwWindow) {
        glfwDestroyWindow(glfwWindow);
    }
    glfwTerminate();
}

void Window::create(int w, int h) {
    width = w;
    height = h;

    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW!");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindow = glfwCreateWindow(width, height, "Placeholder", nullptr, nullptr);
    if (!glfwWindow) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window!");
    }

    loadIcons();
    updateTitle();
}

void Window::loadIcons() {
    std::vector<GLFWimage> icons;
    std::vector<std::string> iconFiles = {
        "./assets/icons/icon_16x16.png",
        "./assets/icons/icon_32x32.png",
        "./assets/icons/icon_48x48.png"
    };

    for (const auto &file : iconFiles) {
        int w, h;
        auto buffer = loadPNG(file, w, h);
        iconBuffers.push_back(std::move(buffer));

        GLFWimage img;
        img.width = w;
        img.height = h;
        img.pixels = iconBuffers.back().data();
        icons.push_back(img);
    }

    glfwSetWindowIcon(glfwWindow, icons.size(), icons.data());
}

VkSurfaceKHR Window::createSurface(VkInstance instance) {
    VkSurfaceKHR surface;
    if (glfwCreateWindowSurface(instance, glfwWindow, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create window surface!");
    }
    return surface;
}

void Window::updateTitle() {
    std::string title = client->getWindowTitle();
    glfwSetWindowTitle(glfwWindow, title.c_str());
}

bool Window::shouldClose() const {
    return glfwWindowShouldClose(glfwWindow);
}

void Window::pollEvents() {
    glfwPollEvents();
}
