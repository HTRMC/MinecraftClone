#include <iostream>
#include <vector>
#include <GLFW/glfw3.h>
#include <spng.h>
#include <fstream>

std::vector<uint8_t> loadPNG(const std::string &path, int &width, int &height) {
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

int main() {
    std::cout << "Hello, World! (client)" << std::endl;

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW!" << std::endl;
        return -1;
    }
    
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    GLFWwindow* window = glfwCreateWindow(854, 480, "My Client", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window!" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    // Load icons (16x16, 32x32, etc.)
    std::vector<GLFWimage> icons;
    std::vector<std::vector<uint8_t>> iconBuffers; // keep memory alive
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

    glfwSetWindowIcon(window, icons.size(), icons.data());

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        glfwSwapBuffers(window);
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}