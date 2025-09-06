#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <optional>
#include <string>

class Window;

struct BufferInfo {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mappedMemory = nullptr;
    VkDeviceSize size = 0;
};

struct BufferPool {
    static constexpr uint32_t BUFFER_COUNT = 3;
    
    BufferInfo buffers[BUFFER_COUNT];
    VkFence fences[BUFFER_COUNT];
    bool inUse[BUFFER_COUNT] = {false, false, false};
    uint32_t currentIndex = 0;
    VkDeviceSize bufferSize = 0;
};

class VulkanContext {
public:
    VulkanContext(Window* window);
    ~VulkanContext();

    void init();
    void cleanup();
    
    BufferInfo createStorageBuffer(VkDeviceSize size);
    void destroyBuffer(BufferInfo& bufferInfo);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

private:
    void createInstance();
    void setupDebugMessenger();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSurface(Window* window);

    bool checkValidationLayerSupport();
    std::vector<const char*> getRequiredExtensions();

private:
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    const std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };
    const std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif
};
