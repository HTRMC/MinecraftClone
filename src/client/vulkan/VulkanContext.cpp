#include "VulkanContext.hpp"
#include "client/util/Window.hpp"

#include <stdexcept>
#include <iostream>
#include <set>
#include <cstring>
#include <vector>
#include <algorithm>
#include <GLFW/glfw3.h>

#include "Logger.hpp"

// ================= Helper Functions =================

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

    std::cerr << "Validation layer: " << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
}

// ================= VulkanContext Methods =================

VulkanContext::VulkanContext(Window* window) {}

VulkanContext::~VulkanContext() {
    cleanup();
}

void VulkanContext::init() {
    createInstance();
    setupDebugMessenger();
    pickPhysicalDevice();
    createLogicalDevice();
}

void VulkanContext::cleanup() {
    if (device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device);
        vkDestroyDevice(device, nullptr);
    }

    if (enableValidationLayers && debugMessenger != VK_NULL_HANDLE) {
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if (func) func(instance, debugMessenger, nullptr);
    }

    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
    }
}

// ---------------- Instance ----------------

void VulkanContext::createInstance() {
    if (enableValidationLayers && !checkValidationLayerSupport()) {
        throw std::runtime_error("Validation layers requested, but not available!");
    }

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "MinecraftClone";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "CustomEngine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    auto extensions = getRequiredExtensions();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();

        debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                          VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                          VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                      VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                      VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugCreateInfo.pfnUserCallback = debugCallback;

        createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
    } else {
        createInfo.enabledLayerCount = 0;
        createInfo.pNext = nullptr;
    }

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan instance!");
    }
}

// ---------------- Debug Messenger ----------------

void VulkanContext::setupDebugMessenger() {
    if (!enableValidationLayers) return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;

    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (!func || func(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
        throw std::runtime_error("Failed to set up debug messenger!");
    }
}

// ---------------- Physical Device ----------------

void VulkanContext::pickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) throw std::runtime_error("Failed to find GPUs with Vulkan support!");

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    // Pick the first device that supports graphics queue
    for (const auto& d : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(d, &props);
        physicalDevice = d;
        Logger::info("Main thread", std::string("Selected GPU: ") + props.deviceName);
        break;
    }

    if (physicalDevice == VK_NULL_HANDLE)
        throw std::runtime_error("Failed to select a physical device!");
}

// ---------------- Logical Device ----------------

void VulkanContext::createLogicalDevice() {
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

    std::optional<uint32_t> graphicsFamily;
    for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            graphicsFamily = i;
            break;
        }
    }

    if (!graphicsFamily.has_value())
        throw std::runtime_error("Failed to find a graphics queue family!");

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = graphicsFamily.value();
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkPhysicalDeviceFeatures deviceFeatures{};

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.queueCreateInfoCount = 1;
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();
    createInfo.enabledLayerCount = enableValidationLayers ? static_cast<uint32_t>(validationLayers.size()) : 0;
    createInfo.ppEnabledLayerNames = enableValidationLayers ? validationLayers.data() : nullptr;

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create logical device!");
    }

    vkGetDeviceQueue(device, graphicsFamily.value(), 0, &graphicsQueue);
}

// ---------------- Validation Layers ----------------

bool VulkanContext::checkValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : validationLayers) {
        bool found = false;
        for (const auto& layerProps : availableLayers) {
            if (strcmp(layerName, layerProps.layerName) == 0) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }

    return true;
}

// ---------------- Extensions ----------------

std::vector<const char*> VulkanContext::getRequiredExtensions() {
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

// ---------------- Buffer Management ----------------

BufferInfo VulkanContext::createStorageBuffer(VkDeviceSize size) {
    BufferInfo bufferInfo;
    bufferInfo.size = size;

    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = size;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferCreateInfo, nullptr, &bufferInfo.buffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create storage buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, bufferInfo.buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferInfo.memory) != VK_SUCCESS) {
        vkDestroyBuffer(device, bufferInfo.buffer, nullptr);
        throw std::runtime_error("Failed to allocate storage buffer memory!");
    }

    vkBindBufferMemory(device, bufferInfo.buffer, bufferInfo.memory, 0);

    if (vkMapMemory(device, bufferInfo.memory, 0, size, 0, &bufferInfo.mappedMemory) != VK_SUCCESS) {
        vkFreeMemory(device, bufferInfo.memory, nullptr);
        vkDestroyBuffer(device, bufferInfo.buffer, nullptr);
        throw std::runtime_error("Failed to map storage buffer memory!");
    }

    Logger::info("Vulkan", "Created storage buffer of size " + std::to_string(size) + " bytes");
    return bufferInfo;
}

void VulkanContext::destroyBuffer(BufferInfo& bufferInfo) {
    if (bufferInfo.mappedMemory) {
        vkUnmapMemory(device, bufferInfo.memory);
        bufferInfo.mappedMemory = nullptr;
    }
    if (bufferInfo.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, bufferInfo.memory, nullptr);
        bufferInfo.memory = VK_NULL_HANDLE;
    }
    if (bufferInfo.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, bufferInfo.buffer, nullptr);
        bufferInfo.buffer = VK_NULL_HANDLE;
    }
    bufferInfo.size = 0;
}

uint32_t VulkanContext::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type!");
}

// ---------------- Buffer Pool Management ----------------

BufferPool VulkanContext::createBufferPool(VkDeviceSize bufferSize) {
    BufferPool pool;
    pool.bufferSize = bufferSize;
    
    for (uint32_t i = 0; i < BufferPool::BUFFER_COUNT; ++i) {
        pool.buffers[i] = createStorageBuffer(bufferSize);
        
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        
        if (vkCreateFence(device, &fenceInfo, nullptr, &pool.fences[i]) != VK_SUCCESS) {
            for (uint32_t j = 0; j < i; ++j) {
                destroyBuffer(pool.buffers[j]);
                vkDestroyFence(device, pool.fences[j], nullptr);
            }
            destroyBuffer(pool.buffers[i]);
            throw std::runtime_error("Failed to create fence for buffer pool!");
        }
    }
    
    Logger::info("Vulkan", "Created buffer pool with " + std::to_string(BufferPool::BUFFER_COUNT) + 
                 " buffers of size " + std::to_string(bufferSize) + " bytes each");
    return pool;
}

void VulkanContext::destroyBufferPool(BufferPool& pool) {
    for (uint32_t i = 0; i < BufferPool::BUFFER_COUNT; ++i) {
        if (pool.fences[i] != VK_NULL_HANDLE) {
            vkWaitForFences(device, 1, &pool.fences[i], VK_TRUE, UINT64_MAX);
            vkDestroyFence(device, pool.fences[i], nullptr);
            pool.fences[i] = VK_NULL_HANDLE;
        }
        destroyBuffer(pool.buffers[i]);
        pool.inUse[i] = false;
    }
    pool.currentIndex = 0;
    pool.bufferSize = 0;
}

BufferInfo* VulkanContext::acquireBuffer(BufferPool& pool) {
    for (uint32_t attempts = 0; attempts < BufferPool::BUFFER_COUNT; ++attempts) {
        uint32_t index = pool.currentIndex;
        pool.currentIndex = (pool.currentIndex + 1) % BufferPool::BUFFER_COUNT;
        
        if (!pool.inUse[index]) {
            VkResult result = vkGetFenceStatus(device, pool.fences[index]);
            if (result == VK_SUCCESS) {
                pool.inUse[index] = true;
                vkResetFences(device, 1, &pool.fences[index]);
                return &pool.buffers[index];
            }
        }
    }
    
    uint32_t oldestIndex = 0;
    for (uint32_t i = 1; i < BufferPool::BUFFER_COUNT; ++i) {
        if (vkGetFenceStatus(device, pool.fences[i]) == VK_SUCCESS) {
            oldestIndex = i;
            break;
        }
    }
    
    if (vkWaitForFences(device, 1, &pool.fences[oldestIndex], VK_TRUE, 1000000) == VK_SUCCESS) {
        pool.inUse[oldestIndex] = true;
        vkResetFences(device, 1, &pool.fences[oldestIndex]);
        return &pool.buffers[oldestIndex];
    }
    
    return nullptr;
}

void VulkanContext::releaseBuffer(BufferPool& pool, uint32_t bufferIndex) {
    if (bufferIndex >= BufferPool::BUFFER_COUNT) return;
    
    pool.inUse[bufferIndex] = false;
}
