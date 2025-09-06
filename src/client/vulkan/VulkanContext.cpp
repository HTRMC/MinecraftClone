#include "VulkanContext.hpp"
#include "client/util/Window.hpp"

#include <stdexcept>
#include <iostream>
#include <set>
#include <cstring>
#include <vector>
#include <algorithm>
#include <functional>
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
    loadExtensionFunctions();
    createCommandPools();
}

void VulkanContext::cleanup() {
    if (device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device);
        
        if (graphicsCommandPool.pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, graphicsCommandPool.pool, nullptr);
        }
        if (transferCommandPool.pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, transferCommandPool.pool, nullptr);
        }
        
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
    queueFamilyIndices = findQueueFamilies(physicalDevice);
    
    if (!queueFamilyIndices.isComplete()) {
        throw std::runtime_error("Failed to find required queue families!");
    }

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {queueFamilyIndices.graphicsFamily.value()};
    
    if (queueFamilyIndices.hasDistinctTransferQueue()) {
        uniqueQueueFamilies.insert(queueFamilyIndices.transferFamily.value());
    }

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceFeatures deviceFeatures{};

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();
    createInfo.enabledLayerCount = enableValidationLayers ? static_cast<uint32_t>(validationLayers.size()) : 0;
    createInfo.ppEnabledLayerNames = enableValidationLayers ? validationLayers.data() : nullptr;

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create logical device!");
    }

    vkGetDeviceQueue(device, queueFamilyIndices.graphicsFamily.value(), 0, &graphicsQueue);
    
    if (queueFamilyIndices.hasDistinctTransferQueue()) {
        vkGetDeviceQueue(device, queueFamilyIndices.transferFamily.value(), 0, &transferQueue);
        Logger::info("Vulkan", "Using dedicated transfer queue");
    } else {
        transferQueue = graphicsQueue;
        Logger::info("Vulkan", "Using graphics queue for transfers");
    }
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
        pool.inUse[i] = false;
    }
    
    Logger::info("Vulkan", "Created buffer pool with " + std::to_string(BufferPool::BUFFER_COUNT) + 
                 " buffers of size " + std::to_string(bufferSize) + " bytes each");
    return pool;
}

void VulkanContext::destroyBufferPool(BufferPool& pool) {
    vkDeviceWaitIdle(device);
    
    for (uint32_t i = 0; i < BufferPool::BUFFER_COUNT; ++i) {
        for (auto& submission : pool.submissions[i]) {
            if (submission.fence != VK_NULL_HANDLE) {
                vkDestroyFence(device, submission.fence, nullptr);
            }
        }
        pool.submissions[i].clear();
        
        destroyBuffer(pool.buffers[i]);
        pool.inUse[i] = false;
    }
    pool.currentIndex = 0;
    pool.bufferSize = 0;
}

BufferInfo* VulkanContext::acquireBuffer(BufferPool& pool) {
    processBufferFences(pool);
    
    for (uint32_t attempts = 0; attempts < BufferPool::BUFFER_COUNT; ++attempts) {
        uint32_t index = pool.currentIndex;
        pool.currentIndex = (pool.currentIndex + 1) % BufferPool::BUFFER_COUNT;
        
        if (isBufferAvailable(pool, index)) {
            pool.inUse[index] = true;
            return &pool.buffers[index];
        }
    }
    
    uint32_t oldestIndex = 0;
    for (uint32_t i = 1; i < BufferPool::BUFFER_COUNT; ++i) {
        if (isBufferAvailable(pool, i)) {
            oldestIndex = i;
            break;
        }
    }
    
    if (isBufferAvailable(pool, oldestIndex)) {
        pool.inUse[oldestIndex] = true;
        return &pool.buffers[oldestIndex];
    }
    
    Logger::warning("Vulkan", "No buffers available in pool, waiting for oldest submission");
    if (!pool.submissions[oldestIndex].empty()) {
        auto& oldestSubmission = pool.submissions[oldestIndex][0];
        if (oldestSubmission.fence != VK_NULL_HANDLE) {
            vkWaitForFences(device, 1, &oldestSubmission.fence, VK_TRUE, UINT64_MAX);
            processBufferFences(pool);
        }
    }
    
    if (isBufferAvailable(pool, oldestIndex)) {
        pool.inUse[oldestIndex] = true;
        return &pool.buffers[oldestIndex];
    }
    
    return nullptr;
}

void VulkanContext::releaseBuffer(BufferPool& pool, uint32_t bufferIndex) {
    if (bufferIndex >= BufferPool::BUFFER_COUNT) return;
    
    pool.inUse[bufferIndex] = false;
}

uint32_t VulkanContext::submitBufferOperation(BufferPool& pool, uint32_t bufferIndex, VkFence fence, 
                                             uint64_t submissionId, std::function<void()> onComplete) {
    if (bufferIndex >= BufferPool::BUFFER_COUNT) return UINT32_MAX;
    
    BufferSubmission submission;
    submission.fence = fence;
    submission.submissionId = submissionId;
    submission.pending = true;
    submission.onComplete = std::move(onComplete);
    
    pool.submissions[bufferIndex].push_back(submission);
    pool.inUse[bufferIndex] = true;
    
    Logger::debug("Vulkan", "Submitted operation " + std::to_string(submissionId) + 
                  " for buffer " + std::to_string(bufferIndex));
    
    return static_cast<uint32_t>(pool.submissions[bufferIndex].size() - 1);
}

void VulkanContext::processBufferFences(BufferPool& pool) {
    for (uint32_t bufferIndex = 0; bufferIndex < BufferPool::BUFFER_COUNT; ++bufferIndex) {
        auto& submissions = pool.submissions[bufferIndex];
        
        auto it = submissions.begin();
        while (it != submissions.end()) {
            if (it->pending && it->fence != VK_NULL_HANDLE) {
                VkResult result = vkGetFenceStatus(device, it->fence);
                if (result == VK_SUCCESS) {
                    it->pending = false;
                    
                    if (it->onComplete) {
                        it->onComplete();
                    }
                    
                    vkDestroyFence(device, it->fence, nullptr);
                    it = submissions.erase(it);
                    
                    Logger::debug("Vulkan", "Completed operation " + std::to_string(it->submissionId) + 
                                 " for buffer " + std::to_string(bufferIndex));
                } else if (result == VK_ERROR_DEVICE_LOST) {
                    Logger::error("Vulkan", "Device lost while checking fence");
                    break;
                } else {
                    ++it;
                }
            } else {
                ++it;
            }
        }
        
        if (submissions.empty() && pool.inUse[bufferIndex]) {
            pool.inUse[bufferIndex] = false;
        }
    }
}

bool VulkanContext::isBufferAvailable(BufferPool& pool, uint32_t bufferIndex) {
    if (bufferIndex >= BufferPool::BUFFER_COUNT) return false;
    
    if (pool.submissions[bufferIndex].empty()) {
        return !pool.inUse[bufferIndex];
    }
    
    for (const auto& submission : pool.submissions[bufferIndex]) {
        if (submission.pending) {
            return false;
        }
    }
    
    return true;
}

VkFence VulkanContext::createFence(bool signaled) {
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (signaled) {
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    }
    
    VkFence fence;
    if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create fence!");
    }
    
    return fence;
}

// ---------------- Queue Family Management ----------------

QueueFamilyIndices VulkanContext::findQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices;
    
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
    
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
    
    for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
        const auto& queueFamily = queueFamilies[i];
        
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphicsFamily = i;
        }
        
        if ((queueFamily.queueFlags & VK_QUEUE_TRANSFER_BIT) && 
            !(queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            indices.transferFamily = i;
        }
        
        if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
            indices.computeFamily = i;
        }
    }
    
    if (!indices.transferFamily.has_value() && indices.graphicsFamily.has_value()) {
        indices.transferFamily = indices.graphicsFamily;
    }
    
    return indices;
}

// ---------------- Command Pool Management ----------------

void VulkanContext::createCommandPools() {
    VkCommandPoolCreateInfo graphicsPoolInfo{};
    graphicsPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    graphicsPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    graphicsPoolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
    
    if (vkCreateCommandPool(device, &graphicsPoolInfo, nullptr, &graphicsCommandPool.pool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create graphics command pool!");
    }
    
    if (queueFamilyIndices.hasDistinctTransferQueue()) {
        VkCommandPoolCreateInfo transferPoolInfo{};
        transferPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        transferPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        transferPoolInfo.queueFamilyIndex = queueFamilyIndices.transferFamily.value();
        
        if (vkCreateCommandPool(device, &transferPoolInfo, nullptr, &transferCommandPool.pool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create transfer command pool!");
        }
    } else {
        transferCommandPool = graphicsCommandPool;
    }
    
    graphicsCommandPool.buffers.resize(3);
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = graphicsCommandPool.pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<uint32_t>(graphicsCommandPool.buffers.size());
    
    if (vkAllocateCommandBuffers(device, &allocInfo, graphicsCommandPool.buffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate graphics command buffers!");
    }
    
    if (queueFamilyIndices.hasDistinctTransferQueue()) {
        transferCommandPool.buffers.resize(3);
        allocInfo.commandPool = transferCommandPool.pool;
        allocInfo.commandBufferCount = static_cast<uint32_t>(transferCommandPool.buffers.size());
        
        if (vkAllocateCommandBuffers(device, &allocInfo, transferCommandPool.buffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate transfer command buffers!");
        }
    }
    
    Logger::info("Vulkan", "Created command pools and allocated command buffers");
}

// ---------------- Transfer Operations ----------------

BufferInfo VulkanContext::createStagingBuffer(VkDeviceSize size) {
    BufferInfo bufferInfo;
    bufferInfo.size = size;

    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = size;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferCreateInfo, nullptr, &bufferInfo.buffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create staging buffer!");
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
        throw std::runtime_error("Failed to allocate staging buffer memory!");
    }

    vkBindBufferMemory(device, bufferInfo.buffer, bufferInfo.memory, 0);

    if (vkMapMemory(device, bufferInfo.memory, 0, size, 0, &bufferInfo.mappedMemory) != VK_SUCCESS) {
        vkFreeMemory(device, bufferInfo.memory, nullptr);
        vkDestroyBuffer(device, bufferInfo.buffer, nullptr);
        throw std::runtime_error("Failed to map staging buffer memory!");
    }

    return bufferInfo;
}

VkCommandBuffer VulkanContext::beginSingleTimeCommands(bool useTransferQueue) {
    CommandPool& pool = useTransferQueue ? transferCommandPool : graphicsCommandPool;
    
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = pool.pool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    return commandBuffer;
}

void VulkanContext::endSingleTimeCommands(VkCommandBuffer commandBuffer, bool useTransferQueue) {
    vkEndCommandBuffer(commandBuffer);

    VkQueue targetQueue = useTransferQueue ? transferQueue : graphicsQueue;
    CommandPool& pool = useTransferQueue ? transferCommandPool : graphicsCommandPool;
    
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(targetQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(targetQueue);

    vkFreeCommandBuffers(device, pool.pool, 1, &commandBuffer);
}

void VulkanContext::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size,
                              VkDeviceSize srcOffset, VkDeviceSize dstOffset) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands(true);

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = srcOffset;
    copyRegion.dstOffset = dstOffset;
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    endSingleTimeCommands(commandBuffer, true);
}

void VulkanContext::copyBufferAsync(const TransferOperation& transfer) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands(true);

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = transfer.srcOffset;
    copyRegion.dstOffset = transfer.dstOffset;
    copyRegion.size = transfer.size;
    vkCmdCopyBuffer(commandBuffer, transfer.srcBuffer, transfer.dstBuffer, 1, &copyRegion);

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    
    if (transfer.completionSemaphore != VK_NULL_HANDLE) {
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &transfer.completionSemaphore;
    }

    vkQueueSubmit(transferQueue, 1, &submitInfo, transfer.completionFence);
    
    vkFreeCommandBuffers(device, transferCommandPool.pool, 1, &commandBuffer);
}

void VulkanContext::loadExtensionFunctions() {
    // Load mesh shader extension functions
    vkCmdDrawMeshTasksEXT = (PFN_vkCmdDrawMeshTasksEXT)vkGetDeviceProcAddr(device, "vkCmdDrawMeshTasksEXT");
    
    if (!vkCmdDrawMeshTasksEXT) {
        throw std::runtime_error("Failed to load vkCmdDrawMeshTasksEXT function!");
    }
    
    Logger::info("VulkanContext", "Loaded mesh shader extension functions");
}
