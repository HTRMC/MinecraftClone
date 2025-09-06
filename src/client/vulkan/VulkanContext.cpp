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

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

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
    
    // Initialize VMA after creating logical device
    VmaAllocatorCreateInfo allocatorInfo{};
    allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;
    allocatorInfo.physicalDevice = physicalDevice;
    allocatorInfo.device = device;
    allocatorInfo.instance = instance;
    
    if (vmaCreateAllocator(&allocatorInfo, &allocator) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create VMA allocator!");
    }
    
    Logger::info("VulkanContext", "VMA allocator initialized");
    
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
        
        if (allocator != VK_NULL_HANDLE) {
            vmaDestroyAllocator(allocator);
            allocator = VK_NULL_HANDLE;
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

    // Enable mesh shader features
    VkPhysicalDeviceMeshShaderFeaturesEXT meshShaderFeatures{};
    meshShaderFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
    meshShaderFeatures.meshShader = VK_TRUE;
    meshShaderFeatures.taskShader = VK_TRUE;

    // Enable Vulkan 1.2 features for bindless textures
    VkPhysicalDeviceVulkan12Features vulkan12Features{};
    vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12Features.runtimeDescriptorArray = VK_TRUE;
    vulkan12Features.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    vulkan12Features.descriptorBindingPartiallyBound = VK_TRUE;
    vulkan12Features.descriptorBindingVariableDescriptorCount = VK_TRUE;
    vulkan12Features.pNext = &meshShaderFeatures;

    // Enable maintenance4 features for LocalSizeId
    VkPhysicalDeviceMaintenance4Features maintenance4Features{};
    maintenance4Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES;
    maintenance4Features.maintenance4 = VK_TRUE;
    maintenance4Features.pNext = &vulkan12Features;

    VkPhysicalDeviceFeatures2 deviceFeatures2{};
    deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    deviceFeatures2.pNext = &maintenance4Features;

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pNext = &deviceFeatures2;
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
    return createBuffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
}

void VulkanContext::destroyBuffer(BufferInfo& bufferInfo) {
    if (bufferInfo.buffer != VK_NULL_HANDLE && bufferInfo.allocation != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator, bufferInfo.buffer, bufferInfo.allocation);
        bufferInfo.buffer = VK_NULL_HANDLE;
        bufferInfo.allocation = VK_NULL_HANDLE;
    }
    bufferInfo.size = 0;
    bufferInfo.allocationInfo = {};
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
    return createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
}

BufferInfo VulkanContext::createUniformBuffer(VkDeviceSize size) {
    return createBuffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
}

BufferInfo VulkanContext::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage) {
    BufferInfo bufferInfo;
    bufferInfo.size = size;

    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = size;
    bufferCreateInfo.usage = usage;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.usage = memoryUsage;
    
    // Map memory for CPU accessible buffers
    if (memoryUsage == VMA_MEMORY_USAGE_CPU_ONLY || memoryUsage == VMA_MEMORY_USAGE_CPU_TO_GPU) {
        allocCreateInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }

    VkResult result = vmaCreateBuffer(allocator, &bufferCreateInfo, &allocCreateInfo,
                                     &bufferInfo.buffer, &bufferInfo.allocation, &bufferInfo.allocationInfo);
    
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer with VMA! Error: " + std::to_string(result));
    }

    Logger::debug("VulkanContext", "Created buffer of size " + std::to_string(size) + " bytes using VMA");
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

// ---------------- VMA Image Management ----------------

ImageInfo VulkanContext::createImage(uint32_t width, uint32_t height, VkFormat format, 
                                    VkImageTiling tiling, VkImageUsageFlags usage, VmaMemoryUsage memoryUsage) {
    ImageInfo imageInfo;
    imageInfo.width = width;
    imageInfo.height = height;
    imageInfo.format = format;

    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.extent.width = width;
    imageCreateInfo.extent.height = height;
    imageCreateInfo.extent.depth = 1;
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.format = format;
    imageCreateInfo.tiling = tiling;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageCreateInfo.usage = usage;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.usage = memoryUsage;

    VkResult result = vmaCreateImage(allocator, &imageCreateInfo, &allocCreateInfo,
                                    &imageInfo.image, &imageInfo.allocation, &imageInfo.allocationInfo);
    
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image with VMA! Error: " + std::to_string(result));
    }

    Logger::debug("VulkanContext", "Created image " + std::to_string(width) + "x" + std::to_string(height) + " using VMA");
    return imageInfo;
}

void VulkanContext::destroyImage(ImageInfo& imageInfo) {
    if (imageInfo.image != VK_NULL_HANDLE && imageInfo.allocation != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator, imageInfo.image, imageInfo.allocation);
        imageInfo.image = VK_NULL_HANDLE;
        imageInfo.allocation = VK_NULL_HANDLE;
    }
    imageInfo.width = 0;
    imageInfo.height = 0;
    imageInfo.format = VK_FORMAT_UNDEFINED;
    imageInfo.allocationInfo = {};
}

void VulkanContext::transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands(true);
    
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    
    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;
    
    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        
        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else {
        throw std::invalid_argument("Unsupported layout transition!");
    }
    
    vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    
    endSingleTimeCommands(commandBuffer, true);
}

void VulkanContext::copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands(true);
    
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};
    
    vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    
    endSingleTimeCommands(commandBuffer, true);
}

void* VulkanContext::mapBuffer(const BufferInfo& bufferInfo) {
    if (bufferInfo.allocationInfo.pMappedData) {
        return bufferInfo.allocationInfo.pMappedData;
    }
    
    void* mappedData;
    VkResult result = vmaMapMemory(allocator, bufferInfo.allocation, &mappedData);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to map buffer memory with VMA! Error: " + std::to_string(result));
    }
    return mappedData;
}

void VulkanContext::unmapBuffer(const BufferInfo& bufferInfo) {
    if (!bufferInfo.allocationInfo.pMappedData) {
        vmaUnmapMemory(allocator, bufferInfo.allocation);
    }
}
