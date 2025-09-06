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

struct BufferSubmission {
    VkFence fence = VK_NULL_HANDLE;
    uint64_t submissionId = 0;
    bool pending = false;
    std::function<void()> onComplete = nullptr;
};

struct BufferPool {
    static constexpr uint32_t BUFFER_COUNT = 3;
    static constexpr uint32_t MAX_SUBMISSIONS_PER_BUFFER = 8;
    
    BufferInfo buffers[BUFFER_COUNT];
    std::vector<BufferSubmission> submissions[BUFFER_COUNT];
    bool inUse[BUFFER_COUNT] = {false, false, false};
    uint32_t currentIndex = 0;
    VkDeviceSize bufferSize = 0;
    
    BufferPool() {
        for (int i = 0; i < BUFFER_COUNT; ++i) {
            submissions[i].reserve(MAX_SUBMISSIONS_PER_BUFFER);
        }
    }
};

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> transferFamily;
    std::optional<uint32_t> computeFamily;
    
    bool isComplete() const {
        return graphicsFamily.has_value();
    }
    
    bool hasDistinctTransferQueue() const {
        return transferFamily.has_value() && transferFamily != graphicsFamily;
    }
};

struct CommandPool {
    VkCommandPool pool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> buffers;
    uint32_t currentBuffer = 0;
};

struct TransferOperation {
    VkBuffer srcBuffer;
    VkBuffer dstBuffer;
    VkDeviceSize size;
    VkDeviceSize srcOffset = 0;
    VkDeviceSize dstOffset = 0;
    VkSemaphore completionSemaphore = VK_NULL_HANDLE;
    VkFence completionFence = VK_NULL_HANDLE;
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
    
    BufferPool createBufferPool(VkDeviceSize bufferSize);
    void destroyBufferPool(BufferPool& pool);
    BufferInfo* acquireBuffer(BufferPool& pool);
    void releaseBuffer(BufferPool& pool, uint32_t bufferIndex);
    
    uint32_t submitBufferOperation(BufferPool& pool, uint32_t bufferIndex, VkFence fence, 
                                  uint64_t submissionId, std::function<void()> onComplete = nullptr);
    void processBufferFences(BufferPool& pool);
    bool isBufferAvailable(BufferPool& pool, uint32_t bufferIndex);
    VkFence createFence(bool signaled = false);
    
    BufferInfo createStagingBuffer(VkDeviceSize size);
    VkCommandBuffer beginSingleTimeCommands(bool useTransferQueue = true);
    void endSingleTimeCommands(VkCommandBuffer commandBuffer, bool useTransferQueue = true);
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size, 
                   VkDeviceSize srcOffset = 0, VkDeviceSize dstOffset = 0);
    void copyBufferAsync(const TransferOperation& transfer);
    
    VkQueue getGraphicsQueue() const { return graphicsQueue; }
    VkQueue getTransferQueue() const { return transferQueue; }
    VkDevice getDevice() const { return device; }

private:
    void createInstance();
    void setupDebugMessenger();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSurface(Window* window);
    void createCommandPools();
    
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
    bool checkValidationLayerSupport();
    std::vector<const char*> getRequiredExtensions();

private:
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue transferQueue = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    
    QueueFamilyIndices queueFamilyIndices;
    CommandPool graphicsCommandPool;
    CommandPool transferCommandPool;

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
