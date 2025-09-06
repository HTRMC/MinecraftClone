#pragma once
#include <functional>
#include <vulkan/vulkan.h>
#include <vector>
#include <optional>
#include <string>
#include <vk_mem_alloc.h>

class Window;

struct BufferInfo {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VmaAllocationInfo allocationInfo = {};
    VkDeviceSize size = 0;
};

struct ImageInfo {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VmaAllocationInfo allocationInfo = {};
    uint32_t width = 0;
    uint32_t height = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
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

struct TimelineSemaphore {
    VkSemaphore semaphore = VK_NULL_HANDLE;
    uint64_t value = 0;
    uint64_t nextSignalValue = 1;
    
    uint64_t getNextSignalValue() {
        return nextSignalValue++;
    }
};

struct TransferOperation {
    VkBuffer srcBuffer;
    VkBuffer dstBuffer;
    VkDeviceSize size;
    VkDeviceSize srcOffset = 0;
    VkDeviceSize dstOffset = 0;
    VkSemaphore completionSemaphore = VK_NULL_HANDLE;
    VkFence completionFence = VK_NULL_HANDLE;
    TimelineSemaphore* timelineSemaphore = nullptr;
    uint64_t signalValue = 0;
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
    
    VmaAllocator getAllocator() const { return allocator; }
    
    BufferPool createBufferPool(VkDeviceSize bufferSize);
    void destroyBufferPool(BufferPool& pool);
    BufferInfo* acquireBuffer(BufferPool& pool);
    void releaseBuffer(BufferPool& pool, uint32_t bufferIndex);
    
    uint32_t submitBufferOperation(BufferPool& pool, uint32_t bufferIndex, VkFence fence, 
                                  uint64_t submissionId, std::function<void()> onComplete = nullptr);
    void processBufferFences(BufferPool& pool);
    bool isBufferAvailable(BufferPool& pool, uint32_t bufferIndex);
    VkFence createFence(bool signaled = false);
    
    TimelineSemaphore createTimelineSemaphore(uint64_t initialValue = 0);
    void destroyTimelineSemaphore(TimelineSemaphore& timelineSemaphore);
    void signalTimelineSemaphore(TimelineSemaphore& timelineSemaphore, uint64_t value);
    void waitForTimelineSemaphore(TimelineSemaphore& timelineSemaphore, uint64_t value, uint64_t timeout = UINT64_MAX);
    uint64_t getTimelineSemaphoreValue(TimelineSemaphore& timelineSemaphore);
    
    BufferInfo createStagingBuffer(VkDeviceSize size);
    BufferInfo createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
    BufferInfo createUniformBuffer(VkDeviceSize size);
    
    ImageInfo createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, 
                         VkImageUsageFlags usage, VmaMemoryUsage memoryUsage);
    void destroyImage(ImageInfo& imageInfo);
    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
    VkCommandBuffer beginSingleTimeCommands(bool useTransferQueue = true);
    void endSingleTimeCommands(VkCommandBuffer commandBuffer, bool useTransferQueue = true);
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size, 
                   VkDeviceSize srcOffset = 0, VkDeviceSize dstOffset = 0);
    void copyBufferAsync(const TransferOperation& transfer);
    
    void* mapBuffer(const BufferInfo& bufferInfo);
    void unmapBuffer(const BufferInfo& bufferInfo);
    
    VkQueue getGraphicsQueue() const { return graphicsQueue; }
    VkQueue getTransferQueue() const { return transferQueue; }
    VkQueue getPresentQueue() const { return graphicsQueue; } // Assuming graphics queue handles present
    VkDevice getDevice() const { return device; }
    VkPhysicalDevice getPhysicalDevice() const { return physicalDevice; }
    VkInstance getInstance() const { return instance; }
    const CommandPool& getGraphicsCommandPool() const { return graphicsCommandPool; }
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
    
    // Mesh shader extension function
    PFN_vkCmdDrawMeshTasksEXT vkCmdDrawMeshTasksEXT = nullptr;

private:
    void createInstance();
    void setupDebugMessenger();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSurface(Window* window);
    void createCommandPools();
    void loadExtensionFunctions();
    
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
    VmaAllocator allocator = VK_NULL_HANDLE;
    
    QueueFamilyIndices queueFamilyIndices;
    CommandPool graphicsCommandPool;
    CommandPool transferCommandPool;

    const std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };
    const std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_EXT_MESH_SHADER_EXTENSION_NAME,
        VK_KHR_MAINTENANCE_4_EXTENSION_NAME,
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME
    };

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif
};
