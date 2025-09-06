#pragma once
#include "VulkanContext.hpp"
#include <queue>
#include <memory>
#include <mutex>

struct SynchronizedTransfer {
    TransferOperation transfer;
    std::function<void()> onComplete;
    uint64_t id;
    bool inProgress = false;
    
    SynchronizedTransfer(TransferOperation t, std::function<void()> callback = nullptr) 
        : transfer(std::move(t)), onComplete(std::move(callback)), id(generateId()) {}
    
private:
    static uint64_t generateId() {
        static std::atomic<uint64_t> counter{0};
        return counter.fetch_add(1);
    }
};

struct GraphicsCommand {
    std::function<void(VkCommandBuffer)> recordFunc;
    VkSemaphore waitSemaphore = VK_NULL_HANDLE;
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
    VkFence signalFence = VK_NULL_HANDLE;
    uint64_t id;
    
    GraphicsCommand(std::function<void(VkCommandBuffer)> func) 
        : recordFunc(std::move(func)), id(generateId()) {}
    
private:
    static uint64_t generateId() {
        static std::atomic<uint64_t> counter{0};
        return counter.fetch_add(1);
    }
};

class QueueManager {
public:
    QueueManager(VulkanContext* vulkanContext);
    ~QueueManager();
    
    void init();
    void cleanup();
    
    uint64_t submitTransfer(const TransferOperation& transfer, std::function<void()> onComplete = nullptr);
    uint64_t submitGraphics(const GraphicsCommand& command);
    
    uint64_t submitMeshDataTransfer(BufferInfo* stagingBuffer, BufferInfo* deviceBuffer, 
                                   VkDeviceSize size, std::function<void()> onComplete = nullptr);
    
    // New parallel command recording methods
    std::vector<uint64_t> submitGraphicsParallel(const std::vector<GraphicsCommand>& commands);
    uint64_t submitGraphicsWithDependency(const GraphicsCommand& command, uint64_t waitForId);
    void submitChunkRenderCommands(const std::vector<GraphicsCommand>& commands, 
                                  std::function<void()> onComplete = nullptr);
    
    void waitForTransfer(uint64_t transferId);
    void waitForGraphics(uint64_t commandId);
    void waitForAll();
    
    bool isTransferComplete(uint64_t transferId);
    bool isGraphicsComplete(uint64_t commandId);
    
    void processCompletedOperations();
    
    size_t getPendingTransfers() const { return pendingTransfers.size(); }
    size_t getPendingGraphics() const { return pendingGraphics.size(); }
    
private:
    void createSynchronizationObjects();
    void destroySynchronizationObjects();
    void createCommandPools();
    void destroyCommandPools();
    VkCommandBuffer getAvailableCommandBuffer(uint32_t& poolIndex);
    void returnCommandBuffer(uint32_t poolIndex, VkCommandBuffer commandBuffer);
    VkSemaphore getAvailableSemaphore();
    VkFence getAvailableFence();
    void returnSemaphore(VkSemaphore semaphore);
    void returnFence(VkFence fence);
    
private:
    VulkanContext* vulkanContext;
    
    std::vector<VkSemaphore> availableSemaphores;
    std::vector<VkFence> availableFences;
    std::mutex syncObjectMutex;
    
    std::unordered_map<uint64_t, std::unique_ptr<SynchronizedTransfer>> pendingTransfers;
    std::unordered_map<uint64_t, std::unique_ptr<GraphicsCommand>> pendingGraphics;
    std::mutex transferMutex;
    std::mutex graphicsMutex;
    
    // Parallel command recording support
    std::vector<VkCommandPool> commandPools;
    std::vector<std::vector<VkCommandBuffer>> commandBufferPools;
    std::vector<std::unique_ptr<std::mutex>> poolMutexes;
    std::atomic<uint32_t> nextPoolIndex{0};
    
    bool initialized = false;
    static constexpr size_t SEMAPHORE_POOL_SIZE = 32;
    static constexpr size_t FENCE_POOL_SIZE = 32;
    static constexpr size_t COMMAND_POOL_COUNT = 4;
    static constexpr size_t BUFFERS_PER_POOL = 8;
};