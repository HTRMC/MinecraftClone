#include "QueueManager.hpp"
#include "Logger.hpp"
#include <thread>

QueueManager::QueueManager(VulkanContext* vulkanContext) : vulkanContext(vulkanContext) {
}

QueueManager::~QueueManager() {
    cleanup();
}

void QueueManager::init() {
    if (initialized) return;
    
    createSynchronizationObjects();
    createCommandPools();
    initialized = true;
    
    Logger::info("QueueManager", "Initialized with " + std::to_string(SEMAPHORE_POOL_SIZE) + 
                 " semaphores, " + std::to_string(FENCE_POOL_SIZE) + " fences, and " + 
                 std::to_string(COMMAND_POOL_COUNT) + " command pools");
}

void QueueManager::cleanup() {
    if (!initialized) return;
    
    waitForAll();
    destroyCommandPools();
    destroySynchronizationObjects();
    
    pendingTransfers.clear();
    pendingGraphics.clear();
    
    initialized = false;
}

uint64_t QueueManager::submitTransfer(const TransferOperation& transfer, std::function<void()> onComplete) {
    auto syncTransfer = std::make_unique<SynchronizedTransfer>(transfer, std::move(onComplete));
    uint64_t id = syncTransfer->id;
    
    syncTransfer->transfer.completionSemaphore = getAvailableSemaphore();
    syncTransfer->transfer.completionFence = getAvailableFence();
    syncTransfer->inProgress = true;
    
    vulkanContext->copyBufferAsync(syncTransfer->transfer);
    
    {
        std::lock_guard<std::mutex> lock(transferMutex);
        pendingTransfers[id] = std::move(syncTransfer);
    }
    
    Logger::debug("QueueManager", "Submitted transfer operation " + std::to_string(id));
    return id;
}

uint64_t QueueManager::submitGraphics(const GraphicsCommand& command) {
    auto graphicsCmd = std::make_unique<GraphicsCommand>(command);
    uint64_t id = graphicsCmd->id;
    
    if (graphicsCmd->signalFence == VK_NULL_HANDLE) {
        graphicsCmd->signalFence = getAvailableFence();
    }
    
    VkCommandBuffer cmdBuffer = vulkanContext->beginSingleTimeCommands(false);
    graphicsCmd->recordFunc(cmdBuffer);
    vkEndCommandBuffer(cmdBuffer);
    
    VkTimelineSemaphoreSubmitInfo timelineSubmitInfo{};
    std::vector<VkSemaphore> waitSemaphores;
    std::vector<uint64_t> waitValues;
    std::vector<VkPipelineStageFlags> waitStages;
    std::vector<VkSemaphore> signalSemaphores;
    std::vector<uint64_t> signalValues;
    
    if (graphicsCmd->waitSemaphore != VK_NULL_HANDLE) {
        waitSemaphores.push_back(graphicsCmd->waitSemaphore);
        waitValues.push_back(0); // Binary semaphore
        waitStages.push_back(graphicsCmd->waitStage);
    }
    
    if (graphicsCmd->waitTimelineSemaphore != nullptr) {
        waitSemaphores.push_back(graphicsCmd->waitTimelineSemaphore->semaphore);
        waitValues.push_back(graphicsCmd->waitTimelineValue);
        waitStages.push_back(graphicsCmd->waitStage);
    }
    
    if (graphicsCmd->signalTimelineSemaphore != nullptr) {
        signalSemaphores.push_back(graphicsCmd->signalTimelineSemaphore->semaphore);
        signalValues.push_back(graphicsCmd->signalTimelineValue);
    }
    
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuffer;
    
    if (!waitSemaphores.empty()) {
        submitInfo.waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.size());
        submitInfo.pWaitSemaphores = waitSemaphores.data();
        submitInfo.pWaitDstStageMask = waitStages.data();
        
        timelineSubmitInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timelineSubmitInfo.waitSemaphoreValueCount = static_cast<uint32_t>(waitValues.size());
        timelineSubmitInfo.pWaitSemaphoreValues = waitValues.data();
        
        if (!signalSemaphores.empty()) {
            submitInfo.signalSemaphoreCount = static_cast<uint32_t>(signalSemaphores.size());
            submitInfo.pSignalSemaphores = signalSemaphores.data();
            timelineSubmitInfo.signalSemaphoreValueCount = static_cast<uint32_t>(signalValues.size());
            timelineSubmitInfo.pSignalSemaphoreValues = signalValues.data();
        }
        
        submitInfo.pNext = &timelineSubmitInfo;
    } else if (!signalSemaphores.empty()) {
        submitInfo.signalSemaphoreCount = static_cast<uint32_t>(signalSemaphores.size());
        submitInfo.pSignalSemaphores = signalSemaphores.data();
        
        timelineSubmitInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timelineSubmitInfo.signalSemaphoreValueCount = static_cast<uint32_t>(signalValues.size());
        timelineSubmitInfo.pSignalSemaphoreValues = signalValues.data();
        submitInfo.pNext = &timelineSubmitInfo;
    }
    
    vkQueueSubmit(vulkanContext->getGraphicsQueue(), 1, &submitInfo, graphicsCmd->signalFence);
    
    {
        std::lock_guard<std::mutex> lock(graphicsMutex);
        pendingGraphics[id] = std::move(graphicsCmd);
    }
    
    Logger::debug("QueueManager", "Submitted graphics command " + std::to_string(id));
    return id;
}

uint64_t QueueManager::submitMeshDataTransfer(BufferInfo* stagingBuffer, BufferInfo* deviceBuffer, 
                                             VkDeviceSize size, std::function<void()> onComplete) {
    TransferOperation transfer;
    transfer.srcBuffer = stagingBuffer->buffer;
    transfer.dstBuffer = deviceBuffer->buffer;
    transfer.size = size;
    transfer.srcOffset = 0;
    transfer.dstOffset = 0;
    
    return submitTransfer(transfer, std::move(onComplete));
}

std::vector<uint64_t> QueueManager::submitGraphicsParallel(const std::vector<GraphicsCommand>& commands) {
    std::vector<uint64_t> commandIds;
    commandIds.reserve(commands.size());
    
    std::vector<VkCommandBuffer> cmdBuffers;
    std::vector<uint32_t> poolIndices;
    cmdBuffers.reserve(commands.size());
    poolIndices.reserve(commands.size());
    
    // Acquire command buffers from different pools for parallel recording
    for (size_t i = 0; i < commands.size(); ++i) {
        uint32_t poolIndex;
        VkCommandBuffer cmdBuffer = getAvailableCommandBuffer(poolIndex);
        cmdBuffers.push_back(cmdBuffer);
        poolIndices.push_back(poolIndex);
        
        // Begin recording
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmdBuffer, &beginInfo);
        
        // Record the command
        commands[i].recordFunc(cmdBuffer);
        vkEndCommandBuffer(cmdBuffer);
    }
    
    // Submit all commands at once with synchronization
    std::vector<VkSubmitInfo> submitInfos;
    std::vector<VkFence> fences;
    std::vector<std::unique_ptr<GraphicsCommand>> graphicsCommands;
    
    submitInfos.reserve(commands.size());
    fences.reserve(commands.size());
    graphicsCommands.reserve(commands.size());
    
    for (size_t i = 0; i < commands.size(); ++i) {
        auto graphicsCmd = std::make_unique<GraphicsCommand>(commands[i]);
        uint64_t id = graphicsCmd->id;
        commandIds.push_back(id);
        
        VkFence fence = getAvailableFence();
        if (graphicsCmd->signalFence == VK_NULL_HANDLE) {
            graphicsCmd->signalFence = fence;
        }
        fences.push_back(graphicsCmd->signalFence);
        
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmdBuffers[i];
        
        if (graphicsCmd->waitSemaphore != VK_NULL_HANDLE) {
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = &graphicsCmd->waitSemaphore;
            submitInfo.pWaitDstStageMask = &graphicsCmd->waitStage;
        }
        
        submitInfos.push_back(submitInfo);
        graphicsCommands.push_back(std::move(graphicsCmd));
    }
    
    // Submit all commands in batch
    if (!submitInfos.empty()) {
        for (size_t i = 0; i < submitInfos.size(); ++i) {
            vkQueueSubmit(vulkanContext->getGraphicsQueue(), 1, &submitInfos[i], fences[i]);
        }
        
        // Store pending graphics commands
        {
            std::lock_guard<std::mutex> lock(graphicsMutex);
            for (auto& cmd : graphicsCommands) {
                pendingGraphics[cmd->id] = std::move(cmd);
            }
        }
    }
    
    // Return command buffers to pools
    for (size_t i = 0; i < cmdBuffers.size(); ++i) {
        returnCommandBuffer(poolIndices[i], cmdBuffers[i]);
    }
    
    Logger::debug("QueueManager", "Submitted " + std::to_string(commands.size()) + " graphics commands in parallel");
    return commandIds;
}

uint64_t QueueManager::submitGraphicsWithDependency(const GraphicsCommand& command, uint64_t waitForId) {
    // Find the fence/semaphore from the dependency
    VkSemaphore dependencySemaphore = VK_NULL_HANDLE;
    {
        std::lock_guard<std::mutex> lock(graphicsMutex);
        auto it = pendingGraphics.find(waitForId);
        if (it != pendingGraphics.end()) {
            dependencySemaphore = getAvailableSemaphore();
        }
    }
    
    GraphicsCommand depCommand = command;
    depCommand.waitSemaphore = dependencySemaphore;
    return submitGraphics(depCommand);
}

void QueueManager::submitChunkRenderCommands(const std::vector<GraphicsCommand>& commands, 
                                            std::function<void()> onComplete) {
    if (commands.empty()) {
        if (onComplete) onComplete();
        return;
    }
    
    auto commandIds = submitGraphicsParallel(commands);
    
    if (onComplete) {
        // Create a completion checker job
        std::thread([this, commandIds, onComplete]() {
            // Wait for all commands to complete
            for (uint64_t id : commandIds) {
                waitForGraphics(id);
            }
            onComplete();
        }).detach();
    }
}

void QueueManager::waitForTransfer(uint64_t transferId) {
    std::lock_guard<std::mutex> lock(transferMutex);
    auto it = pendingTransfers.find(transferId);
    if (it != pendingTransfers.end() && it->second->transfer.completionFence != VK_NULL_HANDLE) {
        vkWaitForFences(vulkanContext->getDevice(), 1, &it->second->transfer.completionFence, VK_TRUE, UINT64_MAX);
    }
}

void QueueManager::waitForGraphics(uint64_t commandId) {
    std::lock_guard<std::mutex> lock(graphicsMutex);
    auto it = pendingGraphics.find(commandId);
    if (it != pendingGraphics.end() && it->second->signalFence != VK_NULL_HANDLE) {
        vkWaitForFences(vulkanContext->getDevice(), 1, &it->second->signalFence, VK_TRUE, UINT64_MAX);
    }
}

void QueueManager::waitForAll() {
    vkDeviceWaitIdle(vulkanContext->getDevice());
    processCompletedOperations();
}

bool QueueManager::isTransferComplete(uint64_t transferId) {
    std::lock_guard<std::mutex> lock(transferMutex);
    auto it = pendingTransfers.find(transferId);
    if (it == pendingTransfers.end()) return true;
    
    if (it->second->transfer.completionFence != VK_NULL_HANDLE) {
        return vkGetFenceStatus(vulkanContext->getDevice(), it->second->transfer.completionFence) == VK_SUCCESS;
    }
    return false;
}

bool QueueManager::isGraphicsComplete(uint64_t commandId) {
    std::lock_guard<std::mutex> lock(graphicsMutex);
    auto it = pendingGraphics.find(commandId);
    if (it == pendingGraphics.end()) return true;
    
    if (it->second->signalFence != VK_NULL_HANDLE) {
        return vkGetFenceStatus(vulkanContext->getDevice(), it->second->signalFence) == VK_SUCCESS;
    }
    return false;
}

void QueueManager::processCompletedOperations() {
    {
        std::lock_guard<std::mutex> lock(transferMutex);
        auto it = pendingTransfers.begin();
        while (it != pendingTransfers.end()) {
            bool isComplete = (it->second->transfer.completionFence != VK_NULL_HANDLE) &&
                             (vkGetFenceStatus(vulkanContext->getDevice(), it->second->transfer.completionFence) == VK_SUCCESS);
            
            if (isComplete) {
                if (it->second->onComplete) {
                    it->second->onComplete();
                }
                
                if (it->second->transfer.completionSemaphore != VK_NULL_HANDLE) {
                    returnSemaphore(it->second->transfer.completionSemaphore);
                }
                if (it->second->transfer.completionFence != VK_NULL_HANDLE) {
                    returnFence(it->second->transfer.completionFence);
                }
                
                it = pendingTransfers.erase(it);
            } else {
                ++it;
            }
        }
    }
    
    {
        std::lock_guard<std::mutex> lock(graphicsMutex);
        auto it = pendingGraphics.begin();
        while (it != pendingGraphics.end()) {
            bool isComplete = (it->second->signalFence != VK_NULL_HANDLE) &&
                             (vkGetFenceStatus(vulkanContext->getDevice(), it->second->signalFence) == VK_SUCCESS);
            
            if (isComplete) {
                if (it->second->signalFence != VK_NULL_HANDLE) {
                    returnFence(it->second->signalFence);
                }
                
                it = pendingGraphics.erase(it);
            } else {
                ++it;
            }
        }
    }
}

void QueueManager::createSynchronizationObjects() {
    availableSemaphores.reserve(SEMAPHORE_POOL_SIZE);
    availableFences.reserve(FENCE_POOL_SIZE);
    
    for (size_t i = 0; i < SEMAPHORE_POOL_SIZE; ++i) {
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        
        VkSemaphore semaphore;
        if (vkCreateSemaphore(vulkanContext->getDevice(), &semaphoreInfo, nullptr, &semaphore) == VK_SUCCESS) {
            availableSemaphores.push_back(semaphore);
        }
    }
    
    for (size_t i = 0; i < FENCE_POOL_SIZE; ++i) {
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        
        VkFence fence;
        if (vkCreateFence(vulkanContext->getDevice(), &fenceInfo, nullptr, &fence) == VK_SUCCESS) {
            availableFences.push_back(fence);
        }
    }
}

void QueueManager::destroySynchronizationObjects() {
    VkDevice device = vulkanContext->getDevice();
    
    for (VkSemaphore semaphore : availableSemaphores) {
        vkDestroySemaphore(device, semaphore, nullptr);
    }
    availableSemaphores.clear();
    
    for (VkFence fence : availableFences) {
        vkDestroyFence(device, fence, nullptr);
    }
    availableFences.clear();
}

VkSemaphore QueueManager::getAvailableSemaphore() {
    std::lock_guard<std::mutex> lock(syncObjectMutex);
    if (!availableSemaphores.empty()) {
        VkSemaphore semaphore = availableSemaphores.back();
        availableSemaphores.pop_back();
        return semaphore;
    }
    
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    
    VkSemaphore semaphore;
    if (vkCreateSemaphore(vulkanContext->getDevice(), &semaphoreInfo, nullptr, &semaphore) == VK_SUCCESS) {
        return semaphore;
    }
    
    return VK_NULL_HANDLE;
}

VkFence QueueManager::getAvailableFence() {
    std::lock_guard<std::mutex> lock(syncObjectMutex);
    if (!availableFences.empty()) {
        VkFence fence = availableFences.back();
        availableFences.pop_back();
        vkResetFences(vulkanContext->getDevice(), 1, &fence);
        return fence;
    }
    
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    
    VkFence fence;
    if (vkCreateFence(vulkanContext->getDevice(), &fenceInfo, nullptr, &fence) == VK_SUCCESS) {
        return fence;
    }
    
    return VK_NULL_HANDLE;
}

void QueueManager::returnSemaphore(VkSemaphore semaphore) {
    if (semaphore == VK_NULL_HANDLE) return;
    
    std::lock_guard<std::mutex> lock(syncObjectMutex);
    if (availableSemaphores.size() < SEMAPHORE_POOL_SIZE) {
        availableSemaphores.push_back(semaphore);
    } else {
        vkDestroySemaphore(vulkanContext->getDevice(), semaphore, nullptr);
    }
}

void QueueManager::returnFence(VkFence fence) {
    if (fence == VK_NULL_HANDLE) return;
    
    std::lock_guard<std::mutex> lock(syncObjectMutex);
    if (availableFences.size() < FENCE_POOL_SIZE) {
        availableFences.push_back(fence);
    } else {
        vkDestroyFence(vulkanContext->getDevice(), fence, nullptr);
    }
}

void QueueManager::createCommandPools() {
    commandPools.resize(COMMAND_POOL_COUNT);
    commandBufferPools.resize(COMMAND_POOL_COUNT);
    poolMutexes.reserve(COMMAND_POOL_COUNT);
    
    // Initialize unique_ptr mutexes
    for (size_t i = 0; i < COMMAND_POOL_COUNT; ++i) {
        poolMutexes.push_back(std::make_unique<std::mutex>());
    }
    
    auto queueFamilyIndices = vulkanContext->findQueueFamilies(vulkanContext->getPhysicalDevice());
    
    for (size_t i = 0; i < COMMAND_POOL_COUNT; ++i) {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
        
        if (vkCreateCommandPool(vulkanContext->getDevice(), &poolInfo, nullptr, &commandPools[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create command pool " + std::to_string(i));
        }
        
        // Allocate command buffers for this pool
        commandBufferPools[i].resize(BUFFERS_PER_POOL);
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPools[i];
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = BUFFERS_PER_POOL;
        
        if (vkAllocateCommandBuffers(vulkanContext->getDevice(), &allocInfo, commandBufferPools[i].data()) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate command buffers for pool " + std::to_string(i));
        }
    }
}

void QueueManager::destroyCommandPools() {
    for (VkCommandPool pool : commandPools) {
        if (pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(vulkanContext->getDevice(), pool, nullptr);
        }
    }
    commandPools.clear();
    commandBufferPools.clear();
    poolMutexes.clear();
}

VkCommandBuffer QueueManager::getAvailableCommandBuffer(uint32_t& poolIndex) {
    poolIndex = nextPoolIndex.fetch_add(1) % COMMAND_POOL_COUNT;
    
    std::lock_guard<std::mutex> lock(*poolMutexes[poolIndex]);
    
    // For now, just use the first buffer in the pool - could implement round-robin later
    VkCommandBuffer cmdBuffer = commandBufferPools[poolIndex][0];
    
    // Reset the command buffer
    vkResetCommandBuffer(cmdBuffer, 0);
    
    return cmdBuffer;
}

void QueueManager::returnCommandBuffer(uint32_t poolIndex, VkCommandBuffer commandBuffer) {
    // Command buffer returns to its pool automatically after submission
    // No explicit action needed due to VK_COMMAND_POOL_CREATE_TRANSIENT_BIT
}