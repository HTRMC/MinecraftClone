#include "QueueManager.hpp"
#include "Logger.hpp"

QueueManager::QueueManager(VulkanContext* vulkanContext) : vulkanContext(vulkanContext) {
}

QueueManager::~QueueManager() {
    cleanup();
}

void QueueManager::init() {
    if (initialized) return;
    
    createSynchronizationObjects();
    initialized = true;
    
    Logger::info("QueueManager", "Initialized with " + std::to_string(SEMAPHORE_POOL_SIZE) + 
                 " semaphores and " + std::to_string(FENCE_POOL_SIZE) + " fences");
}

void QueueManager::cleanup() {
    if (!initialized) return;
    
    waitForAll();
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
    
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuffer;
    
    if (graphicsCmd->waitSemaphore != VK_NULL_HANDLE) {
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &graphicsCmd->waitSemaphore;
        submitInfo.pWaitDstStageMask = &graphicsCmd->waitStage;
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