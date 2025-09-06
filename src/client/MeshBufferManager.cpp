#include "MeshBufferManager.hpp"
#include "Logger.hpp"
#include <cstring>

MeshBufferManager::MeshBufferManager(VulkanContext* vulkanContext, JobSystem* jobSystem)
    : vulkanContext(vulkanContext), jobSystem(jobSystem) {
}

MeshBufferManager::~MeshBufferManager() {
    cleanup();
}

void MeshBufferManager::init(VkDeviceSize faceBufferSize, VkDeviceSize modelBufferSize, VkDeviceSize lightBufferSize) {
    if (initialized) return;
    
    faceBufferPool = vulkanContext->createBufferPool(faceBufferSize);
    modelBufferPool = vulkanContext->createBufferPool(modelBufferSize);
    lightBufferPool = vulkanContext->createBufferPool(lightBufferSize);
    
    initialized = true;
    Logger::info("MeshBufferManager", "Initialized with buffer pools - Face: " + 
                 std::to_string(faceBufferSize) + ", Model: " + std::to_string(modelBufferSize) + 
                 ", Light: " + std::to_string(lightBufferSize));
}

void MeshBufferManager::cleanup() {
    if (!initialized) return;
    
    waitForAllJobs();
    
    {
        std::lock_guard<std::mutex> lock(chunkMutex);
        chunkBuffers.clear();
        chunks.clear();
    }
    
    vulkanContext->destroyBufferPool(faceBufferPool);
    vulkanContext->destroyBufferPool(modelBufferPool);
    vulkanContext->destroyBufferPool(lightBufferPool);
    
    initialized = false;
}

void MeshBufferManager::processChunk(ChunkData* chunk, JobPriority priority) {
    if (!initialized || !chunk) return;
    
    uint64_t chunkKey = getChunkKey(chunk->x, chunk->z);
    
    {
        std::lock_guard<std::mutex> lock(chunkMutex);
        
        if (chunks.find(chunkKey) == chunks.end()) {
            chunks[chunkKey] = std::make_unique<ChunkData>(*chunk);
        } else {
            *chunks[chunkKey] = *chunk;
        }
        
        chunks[chunkKey]->isDirty = true;
    }
    
    jobSystem->submit(priority, [this, chunkKey]() {
        ChunkData* chunkData = nullptr;
        {
            std::lock_guard<std::mutex> lock(chunkMutex);
            auto it = chunks.find(chunkKey);
            if (it != chunks.end()) {
                chunkData = it->second.get();
            }
        }
        
        if (chunkData && chunkData->isDirty) {
            jobSystem->submitChunkGeneration(chunkData);
            writeChunkToBuffers(chunkData);
        }
    });
}

void MeshBufferManager::updateChunkMesh(int32_t chunkX, int32_t chunkZ) {
    uint64_t chunkKey = getChunkKey(chunkX, chunkZ);
    
    std::lock_guard<std::mutex> lock(chunkMutex);
    auto it = chunks.find(chunkKey);
    if (it != chunks.end()) {
        ChunkData* chunk = it->second.get();
        jobSystem->submitChunkMeshUpdate(chunk);
        
        jobSystem->submit(JobPriority::DEFAULT, [this, chunk]() {
            writeChunkToBuffers(chunk);
        });
    }
}

void MeshBufferManager::updateChunkLighting(int32_t chunkX, int32_t chunkZ) {
    uint64_t chunkKey = getChunkKey(chunkX, chunkZ);
    
    std::lock_guard<std::mutex> lock(chunkMutex);
    auto it = chunks.find(chunkKey);
    if (it != chunks.end()) {
        ChunkData* chunk = it->second.get();
        jobSystem->submitChunkLightUpdate(chunk);
        
        jobSystem->submit(JobPriority::LOW, [this, chunk]() {
            if (!chunk->lighting.lightLevels.empty()) {
                BufferInfo* lightBuffer = vulkanContext->acquireBuffer(lightBufferPool);
                if (lightBuffer && lightBuffer->mappedMemory) {
                    size_t dataSize = chunk->lighting.lightLevels.size() + 
                                    chunk->lighting.lightColors.size() * sizeof(uint32_t);
                    if (dataSize <= lightBuffer->size) {
                        std::memcpy(lightBuffer->mappedMemory, chunk->lighting.lightLevels.data(), 
                                   chunk->lighting.lightLevels.size());
                        std::memcpy(static_cast<char*>(lightBuffer->mappedMemory) + chunk->lighting.lightLevels.size(),
                                   chunk->lighting.lightColors.data(), 
                                   chunk->lighting.lightColors.size() * sizeof(uint32_t));
                    }
                }
            }
        });
    }
}

ChunkBufferData* MeshBufferManager::getChunkBuffers(int32_t chunkX, int32_t chunkZ) {
    uint64_t chunkKey = getChunkKey(chunkX, chunkZ);
    
    std::lock_guard<std::mutex> lock(chunkMutex);
    auto it = chunkBuffers.find(chunkKey);
    return (it != chunkBuffers.end()) ? &it->second : nullptr;
}

void MeshBufferManager::releaseChunkBuffers(int32_t chunkX, int32_t chunkZ) {
    uint64_t chunkKey = getChunkKey(chunkX, chunkZ);
    
    std::lock_guard<std::mutex> lock(chunkMutex);
    auto it = chunkBuffers.find(chunkKey);
    if (it != chunkBuffers.end()) {
        ChunkBufferData& bufferData = it->second;
        if (bufferData.bufferIndex != UINT32_MAX) {
            vulkanContext->releaseBuffer(faceBufferPool, bufferData.bufferIndex);
            vulkanContext->releaseBuffer(modelBufferPool, bufferData.bufferIndex);
            vulkanContext->releaseBuffer(lightBufferPool, bufferData.bufferIndex);
        }
        chunkBuffers.erase(it);
    }
}

size_t MeshBufferManager::getPendingJobs() const {
    return jobSystem->getQueueSize() + jobSystem->getActiveJobs();
}

void MeshBufferManager::waitForAllJobs() {
    jobSystem->waitForAll();
}

void MeshBufferManager::writeChunkToBuffers(ChunkData* chunk) {
    if (!chunk || chunk->faces.vertices.empty()) return;
    
    vulkanContext->processBufferFences(faceBufferPool);
    vulkanContext->processBufferFences(modelBufferPool);
    vulkanContext->processBufferFences(lightBufferPool);
    
    BufferInfo* faceBuffer = vulkanContext->acquireBuffer(faceBufferPool);
    BufferInfo* modelBuffer = vulkanContext->acquireBuffer(modelBufferPool);
    BufferInfo* lightBuffer = vulkanContext->acquireBuffer(lightBufferPool);
    
    if (!faceBuffer || !modelBuffer || !lightBuffer) {
        Logger::warning("MeshBufferManager", "Failed to acquire buffers for chunk (" + 
                       std::to_string(chunk->x) + ", " + std::to_string(chunk->z) + ")");
        return;
    }
    
    size_t faceDataSize = chunk->faces.vertices.size() * sizeof(float) + 
                         chunk->faces.indices.size() * sizeof(uint32_t);
    size_t modelDataSize = chunk->models.transforms.size() * sizeof(float) + 
                          chunk->models.modelIds.size() * sizeof(uint32_t);
    size_t lightDataSize = chunk->lighting.lightLevels.size() + 
                          chunk->lighting.lightColors.size() * sizeof(uint32_t);
    
    if (faceDataSize <= faceBuffer->size && faceBuffer->mappedMemory) {
        char* facePtr = static_cast<char*>(faceBuffer->mappedMemory);
        std::memcpy(facePtr, chunk->faces.vertices.data(), chunk->faces.vertices.size() * sizeof(float));
        std::memcpy(facePtr + chunk->faces.vertices.size() * sizeof(float), 
                   chunk->faces.indices.data(), chunk->faces.indices.size() * sizeof(uint32_t));
    }
    
    if (modelDataSize <= modelBuffer->size && modelBuffer->mappedMemory) {
        char* modelPtr = static_cast<char*>(modelBuffer->mappedMemory);
        std::memcpy(modelPtr, chunk->models.transforms.data(), chunk->models.transforms.size() * sizeof(float));
        std::memcpy(modelPtr + chunk->models.transforms.size() * sizeof(float),
                   chunk->models.modelIds.data(), chunk->models.modelIds.size() * sizeof(uint32_t));
    }
    
    if (lightDataSize <= lightBuffer->size && lightBuffer->mappedMemory) {
        char* lightPtr = static_cast<char*>(lightBuffer->mappedMemory);
        std::memcpy(lightPtr, chunk->lighting.lightLevels.data(), chunk->lighting.lightLevels.size());
        std::memcpy(lightPtr + chunk->lighting.lightLevels.size(),
                   chunk->lighting.lightColors.data(), chunk->lighting.lightColors.size() * sizeof(uint32_t));
    }
    
    uint64_t chunkKey = getChunkKey(chunk->x, chunk->z);
    std::lock_guard<std::mutex> lock(chunkMutex);
    
    ChunkBufferData bufferData;
    bufferData.faceBuffer = faceBuffer;
    bufferData.modelBuffer = modelBuffer;
    bufferData.lightBuffer = lightBuffer;
    bufferData.pendingWrite = false;
    
    chunkBuffers[chunkKey] = bufferData;
    
    Logger::debug("MeshBufferManager", "Written chunk (" + std::to_string(chunk->x) + 
                  ", " + std::to_string(chunk->z) + ") data to GPU buffers");
}

uint64_t MeshBufferManager::submitChunkToGPU(ChunkData* chunk, std::function<void()> onComplete) {
    if (!chunk || chunk->faces.vertices.empty()) return 0;
    
    static std::atomic<uint64_t> submissionCounter{1};
    uint64_t submissionId = submissionCounter.fetch_add(1);
    
    writeChunkToBuffers(chunk);
    
    uint64_t chunkKey = getChunkKey(chunk->x, chunk->z);
    auto& bufferData = chunkBuffers[chunkKey];
    
    if (bufferData.faceBuffer) {
        VkFence fence = vulkanContext->createFence();
        
        uint32_t faceBufferIndex = 0;
        for (uint32_t i = 0; i < BufferPool::BUFFER_COUNT; ++i) {
            if (&faceBufferPool.buffers[i] == bufferData.faceBuffer) {
                faceBufferIndex = i;
                break;
            }
        }
        
        vulkanContext->submitBufferOperation(faceBufferPool, faceBufferIndex, fence, submissionId, 
            [this, chunkKey, onComplete]() {
                Logger::debug("MeshBufferManager", "Chunk buffer operation completed");
                if (onComplete) onComplete();
            });
        
        Logger::info("MeshBufferManager", "Submitted chunk (" + std::to_string(chunk->x) + 
                     ", " + std::to_string(chunk->z) + ") to GPU with fence tracking");
    }
    
    return submissionId;
}

void MeshBufferManager::processCompletedSubmissions() {
    vulkanContext->processBufferFences(faceBufferPool);
    vulkanContext->processBufferFences(modelBufferPool);
    vulkanContext->processBufferFences(lightBufferPool);
}

uint64_t MeshBufferManager::getChunkKey(int32_t x, int32_t z) const {
    return (static_cast<uint64_t>(static_cast<uint32_t>(x)) << 32) | static_cast<uint32_t>(z);
}