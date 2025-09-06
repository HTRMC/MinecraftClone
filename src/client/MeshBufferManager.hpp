#pragma once
#include "vulkan/VulkanContext.hpp"
#include "JobSystem.hpp"
#include <unordered_map>
#include <memory>

struct ChunkBufferData {
    BufferInfo* faceBuffer = nullptr;
    BufferInfo* modelBuffer = nullptr; 
    BufferInfo* lightBuffer = nullptr;
    uint32_t bufferIndex = UINT32_MAX;
    bool pendingWrite = false;
};

class MeshBufferManager {
public:
    MeshBufferManager(VulkanContext* vulkanContext, JobSystem* jobSystem);
    ~MeshBufferManager();
    
    void init(VkDeviceSize faceBufferSize, VkDeviceSize modelBufferSize, VkDeviceSize lightBufferSize);
    void cleanup();
    
    void processChunk(ChunkData* chunk, JobPriority priority = JobPriority::DEFAULT);
    void updateChunkMesh(int32_t chunkX, int32_t chunkZ);
    void updateChunkLighting(int32_t chunkX, int32_t chunkZ);
    
    ChunkBufferData* getChunkBuffers(int32_t chunkX, int32_t chunkZ);
    void releaseChunkBuffers(int32_t chunkX, int32_t chunkZ);
    
    size_t getPendingJobs() const;
    void waitForAllJobs();
    
private:
    void writeChunkToBuffers(ChunkData* chunk);
    uint64_t getChunkKey(int32_t x, int32_t z) const;
    
private:
    VulkanContext* vulkanContext;
    JobSystem* jobSystem;
    
    BufferPool faceBufferPool;
    BufferPool modelBufferPool;
    BufferPool lightBufferPool;
    
    std::unordered_map<uint64_t, std::unique_ptr<ChunkData>> chunks;
    std::unordered_map<uint64_t, ChunkBufferData> chunkBuffers;
    
    mutable std::mutex chunkMutex;
    bool initialized = false;
};