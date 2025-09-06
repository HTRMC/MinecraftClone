#include "JobSystem.hpp"
#include "Logger.hpp"
#include <algorithm>

JobSystem::JobSystem(size_t numThreads) : numThreads(numThreads) {
    Logger::info("JobSystem", "Initializing job system with " + std::to_string(numThreads) + " worker threads");
}

JobSystem::~JobSystem() {
    stop();
}

void JobSystem::start() {
    if (running.load()) return;
    
    running.store(true);
    workers.reserve(numThreads);
    
    for (size_t i = 0; i < numThreads; ++i) {
        workers.emplace_back(&JobSystem::workerLoop, this);
    }
    
    Logger::info("JobSystem", "Started " + std::to_string(numThreads) + " worker threads");
}

void JobSystem::stop() {
    if (!running.load()) return;
    
    running.store(false);
    queueCondition.notify_all();
    
    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    
    workers.clear();
    Logger::info("JobSystem", "Stopped all worker threads");
}

void JobSystem::waitForAll() {
    std::unique_lock<std::mutex> lock(queueMutex);
    queueCondition.wait(lock, [this] {
        return jobQueue.empty() && activeJobs.load() == 0;
    });
}

void JobSystem::submitChunkGeneration(ChunkData* chunk) {
    submit(JobPriority::HIGH, [this, chunk]() {
        generateChunkFaces(chunk);
        generateChunkModels(chunk);
        generateChunkLighting(chunk);
        chunk->isDirty = false;
    });
}

std::future<void> JobSystem::submitChunkGenerationParallel(ChunkData* chunk) {
    auto promise = std::make_shared<std::promise<void>>();
    auto future = promise->get_future();
    
    auto completionCounter = std::make_shared<std::atomic<int>>(3);
    
    auto onTaskComplete = [promise, completionCounter, chunk]() {
        if (completionCounter->fetch_sub(1) == 1) {
            chunk->isDirty = false;
            promise->set_value();
        }
    };
    
    submit(JobPriority::HIGH, [this, chunk, onTaskComplete]() {
        generateChunkFaces(chunk);
        onTaskComplete();
    });
    
    submit(JobPriority::HIGH, [this, chunk, onTaskComplete]() {
        generateChunkModels(chunk);
        onTaskComplete();
    });
    
    submit(JobPriority::HIGH, [this, chunk, onTaskComplete]() {
        generateChunkLighting(chunk);
        onTaskComplete();
    });
    
    return future;
}

void JobSystem::submitChunkOperationsParallel(ChunkData* chunk, std::function<void()> onComplete) {
    if (!chunk) {
        if (onComplete) onComplete();
        return;
    }
    
    auto completionCounter = std::make_shared<std::atomic<int>>(3);
    
    auto onTaskComplete = [completionCounter, chunk, onComplete]() {
        if (completionCounter->fetch_sub(1) == 1) {
            chunk->isDirty = false;
            if (onComplete) onComplete();
        }
    };
    
    submit(JobPriority::HIGH, [this, chunk, onTaskComplete]() {
        generateChunkFaces(chunk);
        onTaskComplete();
    });
    
    submit(JobPriority::HIGH, [this, chunk, onTaskComplete]() {
        generateChunkModels(chunk);
        onTaskComplete();
    });
    
    submit(JobPriority::HIGH, [this, chunk, onTaskComplete]() {
        generateChunkLighting(chunk);
        onTaskComplete();
    });
}

void JobSystem::submitChunkMeshUpdate(ChunkData* chunk) {
    submit(JobPriority::DEFAULT, [this, chunk]() {
        generateChunkFaces(chunk);
        chunk->isDirty = false;
    });
}

void JobSystem::submitChunkLightUpdate(ChunkData* chunk) {
    submit(JobPriority::LOW, [this, chunk]() {
        generateChunkLighting(chunk);
    });
}

void JobSystem::submitMultipleChunksParallel(std::vector<ChunkData*> chunks, std::function<void()> onComplete) {
    if (chunks.empty()) {
        if (onComplete) onComplete();
        return;
    }
    
    auto completionCounter = std::make_shared<std::atomic<int>>(static_cast<int>(chunks.size()));
    
    auto onChunkComplete = [completionCounter, onComplete]() {
        if (completionCounter->fetch_sub(1) == 1) {
            if (onComplete) onComplete();
        }
    };
    
    for (ChunkData* chunk : chunks) {
        submitChunkOperationsParallel(chunk, onChunkComplete);
    }
    
    Logger::info("JobSystem", "Submitted " + std::to_string(chunks.size()) + " chunks for parallel processing");
}

size_t JobSystem::getQueueSize() const {
    std::lock_guard<std::mutex> lock(queueMutex);
    return jobQueue.size();
}

size_t JobSystem::getActiveJobs() const {
    return activeJobs.load();
}

void JobSystem::workerLoop() {
    while (running.load()) {
        Job job([]{}, JobPriority::DEFAULT);
        bool hasJob = false;
        
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            queueCondition.wait(lock, [this] {
                return !jobQueue.empty() || !running.load();
            });
            
            if (!running.load()) break;
            
            if (!jobQueue.empty()) {
                job = std::move(const_cast<Job&>(jobQueue.top()));
                jobQueue.pop();
                hasJob = true;
            }
        }
        
        if (hasJob) {
            activeJobs.fetch_add(1);
            
            try {
                job.task();
            } catch (const std::exception& e) {
                Logger::error("JobSystem", std::string("Job execution failed: ") + e.what());
            }
            
            activeJobs.fetch_sub(1);
            queueCondition.notify_all();
        }
    }
}

void JobSystem::gain(ChunkData* chunk) {
    std::lock_guard<std::mutex> lock(chunkMutex);
    
    chunk->faces.vertices.clear();
    chunk->faces.indices.clear();
    
    chunk->faces.vertices.reserve(16 * 16 * 16 * 4 * 5);
    chunk->faces.indices.reserve(16 * 16 * 16 * 6);
    
    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            for (int z = 0; z < 16; ++z) {
                float worldX = chunk->x * 16 + x;
                float worldY = y;
                float worldZ = chunk->z * 16 + z;
                
                chunk->faces.vertices.insert(chunk->faces.vertices.end(), {
                    worldX, worldY, worldZ, 0.0f, 0.0f,
                    worldX + 1, worldY, worldZ, 1.0f, 0.0f,
                    worldX + 1, worldY + 1, worldZ, 1.0f, 1.0f,
                    worldX, worldY + 1, worldZ, 0.0f, 1.0f
                });
                
                uint32_t baseIndex = static_cast<uint32_t>(chunk->faces.vertices.size() / 5) - 4;
                chunk->faces.indices.insert(chunk->faces.indices.end(), {
                    baseIndex, baseIndex + 1, baseIndex + 2,
                    baseIndex, baseIndex + 2, baseIndex + 3
                });
            }
        }
    }
    
    Logger::debug("JobSystem", "Generated " + std::to_string(chunk->faces.vertices.size() / 5) + 
                  " vertices for chunk (" + std::to_string(chunk->x) + ", " + std::to_string(chunk->z) + ")");
}

void JobSystem::generateChunkModels(ChunkData* chunk) {
    std::lock_guard<std::mutex> lock(chunkMutex);
    
    chunk->models.transforms.clear();
    chunk->models.modelIds.clear();
    
    chunk->models.transforms.reserve(10 * 16);
    chunk->models.modelIds.reserve(10);
    
    for (int i = 0; i < 10; ++i) {
        float x = chunk->x * 16 + (i % 16);
        float z = chunk->z * 16 + (i / 16);
        
        chunk->models.transforms.insert(chunk->models.transforms.end(), {
            x, 0.0f, z, 1.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            1.0f, 1.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        });
        
        chunk->models.modelIds.push_back(i % 3);
    }
}

void JobSystem::generateChunkLighting(ChunkData* chunk) {
    std::lock_guard<std::mutex> lock(chunkMutex);
    
    chunk->lighting.lightLevels.clear();
    chunk->lighting.lightColors.clear();
    
    chunk->lighting.lightLevels.reserve(16 * 16 * 16);
    chunk->lighting.lightColors.reserve(16 * 16 * 16);
    
    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            for (int z = 0; z < 16; ++z) {
                uint8_t lightLevel = std::max(0, 15 - y);
                uint32_t lightColor = 0xFFFFFFFF;
                
                chunk->lighting.lightLevels.push_back(lightLevel);
                chunk->lighting.lightColors.push_back(lightColor);
            }
        }
    }
}