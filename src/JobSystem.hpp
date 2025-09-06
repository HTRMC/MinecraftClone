#pragma once
#include <functional>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>
#include <future>

enum class JobPriority : int {
    CRITICAL = 1,
    HIGH = 3,
    DEFAULT = 5,
    LOW = 7,
    BACKGROUND = 10
};

struct Job {
    std::function<void()> task;
    JobPriority priority;
    uint64_t id;
    
    Job(std::function<void()> t, JobPriority p = JobPriority::DEFAULT) 
        : task(std::move(t)), priority(p), id(generateId()) {}
    
    bool operator<(const Job& other) const {
        return static_cast<int>(priority) > static_cast<int>(other.priority);
    }
    
private:
    static uint64_t generateId() {
        static std::atomic<uint64_t> counter{0};
        return counter.fetch_add(1);
    }
};

struct ChunkData {
    int32_t x, z;
    bool isDirty = false;
    
    struct FaceData {
        std::vector<float> vertices;
        std::vector<uint32_t> indices;
    } faces;
    
    struct ModelData {
        std::vector<float> transforms;
        std::vector<uint32_t> modelIds;
    } models;
    
    struct LightData {
        std::vector<uint8_t> lightLevels;
        std::vector<uint32_t> lightColors;
    } lighting;
};

class JobSystem {
public:
    JobSystem(size_t numThreads = std::thread::hardware_concurrency());
    ~JobSystem();
    
    void start();
    void stop();
    void waitForAll();
    
    template<typename Func, typename... Args>
    auto submit(JobPriority priority, Func&& func, Args&&... args) 
        -> std::future<decltype(func(args...))>;
    
    template<typename Func, typename... Args>
    auto submit(Func&& func, Args&&... args) 
        -> std::future<decltype(func(args...))> {
        return submit(JobPriority::DEFAULT, std::forward<Func>(func), std::forward<Args>(args)...);
    }
    
    void submitChunkGeneration(ChunkData* chunk);
    void submitChunkMeshUpdate(ChunkData* chunk);
    void submitChunkLightUpdate(ChunkData* chunk);
    
    std::future<void> submitChunkGenerationParallel(ChunkData* chunk);
    void submitChunkOperationsParallel(ChunkData* chunk, std::function<void()> onComplete = nullptr);
    void submitMultipleChunksParallel(std::vector<ChunkData*> chunks, std::function<void()> onComplete = nullptr);
    
    size_t getQueueSize() const;
    size_t getActiveJobs() const;
    size_t getThreadCount() const { return numThreads; }
    bool isRunning() const { return running.load(); }

    void generateChunkFaces(ChunkData* chunk);
    void generateChunkLighting(ChunkData* chunk);
    void gain(ChunkData* chunk);
    
private:
    void workerLoop();
    void generateChunkModels(ChunkData* chunk);

private:
    std::vector<std::thread> workers;
    std::priority_queue<Job> jobQueue;
    mutable std::mutex queueMutex;
    std::condition_variable queueCondition;
    std::atomic<bool> running{false};
    std::atomic<size_t> activeJobs{0};
    size_t numThreads;
    
    mutable std::mutex chunkMutex;
    std::condition_variable chunkCondition;
};

template<typename Func, typename... Args>
auto JobSystem::submit(JobPriority priority, Func&& func, Args&&... args) 
    -> std::future<decltype(func(args...))> {
    
    using ReturnType = decltype(func(args...));
    
    auto taskPtr = std::make_shared<std::packaged_task<ReturnType()>>(
        std::bind(std::forward<Func>(func), std::forward<Args>(args)...)
    );
    
    std::future<ReturnType> result = taskPtr->get_future();
    
    Job job([taskPtr]() { (*taskPtr)(); }, priority);
    
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        jobQueue.push(std::move(job));
    }
    
    queueCondition.notify_one();
    return result;
}