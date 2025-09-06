#pragma once
#include "VulkanContext.hpp"
#include "DescriptorManager.hpp"
#include "TextureManager.hpp"
#include "MeshShaderPipeline.hpp"
#include "MeshShaderData.hpp"
#include "client/resource/BlockModelLoader.hpp"
#include <mutex>
#include <vector>
#include <atomic>
#include <memory>

struct RenderData {
    std::vector<FaceData> faces;
    std::vector<ModelData> models;
    std::vector<LightData> lights;
    std::vector<glm::ivec4> chunkCoords;
    
    void clear() {
        faces.clear();
        models.clear();
        lights.clear();
        chunkCoords.clear();
    }
};

class ChunkRenderer {
public:
    ChunkRenderer(VulkanContext* vulkanContext, 
                  DescriptorManager* descriptorManager,
                  MeshShaderPipeline* pipeline,
                  TextureManager* textureManager);
    ~ChunkRenderer();
    
    void init();
    void cleanup();
    
    // Thread-safe methods for updating render data
    void updateRenderData(const RenderData& newData);
    void render(VkCommandBuffer commandBuffer, const UniformBufferObject& ubo, bool cameraChanged = false);
    bool hasDataChanged() const { return dataUpdated.load(); }
    
    // Add test data for initial rendering
    void addTestCube();
    void addTestCubeFromJSON();

private:
    void createBuffers();
    void updateBuffers();
    void createDescriptorSet();
    void loadDefaultTextures();
    
    BufferInfo createStorageBuffer(size_t size, const void* data = nullptr);
    void updateStorageBuffer(BufferInfo& buffer, const void* data, size_t size);
    void destroyBuffer(BufferInfo& buffer);

private:
    VulkanContext* vulkanContext;
    DescriptorManager* descriptorManager;
    MeshShaderPipeline* pipeline;
    TextureManager* textureManager;
    
    // Thread-safe render data
    std::mutex renderDataMutex;
    RenderData currentRenderData;
    std::atomic<bool> dataUpdated{false};
    
    // Vulkan buffers
    BufferInfo uboBuffer;
    BufferInfo faceBuffer;
    BufferInfo modelBuffer;
    BufferInfo lightBuffer;
    BufferInfo chunkCoordBuffer;
    
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    bool initialized = false;
};