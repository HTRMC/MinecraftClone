#include "ChunkRenderer.hpp"
#include "Logger.hpp"
#include <cstring>
#include <glm/gtc/matrix_transform.hpp>

ChunkRenderer::ChunkRenderer(VulkanContext* vulkanContext, 
                             DescriptorManager* descriptorManager,
                             MeshShaderPipeline* pipeline)
    : vulkanContext(vulkanContext), descriptorManager(descriptorManager), pipeline(pipeline) {
}

ChunkRenderer::~ChunkRenderer() {
    cleanup();
}

void ChunkRenderer::init() {
    if (initialized) return;
    
    uboBuffer = descriptorManager->createUniformBuffer();
    createBuffers();
    createDescriptorSet();
    
    // Add test data for initial rendering
    addTestCube();
    
    initialized = true;
    Logger::info("ChunkRenderer", "Initialized chunk renderer");
}

void ChunkRenderer::cleanup() {
    if (!initialized) return;
    
    destroyBuffer(uboBuffer);
    destroyBuffer(faceBuffer);
    destroyBuffer(modelBuffer);
    destroyBuffer(lightBuffer);
    destroyBuffer(chunkCoordBuffer);
    
    initialized = false;
}

void ChunkRenderer::updateRenderData(const RenderData& newData) {
    std::lock_guard<std::mutex> lock(renderDataMutex);
    currentRenderData = newData;
    dataUpdated = true;
}

void ChunkRenderer::render(VkCommandBuffer commandBuffer, const UniformBufferObject& ubo) {
    if (!initialized || currentRenderData.faces.empty()) return;
    
    // Update UBO
    descriptorManager->updateUniformBuffer(uboBuffer, ubo);
    
    // Update buffers if data changed
    if (dataUpdated.exchange(false)) {
        std::lock_guard<std::mutex> lock(renderDataMutex);
        updateBuffers();
    }
    
    // Bind pipeline and descriptor set
    pipeline->bind(commandBuffer);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           pipeline->getPipelineLayout(), 0, 1, &descriptorSet, 0, nullptr);
    
    // Dispatch mesh shader workgroups
    // Each workgroup processes 32 faces, so we need (faceCount + 31) / 32 workgroups
    uint32_t faceCount = static_cast<uint32_t>(currentRenderData.faces.size());
    uint32_t workgroupCount = (faceCount + 31) / 32;
    
    vulkanContext->vkCmdDrawMeshTasksEXT(commandBuffer, workgroupCount, 1, 1);
}

void ChunkRenderer::addTestCube() {
    RenderData testData;
    
    // Create a single test chunk at origin
    testData.chunkCoords.push_back(glm::ivec4(0, 0, 0, 0));
    
    // Create test model data for a simple quad facing forward (Z+)
    ModelData quadModel = {};
    // Front face vertices (CCW from front view)
    quadModel.vertices[0] = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f); // Bottom-left
    quadModel.vertices[1] = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f); // Bottom-right
    quadModel.vertices[2] = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f); // Top-right
    quadModel.vertices[3] = glm::vec4(0.0f, 1.0f, 1.0f, 1.0f); // Top-left
    
    // UV coordinates
    quadModel.uvCoords[0] = glm::vec2(0.0f, 1.0f);
    quadModel.uvCoords[1] = glm::vec2(1.0f, 1.0f);
    quadModel.uvCoords[2] = glm::vec2(1.0f, 0.0f);
    quadModel.uvCoords[3] = glm::vec2(0.0f, 0.0f);
    
    // Face normal pointing forward
    quadModel.faceNormal = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
    
    testData.models.push_back(quadModel);
    
    // Create test light data with bright white light
    LightData lightData = {};
    // Pack light data: ao(31) + skyLight(31) + blockLight(0) + r(31) + g(31) + b(31)
    uint32_t ao = 31;
    uint32_t skyLight = 31;
    uint32_t blockLight = 0;
    uint32_t r = 31, g = 31, b = 31;
    
    uint32_t packedLight = ao | (skyLight << 5) | (blockLight << 10) | (r << 15) | (g << 20) | (b << 25);
    
    lightData.vertex0 = packedLight;
    lightData.vertex1 = packedLight;
    lightData.vertex2 = packedLight;
    lightData.vertex3 = packedLight;
    
    testData.lights.push_back(lightData);
    
    // Create test face data
    FaceData face = {};
    // Pack position: block at (8, 8, 8) in chunk
    face.positionAndFlags = (8) | (8 << 5) | (8 << 10) | (0 << 16); // lightIndex = 0
    face.blockAndQuad = (0) | (0 << 16); // quadIndex = 0, textureIndex = 0
    face.chunkIndex = 0;
    
    testData.faces.push_back(face);
    
    // Update render data
    updateRenderData(testData);
    
    Logger::info("ChunkRenderer", "Added test cube with 1 face");
}

void ChunkRenderer::createBuffers() {
    // Create empty buffers initially - they will be resized when data is updated
    faceBuffer = createStorageBuffer(sizeof(FaceData));
    modelBuffer = createStorageBuffer(sizeof(ModelData));
    lightBuffer = createStorageBuffer(sizeof(LightData));
    chunkCoordBuffer = createStorageBuffer(sizeof(glm::ivec4));
}

void ChunkRenderer::updateBuffers() {
    if (currentRenderData.faces.empty()) return;
    
    // Resize and update face buffer
    size_t faceSize = currentRenderData.faces.size() * sizeof(FaceData);
    destroyBuffer(faceBuffer);
    faceBuffer = createStorageBuffer(faceSize, currentRenderData.faces.data());
    
    // Resize and update model buffer
    size_t modelSize = currentRenderData.models.size() * sizeof(ModelData);
    destroyBuffer(modelBuffer);
    modelBuffer = createStorageBuffer(modelSize, currentRenderData.models.data());
    
    // Resize and update light buffer
    size_t lightSize = currentRenderData.lights.size() * sizeof(LightData);
    destroyBuffer(lightBuffer);
    lightBuffer = createStorageBuffer(lightSize, currentRenderData.lights.data());
    
    // Resize and update chunk coord buffer
    size_t chunkSize = currentRenderData.chunkCoords.size() * sizeof(glm::ivec4);
    destroyBuffer(chunkCoordBuffer);
    chunkCoordBuffer = createStorageBuffer(chunkSize, currentRenderData.chunkCoords.data());
    
    // Update descriptor set with new buffers
    descriptorManager->updateDescriptorSet(descriptorSet, uboBuffer.buffer,
                                         faceBuffer.buffer, modelBuffer.buffer,
                                         lightBuffer.buffer, chunkCoordBuffer.buffer);
}

void ChunkRenderer::createDescriptorSet() {
    descriptorSet = descriptorManager->allocateDescriptorSet();
    
    // Initial descriptor set update will happen in updateBuffers()
}

BufferInfo ChunkRenderer::createStorageBuffer(size_t size, const void* data) {
    BufferInfo bufferInfo;
    bufferInfo.size = size;

    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = bufferInfo.size;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(vulkanContext->getDevice(), &bufferCreateInfo, nullptr, &bufferInfo.buffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create storage buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(vulkanContext->getDevice(), bufferInfo.buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = vulkanContext->findMemoryType(memRequirements.memoryTypeBits, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(vulkanContext->getDevice(), &allocInfo, nullptr, &bufferInfo.memory) != VK_SUCCESS) {
        vkDestroyBuffer(vulkanContext->getDevice(), bufferInfo.buffer, nullptr);
        throw std::runtime_error("Failed to allocate storage buffer memory!");
    }

    vkBindBufferMemory(vulkanContext->getDevice(), bufferInfo.buffer, bufferInfo.memory, 0);

    if (data) {
        void* mappedMemory;
        vkMapMemory(vulkanContext->getDevice(), bufferInfo.memory, 0, bufferInfo.size, 0, &mappedMemory);
        std::memcpy(mappedMemory, data, size);
        vkUnmapMemory(vulkanContext->getDevice(), bufferInfo.memory);
    }

    return bufferInfo;
}

void ChunkRenderer::updateStorageBuffer(BufferInfo& buffer, const void* data, size_t size) {
    void* mappedMemory;
    vkMapMemory(vulkanContext->getDevice(), buffer.memory, 0, size, 0, &mappedMemory);
    std::memcpy(mappedMemory, data, size);
    vkUnmapMemory(vulkanContext->getDevice(), buffer.memory);
}

void ChunkRenderer::destroyBuffer(BufferInfo& buffer) {
    if (buffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(vulkanContext->getDevice(), buffer.buffer, nullptr);
        buffer.buffer = VK_NULL_HANDLE;
    }
    if (buffer.memory != VK_NULL_HANDLE) {
        vkFreeMemory(vulkanContext->getDevice(), buffer.memory, nullptr);
        buffer.memory = VK_NULL_HANDLE;
    }
}