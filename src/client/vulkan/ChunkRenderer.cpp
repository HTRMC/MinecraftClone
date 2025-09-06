#include "ChunkRenderer.hpp"
#include "Logger.hpp"
#include <cstring>
#include <glm/gtc/matrix_transform.hpp>

ChunkRenderer::ChunkRenderer(VulkanContext* vulkanContext, 
                             DescriptorManager* descriptorManager,
                             MeshShaderPipeline* pipeline,
                             TextureManager* textureManager)
    : vulkanContext(vulkanContext), descriptorManager(descriptorManager), pipeline(pipeline), textureManager(textureManager) {
}

ChunkRenderer::~ChunkRenderer() {
    cleanup();
}

void ChunkRenderer::init() {
    if (initialized) return;
    
    uboBuffer = descriptorManager->createUniformBuffer();
    createBuffers();
    createDescriptorSet();
    
    // Load default textures to prevent empty texture array
    loadDefaultTextures();
    
    // Initial descriptor set update with textures
    descriptorManager->updateDescriptorSet(descriptorSet, uboBuffer.buffer,
                                         faceBuffer.buffer, modelBuffer.buffer,
                                         lightBuffer.buffer, chunkCoordBuffer.buffer,
                                         textureManager);
    
    // Add test data for initial rendering
    addFullChunk();
    
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
    destroyBuffer(indirectDrawBuffer);
    destroyBuffer(indirectCountBuffer);
    
    initialized = false;
}

void ChunkRenderer::updateRenderData(const RenderData& newData) {
    std::lock_guard<std::mutex> lock(renderDataMutex);
    currentRenderData = newData;
    dataUpdated = true;
}

void ChunkRenderer::render(VkCommandBuffer commandBuffer, const UniformBufferObject& ubo, bool cameraChanged) {
    if (!initialized || currentRenderData.faces.empty()) return;
    
    // Only update UBO if camera changed
    if (cameraChanged) {
        descriptorManager->updateUniformBuffer(uboBuffer, ubo);
    }
    
    // Update buffers if data changed
    if (dataUpdated.exchange(false)) {
        std::lock_guard<std::mutex> lock(renderDataMutex);
        updateBuffers();
    }
    
    // Bind pipeline and descriptor set
    pipeline->bind(commandBuffer);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           pipeline->getPipelineLayout(), 0, 1, &descriptorSet, 0, nullptr);
    
    // Dispatch mesh shader workgroups using indirect count
    uint32_t faceCount = static_cast<uint32_t>(currentRenderData.faces.size());
    uint32_t workgroupCount = (faceCount + 31) / 32;
    
    Logger::debug("ChunkRenderer", "Rendering " + std::to_string(faceCount) + " faces with " + 
                  std::to_string(workgroupCount) + " workgroups using indirect count");
    
    vulkanContext->vkCmdDrawMeshTasksIndirectCountEXT(commandBuffer, 
                                                     indirectDrawBuffer.buffer, 0,  // indirect buffer and offset
                                                     indirectCountBuffer.buffer, 0,  // count buffer and offset
                                                     1,                              // max draw count
                                                     sizeof(VkDrawMeshTasksIndirectCommandEXT)); // stride
}

void ChunkRenderer::renderParallel(std::vector<VkCommandBuffer>& commandBuffers, const UniformBufferObject& ubo, bool cameraChanged) {
    if (!initialized || currentRenderData.faces.empty() || commandBuffers.empty()) return;
    
    // Only update UBO if camera changed
    if (cameraChanged) {
        descriptorManager->updateUniformBuffer(uboBuffer, ubo);
    }
    
    // Update buffers if data changed
    if (dataUpdated.exchange(false)) {
        std::lock_guard<std::mutex> lock(renderDataMutex);
        updateBuffers();
    }
    
    uint32_t faceCount = static_cast<uint32_t>(currentRenderData.faces.size());
    uint32_t facesPerBuffer = (faceCount + commandBuffers.size() - 1) / commandBuffers.size();
    
    for (size_t i = 0; i < commandBuffers.size(); ++i) {
        VkCommandBuffer cmdBuffer = commandBuffers[i];
        
        // Bind pipeline and descriptor set
        pipeline->bind(cmdBuffer);
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                               pipeline->getPipelineLayout(), 0, 1, &descriptorSet, 0, nullptr);
        
        // Calculate workgroup range for this command buffer
        uint32_t startFace = static_cast<uint32_t>(i * facesPerBuffer);
        uint32_t endFace = std::min(startFace + facesPerBuffer, faceCount);
        
        if (startFace < endFace) {
            uint32_t facesToRender = endFace - startFace;
            uint32_t workgroupCount = (facesToRender + 31) / 32;
            
            // Use indirect count draw for parallel rendering too
            vulkanContext->vkCmdDrawMeshTasksIndirectCountEXT(cmdBuffer, 
                                                             indirectDrawBuffer.buffer, 0,  // indirect buffer and offset
                                                             indirectCountBuffer.buffer, 0,  // count buffer and offset
                                                             1,                              // max draw count
                                                             sizeof(VkDrawMeshTasksIndirectCommandEXT)); // stride
            
            Logger::debug("ChunkRenderer", "Command buffer " + std::to_string(i) + " rendering " + 
                          std::to_string(facesToRender) + " faces with " + std::to_string(workgroupCount) + " workgroups using indirect count");
        }
    }
    
    Logger::debug("ChunkRenderer", "Parallel rendering with " + std::to_string(commandBuffers.size()) + 
                  " command buffers for " + std::to_string(faceCount) + " total faces");
}


void ChunkRenderer::createBuffers() {
    // Create empty buffers initially - they will be resized when data is updated
    faceBuffer = createStorageBuffer(sizeof(FaceData));
    modelBuffer = createStorageBuffer(sizeof(ModelData));
    lightBuffer = createStorageBuffer(sizeof(LightData));
    chunkCoordBuffer = createStorageBuffer(sizeof(glm::ivec4));
    
    // Create indirect draw and count buffers with INDIRECT_BUFFER usage
    indirectDrawBuffer = vulkanContext->createBuffer(sizeof(VkDrawMeshTasksIndirectCommandEXT), 
                                                    VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
                                                    VMA_MEMORY_USAGE_CPU_TO_GPU);
    indirectCountBuffer = vulkanContext->createBuffer(sizeof(uint32_t), 
                                                     VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
                                                     VMA_MEMORY_USAGE_CPU_TO_GPU);
}

void ChunkRenderer::updateBuffers() {
    if (currentRenderData.faces.empty()) return;
    
    // Only recreate buffers if size changed, otherwise just update data
    size_t faceSize = currentRenderData.faces.size() * sizeof(FaceData);
    if (faceBuffer.size != faceSize) {
        destroyBuffer(faceBuffer);
        faceBuffer = createStorageBuffer(faceSize, currentRenderData.faces.data());
    } else {
        updateStorageBuffer(faceBuffer, currentRenderData.faces.data(), faceSize);
    }
    
    size_t modelSize = currentRenderData.models.size() * sizeof(ModelData);
    if (modelBuffer.size != modelSize) {
        destroyBuffer(modelBuffer);
        modelBuffer = createStorageBuffer(modelSize, currentRenderData.models.data());
    } else {
        updateStorageBuffer(modelBuffer, currentRenderData.models.data(), modelSize);
    }
    
    size_t lightSize = currentRenderData.lights.size() * sizeof(LightData);
    if (lightBuffer.size != lightSize) {
        destroyBuffer(lightBuffer);
        lightBuffer = createStorageBuffer(lightSize, currentRenderData.lights.data());
    } else {
        updateStorageBuffer(lightBuffer, currentRenderData.lights.data(), lightSize);
    }
    
    size_t chunkSize = currentRenderData.chunkCoords.size() * sizeof(glm::ivec4);
    if (chunkCoordBuffer.size != chunkSize) {
        destroyBuffer(chunkCoordBuffer);
        chunkCoordBuffer = createStorageBuffer(chunkSize, currentRenderData.chunkCoords.data());
    } else {
        updateStorageBuffer(chunkCoordBuffer, currentRenderData.chunkCoords.data(), chunkSize);
    }
    
    // Update indirect draw command
    uint32_t faceCount = static_cast<uint32_t>(currentRenderData.faces.size());
    uint32_t workgroupCount = (faceCount + 31) / 32; // 32 faces per workgroup
    
    VkDrawMeshTasksIndirectCommandEXT indirectCommand = {};
    indirectCommand.groupCountX = workgroupCount;
    indirectCommand.groupCountY = 1;
    indirectCommand.groupCountZ = 1;
    
    updateStorageBuffer(indirectDrawBuffer, &indirectCommand, sizeof(VkDrawMeshTasksIndirectCommandEXT));
    
    // Update indirect count (number of draw commands, which is 1 in our case)
    uint32_t drawCount = (faceCount > 0) ? 1 : 0;
    updateStorageBuffer(indirectCountBuffer, &drawCount, sizeof(uint32_t));
    
    // Update descriptor set if any buffers were recreated (size changed)
    descriptorManager->updateDescriptorSet(descriptorSet, uboBuffer.buffer,
                                         faceBuffer.buffer, modelBuffer.buffer,
                                         lightBuffer.buffer, chunkCoordBuffer.buffer,
                                         textureManager);
}

void ChunkRenderer::createDescriptorSet() {
    descriptorSet = descriptorManager->allocateDescriptorSet();
    
    // Initial descriptor set update will happen in updateBuffers()
}

BufferInfo ChunkRenderer::createStorageBuffer(size_t size, const void* data) {
    BufferInfo bufferInfo = vulkanContext->createStorageBuffer(size);

    if (data) {
        void* mappedData = vulkanContext->mapBuffer(bufferInfo);
        std::memcpy(mappedData, data, size);
        vulkanContext->unmapBuffer(bufferInfo);
    }

    return bufferInfo;
}

void ChunkRenderer::updateStorageBuffer(BufferInfo& buffer, const void* data, size_t size) {
    void* mappedData = vulkanContext->mapBuffer(buffer);
    std::memcpy(mappedData, data, size);
    vulkanContext->unmapBuffer(buffer);
}

void ChunkRenderer::destroyBuffer(BufferInfo& buffer) {
    vulkanContext->destroyBuffer(buffer);
}


void ChunkRenderer::loadDefaultTextures() {
    try {
        // Load a basic stone texture as default (texture ID 0)
        textureManager->loadTexture("assets/minecraft/textures/block/stone.png");
        Logger::info("ChunkRenderer", "Loaded default textures");
    } catch (const std::exception& e) {
        Logger::warning("ChunkRenderer", "Failed to load default texture: " + std::string(e.what()));
    }
}

void ChunkRenderer::addFullChunk() {
    try {
        // Load the stone model from JSON with inheritance support
        BlockModel model = BlockModelLoader::loadModelWithInheritance("assets/minecraft/models/block/stone.json");
        
        // Generate mesh data from the model
        std::vector<ModelData> meshData = BlockModelLoader::generateMeshData(model);
        Logger::info("ChunkRenderer", "Generated " + std::to_string(meshData.size()) + " models from stone.json");
        
        RenderData chunkData;
        
        // Create a single test chunk at origin
        chunkData.chunkCoords.push_back(glm::ivec4(0, 0, 0, 0));
        
        // Use white lighting for proper texture display
        uint32_t faceColors[6][3] = {
            {31, 31, 31},   // West: White
            {31, 31, 31},   // East: White  
            {31, 31, 31},   // North: White
            {31, 31, 31},   // South: White
            {31, 31, 31},   // Down: White
            {31, 31, 31}    // Up: White
        };
        
        // Add the 6 face models once (each face has different orientation)
        for (size_t i = 0; i < meshData.size() && i < 6; ++i) {
            chunkData.models.push_back(meshData[i]);
        }
        
        // Ensure we have at least one model to work with
        if (meshData.empty()) {
            Logger::error("ChunkRenderer", "No mesh data available for stone block");
            return;
        }
        
        // Generate 16x16x16 blocks without culling
        for (int x = 0; x < 16; x++) {
            for (int y = 0; y < 16; y++) {
                for (int z = 0; z < 16; z++) {
                    // For each block, add all 6 faces (no culling) - always create 6 faces
                    for (int faceIndex = 0; faceIndex < 6; ++faceIndex) {
                        // Create lighting data for this face
                        uint32_t ao = 31;
                        uint32_t skyLight = 31;
                        uint32_t blockLight = 0;
                        uint32_t r = faceColors[faceIndex][0];
                        uint32_t g = faceColors[faceIndex][1]; 
                        uint32_t b = faceColors[faceIndex][2];
                        uint32_t packedLight = ao | (skyLight << 5) | (blockLight << 10) | (r << 15) | (g << 20) | (b << 25);
                        
                        LightData lightData = {};
                        lightData.vertex0 = packedLight;
                        lightData.vertex1 = packedLight;
                        lightData.vertex2 = packedLight;
                        lightData.vertex3 = packedLight;
                        chunkData.lights.push_back(lightData);
                        
                        // Create face data referencing the correct model and this lighting
                        FaceData face = {};
                        uint16_t lightIndex = static_cast<uint16_t>(chunkData.lights.size() - 1);
                        
                        // Each face references its corresponding model (with bounds checking)
                        face.setPosition(x, y, z, false, lightIndex);
                        uint16_t modelIndex = static_cast<uint16_t>(faceIndex % chunkData.models.size());
                        face.setBlockAndQuad(0, modelIndex); // texture = 0 (stone), model = modelIndex
                        
                        // Debug: Log model indices for the first block only
                        if (x == 0 && y == 0 && z == 0) {
                            Logger::debug("ChunkRenderer", "Block (0,0,0) face " + std::to_string(faceIndex) + 
                                          " using model " + std::to_string(modelIndex));
                        }
                        face.chunkIndex = 0;
                        chunkData.faces.push_back(face);
                    }
                }
            }
        }
        
        // Update render data
        updateRenderData(chunkData);
        
        Logger::info("ChunkRenderer", "Added full 16x16x16 chunk with " + std::to_string(chunkData.faces.size()) + " faces, " +
                     std::to_string(chunkData.models.size()) + " models, " + std::to_string(chunkData.lights.size()) + " lights");
                     
    } catch (const std::exception& e) {
        Logger::error("ChunkRenderer", "Failed to load full chunk: " + std::string(e.what()));
    }
}