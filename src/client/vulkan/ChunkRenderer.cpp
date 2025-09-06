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
    addTestCubeFromJSON();
    
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
    
    // Dispatch mesh shader workgroups
    // Each workgroup processes 32 faces, so we need (faceCount + 31) / 32 workgroups
    uint32_t faceCount = static_cast<uint32_t>(currentRenderData.faces.size());
    uint32_t workgroupCount = (faceCount + 31) / 32;
    
    Logger::debug("ChunkRenderer", "Rendering " + std::to_string(faceCount) + " faces with " + 
                  std::to_string(workgroupCount) + " workgroups");
    
    vulkanContext->vkCmdDrawMeshTasksEXT(commandBuffer, workgroupCount, 1, 1);
}

void ChunkRenderer::addTestCube() {
    RenderData testData;
    
    // Create a single test chunk at origin
    testData.chunkCoords.push_back(glm::ivec4(0, 0, 0, 0));
    
    // Create cube faces using the proven approach from the working project
    // Cube bounds: min = (-0.5, -0.5, -0.5), max = (0.5, 0.5, 0.5) centered at origin
    constexpr glm::vec3 min(-0.5f, -0.5f, -0.5f);
    constexpr glm::vec3 max(0.5f, 0.5f, 0.5f);
    
    // Define 8 corners of the cube
    const glm::vec3 corner000(min.x, min.y, min.z);  // -X, -Y, -Z
    const glm::vec3 corner001(min.x, min.y, max.z);  // -X, -Y, +Z
    const glm::vec3 corner010(min.x, max.y, min.z);  // -X, +Y, -Z
    const glm::vec3 corner011(min.x, max.y, max.z);  // -X, +Y, +Z
    const glm::vec3 corner100(max.x, min.y, min.z);  // +X, -Y, -Z
    const glm::vec3 corner101(max.x, min.y, max.z);  // +X, -Y, +Z
    const glm::vec3 corner110(max.x, max.y, min.z);  // +X, +Y, -Z
    const glm::vec3 corner111(max.x, max.y, max.z);  // +X, +Y, +Z
    
    // Define faces in order: Left, Right, Front, Back, Bottom, Top
    // All faces use counter-clockwise winding when viewed from outside
    struct FaceInfo {
        glm::vec3 normal;
        glm::vec4 vertices[4];
        int textureSlot;
    };
    
    std::vector<FaceInfo> faces = {
        // Left face (-X) - Counter-clockwise when viewed from outside
        {
            glm::vec3(-1, 0, 0),
            { glm::vec4(corner000, 1.0f), glm::vec4(corner010, 1.0f), glm::vec4(corner011, 1.0f), glm::vec4(corner001, 1.0f) },
            0
        },
        // Right face (+X) - Counter-clockwise when viewed from outside  
        {
            glm::vec3(1, 0, 0),
            { glm::vec4(corner101, 1.0f), glm::vec4(corner111, 1.0f), glm::vec4(corner110, 1.0f), glm::vec4(corner100, 1.0f) },
            1
        },
        // Front face (-Y) - Counter-clockwise when viewed from outside
        {
            glm::vec3(0, -1, 0),
            { glm::vec4(corner100, 1.0f), glm::vec4(corner000, 1.0f), glm::vec4(corner001, 1.0f), glm::vec4(corner101, 1.0f) },
            2
        },
        // Back face (+Y) - Counter-clockwise when viewed from outside
        {
            glm::vec3(0, 1, 0),
            { glm::vec4(corner010, 1.0f), glm::vec4(corner110, 1.0f), glm::vec4(corner111, 1.0f), glm::vec4(corner011, 1.0f) },
            3
        },
        // Bottom face (-Z) - Counter-clockwise when viewed from outside
        {
            glm::vec3(0, 0, -1),
            { glm::vec4(corner000, 1.0f), glm::vec4(corner100, 1.0f), glm::vec4(corner110, 1.0f), glm::vec4(corner010, 1.0f) },
            4
        },
        // Top face (+Z) - Counter-clockwise when viewed from outside
        {
            glm::vec3(0, 0, 1),
            { glm::vec4(corner001, 1.0f), glm::vec4(corner011, 1.0f), glm::vec4(corner111, 1.0f), glm::vec4(corner101, 1.0f) },
            5
        }
    };
    
    // Define different colors for each face for debugging
    uint32_t faceColors[6][3] = {
        {31, 0, 0},   // Left: Red
        {0, 31, 0},   // Right: Green  
        {0, 0, 31},   // Front: Blue
        {31, 31, 0},  // Back: Yellow
        {31, 0, 31},  // Bottom: Magenta
        {0, 31, 31}   // Top: Cyan
    };
    
    // Create model data for each face (6 quads)
    for (size_t i = 0; i < faces.size(); ++i) {
        // Create model data
        ModelData model = {};
        for (int v = 0; v < 4; v++) {
            model.vertices[v] = faces[i].vertices[v];
        }
        // Standard UV coordinates
        model.uvCoords[0] = glm::vec2(0.0f, 0.0f);
        model.uvCoords[1] = glm::vec2(1.0f, 0.0f);
        model.uvCoords[2] = glm::vec2(1.0f, 1.0f);
        model.uvCoords[3] = glm::vec2(0.0f, 1.0f);
        model.faceNormal = glm::vec4(faces[i].normal, 0.0f);
        testData.models.push_back(model);
        
        // Create lighting data for this face with different colors
        uint32_t ao = 31;
        uint32_t skyLight = 31;
        uint32_t blockLight = 0;
        uint32_t r = faceColors[i][0];
        uint32_t g = faceColors[i][1]; 
        uint32_t b = faceColors[i][2];
        uint32_t packedLight = ao | (skyLight << 5) | (blockLight << 10) | (r << 15) | (g << 20) | (b << 25);
        
        LightData lightData = {};
        lightData.vertex0 = packedLight;
        lightData.vertex1 = packedLight;
        lightData.vertex2 = packedLight;
        lightData.vertex3 = packedLight;
        testData.lights.push_back(lightData);
        
        // Create face data referencing this model and lighting
        FaceData face = {};
        // Pack position: block at (0, 0, 0) in chunk, lightIndex = i
        face.positionAndFlags = (0) | (0 << 5) | (0 << 10) | (static_cast<uint32_t>(i) << 16);
        face.blockAndQuad = (static_cast<uint32_t>(faces[i].textureSlot)) | (static_cast<uint32_t>(i) << 16);
        face.chunkIndex = 0;
        testData.faces.push_back(face);
    }
    
    // Update render data
    updateRenderData(testData);
    
    Logger::info("ChunkRenderer", "Added test cube with " + std::to_string(testData.faces.size()) + " faces, " +
                 std::to_string(testData.models.size()) + " models, " + std::to_string(testData.lights.size()) + " lights");
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
    
    // Ensure GPU is idle before updating buffers to prevent device lost
    vkDeviceWaitIdle(vulkanContext->getDevice());
    
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

void ChunkRenderer::addTestCubeFromJSON() {
    try {
        // Load the cube model from JSON with inheritance support
        BlockModel model = BlockModelLoader::loadModelWithInheritance("assets/minecraft/models/block/stone.json");
        
        // Generate mesh data from the model
        std::vector<ModelData> meshData = BlockModelLoader::generateMeshData(model);
        
        RenderData testData;
        
        // Create a single test chunk at origin
        testData.chunkCoords.push_back(glm::ivec4(0, 0, 0, 0));
        
        // Define different colors for each face for debugging
        uint32_t faceColors[6][3] = {
            {31, 0, 0},   // West: Red
            {0, 31, 0},   // East: Green  
            {0, 0, 31},   // North: Blue
            {31, 31, 0},  // South: Yellow
            {31, 0, 31},  // Down: Magenta
            {0, 31, 31}   // Up: Cyan
        };
        
        // Process each face from the loaded model
        for (size_t i = 0; i < meshData.size() && i < 6; ++i) {
            testData.models.push_back(meshData[i]);
            
            // Create lighting data for this face with different colors
            uint32_t ao = 31;
            uint32_t skyLight = 31;
            uint32_t blockLight = 0;
            uint32_t r = faceColors[i][0];
            uint32_t g = faceColors[i][1]; 
            uint32_t b = faceColors[i][2];
            uint32_t packedLight = ao | (skyLight << 5) | (blockLight << 10) | (r << 15) | (g << 20) | (b << 25);
            
            LightData lightData = {};
            lightData.vertex0 = packedLight;
            lightData.vertex1 = packedLight;
            lightData.vertex2 = packedLight;
            lightData.vertex3 = packedLight;
            testData.lights.push_back(lightData);
            
            // Create face data referencing this model and lighting
            FaceData face = {};
            // Pack position: block at (0, 0, 0) in chunk, lightIndex = i
            face.positionAndFlags = (0) | (0 << 5) | (0 << 10) | (static_cast<uint32_t>(i) << 16);
            face.blockAndQuad = (static_cast<uint32_t>(i)) | (static_cast<uint32_t>(i) << 16);
            face.chunkIndex = 0;
            testData.faces.push_back(face);
        }
        
        // Update render data
        updateRenderData(testData);
        
        Logger::info("ChunkRenderer", "Added test cube from JSON with " + std::to_string(testData.faces.size()) + " faces, " +
                     std::to_string(testData.models.size()) + " models, " + std::to_string(testData.lights.size()) + " lights");
                     
    } catch (const std::exception& e) {
        Logger::error("ChunkRenderer", "Failed to load cube from JSON: " + std::string(e.what()));
    }
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