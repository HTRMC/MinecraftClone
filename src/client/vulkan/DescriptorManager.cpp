#include "DescriptorManager.hpp"
#include "Logger.hpp"
#include <array>

DescriptorManager::DescriptorManager(VulkanContext* vulkanContext) 
    : vulkanContext(vulkanContext) {
}

DescriptorManager::~DescriptorManager() {
    cleanup();
}

void DescriptorManager::init() {
    if (initialized) return;
    
    createDescriptorSetLayout();
    createPipelineLayout();
    createDescriptorPool();
    
    initialized = true;
    Logger::info("DescriptorManager", "Initialized descriptor set layout and pipeline layout");
}

void DescriptorManager::cleanup() {
    if (!initialized) return;
    
    VkDevice device = vulkanContext->getDevice();
    
    if (descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        descriptorPool = VK_NULL_HANDLE;
    }
    
    if (pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        pipelineLayout = VK_NULL_HANDLE;
    }
    
    if (descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        descriptorSetLayout = VK_NULL_HANDLE;
    }
    
    initialized = false;
}

void DescriptorManager::createDescriptorSetLayout() {
    std::array<VkDescriptorSetLayoutBinding, 7> bindings{};
    
    // Binding 0: UBO (matrices)
    bindings[0].binding = static_cast<uint32_t>(DescriptorBinding::UBO);
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT;
    
    // Binding 1: FaceData storage buffer
    bindings[1].binding = static_cast<uint32_t>(DescriptorBinding::FACE_DATA);
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT;
    
    // Binding 2: ModelData storage buffer
    bindings[2].binding = static_cast<uint32_t>(DescriptorBinding::MODEL_DATA);
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT;
    
    // Binding 3: LightData storage buffer
    bindings[3].binding = static_cast<uint32_t>(DescriptorBinding::LIGHT_DATA);
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT;
    
    // Binding 4: ChunkCoordBuffer storage buffer
    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT;
    
    // Binding 5: Texture array (bindless)
    bindings[5].binding = 5;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    bindings[5].descriptorCount = 4096; // Max textures
    bindings[5].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    
    // Binding 6: Texture sampler (must be last for variable count)
    bindings[6].binding = 6;
    bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    bindings[6].descriptorCount = 1;
    bindings[6].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    
    // Enable descriptor indexing for bindless textures
    VkDescriptorBindingFlags bindingFlags[7] = {
        0, // UBO
        0, // FaceData
        0, // ModelData  
        0, // LightData
        0, // ChunkCoordBuffer
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT, // Texture array (binding 5)
        VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT  // Sampler (binding 6, must be last for variable count)
    };
    
    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
    bindingFlagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    bindingFlagsInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    bindingFlagsInfo.pBindingFlags = bindingFlags;
    
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    layoutInfo.pNext = &bindingFlagsInfo;
    
    if (vkCreateDescriptorSetLayout(vulkanContext->getDevice(), &layoutInfo, nullptr, 
                                   &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout!");
    }
}

void DescriptorManager::createPipelineLayout() {
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;
    
    if (vkCreatePipelineLayout(vulkanContext->getDevice(), &pipelineLayoutInfo, nullptr, 
                              &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout!");
    }
}

void DescriptorManager::createDescriptorPool() {
    std::array<VkDescriptorPoolSize, 4> poolSizes{};
    
    // Pool for uniform buffers (UBO)
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = MAX_DESCRIPTOR_SETS;
    
    // Pool for storage buffers (FaceData, ModelData, LightData, ChunkCoordBuffer)
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = MAX_DESCRIPTOR_SETS * 4; // 4 storage buffers per set
    
    // Pool for sampled images (bindless texture array)
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    poolSizes[2].descriptorCount = MAX_DESCRIPTOR_SETS * 4096; // Max textures per set
    
    // Pool for samplers
    poolSizes[3].type = VK_DESCRIPTOR_TYPE_SAMPLER;
    poolSizes[3].descriptorCount = MAX_DESCRIPTOR_SETS;
    
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = MAX_DESCRIPTOR_SETS;
    
    if (vkCreateDescriptorPool(vulkanContext->getDevice(), &poolInfo, nullptr, 
                              &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool!");
    }
}

VkDescriptorSet DescriptorManager::allocateDescriptorSet() {
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;
    
    VkDescriptorSet descriptorSet;
    if (vkAllocateDescriptorSets(vulkanContext->getDevice(), &allocInfo, &descriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set!");
    }
    
    return descriptorSet;
}

void DescriptorManager::updateDescriptorSet(VkDescriptorSet descriptorSet,
                                           VkBuffer uboBuffer,
                                           VkBuffer faceDataBuffer,
                                           VkBuffer modelDataBuffer,
                                           VkBuffer lightDataBuffer,
                                           VkBuffer chunkCoordBuffer,
                                           TextureManager* textureManager) {
    uint32_t writeCount = 5; // Base descriptors
    std::vector<VkWriteDescriptorSet> descriptorWrites(7); // Max 7 descriptors
    
    // UBO descriptor
    VkDescriptorBufferInfo uboBufferInfo{};
    uboBufferInfo.buffer = uboBuffer;
    uboBufferInfo.offset = 0;
    uboBufferInfo.range = sizeof(UniformBufferObject);
    
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptorSet;
    descriptorWrites[0].dstBinding = static_cast<uint32_t>(DescriptorBinding::UBO);
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &uboBufferInfo;
    
    // FaceData storage buffer descriptor
    VkDescriptorBufferInfo faceDataBufferInfo{};
    faceDataBufferInfo.buffer = faceDataBuffer;
    faceDataBufferInfo.offset = 0;
    faceDataBufferInfo.range = VK_WHOLE_SIZE;
    
    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = descriptorSet;
    descriptorWrites[1].dstBinding = static_cast<uint32_t>(DescriptorBinding::FACE_DATA);
    descriptorWrites[1].dstArrayElement = 0;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pBufferInfo = &faceDataBufferInfo;
    
    // ModelData storage buffer descriptor
    VkDescriptorBufferInfo modelDataBufferInfo{};
    modelDataBufferInfo.buffer = modelDataBuffer;
    modelDataBufferInfo.offset = 0;
    modelDataBufferInfo.range = VK_WHOLE_SIZE;
    
    descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[2].dstSet = descriptorSet;
    descriptorWrites[2].dstBinding = static_cast<uint32_t>(DescriptorBinding::MODEL_DATA);
    descriptorWrites[2].dstArrayElement = 0;
    descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[2].descriptorCount = 1;
    descriptorWrites[2].pBufferInfo = &modelDataBufferInfo;
    
    // LightData storage buffer descriptor
    VkDescriptorBufferInfo lightDataBufferInfo{};
    lightDataBufferInfo.buffer = lightDataBuffer;
    lightDataBufferInfo.offset = 0;
    lightDataBufferInfo.range = VK_WHOLE_SIZE;
    
    descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[3].dstSet = descriptorSet;
    descriptorWrites[3].dstBinding = static_cast<uint32_t>(DescriptorBinding::LIGHT_DATA);
    descriptorWrites[3].dstArrayElement = 0;
    descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[3].descriptorCount = 1;
    descriptorWrites[3].pBufferInfo = &lightDataBufferInfo;
    
    // ChunkCoordBuffer storage buffer descriptor
    VkDescriptorBufferInfo chunkCoordBufferInfo{};
    chunkCoordBufferInfo.buffer = chunkCoordBuffer;
    chunkCoordBufferInfo.offset = 0;
    chunkCoordBufferInfo.range = VK_WHOLE_SIZE;
    
    descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[4].dstSet = descriptorSet;
    descriptorWrites[4].dstBinding = 4;
    descriptorWrites[4].dstArrayElement = 0;
    descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[4].descriptorCount = 1;
    descriptorWrites[4].pBufferInfo = &chunkCoordBufferInfo;
    
    // Texture array and sampler descriptors (if textureManager provided)
    std::vector<VkDescriptorImageInfo> imageInfos;
    VkDescriptorImageInfo samplerInfo{};
    
    if (textureManager && textureManager->getTextureCount() > 0) {
        // Texture array descriptor (binding 5 = SAMPLED_IMAGE)
        const auto& imageViews = textureManager->getTextureImageViews();
        imageInfos.resize(imageViews.size());
        
        for (size_t i = 0; i < imageViews.size(); ++i) {
            imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfos[i].imageView = imageViews[i];
            imageInfos[i].sampler = VK_NULL_HANDLE;
        }
        
        descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[5].dstSet = descriptorSet;
        descriptorWrites[5].dstBinding = 5;
        descriptorWrites[5].dstArrayElement = 0;
        descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        descriptorWrites[5].descriptorCount = static_cast<uint32_t>(imageInfos.size());
        descriptorWrites[5].pImageInfo = imageInfos.data();
        
        // Sampler descriptor (binding 6 = SAMPLER)
        samplerInfo.sampler = textureManager->getTextureSampler();
        samplerInfo.imageView = VK_NULL_HANDLE;
        samplerInfo.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        
        descriptorWrites[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[6].dstSet = descriptorSet;
        descriptorWrites[6].dstBinding = 6;
        descriptorWrites[6].dstArrayElement = 0;
        descriptorWrites[6].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        descriptorWrites[6].descriptorCount = 1;
        descriptorWrites[6].pImageInfo = &samplerInfo;
        
        writeCount = 7; // Include texture descriptors
    }
    
    vkUpdateDescriptorSets(vulkanContext->getDevice(), writeCount, descriptorWrites.data(), 0, nullptr);
}

BufferInfo DescriptorManager::createUniformBuffer() {
    return vulkanContext->createUniformBuffer(sizeof(UniformBufferObject));
}

void DescriptorManager::updateUniformBuffer(BufferInfo& uboBuffer, const UniformBufferObject& ubo) {
    void* mappedData = vulkanContext->mapBuffer(uboBuffer);
    std::memcpy(mappedData, &ubo, sizeof(ubo));
    vulkanContext->unmapBuffer(uboBuffer);
}