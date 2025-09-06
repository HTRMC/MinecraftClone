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
    std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
    
    // Binding 0: UBO (matrices)
    bindings[0].binding = static_cast<uint32_t>(DescriptorBinding::UBO);
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_MESH_BIT_NV | VK_SHADER_STAGE_FRAGMENT_BIT;
    
    // Binding 1: FaceData storage buffer
    bindings[1].binding = static_cast<uint32_t>(DescriptorBinding::FACE_DATA);
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_MESH_BIT_NV;
    
    // Binding 2: ModelData storage buffer
    bindings[2].binding = static_cast<uint32_t>(DescriptorBinding::MODEL_DATA);
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_MESH_BIT_NV;
    
    // Binding 3: LightData storage buffer
    bindings[3].binding = static_cast<uint32_t>(DescriptorBinding::LIGHT_DATA);
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_MESH_BIT_NV;
    
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    
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
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    
    // Pool for uniform buffers (UBO)
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = MAX_DESCRIPTOR_SETS;
    
    // Pool for storage buffers (FaceData, ModelData, LightData)
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = MAX_DESCRIPTOR_SETS * 3; // 3 storage buffers per set
    
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
                                           VkBuffer lightDataBuffer) {
    std::array<VkWriteDescriptorSet, 4> descriptorWrites{};
    
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
    
    vkUpdateDescriptorSets(vulkanContext->getDevice(), static_cast<uint32_t>(descriptorWrites.size()),
                          descriptorWrites.data(), 0, nullptr);
}

BufferInfo DescriptorManager::createUniformBuffer() {
    BufferInfo bufferInfo;
    bufferInfo.size = sizeof(UniformBufferObject);

    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = bufferInfo.size;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(vulkanContext->getDevice(), &bufferCreateInfo, nullptr, &bufferInfo.buffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create uniform buffer!");
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
        throw std::runtime_error("Failed to allocate uniform buffer memory!");
    }

    vkBindBufferMemory(vulkanContext->getDevice(), bufferInfo.buffer, bufferInfo.memory, 0);

    if (vkMapMemory(vulkanContext->getDevice(), bufferInfo.memory, 0, bufferInfo.size, 0, &bufferInfo.mappedMemory) != VK_SUCCESS) {
        vkFreeMemory(vulkanContext->getDevice(), bufferInfo.memory, nullptr);
        vkDestroyBuffer(vulkanContext->getDevice(), bufferInfo.buffer, nullptr);
        throw std::runtime_error("Failed to map uniform buffer memory!");
    }

    return bufferInfo;
}

void DescriptorManager::updateUniformBuffer(BufferInfo& uboBuffer, const UniformBufferObject& ubo) {
    if (uboBuffer.mappedMemory) {
        std::memcpy(uboBuffer.mappedMemory, &ubo, sizeof(ubo));
    }
}