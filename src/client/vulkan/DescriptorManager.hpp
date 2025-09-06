#pragma once
#include "VulkanContext.hpp"
#include "MeshShaderData.hpp"

class DescriptorManager {
public:
    DescriptorManager(VulkanContext* vulkanContext);
    ~DescriptorManager();
    
    void init();
    void cleanup();
    
    VkDescriptorSetLayout getDescriptorSetLayout() const { return descriptorSetLayout; }
    VkPipelineLayout getPipelineLayout() const { return pipelineLayout; }
    
    VkDescriptorSet allocateDescriptorSet();
    void updateDescriptorSet(VkDescriptorSet descriptorSet, 
                           VkBuffer uboBuffer,
                           VkBuffer faceDataBuffer,
                           VkBuffer modelDataBuffer, 
                           VkBuffer lightDataBuffer,
                           VkBuffer chunkCoordBuffer);
    
    BufferInfo createUniformBuffer();
    void updateUniformBuffer(BufferInfo& uboBuffer, const UniformBufferObject& ubo);
    
private:
    void createDescriptorSetLayout();
    void createPipelineLayout();
    void createDescriptorPool();
    
private:
    VulkanContext* vulkanContext;
    
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    
    bool initialized = false;
    static constexpr uint32_t MAX_DESCRIPTOR_SETS = 1000;
};