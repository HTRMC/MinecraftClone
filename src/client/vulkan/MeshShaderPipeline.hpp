#pragma once
#include "VulkanContext.hpp"
#include "ShaderManager.hpp"
#include "DescriptorManager.hpp"

class MeshShaderPipeline {
public:
    MeshShaderPipeline(VulkanContext* vulkanContext, 
                       ShaderManager* shaderManager,
                       DescriptorManager* descriptorManager);
    ~MeshShaderPipeline();
    
    void init(VkRenderPass renderPass);
    void initWithDynamicRendering(VkFormat colorFormat, VkFormat depthFormat);
    void cleanup();
    
    VkPipeline getPipeline() const { return pipeline; }
    VkPipelineLayout getPipelineLayout() const;
    
    void bind(VkCommandBuffer commandBuffer);

private:
    void createPipeline(VkRenderPass renderPass);
    void createPipelineWithDynamicRendering(VkFormat colorFormat, VkFormat depthFormat);
    
private:
    VulkanContext* vulkanContext;
    ShaderManager* shaderManager;
    DescriptorManager* descriptorManager;
    
    VkPipeline pipeline = VK_NULL_HANDLE;
    bool initialized = false;
};