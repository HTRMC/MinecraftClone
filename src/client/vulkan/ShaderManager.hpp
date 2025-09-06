#pragma once
#include "VulkanContext.hpp"
#include <string>
#include <vector>
#include <unordered_map>

class ShaderManager {
public:
    ShaderManager(VulkanContext* vulkanContext);
    ~ShaderManager();
    
    void init();
    void cleanup();
    
    VkShaderModule loadShaderModule(const std::string& filePath);
    void destroyShaderModule(VkShaderModule shaderModule);
    
    VkShaderModule getMeshShaderModule() const { return meshShaderModule; }
    VkShaderModule getFragmentShaderModule() const { return fragmentShaderModule; }
    
private:
    std::vector<char> readFile(const std::string& filePath);
    VkShaderModule createShaderModule(const std::vector<char>& code);
    
private:
    VulkanContext* vulkanContext;
    
    VkShaderModule meshShaderModule = VK_NULL_HANDLE;
    VkShaderModule fragmentShaderModule = VK_NULL_HANDLE;
    
    std::unordered_map<std::string, VkShaderModule> shaderModules;
    bool initialized = false;
};