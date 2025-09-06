#include "ShaderManager.hpp"
#include "Logger.hpp"
#include <fstream>
#include <stdexcept>

ShaderManager::ShaderManager(VulkanContext* vulkanContext) 
    : vulkanContext(vulkanContext) {
}

ShaderManager::~ShaderManager() {
    cleanup();
}

void ShaderManager::init() {
    if (initialized) return;
    
    meshShaderModule = loadShaderModule("assets/minecraft/shaders/core/chunk.msh.spv");
    fragmentShaderModule = loadShaderModule("assets/minecraft/shaders/core/chunk.fsh.spv");
    
    initialized = true;
    Logger::info("ShaderManager", "Loaded mesh and fragment shader modules");
}

void ShaderManager::cleanup() {
    if (!initialized) return;
    
    for (auto& [path, module] : shaderModules) {
        destroyShaderModule(module);
    }
    shaderModules.clear();
    
    // Reset handles to avoid double destruction
    meshShaderModule = VK_NULL_HANDLE;
    fragmentShaderModule = VK_NULL_HANDLE;
    
    initialized = false;
}

VkShaderModule ShaderManager::loadShaderModule(const std::string& filePath) {
    auto it = shaderModules.find(filePath);
    if (it != shaderModules.end()) {
        return it->second;
    }
    
    std::vector<char> shaderCode = readFile(filePath);
    VkShaderModule shaderModule = createShaderModule(shaderCode);
    
    shaderModules[filePath] = shaderModule;
    return shaderModule;
}

void ShaderManager::destroyShaderModule(VkShaderModule shaderModule) {
    if (shaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(vulkanContext->getDevice(), shaderModule, nullptr);
    }
}

std::vector<char> ShaderManager::readFile(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::ate | std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + filePath);
    }
    
    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);
    
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    
    return buffer;
}

VkShaderModule ShaderManager::createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
    
    VkShaderModule shaderModule;
    if (vkCreateShaderModule(vulkanContext->getDevice(), &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module!");
    }
    
    return shaderModule;
}