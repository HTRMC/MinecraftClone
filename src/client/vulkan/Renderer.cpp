#include "Renderer.hpp"
#include "client/util/Window.hpp"
#include "client/util/Camera.hpp"
#include "Logger.hpp"
#include <GLFW/glfw3.h>
#include <stdexcept>
#include <algorithm>
#include <array>

Renderer::Renderer(VulkanContext* vulkanContext, Window* window)
    : vulkanContext(vulkanContext), window(window) {
}

Renderer::~Renderer() {
    cleanup();
}

void Renderer::init() {
    if (initialized) return;
    
    createSurface();
    createSwapChain();
    createImageViews();
    createDepthResources();
    createDynamicRenderingInfo();
    createCommandBuffers();
    
    // Initialize rendering components
    shaderManager = std::make_unique<ShaderManager>(vulkanContext);
    shaderManager->init();
    
    descriptorManager = std::make_unique<DescriptorManager>(vulkanContext);
    descriptorManager->init();
    
    textureManager = std::make_unique<TextureManager>(vulkanContext);
    textureManager->init();
    
    meshPipeline = std::make_unique<MeshShaderPipeline>(vulkanContext, shaderManager.get(), descriptorManager.get());
    meshPipeline->initWithDynamicRendering(swapChainImageFormat, findDepthFormat());
    
    chunkRenderer = std::make_unique<ChunkRenderer>(vulkanContext, descriptorManager.get(), meshPipeline.get(), textureManager.get());
    chunkRenderer->init();
    
    // Create sync objects - one set per frame in flight for proper synchronization
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    
    // Create per-swapchain-image render finished semaphores
    perImageRenderFinishedSemaphores.resize(swapChainImages.size());
    
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    
    // Create semaphores per frame in flight (for imageAvailable)
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (vkCreateSemaphore(vulkanContext->getDevice(), &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(vulkanContext->getDevice(), &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create synchronization objects for frames!");
        }
    }
    
    // Create per-swapchain-image render finished semaphores
    for (size_t i = 0; i < swapChainImages.size(); i++) {
        if (vkCreateSemaphore(vulkanContext->getDevice(), &semaphoreInfo, nullptr, &perImageRenderFinishedSemaphores[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create per-image render finished semaphores!");
        }
    }
    
    // Create fences per frame in flight
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (vkCreateFence(vulkanContext->getDevice(), &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create synchronization objects for frames!");
        }
    }
    
    // Create timeline semaphore for advanced synchronization
    frameTimelineSemaphore = vulkanContext->createTimelineSemaphore(0);
    
    initialized = true;
    Logger::info("Renderer", "Initialized renderer with mesh shaders");
}

void Renderer::cleanup() {
    if (!initialized) return;
    
    vkDeviceWaitIdle(vulkanContext->getDevice());
    
    for (size_t i = 0; i < imageAvailableSemaphores.size(); i++) {
        vkDestroySemaphore(vulkanContext->getDevice(), renderFinishedSemaphores[i], nullptr);
        vkDestroySemaphore(vulkanContext->getDevice(), imageAvailableSemaphores[i], nullptr);
    }
    
    for (size_t i = 0; i < perImageRenderFinishedSemaphores.size(); i++) {
        vkDestroySemaphore(vulkanContext->getDevice(), perImageRenderFinishedSemaphores[i], nullptr);
    }
    
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroyFence(vulkanContext->getDevice(), inFlightFences[i], nullptr);
    }
    
    // Destroy timeline semaphore
    vulkanContext->destroyTimelineSemaphore(frameTimelineSemaphore);
    
    chunkRenderer.reset();
    meshPipeline.reset();
    textureManager.reset();
    descriptorManager.reset();
    shaderManager.reset();
    
    cleanupSwapChain();
    
    if (surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(vulkanContext->getInstance(), surface, nullptr);
    }
    
    initialized = false;
}

void Renderer::render() {
    if (!initialized) return;
    
    // Wait for fence first to ensure previous work is done (critical for semaphore safety)
    VkResult fenceWaitResult = vkWaitForFences(vulkanContext->getDevice(), 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
    if (fenceWaitResult != VK_SUCCESS) {
        if (fenceWaitResult == VK_ERROR_DEVICE_LOST) {
            Logger::error("Renderer", "Device lost during fence wait! Attempting recovery...");
            // Try to recreate the swapchain which may help recover
            recreateSwapChain();
            return;
        }
        Logger::error("Renderer", "Failed to wait for fence: " + std::to_string(fenceWaitResult));
        return;
    }
    
    // Reset fence only after successfully waiting for it
    VkResult fenceResetResult = vkResetFences(vulkanContext->getDevice(), 1, &inFlightFences[currentFrame]);
    if (fenceResetResult != VK_SUCCESS) {
        Logger::error("Renderer", "Failed to reset fence: " + std::to_string(fenceResetResult));
        return;
    }
    
    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(vulkanContext->getDevice(), swapChain, UINT64_MAX, 
                                           imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
    
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_ERROR_SURFACE_LOST_KHR) {
        recreateSwapChain();
        return;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        // Log the specific error code and try to recover
        Logger::warning("Renderer", "vkAcquireNextImageKHR returned error: " + std::to_string(result));
        recreateSwapChain();
        return;
    }
    
    // Check if we need to update anything (camera changed or data changed)
    bool cameraChanged = camera && camera->hasChanged();
    bool dataChanged = chunkRenderer && chunkRenderer->hasDataChanged();
    
    // Check if we need to re-record this frame's command buffer
    bool needsRecording = cameraChanged || dataChanged || 
                         !commandBuffersRecorded[currentFrame] || 
                         recordedForImageIndex[currentFrame] != imageIndex;
    
    if (needsRecording) {
        // Only reset if we're sure the previous command buffer has finished
        // The fence wait above ensures this
        VkResult resetResult = vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        if (resetResult != VK_SUCCESS) {
            Logger::error("Renderer", "Failed to reset command buffer: " + std::to_string(resetResult));
            return;
        }
        
        recordDynamicRenderingCommands(commandBuffers[currentFrame], imageIndex);
        
        commandBuffersRecorded[currentFrame] = true;
        recordedForImageIndex[currentFrame] = imageIndex;
    }
    
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    
    // Get next timeline value for this frame
    uint64_t currentTimelineValue = frameTimelineSemaphore.getNextSignalValue();
    
    // Set up semaphores (mix of binary and timeline)
    VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame], frameTimelineSemaphore.semaphore};
    uint64_t waitValues[] = {0, frameTimelineSemaphore.value}; // Binary semaphore value is 0
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};
    submitInfo.waitSemaphoreCount = 2;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    
    VkSemaphore signalSemaphores[] = {perImageRenderFinishedSemaphores[imageIndex], frameTimelineSemaphore.semaphore};
    uint64_t signalValues[] = {0, currentTimelineValue}; // Binary semaphore value is 0
    submitInfo.signalSemaphoreCount = 2;
    submitInfo.pSignalSemaphores = signalSemaphores;
    
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
    
    // Set up timeline semaphore submit info
    VkTimelineSemaphoreSubmitInfo timelineSubmitInfo{};
    timelineSubmitInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineSubmitInfo.waitSemaphoreValueCount = 2;
    timelineSubmitInfo.pWaitSemaphoreValues = waitValues;
    timelineSubmitInfo.signalSemaphoreValueCount = 2;
    timelineSubmitInfo.pSignalSemaphoreValues = signalValues;
    
    submitInfo.pNext = &timelineSubmitInfo;
    
    VkResult submitResult = vkQueueSubmit(vulkanContext->getGraphicsQueue(), 1, &submitInfo, inFlightFences[currentFrame]);
    
    // Update timeline semaphore value after successful submission
    if (submitResult == VK_SUCCESS) {
        frameTimelineSemaphore.value = currentTimelineValue;
    }
    if (submitResult != VK_SUCCESS) {
        if (submitResult == VK_ERROR_DEVICE_LOST) {
            Logger::error("Renderer", "Device lost during queue submit! Attempting recovery...");
            recreateSwapChain();
            return;
        }
        Logger::error("Renderer", "Failed to submit draw command buffer: " + std::to_string(submitResult));
        return;
    }
    
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    
    VkSwapchainKHR swapChains[] = {swapChain};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;
    
    result = vkQueuePresentKHR(vulkanContext->getPresentQueue(), &presentInfo);
    
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || result == VK_ERROR_SURFACE_LOST_KHR || framebufferResized) {
        framebufferResized = false;
        recreateSwapChain();
        // Reset all tracking after swapchain recreation
        std::fill(commandBuffersRecorded.begin(), commandBuffersRecorded.end(), false);
        std::fill(recordedForImageIndex.begin(), recordedForImageIndex.end(), UINT32_MAX);
        std::fill(lastSubmittedFrame.begin(), lastSubmittedFrame.end(), 0);
    } else if (result != VK_SUCCESS) {
        Logger::warning("Renderer", "vkQueuePresentKHR returned error: " + std::to_string(result));
        // Try to recover by recreating swapchain
        recreateSwapChain();
        std::fill(commandBuffersRecorded.begin(), commandBuffersRecorded.end(), false);
        std::fill(recordedForImageIndex.begin(), recordedForImageIndex.end(), UINT32_MAX);
        std::fill(lastSubmittedFrame.begin(), lastSubmittedFrame.end(), 0);
    }
    
    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    frameNumber++;
}

UniformBufferObject Renderer::createUBO() {
    UniformBufferObject ubo{};
    
    ubo.view = camera->getViewMatrix();
    ubo.proj = glm::perspective(glm::radians(45.0f), 
                               swapChainExtent.width / static_cast<float>(swapChainExtent.height),
                               0.1f, 1000.0f);
    
    return ubo;
}

void Renderer::createSurface() {
    if (glfwCreateWindowSurface(vulkanContext->getInstance(), window->getHandle(), nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create window surface!");
    }
}

void Renderer::createSwapChain() {
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(vulkanContext->getPhysicalDevice());
    
    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);
    
    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }
    
    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;
    
    if (vkCreateSwapchainKHR(vulkanContext->getDevice(), &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create swap chain!");
    }
    
    vkGetSwapchainImagesKHR(vulkanContext->getDevice(), swapChain, &imageCount, nullptr);
    swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(vulkanContext->getDevice(), swapChain, &imageCount, swapChainImages.data());
    
    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;
}

void Renderer::createImageViews() {
    swapChainImageViews.resize(swapChainImages.size());
    
    for (size_t i = 0; i < swapChainImages.size(); i++) {
        VkImageViewCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = swapChainImages[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = swapChainImageFormat;
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;
        
        if (vkCreateImageView(vulkanContext->getDevice(), &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create image views!");
        }
    }
}

void Renderer::createDepthResources() {
    VkFormat depthFormat = findDepthFormat();
    
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = swapChainExtent.width;
    imageInfo.extent.height = swapChainExtent.height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = depthFormat;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(vulkanContext->getDevice(), &imageInfo, nullptr, &depthImage) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create depth image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(vulkanContext->getDevice(), depthImage, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = vulkanContext->findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(vulkanContext->getDevice(), &allocInfo, nullptr, &depthImageMemory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate depth image memory!");
    }

    vkBindImageMemory(vulkanContext->getDevice(), depthImage, depthImageMemory, 0);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = depthImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = depthFormat;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(vulkanContext->getDevice(), &viewInfo, nullptr, &depthImageView) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create depth texture image view!");
    }
}

VkFormat Renderer::findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
    for (VkFormat format : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(vulkanContext->getPhysicalDevice(), format, &props);

        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
            return format;
        } else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }

    throw std::runtime_error("Failed to find supported format!");
}

VkFormat Renderer::findDepthFormat() {
    return findSupportedFormat(
        {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
    );
}

bool Renderer::hasStencilComponent(VkFormat format) {
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

void Renderer::createCommandBuffers() {
    commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    commandBuffersRecorded.resize(MAX_FRAMES_IN_FLIGHT, false);
    recordedForImageIndex.resize(MAX_FRAMES_IN_FLIGHT, UINT32_MAX);
    lastSubmittedFrame.resize(swapChainImages.size(), 0);
    
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = vulkanContext->getGraphicsCommandPool().pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());
    
    if (vkAllocateCommandBuffers(vulkanContext->getDevice(), &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffers!");
    }
}

void Renderer::createDynamicRenderingInfo() {
    dynamicRenderingInfo = {};
    dynamicRenderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    dynamicRenderingInfo.pNext = nullptr;
    dynamicRenderingInfo.flags = 0;
    dynamicRenderingInfo.renderArea = {{0, 0}, swapChainExtent};
    dynamicRenderingInfo.layerCount = 1;
    dynamicRenderingInfo.viewMask = 0;
    dynamicRenderingInfo.colorAttachmentCount = 1;
    
    Logger::info("Renderer", "Dynamic rendering info created");
}

void Renderer::recordDynamicRenderingCommands(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    
    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin recording command buffer!");
    }
    
    // Transition swapchain image to color attachment optimal layout
    VkImageMemoryBarrier imageBarrier{};
    imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imageBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.image = swapChainImages[imageIndex];
    imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBarrier.subresourceRange.baseMipLevel = 0;
    imageBarrier.subresourceRange.levelCount = 1;
    imageBarrier.subresourceRange.baseArrayLayer = 0;
    imageBarrier.subresourceRange.layerCount = 1;
    imageBarrier.srcAccessMask = 0;
    imageBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageBarrier);
    
    // Setup color attachment
    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = swapChainImageViews[imageIndex];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {{0.2f, 0.3f, 0.8f, 1.0f}};
    
    // Setup depth attachment
    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView = depthImageView;
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.clearValue.depthStencil = {1.0f, 0};
    
    // Begin dynamic rendering
    VkRenderingInfo renderingInfo = dynamicRenderingInfo;
    renderingInfo.renderArea = {{0, 0}, swapChainExtent};
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = &depthAttachment;
    
    vkCmdBeginRendering(commandBuffer, &renderingInfo);
    
    // Set dynamic viewport and scissor
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float) swapChainExtent.width;
    viewport.height = (float) swapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    
    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = swapChainExtent;
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
    
    // Check if camera has changed and create UBO
    bool cameraChanged = camera && camera->hasChanged();
    UniformBufferObject ubo = createUBO();
    
    // Reset camera change flag after creating UBO
    if (cameraChanged) {
        camera->resetChangeFlag();
    }
    
    // Render chunks using mesh shaders
    chunkRenderer->render(commandBuffer, ubo, cameraChanged);
    
    // End dynamic rendering
    vkCmdEndRendering(commandBuffer);
    
    // Transition swapchain image to present layout
    imageBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    imageBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    imageBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    imageBarrier.dstAccessMask = 0;
    
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 
                        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageBarrier);
    
    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to record command buffer!");
    }
}

SwapChainSupportDetails Renderer::querySwapChainSupport(VkPhysicalDevice device) {
    SwapChainSupportDetails details;
    
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
    
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
    
    if (formatCount != 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
    }
    
    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
    
    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
    }
    
    return details;
}

VkSurfaceFormatKHR Renderer::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
        }
    }
    return availableFormats[0];
}

VkPresentModeKHR Renderer::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return availablePresentMode;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D Renderer::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    } else {
        int width, height;
        glfwGetFramebufferSize(window->getHandle(), &width, &height);
        
        VkExtent2D actualExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
        
        actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
        
        return actualExtent;
    }
}

void Renderer::recreateSwapChain() {
    int width = 0, height = 0;
    glfwGetFramebufferSize(window->getHandle(), &width, &height);
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(window->getHandle(), &width, &height);
        glfwWaitEvents();
    }
    
    vkDeviceWaitIdle(vulkanContext->getDevice());
    
    // Clean up old per-image render finished semaphores
    for (size_t i = 0; i < perImageRenderFinishedSemaphores.size(); i++) {
        vkDestroySemaphore(vulkanContext->getDevice(), perImageRenderFinishedSemaphores[i], nullptr);
    }
    
    cleanupSwapChain();
    
    createSwapChain();
    createImageViews();
    createDepthResources();
    createDynamicRenderingInfo();
    
    // Recreate per-swapchain-image render finished semaphores with new size
    perImageRenderFinishedSemaphores.resize(swapChainImages.size());
    
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    
    for (size_t i = 0; i < swapChainImages.size(); i++) {
        if (vkCreateSemaphore(vulkanContext->getDevice(), &semaphoreInfo, nullptr, &perImageRenderFinishedSemaphores[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to recreate per-image render finished semaphores!");
        }
    }
    
    // Recreate mesh pipeline with dynamic rendering
    if (meshPipeline) {
        meshPipeline->cleanup();
        meshPipeline->initWithDynamicRendering(swapChainImageFormat, findDepthFormat());
    }
}

void Renderer::cleanupSwapChain() {
    for (auto imageView : swapChainImageViews) {
        vkDestroyImageView(vulkanContext->getDevice(), imageView, nullptr);
    }
    
    // Clean up depth resources
    if (depthImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(vulkanContext->getDevice(), depthImageView, nullptr);
        depthImageView = VK_NULL_HANDLE;
    }
    if (depthImage != VK_NULL_HANDLE) {
        vkDestroyImage(vulkanContext->getDevice(), depthImage, nullptr);
        depthImage = VK_NULL_HANDLE;
    }
    if (depthImageMemory != VK_NULL_HANDLE) {
        vkFreeMemory(vulkanContext->getDevice(), depthImageMemory, nullptr);
        depthImageMemory = VK_NULL_HANDLE;
    }
    
    if (swapChain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(vulkanContext->getDevice(), swapChain, nullptr);
    }
}