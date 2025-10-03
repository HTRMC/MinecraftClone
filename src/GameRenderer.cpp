#include "GameRenderer.hpp"

#include <algorithm>

float GameRenderer::getFarPlaneDistance() {
    return std::max(viewDistanceBlocks * 4.0f,  static_cast<float>(gameOptions.getCloudRenderDistance().getValue() * 16));
}

void GameRenderer::renderWorld() {
    viewDistanceBlocks = static_cast<float>(gameOptions.getClampedViewDistance() * 16);
}
