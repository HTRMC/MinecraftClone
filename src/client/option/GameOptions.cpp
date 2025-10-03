#include "GameOptions.hpp"

#include <algorithm>

GameOptions::GameOptions()
    : viewDistance("viewDistance", 12)
    , serverViewDistance(0)
    , cloudRenderDistance("cloudRenderDistance", 128) {
}

int GameOptions::getClampedViewDistance() {
    return serverViewDistance > 0 ? std::min(viewDistance.getValue(), serverViewDistance) : viewDistance.getValue();
}

SimpleOption<int>& GameOptions::getCloudRenderDistance() {
    return cloudRenderDistance;
}
