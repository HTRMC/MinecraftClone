#include "client/render/Camera.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <algorithm>
#include <cmath>

Camera::Camera(const glm::vec3& position, float yaw, float pitch,
               float movementSpeed, float mouseSensitivity)
    : position(position)
    , worldUp(0.0f, -1.0f, 0.0f)  // Vulkan Y- is up
    , yaw(yaw)
    , pitch(pitch)
    , movementSpeed(movementSpeed)
    , mouseSensitivity(mouseSensitivity)
    , rotation(1.0f, 0.0f, 0.0f, 0.0f)
{
    updateRotation();
}

glm::mat4 Camera::getViewMatrix() const {
    glm::vec3 front = getFront();
    glm::vec3 up = getUp();
    return glm::lookAt(position, position + front, up);
}

glm::vec3 Camera::getFront() const {
    return rotation * glm::vec3(0.0f, 0.0f, -1.0f);
}

glm::vec3 Camera::getRight() const {
    return rotation * glm::vec3(-1.0f, 0.0f, 0.0f);
}

glm::vec3 Camera::getUp() const {
    return rotation * glm::vec3(0.0f, 1.0f, 0.0f);
}

void Camera::processMouseMovement(float xOffset, float yOffset) {
    // Apply mouse sensitivity (Minecraft-style)
    double sensitivity = mouseSensitivity * 0.6 + 0.2;
    double multiplier = sensitivity * sensitivity * sensitivity * 8.0;

    xOffset *= static_cast<float>(multiplier * 0.15);
    yOffset *= static_cast<float>(multiplier * 0.15);

    yaw -= xOffset;
    pitch += yOffset;

    // Constrain pitch
    pitch = glm::clamp(pitch, -90.0f, 90.0f);

    // Wrap yaw to [-180, 180]
    yaw = wrapDegrees(yaw);

    updateRotation();
}

void Camera::processKeyboardMovement(bool forward, bool backward, bool left, bool right,
                                      bool up, bool down, float deltaTime) {
    float velocity = movementSpeed * deltaTime;

    glm::vec3 front = getFront();
    glm::vec3 rightVec = getRight();

    // WASD movement (horizontal plane only)
    if (forward) {
        glm::vec3 horizontalFront = glm::normalize(glm::vec3(front.x, 0.0f, front.z));
        position += horizontalFront * velocity;
    }
    if (backward) {
        glm::vec3 horizontalFront = glm::normalize(glm::vec3(front.x, 0.0f, front.z));
        position -= horizontalFront * velocity;
    }
    if (left) {
        position += rightVec * velocity;
    }
    if (right) {
        position -= rightVec * velocity;
    }

    // Space and Shift for vertical movement
    if (up) {
        position -= worldUp * velocity;  // worldUp is (0, -1, 0), so subtract to go up
    }
    if (down) {
        position += worldUp * velocity;  // Add to go down (since worldUp is negative Y)
    }
}

void Camera::updateRotation() {
    // Convert yaw and pitch to quaternion
    float yawRad = glm::radians(yaw);
    float pitchRad = glm::radians(pitch);

    rotation = glm::angleAxis(yawRad, glm::vec3(0.0f, 1.0f, 0.0f)) *
               glm::angleAxis(-pitchRad, glm::vec3(1.0f, 0.0f, 0.0f));
}

float Camera::wrapDegrees(float degrees) {
    float f = fmod(degrees, 360.0f);
    if (f >= 180.0f) f -= 360.0f;
    if (f < -180.0f) f += 360.0f;
    return f;
}