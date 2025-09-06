#include "Camera.hpp"
#include "Logger.hpp"
#include <GLFW/glfw3.h>

Camera::Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch) 
    : position(position), worldUp(up), yaw(yaw), pitch(pitch) {
    updateCameraVectors();
}

void Camera::processKeyboard(int key, float deltaTime) {
    float velocity = movementSpeed * deltaTime;
    glm::vec3 oldPosition = position;
    
    // WASD movement on horizontal plane only
    glm::vec3 horizontalFront = glm::normalize(glm::vec3(front.x, 0.0f, front.z));
    glm::vec3 horizontalRight = glm::normalize(glm::cross(horizontalFront, worldUp));
    
    if (key == GLFW_KEY_W)
        position += horizontalFront * velocity;
    if (key == GLFW_KEY_S)
        position -= horizontalFront * velocity;
    if (key == GLFW_KEY_A)
        position -= horizontalRight * velocity;
    if (key == GLFW_KEY_D)
        position += horizontalRight * velocity;
    
    // Debug print when integer position changes
    glm::ivec3 intPos = glm::ivec3(floor(position.x), floor(position.y), floor(position.z));
    if (intPos != lastLoggedIntPos) {
        Logger::debug("Camera", "Player position: (" + std::to_string(intPos.x) + ", " + 
                                std::to_string(intPos.y) + ", " + std::to_string(intPos.z) + ")");
        lastLoggedIntPos = intPos;
    }
}

void Camera::processMouseMovement(float xOffset, float yOffset) {
    xOffset *= mouseSensitivity;
    yOffset *= mouseSensitivity;
    
    yaw += xOffset;
    pitch += yOffset;
    
    // Constrain pitch
    if (pitch > 89.0f)
        pitch = 89.0f;
    if (pitch < -89.0f)
        pitch = -89.0f;
    
    updateCameraVectors();
}

glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(position, position + front, up);
}

void Camera::updateCameraVectors() {
    glm::vec3 frontVec;
    frontVec.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    frontVec.y = sin(glm::radians(pitch));
    frontVec.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    front = glm::normalize(frontVec);
    
    right = glm::normalize(glm::cross(front, worldUp));
    up = glm::normalize(glm::cross(right, front));
}