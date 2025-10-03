#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

class Camera {
public:
    Camera(const glm::vec3& position = glm::vec3(24.0f, 24.0f, 24.0f),
           float yaw = 45.0f,
           float pitch = 35.0f,
           float movementSpeed = 10.0f,
           float mouseSensitivity = 1.0f);

    // Get view matrix
    glm::mat4 getViewMatrix() const;

    // Get front vector
    glm::vec3 getFront() const;

    // Get right vector
    glm::vec3 getRight() const;

    // Get up vector
    glm::vec3 getUp() const;

    // Get position
    glm::vec3 getPosition() const { return position; }

    // Process mouse movement
    void processMouseMovement(float xOffset, float yOffset);

    // Process keyboard movement
    void processKeyboardMovement(bool forward, bool backward, bool left, bool right,
                                  bool up, bool down, float deltaTime);

    // Setters
    void setMovementSpeed(float speed) { movementSpeed = speed; }
    void setMouseSensitivity(float sensitivity) { mouseSensitivity = sensitivity; }

private:
    glm::vec3 position;
    glm::quat rotation;
    glm::vec3 worldUp;

    float yaw;
    float pitch;
    float movementSpeed;
    float mouseSensitivity;

    void updateRotation();
    float wrapDegrees(float degrees);
};