#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera {
public:
    Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f),
           glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f),
           float yaw = -90.0f, float pitch = 0.0f);
    
    void processKeyboard(int key, float deltaTime);
    void processMouseMovement(float xOffset, float yOffset);
    
    glm::mat4 getViewMatrix() const;
    glm::vec3 getPosition() const { return position; }
    bool hasChanged() const { return viewMatrixChanged; }
    void resetChangeFlag() { viewMatrixChanged = false; }
    
    void setMovementSpeed(float speed) { movementSpeed = speed; }
    void setSensitivity(float sensitivity) { mouseSensitivity = sensitivity; }

private:
    void updateCameraVectors();

    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec3 worldUp;
    
    float yaw;
    float pitch;
    
    float movementSpeed = 5.0f;
    float mouseSensitivity = 0.1f;
    
    mutable bool viewMatrixChanged = true;
    glm::ivec3 lastLoggedIntPos = glm::ivec3(INT_MAX);
};