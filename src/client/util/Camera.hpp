#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <GLFW/glfw3.h>

class Camera {
public:
    Camera(const glm::vec3& position = glm::vec3(0.0f, 0.0f, 3.0f), 
           const glm::vec3& up = glm::vec3(0.0f, -1.0f, 0.0f), 
           float yaw = -90.0f, 
           float pitch = 0.0f);
    
    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;
    Camera(Camera&&) = delete;
    Camera& operator=(Camera&&) = delete;
    
    glm::mat4 getViewMatrix() const;
    
    void processKeyboard(GLFWwindow* window, float deltaTime);
    void processMouseMovement(double xOffset, double yOffset, bool constrainPitch = true);
    const glm::vec3& getPosition() const { return pos; }
    const glm::vec3& getFront() const { return front; }
    bool hasChanged() const { return viewMatrixChanged; }
    void resetChangeFlag() { viewMatrixChanged = false; }
    
    void printFacingDirection() const;

private:
    void updateCameraVectors();
    float wrapDegrees(float degrees);
    void setRotation(float yaw, float pitch);

    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec3 worldUp;

    float lastYaw;
    float lastPitch;
    
    float movementSpeed = 10.0f;
    float mouseSensitivity = 0.69f;

    static constexpr float BASE_CAMERA_DISTANCE = 4.0F;

    static constexpr glm::vec3 HORIZONTAL = glm::vec3(0.0f, 0.0f, -1.0f);
    static constexpr glm::vec3 VERTICAL = glm::vec3(0.0f, -1.0f, 0.0f);
    static constexpr glm::vec3 DIAGONAL = glm::vec3(-1.0f, 0.0f, 0.0f);

    glm::vec3 pos = glm::vec3(0.0f, 0.0f, 0.0f);

    glm::vec3 horizontalPlane = HORIZONTAL;
    glm::vec3 verticalPlane = VERTICAL;
    glm::vec3 diagonalPlane = DIAGONAL;

    float pitch;
    float yaw;

    glm::quat rotation;
    
    mutable bool viewMatrixChanged = true;
    glm::ivec3 lastLoggedIntPos = glm::ivec3(INT_MAX);
};