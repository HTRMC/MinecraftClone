#include "Camera.hpp"
#include "Logger.hpp"
#include <cmath>

Camera::Camera(const glm::vec3& position, const glm::vec3& up, float yaw, float pitch)
    : pos(position)
    , worldUp(up)
    , yaw(yaw)
    , pitch(pitch)
    ,lastYaw(yaw)
    ,lastPitch(pitch)
    , front(glm::vec3(0.0f, 0.0f, -1.0f))
    , horizontalPlane(HORIZONTAL)
    , verticalPlane(VERTICAL)
    , diagonalPlane(DIAGONAL)
    , rotation(glm::quat())
{
    setRotation(yaw, pitch);
}

glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(pos, pos + front, up);
}

void Camera::processKeyboard(GLFWwindow* window, float deltaTime) {
    float velocity = movementSpeed * deltaTime;
    glm::vec3 oldPosition = pos;
    
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        glm::vec3 horizontalFront = glm::normalize(glm::vec3(front.x, 0.0f, front.z));
        pos += horizontalFront * velocity;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        glm::vec3 horizontalFront = glm::normalize(glm::vec3(front.x, 0.0f, front.z));
        pos -= horizontalFront * velocity;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        pos -= right * velocity;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        pos += right * velocity;
    }
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
        pos -= worldUp * velocity;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
        pos += worldUp * velocity;
    }
    
    // Mark view matrix as changed if position changed
    if (pos != oldPosition) {
        viewMatrixChanged = true;
    }
    
    static bool f1Pressed = false;
    if (glfwGetKey(window, GLFW_KEY_F1) == GLFW_PRESS) {
        if (!f1Pressed) {
            printFacingDirection();
            f1Pressed = true;
        }
    } else {
        f1Pressed = false;
    }
    
    // Debug print when integer position changes
    glm::ivec3 intPos = glm::ivec3(floor(pos.x), floor(pos.y), floor(pos.z));
    if (intPos != lastLoggedIntPos) {
        Logger::debug("Camera", "Player position: (" + std::to_string(intPos.x) + ", " + 
                                std::to_string(intPos.y) + ", " + std::to_string(intPos.z) + ")");
        lastLoggedIntPos = intPos;
    }
}

float Camera::wrapDegrees(float degrees) {
    float f = fmod(degrees, 360.0f);
    if (f >= 180.0f) f -= 360.0f;
    if (f < -180.0f) f += 360.0f;
    return f;
}

void Camera::processMouseMovement(double xOffset, double yOffset, bool constrainPitch) {
    double sensitivity = mouseSensitivity * 0.6 + 0.2;
    double multiplier = sensitivity * sensitivity * sensitivity * 8.0;
    
    xOffset *= multiplier * 0.15;
    yOffset *= multiplier * 0.15;

    float oldYaw = lastYaw;
    float oldPitch = lastPitch;

    lastYaw -= static_cast<float>(xOffset);
    lastPitch += static_cast<float>(yOffset);

    lastPitch = glm::clamp(lastPitch, -90.0f, 90.0f);
    lastYaw = wrapDegrees(lastYaw);

    yaw = lastYaw;
    pitch = lastPitch;

    // Mark view matrix as changed if rotation changed
    if (yaw != oldYaw || pitch != oldPitch) {
        viewMatrixChanged = true;
        setRotation(yaw, pitch);
    }
}

void Camera::setRotation(float yaw, float pitch) {
    this->pitch = pitch;
    this->yaw = yaw;
    
    float yawRad = glm::radians(yaw);
    float pitchRad = glm::radians(pitch);
    
    rotation = glm::angleAxis(yawRad, glm::vec3(0.0f, -1.0f, 0.0f)) *
               glm::angleAxis(pitchRad, glm::vec3(1.0f, 0.0f, 0.0f));
    
    horizontalPlane = rotation * HORIZONTAL;
    verticalPlane = rotation * VERTICAL;  
    diagonalPlane = rotation * DIAGONAL;
    
    front = glm::normalize(horizontalPlane);
    right = glm::normalize(diagonalPlane);
    up = glm::normalize(verticalPlane);
}

void Camera::updateCameraVectors() {
    setRotation(yaw, pitch);
}

void Camera::printFacingDirection() const {
    float x = front.x;
    float y = front.y;
    float z = front.z;
    
    float absX = fabs(x);
    float absY = fabs(y);
    float absZ = fabs(z);
    
    const char* axis;
    if (absX > absY && absX > absZ) {
        axis = x > 0 ? "X+" : "X-";
    } else if (absY > absX && absY > absZ) {
        axis = y > 0 ? "Y+" : "Y-";
    } else {
        axis = z > 0 ? "Z+" : "Z-";
    }
    
    printf("Facing: %s (front: %.2f, %.2f, %.2f)\n", axis, x, y, z);
}