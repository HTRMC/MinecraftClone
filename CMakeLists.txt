cmake_minimum_required(VERSION 4.0)
project(MinecraftClone)

set(CMAKE_CXX_STANDARD 23)

include(FetchContent)

find_package(Vulkan REQUIRED)

# GLFW
FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG        3.4
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "Don't build GLFW docs")
set(GLFW_BUILD_TESTS OFF CACHE BOOL "Don't build GLFW tests")
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Don't build GLFW examples")
FetchContent_MakeAvailable(glfw)

# glm
FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG        1.0.1
)
FetchContent_MakeAvailable(glm)

# simdjson
FetchContent_Declare(
        simdjson
        GIT_REPOSITORY https://github.com/simdjson/simdjson.git
        GIT_TAG        v3.13.0
)
FetchContent_MakeAvailable(simdjson)


# SPNG
FetchContent_Declare(
        spng
        GIT_REPOSITORY https://github.com/randy408/libspng.git
        GIT_TAG        v0.7.4
)
set(SPNG_USE_SYSTEM_ZLIB OFF CACHE BOOL "" FORCE)
set(SPNG_STATIC ON CACHE BOOL "" FORCE)
set(SPNG_SHARED OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(spng)

# VMA
FetchContent_Declare(
        VMA
        GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
        GIT_TAG        v3.3.0
)
FetchContent_MakeAvailable(VMA)

find_program(GLSLC glslc HINTS $ENV{VULKAN_SDK}/Bin)
if(NOT GLSLC)
    message(FATAL_ERROR "glslc not found. Please install Vulkan SDK.")
endif()

set(BIN_DIR ${CMAKE_BINARY_DIR}/bin)

# Shader compilation function
function(add_shader TARGET SHADER)
    set(SHADER_SOURCE ${CMAKE_SOURCE_DIR}/assets/minecraft/shaders/${SHADER})
    set(SHADER_OUTPUT ${BIN_DIR}/assets/minecraft/shaders/${SHADER}.spv)

    # Create output directory
    get_filename_component(SHADER_OUTPUT_DIR ${SHADER_OUTPUT} DIRECTORY)

    # Determine shader stage from file extension
    get_filename_component(SHADER_EXT ${SHADER} EXT)
    if(SHADER_EXT STREQUAL ".msh")
        set(SHADER_STAGE mesh)
    elseif(SHADER_EXT STREQUAL ".tsh")
        set(SHADER_STAGE task)
    elseif(SHADER_EXT STREQUAL ".fsh")
        set(SHADER_STAGE frag)
    elseif(SHADER_EXT STREQUAL ".vsh")
        set(SHADER_STAGE vert)
    elseif(SHADER_EXT STREQUAL ".csh")
        set(SHADER_STAGE comp)
    else()
        message(FATAL_ERROR "Unknown shader extension: ${SHADER_EXT}")
    endif()

    add_custom_command(
            OUTPUT ${SHADER_OUTPUT}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_DIR}
            COMMAND ${GLSLC} -fshader-stage=${SHADER_STAGE} --target-spv=spv1.6 -o ${SHADER_OUTPUT} ${SHADER_SOURCE}
            DEPENDS ${SHADER_SOURCE}
            COMMENT "Compiling shader ${SHADER}"
    )

    # Sanitize the shader path to a valid target name (replace slashes with underscores)
    string(REPLACE "/" "_" SHADER_SAFE_NAME ${SHADER})
    add_custom_target(${SHADER_SAFE_NAME}_target DEPENDS ${SHADER_OUTPUT})
    add_dependencies(${TARGET} ${SHADER_SAFE_NAME}_target)
endfunction()

# Grab all source files recursively
file(GLOB_RECURSE ALL_SOURCES CONFIGURE_DEPENDS
        ${CMAKE_SOURCE_DIR}/src/*.cpp
)

# Filter out client and server
set(COMMON_SOURCES "")
foreach(SRC ${ALL_SOURCES})
    if(NOT SRC MATCHES "/client/" AND NOT SRC MATCHES "/server/")
        list(APPEND COMMON_SOURCES ${SRC})
    endif()
endforeach()

# Client-only sources
file(GLOB_RECURSE CLIENT_SOURCES CONFIGURE_DEPENDS
        ${CMAKE_SOURCE_DIR}/src/client/*.cpp
)

# Server-only sources
file(GLOB_RECURSE SERVER_SOURCES CONFIGURE_DEPENDS
        ${CMAKE_SOURCE_DIR}/src/server/*.cpp
)

# Client executable
add_executable(${CMAKE_PROJECT_NAME}Client
        ${COMMON_SOURCES}
        ${CLIENT_SOURCES}
)

# Server executable
add_executable(${CMAKE_PROJECT_NAME}Server
        ${COMMON_SOURCES}
        ${SERVER_SOURCES}
)

set_target_properties(${CMAKE_PROJECT_NAME}Client PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR}
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${BIN_DIR}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${BIN_DIR}
)

set_target_properties(${CMAKE_PROJECT_NAME}Server PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR}
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${BIN_DIR}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${BIN_DIR}
)

# Set MSVC optimization flags
if(MSVC)
    target_compile_options(${CMAKE_PROJECT_NAME}Client PRIVATE
            $<$<CONFIG:Release>:/O2>                    # Maximum optimization for Release builds
            $<$<CONFIG:Debug>:/Od>                      # Disable optimization for Debug builds
            $<$<CONFIG:RelWithDebInfo>:/Od>             # Maximum optimization for RelWithDebInfo builds
            /permissive-                                # Disable permissive mode for stricter conformance
    )
    target_compile_options(${CMAKE_PROJECT_NAME}Server PRIVATE
            $<$<CONFIG:Release>:/O2>                    # Maximum optimization for Release builds
            $<$<CONFIG:Debug>:/Od>                      # Disable optimization for Debug builds
            $<$<CONFIG:RelWithDebInfo>:/Od>             # Maximum optimization for RelWithDebInfo builds
            /permissive-                                # Disable permissive mode for stricter conformance
    )
endif()

# Compile definitions
target_compile_definitions(${CMAKE_PROJECT_NAME}Client PRIVATE
        CLIENT_BUILD
        VULKAN_HPP_ENABLE_STRINGIZE
        GLFW_INCLUDE_VULKAN
        GLM_ENABLE_EXPERIMENTAL
)

target_compile_definitions(${CMAKE_PROJECT_NAME}Server PRIVATE
        SERVER_BUILD
        VULKAN_HPP_ENABLE_STRINGIZE
        GLFW_INCLUDE_VULKAN
        GLM_ENABLE_EXPERIMENTAL
)

# Collect all shader files recursively from shaders/ with the supported extensions
file(GLOB_RECURSE SHADER_FILES CONFIGURE_DEPENDS
        ${CMAKE_SOURCE_DIR}/assets/minecraft/shaders/*.msh
        ${CMAKE_SOURCE_DIR}/assets/minecraft/shaders/*.tsh
        ${CMAKE_SOURCE_DIR}/assets/minecraft/shaders/*.fsh
        ${CMAKE_SOURCE_DIR}/assets/minecraft/shaders/*.vsh
        ${CMAKE_SOURCE_DIR}/assets/minecraft/shaders/*.csh
)

# Add each shader via the add_shader function
foreach(SHADER_PATH ${SHADER_FILES})
    # Compute the relative path from shaders/ folder
    file(RELATIVE_PATH REL_SHADER_FILE ${CMAKE_SOURCE_DIR}/assets/minecraft/shaders ${SHADER_PATH})
    add_shader(${CMAKE_PROJECT_NAME}Client ${REL_SHADER_FILE})
endforeach()

# Automatically copy assets to the binary directory
set(ASSETS_SOURCE_DIR ${CMAKE_SOURCE_DIR}/assets)
set(ASSETS_BINARY_DIR ${CMAKE_BINARY_DIR}/assets)

add_custom_command(
        TARGET ${CMAKE_PROJECT_NAME}Client
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${ASSETS_SOURCE_DIR}
        ${BIN_DIR}/assets
        COMMENT "Copying assets to binary directory"
)

# Include directories
target_include_directories(${CMAKE_PROJECT_NAME}Client PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${Vulkan_INCLUDE_DIRS}
        ${vma_SOURCE_DIR}/include
)

target_include_directories(${CMAKE_PROJECT_NAME}Server PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

# Platform-specific libraries
if(WIN32)
    target_link_libraries(${CMAKE_PROJECT_NAME}Client PRIVATE glfw ${Vulkan_LIBRARIES} spng_static simdjson dwmapi)
    target_link_libraries(${CMAKE_PROJECT_NAME}Server PRIVATE glfw ${Vulkan_LIBRARIES} spng_static simdjson dwmapi)
elseif(UNIX AND NOT APPLE)
    target_link_libraries(${CMAKE_PROJECT_NAME}Client PRIVATE glfw ${Vulkan_LIBRARIES} spng_static simdjson)
    target_link_libraries(${CMAKE_PROJECT_NAME}Server PRIVATE glfw ${Vulkan_LIBRARIES} spng_static simdjson)
elseif(APPLE)
    target_link_libraries(${CMAKE_PROJECT_NAME}Client PRIVATE glfw ${Vulkan_LIBRARIES} spng_static simdjson)
    target_link_libraries(${CMAKE_PROJECT_NAME}Server PRIVATE glfw ${Vulkan_LIBRARIES} spng_static simdjson)
endif()

