cmake_minimum_required(VERSION 3.30)
project(MinecraftClone)

set(CMAKE_CXX_STANDARD 23)

# Include FetchContent module
include(FetchContent)

# If Vulkan SDK not found, fetch Vulkan headers and loader
if(NOT Vulkan_FOUND)
    message(STATUS "Vulkan SDK not found. Fetching Vulkan headers and loader...")

    # Fetch Vulkan Headers
    FetchContent_Declare(
            VulkanHeaders
            GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
            GIT_TAG        v1.4.328
    )

    # Fetch Vulkan Loader
    FetchContent_Declare(
            VulkanLoader
            GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Loader.git
            GIT_TAG        v1.4.328
    )

    # Make Vulkan Headers available first
    FetchContent_MakeAvailable(VulkanHeaders)

    # Configure and build Vulkan Loader
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_STATIC_LOADER OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(VulkanLoader)

    # Set Vulkan variables manually
    set(Vulkan_INCLUDE_DIRS ${vulkanheaders_SOURCE_DIR}/include)
    set(Vulkan_LIBRARIES vulkan)
    set(Vulkan_FOUND TRUE)
endif()

# Fetch GLFW
FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG        3.4
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "Don't build GLFW docs")
set(GLFW_BUILD_TESTS OFF CACHE BOOL "Don't build GLFW tests")
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Don't build GLFW examples")

# Fetch GLM
FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG        1.0.1
)

# VMA
FetchContent_Declare(
        VMA
        GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
        GIT_TAG        v3.3.0
)

# fmt
FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG        12.0.0
)

# spdlog
FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG        v1.15.3
)

# Make available
FetchContent_MakeAvailable(glfw glm VMA fmt spdlog)

# Option to pre-compile shaders (can be disabled if shaders are pre-compiled)
option(COMPILE_SHADERS "Compile shaders from source" ON)

if(COMPILE_SHADERS)
    # Find shader compiler - first try system glslc
    find_program(GLSLC glslc HINTS $ENV{VULKAN_SDK}/Bin)

    if(NOT GLSLC)
        message(STATUS "glslc not found. Fetching glslang to build shader compiler...")

        # Fetch glslang for shader compilation
        FetchContent_Declare(
                glslang
                GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
                GIT_TAG        16.0.0
        )

        # Configure glslang options
        set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
        set(BUILD_EXTERNAL OFF CACHE BOOL "" FORCE)
        set(ENABLE_GLSLANG_BINARIES ON CACHE BOOL "" FORCE)
        set(ENABLE_HLSL OFF CACHE BOOL "" FORCE)
        set(ENABLE_CTEST OFF CACHE BOOL "" FORCE)
        set(ENABLE_OPT OFF CACHE BOOL "" FORCE)

        FetchContent_MakeAvailable(glslang)

        # Use glslangValidator instead of glslc
        set(USE_GLSLANG_VALIDATOR TRUE)
        set(GLSLANG_VALIDATOR $<TARGET_FILE:glslang-standalone>)
    endif()
else()
    message(STATUS "Shader compilation disabled. Using pre-compiled shaders.")
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

# Automatically collect all .cpp files under src/
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
        ${CMAKE_SOURCE_DIR}/src/*.cpp
)

add_executable(${CMAKE_PROJECT_NAME})
target_sources(${CMAKE_PROJECT_NAME} PRIVATE ${SOURCES})

set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR}
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${BIN_DIR}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${BIN_DIR}
)

# Set MSVC optimization flags and PDB handling
if(MSVC)
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE
            $<$<CONFIG:Release>:/O2>  # Maximum optimization for Release builds
            $<$<CONFIG:Debug>:/Od>    # Disable optimization for Debug builds
            $<$<CONFIG:RelWithDebInfo>:/Od>       # Maximum optimization for RelWithDebInfo builds
            /permissive-              # Disable permissive mode for stricter conformance
    )

    # Fix MSVC PDB issues
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
            COMPILE_PDB_NAME "${CMAKE_PROJECT_NAME}"
            COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pdb"
    )

    # Ensure PDB directory exists
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/pdb")

    # Set linker flags to handle PDB properly
    target_link_options(${CMAKE_PROJECT_NAME} PRIVATE
            $<$<CONFIG:Debug>:/DEBUG:FULL>
            $<$<CONFIG:Debug>:/PDB:${CMAKE_BINARY_DIR}/pdb/${CMAKE_PROJECT_NAME}.pdb>
    )
endif()

target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
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
    add_shader(${CMAKE_PROJECT_NAME} ${REL_SHADER_FILE})
endforeach()

# Automatically copy assets to the binary directory if assets exist
set(ASSETS_SOURCE_DIR ${CMAKE_SOURCE_DIR}/assets)
set(ASSETS_BINARY_DIR ${CMAKE_BINARY_DIR}/assets)

if(EXISTS ${ASSETS_SOURCE_DIR})
    add_custom_command(
            TARGET ${CMAKE_PROJECT_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${ASSETS_SOURCE_DIR}
            ${BIN_DIR}/assets
            COMMENT "Copying assets to binary directory"
    )
else()
    message(STATUS "Assets directory not found, skipping asset copying")
endif()

# Include directories
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${Vulkan_INCLUDE_DIRS}
        ${vma_SOURCE_DIR}/include
        ${glm_SOURCE_DIR}
)

# Platform-specific libraries
if(WIN32)
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE glfw ${Vulkan_LIBRARIES} fmt::fmt spdlog::spdlog)
elseif(UNIX AND NOT APPLE)
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE glfw ${Vulkan_LIBRARIES} fmt::fmt spdlog::spdlog)
elseif(APPLE)
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE glfw ${Vulkan_LIBRARIES} fmt::fmt spdlog::spdlog)
endif()
