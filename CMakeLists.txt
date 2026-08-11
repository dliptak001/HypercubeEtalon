cmake_minimum_required(VERSION 3.21)
project(HypercubeEtalon VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Single-config generators (Ninja/Make): empty CMAKE_BUILD_TYPE disables
# $<CONFIG:…> flags below. Multi-config (VS, Xcode) leaves this alone.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- HypercubeCNN (vendored core; see third_party/HypercubeCNN/VENDORED.md) ---
add_subdirectory(third_party/HypercubeCNN)

# Shared compile / link flags (host + vendored core)
add_library(etalon_compile_flags INTERFACE)
target_compile_options(etalon_compile_flags INTERFACE
    $<$<CONFIG:Release>:-O3 -march=x86-64-v2 -ffast-math>
    $<$<CONFIG:Debug>:-O0 -g>
    -Wall -Wextra -Wno-unknown-pragmas
)
if(MINGW)
    target_link_options(etalon_compile_flags INTERFACE
        -static-libgcc
        -static-libstdc++
    )
endif()

# Match host Release/Debug options on the vendored static library.
target_link_libraries(HypercubeCNNCore PRIVATE etalon_compile_flags)

add_executable(HypercubeEtalon
    main.cpp
    Exciter.cpp
)

target_include_directories(HypercubeEtalon
    PRIVATE
        ${PROJECT_SOURCE_DIR}
)

target_link_libraries(HypercubeEtalon
    PRIVATE
        HypercubeCNNCore
        etalon_compile_flags
)

if(MINGW)
    target_link_options(HypercubeEtalon PRIVATE
        # Debug MinGW + static libs can hit duplicate symbols on some hosts.
        $<$<CONFIG:Debug>:-Wl,--allow-multiple-definition>
    )
endif()
