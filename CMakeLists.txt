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

# Shared compile / link flags (host core + demos)
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

# --- Core: Exciter + Readout + Etalon ---
add_library(HypercubeEtalonCore STATIC
    Exciter.cpp
    Readout.cpp
    Etalon.cpp
)

target_include_directories(HypercubeEtalonCore
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)

target_link_libraries(HypercubeEtalonCore
    PUBLIC  HypercubeCNNCore
    PRIVATE etalon_compile_flags
)

# --- Smoke executable ---
add_executable(HypercubeEtalon
    main.cpp
)

target_link_libraries(HypercubeEtalon
    PRIVATE
        HypercubeEtalonCore
        etalon_compile_flags
)

# ——— Demos (only when this repo is the top-level project) ———
# Layout: examples/<demo>/…  shared headers: examples/common/
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    function(etalon_add_exe name)
        cmake_parse_arguments(ET "" "" "SOURCES;INCLUDES" ${ARGN})
        add_executable(${name} ${ET_SOURCES})
        target_link_libraries(${name} PRIVATE HypercubeEtalonCore etalon_compile_flags)
        target_include_directories(${name} PRIVATE
            ${PROJECT_SOURCE_DIR}/examples/common
            ${ET_INCLUDES}
        )
    endfunction()

    etalon_add_exe(etalon_synth SOURCES examples/synth/etalon_synth.cpp)
    etalon_add_exe(etalon_mnist
        SOURCES examples/mnist/etalon_mnist.cpp
        INCLUDES ${PROJECT_SOURCE_DIR}/examples/mnist)

    etalon_add_exe(etalon_raman
        SOURCES
            examples/RamanBaselineExtraction/etalon_raman.cpp
            examples/RamanBaselineExtraction/BaselineExtractor.cpp
            examples/RamanBaselineExtraction/RamanDataset.cpp
        INCLUDES ${PROJECT_SOURCE_DIR}/examples/RamanBaselineExtraction)
endif()
