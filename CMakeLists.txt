cmake_minimum_required(VERSION 3.20)
project(parcour_bot CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

option(PARCOUR_BUILD_VIEWER "Build the GLFW viewer" ON)

# --- MuJoCo (prebuilt release, fetched by scripts/setup_deps.sh) -------------------------
set(MUJOCO_VERSION 3.17.0)
set(MUJOCO_DIR ${CMAKE_SOURCE_DIR}/third_party/mujoco-${MUJOCO_VERSION})
if(NOT EXISTS ${MUJOCO_DIR}/lib/libmujoco.so)
  message(FATAL_ERROR "MuJoCo not found. Run scripts/setup_deps.sh first.")
endif()
add_library(mujoco SHARED IMPORTED)
set_target_properties(mujoco PROPERTIES
  IMPORTED_LOCATION ${MUJOCO_DIR}/lib/libmujoco.so
  INTERFACE_INCLUDE_DIRECTORIES ${MUJOCO_DIR}/include)
set(CMAKE_BUILD_RPATH ${MUJOCO_DIR}/lib)

# --- core library -----------------------------------------------------------------------
add_library(parcour_core
  src/core/humanoid.cpp
  src/core/injury.cpp
  src/core/body_state.cpp
  src/core/simulation.cpp
  src/core/course.cpp
  src/core/perception.cpp
  src/core/parkour_env.cpp)
target_include_directories(parcour_core PUBLIC src)
target_link_libraries(parcour_core PUBLIC mujoco)
target_compile_options(parcour_core PRIVATE -Wall -Wextra)
target_compile_definitions(parcour_core PUBLIC PARCOUR_ASSET_DIR="${CMAKE_SOURCE_DIR}/assets")

add_executable(parcour_selftest apps/selftest.cpp)
target_link_libraries(parcour_selftest PRIVATE parcour_core)

# --- viewer -----------------------------------------------------------------------------
if(PARCOUR_BUILD_VIEWER)
  include(FetchContent)
  set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)  # WSLg provides X11 via XWayland
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(glfw
    URL https://github.com/glfw/glfw/releases/download/3.4/glfw-3.4.zip
    URL_HASH SHA256=b5ec004b2712fd08e8861dc271428f048775200a2df719ccf575143ba749a3e9)
  FetchContent_MakeAvailable(glfw)

  add_executable(parcour_viewer apps/viewer.cpp)
  target_link_libraries(parcour_viewer PRIVATE parcour_core glfw)
endif()
