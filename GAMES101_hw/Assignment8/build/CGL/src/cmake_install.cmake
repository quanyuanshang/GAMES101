# Install script for directory: C:/Users/ASUS/Desktop/cs/CG/GAMES101_hw/assignment8/CGL/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/RopeSim")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "C:/mingw64/bin/objdump.exe")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/ASUS/Desktop/cs/CG/GAMES101_hw/assignment8/build/CGL/src/libCGL.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("C:/Users/ASUS/Desktop/cs/CG/GAMES101_hw/assignment8/build/CGL/src/CMakeFiles/CGL.dir/install-cxx-module-bmi-Debug.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CGL" TYPE FILE FILES
    "C:/Users/ASUS/Desktop/cs/CG/GAMES101_hw/assignment8/CGL/src/CGL.h"
    "C:/Users/ASUS/Desktop/cs/CG/GAMES101_hw/assignment8/CGL/src/vector2D.h"
    "C:/Users/ASUS/Desktop/cs/CG/GAMES101_hw/assignment8/CGL/src/vector3D.h"
    "C:/Users/ASUS/Desktop/cs/CG/GAMES101_hw/assignment8/CGL/src/vector4D.h"
    "C:/Users/ASUS/Desktop/cs/CG/GAMES101_hw/assignment8/CGL/src/matrix3x3.h"
    "C:/Users/ASUS/Desktop/cs/CG/GAMES101_hw/assignment8/CGL/src/matrix4x4.h"
    "C:/Users/ASUS/Desktop/cs/CG/GAMES101_hw/assignment8/CGL/src/quaternion.h"
    "C:/Users/ASUS/Desktop/cs/CG/GAMES101_hw/assignment8/CGL/src/complex.h"
    "C:/Users/ASUS/Desktop/cs/CG/GAMES101_hw/assignment8/CGL/src/color.h"
    "C:/Users/ASUS/Desktop/cs/CG/GAMES101_hw/assignment8/CGL/src/osdtext.h"
    "C:/Users/ASUS/Desktop/cs/CG/GAMES101_hw/assignment8/CGL/src/viewer.h"
    "C:/Users/ASUS/Desktop/cs/CG/GAMES101_hw/assignment8/CGL/src/base64.h"
    "C:/Users/ASUS/Desktop/cs/CG/GAMES101_hw/assignment8/CGL/src/tinyxml2.h"
    "C:/Users/ASUS/Desktop/cs/CG/GAMES101_hw/assignment8/CGL/src/renderer.h"
    )
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "C:/Users/ASUS/Desktop/cs/CG/GAMES101_hw/assignment8/build/CGL/src/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
