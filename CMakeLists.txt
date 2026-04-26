cmake_minimum_required(VERSION 3.25)
project(AVM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/W4 /permissive- /wd4459)
else()
  add_compile_options(-Wall -Wextra -Wpedantic -Wno-ignored-qualifiers)
endif()

option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
include(FetchContent)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 5.0.0
  GIT_SHALLOW ON
)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)

if(ENABLE_ASAN)
  if(MSVC)
    add_compile_options(/fsanitize=address)
    add_link_options(/fsanitize=address)
  else()
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address)
  endif()
endif()

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/gtest)
#include_directories(${eigen_SOURCE_DIR})

add_executable(unit_tests
  test/unit_tests.cpp
  test/test.cpp
)

add_executable(dsl_tests
  test/dsl_tests.cpp
  test/dsl_const_tests.cpp
  test/test.cpp
)

add_executable(eigen_test
  test/eigen_test.cpp
  test/test.cpp
)

add_executable(extended_tests
  test/extended_tests.cpp
  test/test.cpp
)

add_executable(scalar_tests
  test/scalar_tests.cpp
  test/test.cpp
)
target_link_libraries(eigen_test PRIVATE Eigen3::Eigen)

enable_testing()
add_test(NAME UnitTests COMMAND unit_tests)
add_test(NAME DslTests COMMAND dsl_tests)
add_test(NAME EigenTests COMMAND eigen_test)
add_test(NAME ExtendedTests COMMAND extended_tests)
add_test(NAME ScalarTests COMMAND scalar_tests)
