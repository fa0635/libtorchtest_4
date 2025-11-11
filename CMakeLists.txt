cmake_minimum_required(VERSION 3.18)
project(libtorchtest_3)

set(Torch_ROOT "/home/andrej/libtorch")
set(yaml-cpp_ROOT "/home/andrej/miniconda3")
set(ROOT_ROOT "/home/andrej/root_install")

find_package(Torch REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(ROOT 6.36 CONFIG REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} ${ROOT_CXX_FLAGS}")

add_executable(main main.cpp)
target_include_directories(main PUBLIC ${YAML_CPP_INCLUDE_DIR} ${ROOT_INCLUDE_DIRS})
target_link_libraries(main PUBLIC ${TORCH_LIBRARIES}
                                  yaml-cpp::yaml-cpp
                                  ${ROOT_LIBRARIES}
                                  ${ROOT_EXE_LINKER_FLAGS})
set_property(TARGET main PROPERTY CXX_STANDARD 17)
