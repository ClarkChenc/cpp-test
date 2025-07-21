cmake_minimum_required(VERSION 3.15)
project(CPPTest)

set(CMAKE_CXX_STANDARD 17)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/../bin)

# set(FORBID_EIGEN "-DEIGEN_DONT_VECTORIZE -DEIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -pthread -march=x86-64  -msse3 -mavx2")
add_compile_options(-msse3 -mavx2)

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} ")
else()
    message(WARNING "OpenMP is not found")
endif()

find_package(absl CONFIG REQUIRED)

add_subdirectory(${PROJECT_SOURCE_DIR}/include)

add_executable(main main.cc)

find_library(TCMALLOC_LIB tcmalloc REQUIRED)

if (APPLE)
else()
    target_link_libraries(main PRIVATE 
        # ${TCMALLOC_LIB}
        # absl::flat_hash_map
        mylib
    )
    # # 添加编译器和链接器标志
    # target_compile_options(main PRIVATE ${SANITIZER_FLAGS})
    # target_link_options(main PRIVATE ${SANITIZER_FLAGS})
endif()
