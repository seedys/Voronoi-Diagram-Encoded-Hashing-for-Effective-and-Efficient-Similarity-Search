cmake_minimum_required(VERSION 3.15)
project(IKE VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# if build test program
option(IKE_BUILD_TESTS "Build IKE tests" ON)

set(COMPILE_OPTIONS
    -O3
    -funroll-loops
    -fomit-frame-pointer
    -march=native
    # -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw -mavx512vpopcntdq
    -mpopcnt
    -fopenmp
)

# Find dependency packages
find_package(OpenMP REQUIRED)

# Set FAISS path
set(FAISS_INCLUDE_ROOT "/path/to/faiss/include" CACHE PATH "Path to FAISS include")
set(FAISS_LIB_ROOT "/path/to/faiss/lib" CACHE PATH "Path to FAISS library")

message(STATUS "FAISS include: ${FAISS_INCLUDE_ROOT}")
message(STATUS "FAISS lib directory: ${FAISS_LIB_ROOT}")

# Source files
set(IKE_SOURCES
    src/IForest_index/gemc.cpp
    src/IForest_index/IForest_Flat.cpp
    src/IForest_index/IVFIForest.cpp
    src/IForest_index/IForest_HNSW.cpp
    src/IForest_index/myHNSW.cpp
)

add_library(IKE STATIC ${IKE_SOURCES})
target_include_directories(IKE PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${FAISS_INCLUDE_ROOT}
)

target_compile_options(IKE PRIVATE 
    ${COMPILE_OPTIONS}
)

target_link_libraries(IKE PRIVATE 
    OpenMP::OpenMP_CXX
)

target_link_directories(IKE PUBLIC ${FAISS_LIB_ROOT})
target_link_libraries(IKE PUBLIC faiss_avx512)

# Build test program
if(IKE_BUILD_TESTS)
    set(TEST_PROGRAMS
        Exhaustive_Search_with_IKE
        HNSW_with_IKE  
        IVF_with_IKE
    )
    
    foreach(test_program IN LISTS TEST_PROGRAMS)
        add_executable(${test_program} test/${test_program}.cpp test/utils.cpp)
        target_include_directories(${test_program} PRIVATE 
          ${CMAKE_CURRENT_SOURCE_DIR}/test
        )
        target_link_libraries(${test_program} PRIVATE IKE OpenMP::OpenMP_CXX)
        
        target_compile_options(${test_program} PRIVATE 
          ${COMPILE_OPTIONS}
        )
        
    endforeach()
endif()
