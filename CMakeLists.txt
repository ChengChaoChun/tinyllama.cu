cmake_minimum_required(VERSION 3.28)

project(Llama LANGUAGES CXX CUDA) 

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  

set(CMAKE_CUDA_STANDARD 20)  
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

include(FetchContent)

FetchContent_Declare(
    sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece.git
    GIT_TAG v0.2.2
)

FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0
)

FetchContent_MakeAvailable(sentencepiece json)

find_package(CUDAToolkit REQUIRED)

add_executable(tinyllama-cu
    src/main.cpp
    src/tokenizer.cpp  
    src/safetensor_reader.cpp
    src/cpu_model_loader.cpp 
    src/gpu_model_loader.cpp 
    src/transformer.cpp
    src/gpu_transformer.cpp


    src/backend/cpu_backend.cpp
    src/backend/gpu_backend.cpp
    src/backend/kernel.cu
    
    #src/cuda/cuda_backend.cpp   
)  

target_compile_features(tinyllama-cu PRIVATE cxx_std_20 cuda_std_20)

message(STATUS "SentencePiece source: ${sentencepiece_SOURCE_DIR}")

target_include_directories(
    tinyllama-cu
    PRIVATE
    ${sentencepiece_SOURCE_DIR}
    ${sentencepiece_SOURCE_DIR}/src
)


target_link_libraries(
    tinyllama-cu
    PRIVATE
    sentencepiece
    nlohmann_json::nlohmann_json
    
    CUDA::cudart
    CUDA::cublas
)  