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
  
add_library(tinyllama-core STATIC
    src/tokenizer.cpp
    src/chat_template.cpp
    src/safetensor_reader.cpp

    src/cpu_model_loader.cpp
    src/gpu_model_loader.cpp

    src/transformer.cpp
    src/gpu_transformer.cpp

    src/backend/cpu_backend.cpp
    src/backend/gpu_backend.cpp
    src/backend/kernel.cu
)

target_include_directories(tinyllama-core
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend
PRIVATE
    ${sentencepiece_SOURCE_DIR}
    ${sentencepiece_SOURCE_DIR}/src
)

target_link_libraries(tinyllama-core  
PUBLIC
    sentencepiece
    nlohmann_json::nlohmann_json

    CUDA::cudart
    CUDA::cublas
)

target_compile_features(tinyllama-core
PUBLIC
    cxx_std_20
    cuda_std_20
)


# ============================================================
# Normal Inference Executable
# src/main.cpp
# This is the program users normally run.
# ============================================================
add_executable(tinyllama-cu
    src/main.cpp
)
  
target_link_libraries(tinyllama-cu
PRIVATE
    tinyllama-core
)

# ============================================================
# Test Executable
#
# test/inference_test.cpp
#
# This executable is only used for numerical validation.
# It uses the same inference engine as tinyllama-cu.
# ============================================================
add_executable(gpu-logits-test
    test/gpu_logits_test.cpp
)

target_link_libraries(gpu-logits-test  
PRIVATE
    tinyllama-core  
)  

add_executable(chat_template_test
    test/chat_template_test.cpp
)

target_link_libraries(chat_template_test
PRIVATE
    tinyllama-core    
) 

add_executable(placeholder_test
    test/placeholder_test.cpp
)

target_link_libraries(placeholder_test
PRIVATE
    tinyllama-core    
) 