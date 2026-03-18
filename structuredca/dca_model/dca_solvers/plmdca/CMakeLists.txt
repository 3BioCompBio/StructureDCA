cmake_minimum_required(VERSION 3.10)
project(plmdcaBackend)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(include)
include_directories(lbfgs/include)

# Source files
set(SRC_FILES 
    plmdcaBackend.cpp 
    plmdca.cpp 
    lbfgs/lib/lbfgs.cpp
)

# Shared library output
add_library(plmdcaBackend SHARED ${SRC_FILES})

# Change output name to _plmdcaBackend.so
set_target_properties(plmdcaBackend PROPERTIES
    OUTPUT_NAME "_plmdcaBackend"
    SUFFIX ".so"
)

# Compiler options (optional: add any optimization/debug flags here)
target_compile_options(plmdcaBackend PRIVATE -fPIC -O3)

# Link libraries (add more if needed, e.g., -lpthread)
target_link_libraries(plmdcaBackend PRIVATE)