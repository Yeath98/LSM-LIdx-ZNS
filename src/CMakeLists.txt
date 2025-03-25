include_directories(include LSM-Tree)
project(ZNS-SSD)

# 设置 Python 和 NumPy 的头文件路径
include_directories(
    /usr/include/python3.6m
    /usr/local/lib64/python3.6/site-packages/numpy/core/include
)

# 设置 Python 库路径
link_directories(/usr/lib64/python3.6/config-3.6m-x86_64-linux-gnu)

# 添加子目录
add_subdirectory(femu)
add_subdirectory(LSM-Tree)

# 设置源文件
add_executable(ZNS-SSD main.cpp test.h include/utils.cpp readThreadPool.h)

# 链接 Python 库
target_link_libraries(ZNS-SSD LSM-Tree femu python3.6m)

# 设置动态链接路径
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,/usr/lib64/python3.6/config-3.6m-x86_64-linux-gnu")