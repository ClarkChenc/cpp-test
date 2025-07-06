# 自动收集当前目录下所有 .cpp 文件
file(GLOB MYLIB_SOURCES "*.cc")

# 构建静态库 mylib.a
add_library(mylib STATIC ${MYLIB_SOURCES})

# 将头文件路径导出给其他模块使用
target_include_directories(mylib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
