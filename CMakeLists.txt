cmake_minimum_required(VERSION 3.20)
project(WindowsFaceIdCpp20 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /W4 /utf-8)
else()
    add_compile_options(-O3 -mavx2 -mfma -Wall -Wextra)
endif()

add_subdirectory(src/credential_provider)

add_executable(FaceIdManager WIN32
    src/manager/FaceIdManager.cpp
    src/credential_provider/FaceIdHud.cpp
    src/credential_provider/NativeFaceEngine.cpp
)

target_include_directories(FaceIdManager PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/credential_provider"
)

target_compile_definitions(FaceIdManager PRIVATE
    UNICODE _UNICODE
    WIN32_LEAN_AND_MEAN
)

target_link_libraries(FaceIdManager PRIVATE
    comctl32
    shell32
    crypt32
    mfplat
    mf
    mfreadwrite
    mfuuid
    ole32
    oleaut32
    advapi32
    user32
    gdi32
    gdiplus
    uuid
)
