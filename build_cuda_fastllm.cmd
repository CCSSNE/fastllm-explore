@echo off
setlocal
chcp 65001 >nul
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
set CUDA_PATH=%CD%\cuda-12.4-toolkit
set CUDA_PATH_V12_4=%CUDA_PATH%
set PATH=%CUDA_PATH%\bin;%PATH%
cmake -S fastllm-master -B fastllm-master\build-cuda-ninja -G Ninja -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DUSE_CUDA=ON -DUSE_NUMAS=OFF -DBUILD_CLI=OFF -DCUDA_ARCH=89 -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe" -DCUDAToolkit_ROOT="%CUDA_PATH%" "-DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -Xcompiler=/utf-8" "-DCMAKE_EXE_LINKER_FLAGS=/LIBPATH:%CUDA_PATH%\lib\x64" "-DCMAKE_SHARED_LINKER_FLAGS=/LIBPATH:%CUDA_PATH%\lib\x64" "-DCMAKE_CUDA_STANDARD_LIBRARIES=cudart.lib"
cmake --build fastllm-master\build-cuda-ninja --target fastllm_tools -j 6
