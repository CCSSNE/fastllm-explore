@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo ERROR: vswhere.exe not found at "%VSWHERE%".
    exit /b 1
)

for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%I"
if not defined VSINSTALL (
    echo ERROR: MSVC C++ x64 tools are not installed.
    echo Install the Visual Studio component: Microsoft.VisualStudio.Component.VC.Tools.x86.x64
    exit /b 1
)

if not exist "%VSINSTALL%\Common7\Tools\VsDevCmd.bat" (
    echo ERROR: VsDevCmd.bat not found under "%VSINSTALL%".
    exit /b 1
)

call "%VSINSTALL%\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
where cl >nul 2>nul
if errorlevel 1 (
    echo ERROR: cl.exe is still unavailable after VsDevCmd.bat.
    exit /b 1
)

set "CUDA_PATH=%ROOT%\cuda-12.4-toolkit"
set "CUDA_PATH_V12_4=%CUDA_PATH%"
set "PATH=%CUDA_PATH%\bin;%PATH%"

if not exist "%CUDA_PATH%\bin\nvcc.exe" (
    echo ERROR: nvcc.exe not found at "%CUDA_PATH%\bin\nvcc.exe".
    exit /b 1
)

set "BUILD_DIR=%ROOT%\fastllm-master\build-cuda-ninja-clean"
cmake -S "%ROOT%\fastllm-master" -B "%BUILD_DIR%" -G Ninja ^
    -DCMAKE_C_COMPILER=cl ^
    -DCMAKE_CXX_COMPILER=cl ^
    -DUSE_CUDA=ON ^
    -DUSE_NUMAS=OFF ^
    -DBUILD_CLI=OFF ^
    -DCUDA_ARCH=89 ^
    -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe" ^
    -DCUDAToolkit_ROOT="%CUDA_PATH%" ^
    "-DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -Xcompiler=/utf-8" ^
    "-DCMAKE_EXE_LINKER_FLAGS=/LIBPATH:%CUDA_PATH%\lib\x64" ^
    "-DCMAKE_SHARED_LINKER_FLAGS=/LIBPATH:%CUDA_PATH%\lib\x64" ^
    "-DCMAKE_CUDA_STANDARD_LIBRARIES=cudart.lib"
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%" --target fastllm_tools -j 6
if errorlevel 1 exit /b 1

echo Build succeeded. Review "%BUILD_DIR%\tools\ftllm\fastllm_tools.dll" before replacing the runtime DLL.
