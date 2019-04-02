#include "../include/Core.h"
#include "../include/Graphics.h"
#include "../include/Application.h"

#ifdef WIN32
// --- Make discrete GPU by default.
extern "C" {
    // --- developer.download.nvidia.com/devzone/devcenter/gamegraphics/files/OptimusRenderingPolicies.pdf
    __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
    // --- developer.amd.com/community/blog/2015/10/02/amd-enduro-system-for-developers/
    __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 0x00000001;
}
#endif

int main(int argc, char **argv) {
#ifdef _MSC_VER
    _setmode(_fileno(stdout), _O_U8TEXT);
#endif
    Debug::Log(Debug::LogInfo, L"Initalizing Application:\n");
    CoreApplication::Initalize();
    CoreApplication::MainLoop();
    CoreApplication::Close();
    
#ifdef WIN32
    Debug::Log(Debug::LogInfo, L"Press any key to close...");
    _getch();
#endif
}

// --- Build using clang-cl
// "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
// "C:\Program Files\LLVM\bin\clang-cl.exe" "C:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\External\GLAD\glad.c" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\Texture3D.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\Texture2D.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\Text2DGenerator.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\Text.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\Space.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\Shape2DContour.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\Shape2D.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\ShaderStage.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\ShaderProgram.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\Texture.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\RenderTarget.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\Object.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\ImageLoader.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\IIdentifier.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\Glyph.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\Font.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\FileStream.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\FileManager.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\EdgeSegments.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\EdgeHolder.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\Material.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\Window.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\Bitmap.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\Utility\TexturePacking.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\Utility\SDFGenerator.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\main.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\Utility\LogCore.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\Application.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\Utility\Timer.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\CoreTime.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\Mesh.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\MeshLoader.cpp" "c:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\source\Utility\DeviceFunctions.cpp" /Z7 /MDd /W4 /EHsc /std:c++17 /I"C:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source\External" /I"C:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Source" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include" /D"_UNICODE" /D"UNICODE" /Od /o"C:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\x64\Debug\_EmptySource.exe" /link "C:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Libraries\freetype.lib" "C:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Libraries\glfw3.lib" "C:\Users\Consalvo\Workspace\Tesis\EmptySource\Application\EmptySource\Libraries\glfw3dll.lib" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib\x64\cudart_static.lib" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib\x64\nvml.lib" "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.17763.0\um\x64\opengl32.lib" "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.17763.0\um\x64\kernel32.lib" "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.17763.0\um\x64\user32.lib" "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.17763.0\um\x64\gdi32.lib" "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.17763.0\um\x64\shell32.lib" "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.17763.0\um\x64\winspool.lib" "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.17763.0\um\x64\uuid.lib" "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.17763.0\um\x64\comdlg32.lib"