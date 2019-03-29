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
#ifdef WIN32
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

