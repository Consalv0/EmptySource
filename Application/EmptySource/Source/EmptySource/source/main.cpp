#include "..\include\Core.h"
#include "..\include\Graphics.h"
#include "..\include\Application.h"

#ifdef _WIN32
// --- Make discrete GPU by default.
extern "C" {
	// --- developer.download.nvidia.com/devzone/devcenter/gamegraphics/files/OptimusRenderingPolicies.pdf
	__declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
	// --- developer.amd.com/community/blog/2015/10/02/amd-enduro-system-for-developers/
	__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 0x00000001;
}
#endif

int main(int argc, char **argv) {
	_setmode(_fileno(stdout), _O_U16TEXT);

	Debug::Log(Debug::LogNormal, L"Initalizing Application:\n");
	CoreApplication::Initalize();
	CoreApplication::MainLoop();
	CoreApplication::Close();

	Debug::Log(Debug::LogNormal, L"Press any key to close...");
	_getch();
}