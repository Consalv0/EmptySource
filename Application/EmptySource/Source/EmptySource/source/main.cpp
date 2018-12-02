#include "../include/Core.h"
#include "../include/Application.h"

#ifdef _WIN32
// Make discrete GPU by default.
extern "C" {
	// developer.download.nvidia.com/devzone/devcenter/gamegraphics/files/OptimusRenderingPolicies.pdf
	__declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
	// developer.amd.com/community/blog/2015/10/02/amd-enduro-system-for-developers/
	__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 0x00000001;
}
#endif

 CoreApplication Application;

int main() {
	_setmode(_fileno(stdout), _O_U16TEXT);

	wprintf(L"Initalizing Application:\n");
	Application.Initalize();
	wprintf(L"...............\n");
	Application.PrintGraphicsInformation();
	wprintf(L"...............\n");

	Application.MainLoop();
	Application.Close();

	wprintf(L"\nPress any key to close...\n");
	_getch();
}