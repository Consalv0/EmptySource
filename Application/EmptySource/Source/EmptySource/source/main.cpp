#include "EmptySource/include/EmptyHeaders.h"
#include "EmptySource/include/SApplication.h"

#ifdef _WIN32
// Make discrete GPU by default.
extern "C" {
	// developer.download.nvidia.com/devzone/devcenter/gamegraphics/files/OptimusRenderingPolicies.pdf
	__declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
	// developer.amd.com/community/blog/2015/10/02/amd-enduro-system-for-developers/
	__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 0x00000001;
}
#endif

 SApplication Application;

int main() {
	printf("Initalizing Application:\n");
	Application.Initalize();
	printf("...............\n");
	Application.GraphicsInformation();
	printf("...............\n");

	Application.MainLoop();
	Application.Close();

	printf("\nPress any key to close...\n");
	_getch();
}