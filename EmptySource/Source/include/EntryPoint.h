#pragma once

#include "../include/Utility/LogCore.h"
#include "../include/Application.h"

#ifdef ES_PLATFORM_WINDOWS
#include <windows.h>
#include <io.h>
#include <conio.h>
#include <fcntl.h>

// --- Make discrete GPU by default.
extern "C" {
	// --- developer.download.nvidia.com/devzone/devcenter/gamegraphics/files/OptimusRenderingPolicies.pdf
	__declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
	// --- developer.amd.com/community/blog/2015/10/02/amd-enduro-system-for-developers/
	__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 0x00000001;
}
#endif

extern EmptySource::Application * CreateInstance();

int main(int argc, char **argv) {
#ifdef ES_PLATFORM_WINDOWS
	_setmode(_fileno(stdout), _O_U8TEXT);
#endif

	EmptySource::Application::GetInstance().Run();

#ifdef ES_DEBUG
	Debug::Log(Debug::LogInfo, L"Press any key to close...");
	_getch();
#endif

	return 0;
}