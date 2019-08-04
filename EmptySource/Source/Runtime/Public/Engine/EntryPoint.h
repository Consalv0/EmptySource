#pragma once

#include "Utility/LogCore.h"
#include "Engine/Application.h"

#ifdef ES_PLATFORM_WINDOWS

#ifdef ES_DLLEXPORT

// --- Make discrete GPU by default.
extern "C" {
	// --- developer.download.nvidia.com/devzone/devcenter/gamegraphics/files/OptimusRenderingPolicies.pdf
	__declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
	// --- developer.amd.com/community/blog/2015/10/02/amd-enduro-system-for-developers/
	__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 0x00000001;
}
#endif

#include <iostream>
#include <io.h>
#include <conio.h>
#include <fcntl.h>
#endif

extern EmptySource::Application * EmptySource::CreateApplication();

int main(int argc, char **argv) {
#ifdef ES_PLATFORM_WINDOWS
	_setmode(_fileno(stdout), _O_U8TEXT);
#endif

	EmptySource::Application::GetInstance().Run();

#ifdef ES_DEBUG
	EmptySource::Debug::Log(EmptySource::Debug::LogInfo, L"Press any key to close...");
	_getch();
#endif

	return 0;
}