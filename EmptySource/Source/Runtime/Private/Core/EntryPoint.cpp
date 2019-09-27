#pragma once

#include "CoreMinimal.h"
#include "Core/Application.h"

#include <SDL_main.h>

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

extern ESource::Application * ESource::CreateApplication();

int main(int argc, char **argv) {
#ifdef ES_PLATFORM_WINDOWS
	_setmode(_fileno(stdout), _O_U8TEXT);
#endif

	ESource::Log::Initialize();
	ESource::Application::GetInstance()->Run();

#ifdef ES_DEBUG
	_getch();
#endif // ES_DEBUG

	return 0;
}