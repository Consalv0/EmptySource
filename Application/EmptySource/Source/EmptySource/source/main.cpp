#include "..\include\Core.h"
#include "..\include\Graphics.h"
#include "..\include\Application.h"

#include "..\include\Utility\CUDAUtility.h"

#ifdef _WIN32
// Make discrete GPU by default.
extern "C" {
	// developer.download.nvidia.com/devzone/devcenter/gamegraphics/files/OptimusRenderingPolicies.pdf
	__declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
	// developer.amd.com/community/blog/2015/10/02/amd-enduro-system-for-developers/
	__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 0x00000001;
}
#endif

extern "C" bool RunTest(const int argc, const char **argv,
                        char *data, int2 *data_int2, unsigned int len);


int main(int argc, char **argv) {
	_setmode(_fileno(stdout), _O_U16TEXT);

	Debug::Log(Debug::LogNormal, L"Initalizing Application:\n");

	// input data
	int len = 16;
	// the data has some zero padding at the end so that the size is a multiple of
	// four, this simplifies the processing as each thread can process four
	// elements (which is necessary to avoid bank conflicts) but no branching is
	// necessary to avoid out of bounds reads
	char str[] = { 82, 111, 118, 118, 121, 42, 97, 121, 124, 118, 110, 56,
				   10, 10, 10, 10
	};

	// Use int2 showing that CUDA vector types can be used in cpp code
	int2 i2[16];

	for (int i = 0; i < len; i++) {
		i2[i].x = str[i];
		i2[i].y = 10;
	}

	bool bTestResult;

	// run the device part of the program
	bTestResult = RunTest(argc, (const char **)argv, str, i2, len);

	Debug::Log(Debug::LogDebug, L"%s", CharToWChar(str));

	char str_device[16];

	for (int i = 0; i < len; i++) {
		str_device[i] = (char)(i2[i].x);
	}

	Debug::Log(Debug::LogDebug, L"%s", CharToWChar(str_device));

	CoreApplication::Initalize();
	CoreApplication::PrintGraphicsInformation();

	CoreApplication::MainLoop();
	CoreApplication::Close();

	Debug::Log(Debug::LogNormal, L"Press any key to close...");
	_getch();
}