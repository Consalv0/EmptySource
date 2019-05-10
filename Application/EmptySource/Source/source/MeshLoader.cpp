
#include "../include/Core.h"
#include "../include/OBJLoader.h"
#include "../include/FBXLoader.h"
#include "../include/MeshLoader.h"

bool MeshLoader::LoadFromFile(FileStream * File, FileData & Data, bool Optimize) {
	WString Extension = File->GetExtension();
	if (Text::CompareIgnoreCase(Extension, L"FBX")) {
		return FBXLoader::Load(File, Data, Optimize);
	}
	if (Text::CompareIgnoreCase(Extension, L"OBJ")) {
		return OBJLoader::Load(File, Data, Optimize);
	}

	return false;
}

bool MeshLoader::Load(FileData & Data, FileStream * File, bool Optimize) {
	if (File == NULL) return false;

	Debug::Log(Debug::LogInfo, L"Reading File Model '%ls'", File->GetShortPath().c_str());
	bool bNoError = LoadFromFile(File, Data, Optimize);
	Data.bLoaded = bNoError;
	return bNoError;
}
