
#include <algorithm>
#include <stdlib.h>
#include "../include/Core.h"
#include "../include/FileManager.h"

FileList FileManager::Files = FileList();

FileStream* FileManager::Open(const WString & FilePath) {
	WString FullFilePath = GetFullPath(FilePath);
	FileList::iterator Found = FindInFiles(FullFilePath);

	if (Found != Files.end()) (*Found)->Reset();
	if (Found != Files.end() && (*Found)->IsValid()) return *Found;

	FileStream* NewStream = new FileStream(FullFilePath);

	if (!NewStream->IsValid()) return NULL;

	Files.push_back(NewStream);
	return NewStream;
}

WString FileManager::GetFullPath(const WString & Path) {
#ifdef WIN32
	WChar FullPath[MAX_PATH + 1];
	GetFullPathName(Path.c_str(), MAX_PATH, FullPath, NULL);
    return FullPath;
#else
    Char FullPath[PATH_MAX + 1];
    Char * Ptr;
    Ptr = realpath(WStringToString(Path).c_str(), FullPath);
    return CharToWChar(FullPath);
#endif
}

FileList::iterator FileManager::FindInFiles(const WString & FilePath) {
	return std::find_if(Files.begin(), Files.end(), [FilePath](FileStream* & File)
		-> bool { return File->GetPath() == FilePath; }
	);
}

WString FileManager::ReadStream(FileStream* Stream) {
	WString ShaderCode = L"";

	// ReadStreams the Vertex Shader code from the file
	if (Stream != NULL && Stream->IsValid()) {
		ShaderCode = Stream->ReadStream().str();
		Stream->Close();
	}

	return ShaderCode;
}
