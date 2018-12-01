
#include "..\include\SCore.h"
#include "..\include\SFileManager.h"

FileStringMap SFileManager::Files = FileStringMap();

std::fstream* SFileManager::Open(const std::wstring & FilePath) {
	std::fstream* NewStream(new std::fstream(FilePath));

	if (NewStream->fail() || !NewStream->good() || !*NewStream) return NULL;

	Files[NewStream] = FilePath;
	return NewStream;
}

std::wstring SFileManager::GetFilePath(std::fstream * FileStream) {

	if (FileStream == NULL) {
		wprintf(L"Error:: File stream not found");
		return L"";
	}

	FileStringMap::const_iterator Found = Files.find(FileStream);

	if (Found != Files.end()) return (*Found).second;
	else {
		wprintf(L"Error:: File stream not found");
		return L"";
	}
}
