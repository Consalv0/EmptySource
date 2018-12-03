#pragma once

#include "..\include\FileStream.h"

typedef std::vector<FileStream*> FileList;

class FileManager {
private:
	static FileList Files;

public:
	static FileStream* Open(const WString& FilePath);
};