#pragma once

#include "..\include\FileStream.h"

typedef std::vector<FileStream*> FileList;

class FileManager {
private:
	static FileList Files;

	static FileList::iterator FindInFiles(const WString& FilePath);

public:
	static FileStream* Open(const WString& FilePath);
	static WString GetFullPath(const WString& Path);

	//* ReadStreams the file streams of the shader code
	static WString ReadStream(FileStream* Stream);
};