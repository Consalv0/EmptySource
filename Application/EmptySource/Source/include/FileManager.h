#pragma once

#include "../include/FileStream.h"

typedef TArray<FileStream*> FileList;

class FileManager {
private:
	static FileList Files;

	static FileList::iterator FindInFiles(const WString& FilePath);

public:
	static FileStream* Open(const WString& FilePath);

	static WString GetFileExtension(const WString& Path);

	static WString GetFullPath(const WString& Path);
    
    static WString GetAppDirectory();

	//* ReadStreams the file streams of the shader code
	static WString ReadStream(FileStream* Stream);
};
