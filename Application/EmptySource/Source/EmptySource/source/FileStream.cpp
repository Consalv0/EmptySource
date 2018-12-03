
#include "..\include\Core.h"
#include "..\include\FileStream.h"

#include "..\include\LogCore.h"

FileStream::FileStream() {
	Stream = NULL;
	Path = L"";
}

FileStream::FileStream(WString FilePath) {
	Stream = new std::fstream(FilePath);
	Path = FilePath;
}

WString FileStream::GetExtension() const {
	return WString();
}

WString FileStream::GetPath() const {
	return Path;
}

std::stringstream FileStream::ReadStream() const {
	std::stringstream stringStream;
	if (IsValid()) {
		try {
			stringStream << Stream->rdbuf();
		} catch (...) {}
	} else {
		_LOG(LogError, L"File '%s' is not valid or do not exist", Path);
	}

	return stringStream;
}

bool FileStream::IsValid() const {
	return !Stream->fail() && Stream->good() && Stream != NULL;
}

bool FileStream::Open() {
	Stream->open(Path);
	return Stream->is_open();
}

void FileStream::Close() {
	Stream->close();
}
