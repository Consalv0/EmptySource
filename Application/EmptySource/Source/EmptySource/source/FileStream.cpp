
#include "..\include\Core.h"
#include "..\include\FileStream.h"

#include "..\include\LogCore.h"

FileStream::FileStream() {
	Stream = NULL;
	Path = L"";
}

FileStream::FileStream(WString FilePath) {
	Stream = new std::wfstream(FilePath);
	Path = FilePath;
	if (!IsValid()) _LOG(LogError, L"File '%s' is not valid or do not exist", FilePath.c_str());
}

WString FileStream::GetExtension() const {
	return WString();
}

WString FileStream::GetPath() const {
	return Path;
}

std::wstringstream FileStream::ReadStream() const {
	std::wstringstream stringStream;
	if (IsValid()) {
		try {
			stringStream << Stream->rdbuf();
		} catch (...) {}
	} else {
		_LOG(LogError, L"File '%s' is not valid or do not exist", Path.c_str());
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
