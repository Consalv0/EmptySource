
#ifdef __APPLE__
#include <unistd.h>
#endif
#include "../include/Core.h"
#include "../include/FileStream.h"
#include "../include/FileManager.h"

FileStream::FileStream() {
	Stream = NULL;
	Path = L"";
	Lenght = 0;
}

FileStream::FileStream(WString FilePath) {
#ifdef WIN32
	Stream = new std::wfstream(FilePath);
#else
    Stream = new std::wfstream(WStringToString(FilePath));
#endif
	Path = FilePath;
	Open();
	if (!IsValid()) Debug::Log(Debug::LogError, L"File '%ls' is not valid or do not exist", FilePath.c_str());
}

WString FileStream::GetExtension() const {
	WString::size_type ExtensionIndex;

	ExtensionIndex = Path.rfind('.');

	if (ExtensionIndex != WString::npos) {
		return Path.substr(ExtensionIndex + 1);
	} else {
		return L"";
	}
}

WString FileStream::GetPath() const {
	return Path;
}

WString FileStream::GetShortPath() const {
    WString CurrentDirectory = FileManager::GetAppDirectory();
	WString ReturnValue = Path;
	Text::Replace(ReturnValue, CurrentDirectory, L"..");

	return ReturnValue;
}

std::wstringstream FileStream::ReadStream() const {
	std::wstringstream stringStream;
	if (IsValid()) {
		try {
			stringStream << Stream->rdbuf();
		} catch (...) {}
	} else {
		Debug::Log(Debug::LogError, L"File '%ls' is not valid or do not exist", Path.c_str());
	}

	return stringStream;
}

bool FileStream::ReadNarrowStream(String* Output) const {
	if (IsValid()) {
		try {
			std::fstream NarrowStream;
#ifdef WIN32
            NarrowStream.open(Path, std::ios::in | std::ios::binary);
#else
            NarrowStream.open(WStringToString(Path), std::ios::in | std::ios::binary);
#endif
			NarrowStream.seekg(0, std::ios::end);
			Output->resize(NarrowStream.tellg());
			NarrowStream.seekg(0, std::ios::beg);
			NarrowStream.read(&(*Output)[0], Output->size());
			NarrowStream.close();
		} catch (...) {
			return false;
		}
	} else {
		Debug::Log(Debug::LogError, L"File '%ls' is not valid or do not exist", Path.c_str());
		return false;
	}

	return true;
}

WChar* FileStream::GetLine(long long MaxCount) {
	WChar* String = new WChar[MaxCount + 1];
	Stream->getline(String, MaxCount);
	return String;
}

bool FileStream::IsValid() const {
	return !Stream->fail() && Stream->good() && Stream != NULL;
}

long FileStream::GetLenght() {
	return Lenght;
}

bool FileStream::Open() {
#ifdef WIN32
    Stream->open(Path, std::ios::in | std::ios::binary);
#else
    Stream->open(WStringToString(Path), std::ios::in | std::ios::binary);
#endif

	if (Stream->is_open()) Reset();

	return Stream->is_open();
}

void FileStream::Reset() {
	Stream->clear();
	Stream->seekg(0, Stream->end);
	Lenght = long(Stream->tellg());
	Stream->seekg(0, Stream->beg);
}

void FileStream::Close() {
	Stream->close();
}
