
#include "..\include\Core.h"
#include "..\include\FileStream.h"

#include "..\include\LogCore.h"

FileStream::FileStream() {
	Stream = NULL;
	Path = L"";
	Lenght = 0;
}

FileStream::FileStream(WString FilePath) {
	Stream = new std::wfstream(FilePath);
	Path = FilePath;
	Open();
	if (!IsValid()) Debug::Log(Debug::LogError, L"File '%s' is not valid or do not exist", FilePath.c_str());
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
	WChar CurrentDirectory[_MAX_DIR + 1];
	GetCurrentDirectory(_MAX_DIR, CurrentDirectory);

	WString ReturnValue = Path;
	TextReplace(ReturnValue, WString(CurrentDirectory), WString(L".."));

	return ReturnValue;
}

std::wstringstream FileStream::ReadStream() const {
	std::wstringstream stringStream;
	if (IsValid()) {
		try {
			stringStream << Stream->rdbuf();
		} catch (...) {}
	} else {
		Debug::Log(Debug::LogError, L"File '%s' is not valid or do not exist", Path.c_str());
	}

	return stringStream;
}

std::stringstream FileStream::ReadNarrowStream() const {
	std::stringstream stringStream;
	if (IsValid()) {
		try {
			std::fstream NarrowStream;
			NarrowStream.open(Path, std::fstream::in);
			stringStream << NarrowStream.rdbuf();
		} catch (...) {}
	} else {
		Debug::Log(Debug::LogError, L"File '%s' is not valid or do not exist", Path.c_str());
	}

	return stringStream;
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
	Stream->open(Path);

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
