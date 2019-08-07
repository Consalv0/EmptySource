
#include "Engine/Log.h"
#include "Engine/Core.h"
#include "Files/FileStream.h"
#include "Files/FileManager.h"

#ifdef __APPLE__
#include <unistd.h>
#endif
#include <sstream>
#include <fstream>
#include <iostream>

namespace EmptySource {

	FileStream::FileStream() {
		Stream = NULL;
		Path = L"";
		Lenght = 0;
	}

	FileStream::FileStream(WString FilePath) {
#ifdef ES_PLATFORM_WINDOWS
		Stream = new std::wfstream(FilePath);
#else
		Stream = new std::wfstream(Text::WideToNarrow(FilePath));
#endif

		Path = FilePath;
		Open();

		if (!IsValid())
			LOG_CORE_ERROR(L"File '{}' is not valid or do not exist", FilePath);
		else
			LocaleToUTF8();
	}

	WString FileStream::GetExtension() const {
		WString::size_type ExtensionIndex;

		ExtensionIndex = Path.rfind('.');

		if (ExtensionIndex != WString::npos) {
			return Path.substr(ExtensionIndex + 1);
		}
		else {
			return L"";
		}
	}

	WString FileStream::GetFileName() const {
		WChar Separator = L'/';

#ifdef ES_PLATFORM_WINDOWS
		Separator = L'\\';
#endif

		size_t i = Path.rfind(Separator, Path.length());
		if (i != WString::npos) {
			return(Path.substr(i + 1, Path.length() - i));
		}

		return L"";
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
			}
			catch (...) {}
		}
		else {
			LOG_CORE_ERROR(L"File '{}' is not valid or do not exist", Path);
		}

		return stringStream;
	}

	bool FileStream::ReadNarrowStream(NString* Output) const {
		if (IsValid()) {
			try {
				std::fstream NarrowStream;
#ifdef ES_PLATFORM_WINDOWS
				NarrowStream.open(Path, std::ios::in);
#else
				NarrowStream.open(Text::WideToNarrow(Path), std::ios::in | std::ios::binary);
#endif
				NarrowStream.seekg(0, std::ios::end);
				Output->resize(NarrowStream.tellg());
				NarrowStream.seekg(0, std::ios::beg);
				NarrowStream.read(&(*Output)[0], Output->size());
				NarrowStream.close();
			}
			catch (...) {
				return false;
			}
		}
		else {
			LOG_CORE_ERROR(L"File '{}' is not valid or do not exist", Path);
			return false;
		}

		return true;
	}

	WString FileStream::GetLine() {
		WString String;
		std::getline(*Stream, String);
		return String;
	}

	bool FileStream::IsValid() const {
		return Stream != NULL && !Stream->fail() && Stream->good();
	}

	void FileStream::LocaleToUTF8() {
		static std::locale Locale("en_US.UTF-8");
		Stream->imbue(Locale);
	}

	long FileStream::GetLenght() {
		return Lenght;
	}

	bool FileStream::Open() {
#ifdef ES_PLATFORM_WINDOWS
		Stream->open(Path, std::ios::in | std::ios::out);
#else
		Stream->open(Text::WideToNarrow(Path), std::ios::in);
#endif
		if (Stream->is_open())
			Reset();

		return Stream->is_open();
	}

	void FileStream::Clean() {
		if (Stream->is_open())
			Stream->close();
#ifdef ES_PLATFORM_WINDOWS
		Stream->open(Path, std::ios::in | std::ios::out | std::ios::trunc);
#else
		Stream->open(Text::WideToNarrow(Path), std::ios::in | std::ios::out | std::ios::trunc);
#endif
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

}