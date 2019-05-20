#pragma once

#include <fstream>
#include "../include/Text.h"
#include "../include/Utility/Hasher.h"

struct FileStream {
private:
	std::wfstream* Stream;
	long Lenght;
	WString Path;

public:

	FileStream();
	FileStream(WString Path);

	WString GetExtension() const;
	WString GetFileName() const;
	WString GetPath() const;
	WString GetShortPath() const;
	std::wstringstream ReadStream() const;
	bool ReadNarrowStream(String* Output) const;
	WString GetLine();
	bool IsValid() const;

	template <typename T>
	inline const std::wistream& operator>>(T Value) {
        return (std::move(*Stream) >> Value);
	}

	template <typename T>
	inline const std::wostream& operator<<(T Value) {
		return (std::move(*Stream) << Value);
	}

	inline float GetProgress() const {
		long Progress = long(Stream->tellg());
		return Progress / float(Lenght);
	}

	inline long GetPosition() const {
		return (long)Stream->tellg();
	}

	void LocaleToUTF8();
	long GetLenght();
	bool Open();
	void Clean();
	void Reset();
	void Close();
};