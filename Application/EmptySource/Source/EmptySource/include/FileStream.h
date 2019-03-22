#pragma once

struct FileStream {
private:
	std::wfstream* Stream;
	long Lenght;
	WString Path;

public:

	FileStream();
	FileStream(WString Path);

	WString GetExtension() const;
	WString GetPath() const;
	WString GetShortPath() const;
	std::wstringstream ReadStream() const;
	bool ReadNarrowStream(String* Output) const;
	WChar* GetLine(long long MaxCount);
	bool IsValid() const;

	inline const std::wistream& operator>>(WString& _Str) {
        return (std::move(*Stream) >> _Str);
	}

	inline const float GetProgress() const {
		long Progress = long(Stream->tellg());
		return Progress / float(Lenght);
	}

	inline const long GetPosition() const {
		return (long)Stream->tellg();
	}

	long GetLenght();
	bool Open();
	void Reset();
	void Close();
};
