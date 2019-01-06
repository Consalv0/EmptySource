#pragma once

struct FileStream {
private:
	std::wfstream* Stream;
	WString Path;

public:

	FileStream();
	FileStream(WString Path);

	WString GetExtension() const;
	WString GetPath() const;
	WString GetShortPath() const;
	std::wstringstream ReadStream() const;
	WChar* GetLine(long long MaxCount);
	bool IsValid() const;

	inline const std::wistream& operator>>(WString& _Str) {
		return (_STD move(*Stream) >> _Str);
	}

	bool Open();
	void Reset();
	void Close();
};