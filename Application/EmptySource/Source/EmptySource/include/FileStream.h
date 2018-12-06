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
	std::wstringstream ReadStream() const;
	bool IsValid() const;

	bool Open();
	void Close();
};