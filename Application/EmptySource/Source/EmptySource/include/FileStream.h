#pragma once

struct FileStream {
private:
	std::fstream* Stream;
	WString Path;

public:

	FileStream();
	FileStream(WString Path);

	WString GetExtension() const;
	WString GetPath() const;
	std::stringstream ReadStream() const;
	bool IsValid() const;

	bool Open();
	void Close();
};