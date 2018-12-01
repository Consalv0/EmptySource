#pragma once

typedef std::map<std::fstream*, std::wstring> FileStringMap;

class SFileManager {
private:
	static FileStringMap Files;

public:
	static std::fstream* Open(const std::wstring& FilePath);
	static std::wstring GetFilePath(std::fstream* FileStream);
};