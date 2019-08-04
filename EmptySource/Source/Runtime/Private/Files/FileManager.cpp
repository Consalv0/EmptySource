
#ifdef __APPLE__
#include <unistd.h>
#import <Foundation/Foundation.h>
#endif

#include "Engine/Core.h"
#include "Engine/Text.h"
#include "Files/FileManager.h"

#include <algorithm>
#include <stdlib.h>

namespace EmptySource {

	FileList FileManager::Files = FileList();

	FileStream* FileManager::GetFile(const WString & FilePath) {
		WString FullFilePath = GetFullPath(FilePath);
		FileList::iterator Found = FindInFiles(FullFilePath);

		if (Found != Files.end()) (*Found)->Reset();
		if (Found != Files.end() && (*Found)->IsValid()) return *Found;

		FileStream* NewStream = new FileStream(FullFilePath);

		if (!NewStream->IsValid()) return NULL;

		Files.push_back(NewStream);
		return NewStream;
	}

	FileStream * FileManager::MakeFile(const WString & FilePath) {
		std::fstream Stream;
#ifdef WIN32
		Stream.open(FilePath, std::ios::in | std::ios::out | std::ios::trunc);
#else
		Stream.open(WStringToString(FilePath), std::ios::in | std::ios::out | std::ios::binary | std::ios::trunc);
#endif
		Stream.close();

		return GetFile(FilePath);
	}

	WString FileManager::GetFileExtension(const WString & Path) {
		WString::size_type ExtensionIndex;

		ExtensionIndex = Path.rfind('.');

		if (ExtensionIndex != WString::npos) {
			return Path.substr(ExtensionIndex + 1);
		}
		else {
			return L"";
		}
	}

	WString FileManager::GetFullPath(const WString & Path) {
#ifdef WIN32
		WChar FullPath[MAX_PATH + 1];
		GetFullPathName(Path.c_str(), MAX_PATH, FullPath, NULL);
		return FullPath;
#elif __APPLE__
		WString ResourcesPath = Path;
		WString ResourcesFolderName = L"Resources/";
		Text::Replace(ResourcesPath, ResourcesFolderName, WString(L""));
		NSString * NSStringToResourcesPath = [[NSString alloc] initWithBytes:ResourcesPath.data()
			length : ResourcesPath.size() * sizeof(wchar_t)
			encoding : NSUTF32LittleEndianStringEncoding];
		NSString * FilePath = [[NSBundle mainBundle] pathForResource:NSStringToResourcesPath ofType : @""];
		NSStringEncoding Encode = CFStringConvertEncodingToNSStringEncoding(kCFStringEncodingUTF32LE);
		NSData * SData = [FilePath dataUsingEncoding : Encode];

		return WString((wchar_t*)[SData bytes], [SData length] / sizeof(wchar_t));
#else
		Char FullPath[PATH_MAX + 1];
		Char * Ptr;
		Ptr = realpath(WStringToString(Path).c_str(), FullPath);
		return CharToWChar(FullPath);
#endif
	}

	WString FileManager::GetAppDirectory() {
#ifdef ES_PLATFORM_WINDOWS
		WChar Buffer[MAX_PATH];
		GetCurrentDirectory(_MAX_DIR, Buffer);
		WString CurrentDirectory(Buffer);
#else
		NSStringEncoding Encode = CFStringConvertEncodingToNSStringEncoding(kCFStringEncodingUTF32LE);
		NSData * SData = [[[NSBundle mainBundle] resourcePath] dataUsingEncoding:Encode];

		WString CurrentDirectory((wchar_t*)[SData bytes], [SData length] / sizeof(wchar_t));
#endif
		return CurrentDirectory;
	}

	FileList::iterator FileManager::FindInFiles(const WString & FilePath) {
		return std::find_if(Files.begin(), Files.end(), [FilePath](FileStream* & File)
			-> bool { return File->GetPath() == FilePath; }
		);
	}

	WString FileManager::ReadStream(FileStream* Stream) {
		WString ShaderCode = L"";

		// ReadStreams the Vertex Shader code from the file
		if (Stream != NULL && Stream->IsValid()) {
			ShaderCode = Stream->ReadStream().str();
			Stream->Close();
		}

		return ShaderCode;
	}

}