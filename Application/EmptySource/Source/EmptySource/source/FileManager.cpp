
#include <algorithm>
#include "..\include\Core.h"
#include "..\include\FileManager.h"

FileList FileManager::Files = FileList();

FileStream* FileManager::Open(const WString & FilePath) {
	FileList::iterator Found = std::find_if(Files.begin(), Files.end(), [FilePath](FileStream* & File)
		-> bool { return File->GetPath() == FilePath; }
	);
	
	if (Found != Files.end() && (*Found)->IsValid()) return *Found;

	FileStream* NewStream = new FileStream(FilePath);

	if (!NewStream->IsValid()) return NULL;

	Files.push_back(NewStream);
	return NewStream;
}
