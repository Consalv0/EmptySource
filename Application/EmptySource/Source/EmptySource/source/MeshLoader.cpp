#include <ctime>
#include <cstdio>
#include <string>

#include "..\include\Core.h"
#include "..\include\Math\Math.h"
#include "..\include\MeshLoader.h"
#include "..\include\FileStream.h"

std::locale MeshLoader::Locale = std::locale();

bool MeshLoader::GetSimilarVertexIndex(const MeshVertex & Vertex, std::unordered_map<MeshVertex, unsigned>& VertexToIndex, unsigned & Result) {
	std::unordered_map<MeshVertex, unsigned>::iterator it = VertexToIndex.find(Vertex);
	if (it == VertexToIndex.end()) {
		return false;
	} else {
		Result = it->second;
		return true;
	}
}

void MeshLoader::ExtractVector3(const Char * Text, Vector3* Vector) {
	Char* LineState;
	Vector->x = std::strtof(Text, &LineState);
	Vector->y = std::strtof(LineState, &LineState);
	Vector->z = std::strtof(LineState, NULL);
}

void MeshLoader::ExtractVector2(const Char * Text, Vector2* Vector) {
	Char* LineState;
	Vector->x = std::strtof(Text, &LineState);
	Vector->y = std::strtof(LineState, &LineState);
}

void MeshLoader::ExtractIntVector3(const Char * Text, IntVector3 * Vector) {
	Char* LineState;
	Vector->x = (int)std::strtof(Text, &LineState);
	Vector->y = (int)std::strtof(LineState, &LineState);
	Vector->z = (int)std::strtof(LineState, NULL);
}

void MeshLoader::ParseOBJLine(
	const OBJKeyword& Keyword,
	Char * Line,
	ParseData& Data)
{
	bool bWarned = false;

	if (Keyword == CSType && !bWarned) {
		bWarned = true;
		Debug::Log(Debug::NoLog, L"\r");
		Debug::Log(Debug::LogWarning, L"The object contains free-form geometry, this is not supported");
		return;
	}
		
	if (Keyword == Object) {
		Debug::Log(Debug::NoLog, L"\r");
		Debug::Log(Debug::LogNormal, L"├> Parsing Object: %s", CharToWChar(Line));
		return;
	}
		
	if (Keyword == Vertex) {
		ExtractVector3(Line, &Data.ListPositions[Data.PositionCount++]);
		// Debug::Log(Debug::LogDebug, L"Vertex %s", Vert.ToString().c_str());
		return;
	}
		
	if (Keyword == Normal) {
		ExtractVector3(Line, &Data.ListNormals[Data.NormalCount++]);
		// Debug::Log(Debug::LogDebug, L"Normal %s", Normal.ToString().c_str());
		return;
	}
		
	if (Keyword == TextureCoord) {
		ExtractVector2(Line, &Data.ListUVs[Data.UVsCount++]);
		// Debug::Log(Debug::LogDebug, L"UV %s", TexCoords.ToString().c_str());
		return;
	}
		
	if (Keyword == Face) {
		// 0 = Vertex, 1 = TextureCoords, 2 = Normal
		IntVector3 VertexIndex = IntVector3(1);
		Char *LineState, *Token;
		Token = strtok_s(Line, " ", &LineState);
		int VertexCount = 0;

		while (Token != NULL) {
			int Empty;

			if (Data.ListUVs.size() <= 0) {
				if (Data.ListNormals.size() <= 0) {
					Empty = sscanf_s(
						Token, "%d",
						&VertexIndex[0]
					);
				} else {
					Empty = sscanf_s(
						Token, "%d//%d",
						&VertexIndex[0],
						&VertexIndex[2]
					);
				}
			} else if (Data.ListNormals.size() <= 0) {
				if (Data.ListUVs.size() <= 0) {
					Empty = sscanf_s(
						Token, "%d",
						&VertexIndex[0]
					);
				} else {
					Empty = sscanf_s(
						Token, "%d/%d",
						&VertexIndex[0],
						&VertexIndex[1]
					);
				}
			} else {
				Empty = sscanf_s(
					Token, "%d/%d/%d",
					&VertexIndex[0],
					&VertexIndex[1],
					&VertexIndex[2]
				);
			}

			Token = strtok_s(NULL, " ", &LineState);
			if (VertexIndex[0] < 0 && !bWarned) {
				bWarned = true;
				Debug::Log(Debug::NoLog, L"\r");
				Debug::Log(Debug::LogWarning, L"The model contains negative references, this is not supported");
				continue;
			}

			if (Empty < 0) {
				continue;
			}

			VertexCount++;
			if (VertexCount == 4) {
				Data.VertexIndices.push_back(Data.VertexIndices[Data.VertexIndices.size() - 3]);
				Data.VertexIndices.push_back(Data.VertexIndices[Data.VertexIndices.size() - 2]);
				Data.VertexIndices.push_back(VertexIndex);
			} else {
				Data.VertexIndices.push_back(VertexIndex);
			}
		}

		if (VertexCount > 4 && !bWarned) {
			bWarned = true;
			Debug::Log(Debug::NoLog, L"\r");
			Debug::Log(Debug::LogWarning, L"The model has n-gons and this is no supported yet");
		}

		return;
	}
}

MeshLoader::OBJKeyword MeshLoader::GetOBJKeyword(const Char * Word) {
	OBJKeyword Keyword = Undefined;

	if (std::strcmp(Word, "v") == 0) Keyword = Vertex;
	if (Keyword == Undefined && std::strcmp(Word, "vn") == 0) Keyword = Normal;
	if (Keyword == Undefined && std::strcmp(Word, "vt") == 0) Keyword = TextureCoord;
	if (Keyword == Undefined && std::strcmp(Word, "f") == 0) Keyword = Face;
	if (Keyword == Undefined && std::strcmp(Word, "#") == 0) Keyword = Comment;
	if (Keyword == Undefined && std::strcmp(Word, "o") == 0) Keyword = Object;
	if (Keyword == Undefined && std::strcmp(Word, "cstype") == 0) Keyword = CSType;

	return Keyword;
}

void MeshLoader::ReadOBJByLine(
	const Char * InFile,
	ParseData& Data)
{
	const size_t LogCountBottleNeck = 86273;
	size_t LogCount = 1;
	size_t CharacterCount = 0;
	size_t LastSplitPosition = 0;
	size_t MaxCharacterCount = std::strlen(InFile);
	const Char* Pointer = InFile;
	size_t LineCount = 0;
	int VertexIndicesCount = 0;
	int PositionCount = 0;
	int NormalCount = 0;
	int UVsCount = 0;

	while (CharacterCount <= MaxCharacterCount) {
		CharacterCount++;
		if (std::isspace(*(Pointer++), Locale)) {
			size_t KeySize = CharacterCount - LastSplitPosition;
			Char* Key = new Char[KeySize + 1];
			std::copy(&InFile[LastSplitPosition], Pointer, Key);
			Key[KeySize - 1] = '\0';
			LastSplitPosition = CharacterCount;

			while (CharacterCount <= MaxCharacterCount) {
				CharacterCount++;
				if (*(Pointer++) == '\n') {
					size_t LineSize = CharacterCount - LastSplitPosition;
					Char* Line = new Char[LineSize + 1];
					std::copy(&InFile[LastSplitPosition], Pointer, Line);
					Line[LineSize - 1] = '\0';

					ParseOBJLine(
						GetOBJKeyword(Key), Line, Data
					);

					delete[] Line;
					LineCount++;
					LastSplitPosition = CharacterCount;
					break;
				}
			}
			delete[] Key;
		}

		float Progress = CharacterCount / float(MaxCharacterCount);
		if (--LogCount <= 0) {
			LogCount = LogCountBottleNeck;
			float cur = std::ceil(Progress * 25);
			Debug::Log(Debug::NoLog, L"\r [%s%s] %.2f%% %d lines", WString(int(cur), L'#').c_str(), WString(int(25 + 1 - cur), L' ').c_str(), 100 * Progress, LineCount);
		}
		
		if (CharacterCount == MaxCharacterCount) {
			Debug::Log(Debug::NoLog, L"\r");
			Debug::Log(Debug::LogNormal, L"├> [%s] %.2f%% %d lines", WString(25, L'#').c_str(), 100 * Progress, LineCount);
		}
	}
}

void MeshLoader::PrepareOBJData(const Char * InFile, ParseData& Data) {
	size_t CharacterCount = 0;
	size_t MaxCharacterCount = std::strlen(InFile);
	const Char* Pointer = InFile;
	int VertexCount = 0;
	int NormalCount = 0;
	int UVCount = 0;
	int FaceCount = 0;
	int ObjectCount = 0;

	while (CharacterCount <= MaxCharacterCount) {
		CharacterCount++;
		Pointer++;
		if (*Pointer == 'v' && *(Pointer + 1) == ' ') { VertexCount++; continue; };
		if (*Pointer == 'n') { NormalCount++; continue; };
		if (*Pointer == 't') { UVCount++;     continue; };
		if (*Pointer == 'f') { FaceCount++;   continue; };
	}

	Data.ListPositions.resize(VertexCount);
	Data.ListNormals.resize(NormalCount);
	Data.ListUVs.resize(UVCount);
	Data.VertexIndices.reserve(FaceCount * 4);
}

bool MeshLoader::FromOBJ(FileStream * File, MeshFaces * Faces, MeshVertices * Vertices, bool hasOptimize) {
	if (File == NULL || !File->IsValid()) return false;

	ParseData Data;

	{
		bool bWarned = false;
		clock_t StartTime = clock();
		Debug::Log(Debug::LogNormal, L"Parsing Model '%s'", File->GetShortPath().c_str());
		
		String* MemoryText = new String();
		File->ReadNarrowStream(MemoryText);

		String KeyWord;
		long LineCount = 0;

		PrepareOBJData(MemoryText->c_str(), Data);
		ReadOBJByLine(MemoryText->c_str(), Data);
		delete MemoryText;

		clock_t EndTime = clock();
		float TotalTime = float(EndTime - StartTime) / CLOCKS_PER_SEC;
		Debug::Log(Debug::LogNormal, L"├> Parsed %d vertices and %d triangles in %.3fs", Data.VertexIndices.size(), Data.VertexIndices.size() / 3, TotalTime);
	}

	std::unordered_map<MeshVertex, unsigned> VertexToIndex;
	Data.VertexIndices.shrink_to_fit();
	VertexToIndex.reserve(Data.VertexIndices.size());
	int* Indices = new int[Data.VertexIndices.size()];
	Faces->reserve(Data.VertexIndices.size() / 3);
	Vertices->reserve(Data.VertexIndices.size());

	const size_t LogCountBottleNeck = 86273;
	size_t LogCount = 1;

	clock_t StartTime = clock();
	for (unsigned int Count = 0; Count < Data.VertexIndices.size(); Count++) {

		float prog = Count / float(Data.VertexIndices.size());
		if (--LogCount <= 0) {
			LogCount = LogCountBottleNeck;
			float cur = std::ceil(prog * 25);
			Debug::Log(Debug::NoLog, L"\r [%s%s] %.2f%% %d vertices", WString(int(cur), L'#').c_str(), WString(int(25 + 1 - cur), L' ').c_str(), 100 * prog, Count);
		}

		MeshVertex NewVertex = {
			Data.ListPositions.size() > 0 ?
				Data.ListPositions[Data.VertexIndices[Count][0] - 1] : 0,
			Data.ListNormals.size() > 0 ?
				Data.ListNormals[Data.VertexIndices[Count][2] - 1] : Vector3(0.3F, 0.3F, 0.4F),
			0,
			Data.ListUVs.size() > 0 ?
				Data.ListUVs[Data.VertexIndices[Count][1] - 1] : 0,
			Data.ListUVs.size() > 0 ?
				Data.ListUVs[Data.VertexIndices[Count][1] - 1] : 0,
			1
		};
		// NewVertex.Color = Vector4(1);
		// NewVertex.Tangent = Vector3();

		unsigned Index = Count;
		bool bFoundIndex = false;
		if (hasOptimize) {
			bFoundIndex = GetSimilarVertexIndex(NewVertex, VertexToIndex, Index);
		}

		if (bFoundIndex) { // A similar vertex is already in the VBO, use it instead !
			Indices[Count] = Index;
		} else { // If not, it needs to be added in the output data.
			Vertices->push_back(NewVertex);
			unsigned NewIndex = (unsigned)Vertices->size() - 1;
			Indices[Count] = NewIndex;
			if (hasOptimize) VertexToIndex[NewVertex] = NewIndex;
		}

		if ((Count + 1) % 3 == 0) {
			Faces->push_back({ Indices[Count - 2], Indices[Count - 1], Indices[Count] });
			// _LOG(LogDebug, L"Face {%d, %d, %d}",
			// 	Faces->back()[0], Faces->back()[1], Faces->back()[2]
			// );
		}
	}

	Debug::Log(Debug::NoLog, L"\r");
	Debug::Log(Debug::LogNormal, L"├> [%s] 100.00%% %d vertices", WString(25, L'#').c_str(), Data.VertexIndices.size());

	clock_t EndTime = clock();
	float TotalTime = float(EndTime - StartTime) / CLOCKS_PER_SEC;
	size_t AllocatedSize = sizeof(IntVector3) * Faces->size() + sizeof(MeshVertex) * Vertices->size();
	Debug::Log(Debug::LogNormal, L"└> Allocated %.2fKB in %.2fs", AllocatedSize / 1024.F, TotalTime);

	return true;
}
