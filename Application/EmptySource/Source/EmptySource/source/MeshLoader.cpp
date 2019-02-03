
#include <cstdio>
#include <string>

#include "..\include\Core.h"
#include "..\include\Utility\Timer.h"
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
	Vector->x = (float)StringToDouble(Text, &LineState);
	Vector->y = (float)StringToDouble(LineState, &LineState);
	Vector->z = (float)StringToDouble(LineState, &LineState);
}

void MeshLoader::ExtractVector2(const Char * Text, Vector2* Vector) {
	Char* LineState;
	Vector->x = (float)StringToDouble(Text, &LineState);
	Vector->y = (float)StringToDouble(LineState, &LineState);
}

void MeshLoader::ExtractIntVector3(const Char * Text, IntVector3 * Vector) {
	Char* LineState;
	Vector->x = (int)StringToDouble(Text, &LineState);
	Vector->y = (int)StringToDouble(LineState, &LineState);
	Vector->z = (int)StringToDouble(LineState, NULL);
}

void MeshLoader::ParseOBJLine(
	const OBJKeyword& Keyword,
	Char * Line,
	OBJFileData& ModelData,
	int ObjectCount)
{
	bool bWarned = false;
	OBJObjectData* Data = &ModelData.Objects[ObjectCount];

	if (Keyword == CSType && !bWarned) {
		bWarned = true;
		Debug::Log(Debug::NoLog, L"\r");
		Debug::Log(Debug::LogWarning, L"The object contains free-form geometry, this is not supported");
		return;
	}
		
	if (Keyword == Object) {
		Data->Name = Line;
		return;
	}
		
	if (Keyword == Vertex) {
		ExtractVector3(Line, &ModelData.ListPositions[ModelData.PositionCount++]);
		Data->PositionCount++;
		// Debug::Log(Debug::LogDebug, L"Vertex %s", Vert.ToString().c_str());
		return;
	}
		
	if (Keyword == Normal) {
		ExtractVector3(Line, &ModelData.ListNormals[ModelData.NormalCount++]);
		Data->NormalCount++;
		// Debug::Log(Debug::LogDebug, L"Normal %s", Normal.ToString().c_str());
		return;
	}
		
	if (Keyword == TextureCoord) {
		ExtractVector2(Line, &ModelData.ListUVs[ModelData.UVsCount++]);
		Data->UVsCount++;
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

			if (Data->UVsCount <= 0) {
				if (Data->NormalCount <= 0) {
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
			} else if (Data->NormalCount <= 0) {
				if (Data->UVsCount <= 0) {
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
				ModelData.VertexIndices.push_back(ModelData.VertexIndices[ModelData.VertexIndices.size() - 3]);
				ModelData.VertexIndices.push_back(ModelData.VertexIndices[ModelData.VertexIndices.size() - 2]);
				ModelData.VertexIndices.push_back(VertexIndex);
				Data->VertexIndicesCount += 3;
				ModelData.VertexIndicesCount += 3;
			} else {
				ModelData.VertexIndices.push_back(VertexIndex);
				Data->VertexIndicesCount++;
				ModelData.VertexIndicesCount++;
			}
		}

		if (VertexCount > 4 && !bWarned) {
			bWarned = true;
			Debug::Log(Debug::NoLog, L"\r");
			Debug::Log(Debug::LogWarning, L"The model has n-gons, this may lead to unwanted geometry");
		}

		return;
	}
}

MeshLoader::OBJKeyword MeshLoader::GetOBJKeyword(const Char * Word) {
	if (Word[0] == 'f' ) return Face;
	if (Word[0] == 'v') {
		if (Word[1] == ' ') return Vertex;
		if (Word[1] == 'n' && Word[2] == ' ') return Normal;
		if (Word[1] == 't' && Word[2] == ' ') return TextureCoord;
	}
	if (Word[0] == '#' && Word[1] == ' ') return Comment;
	if (Word[0] == 'o' && Word[1] == ' ') return Object;
	if (Word[0] == 'c') return CSType;

	return Undefined;
}

void MeshLoader::ReadOBJByLine(
	const Char * InFile,
	OBJFileData& ModelData)
{
	const size_t LogCountBottleNeck = 86273;
	size_t LogCount = 1;
	size_t CharacterCount = 0;
	size_t LastSplitPosition = 0;
	size_t MaxCharacterCount = std::strlen(InFile);
	const Char* Pointer = InFile;
	size_t LineCount = 0;
	int ObjectCount = 0;

	while (CharacterCount <= MaxCharacterCount) {
		CharacterCount++;
		if (std::isspace(*(Pointer++), Locale)) {
			size_t KeySize = CharacterCount - LastSplitPosition;
			Char* Key = (Char*)&InFile[LastSplitPosition++];
			LastSplitPosition = CharacterCount;

			while (CharacterCount <= MaxCharacterCount) {
				CharacterCount++;
				if (*(Pointer++) == '\n') {
					size_t LineSize = CharacterCount - LastSplitPosition;
					Char* Line = (Char*)&InFile[LastSplitPosition];
					Line[LineSize - 1] = '\0';
					
					OBJKeyword LineKey = GetOBJKeyword(Key);
					if (LineKey == Object) {
						++ObjectCount;
					}

					ParseOBJLine(
						LineKey, Line, ModelData, ObjectCount == 0 ? ObjectCount : ObjectCount - 1
					);

					LineCount++;
					LastSplitPosition = CharacterCount;
					break;
				}
			}
		}

		float Progress = CharacterCount / float(MaxCharacterCount);
		if (--LogCount <= 0) {
			LogCount = LogCountBottleNeck;
			float cur = std::ceil(Progress * 25);
			Debug::Log(Debug::NoLog, L"\r [%s%s] %.2f%% %s lines",
				WString(int(cur), L'#').c_str(), WString(int(25 + 1 - cur), L' ').c_str(),
				50 * Progress, Text::FormattedUnit(LineCount, 2).c_str()
			);
		}
		
		if (CharacterCount == MaxCharacterCount) {
			Debug::Log(Debug::NoLog, L"\r");
			Debug::Log(Debug::LogNormal, L"├> [%s] %.2f%% %d lines", WString(25, L'#').c_str(), 50 * Progress, LineCount);
		}
	}
}

void MeshLoader::PrepareOBJData(const Char * InFile, OBJFileData& ModelData) {
	const Char* Pointer = InFile;
	int VertexCount = 0;
	int NormalCount = 0;
	int UVCount = 0;
	int FaceCount = 0;
	int ObjectCount = 0;
	OBJKeyword Keyword;

	while (*Pointer != '\0') {
		Keyword = GetOBJKeyword(Pointer);
		if (Keyword == Comment) {
			while (*Pointer != '\n' && *Pointer != '\0') ++Pointer;
			++Pointer;
			continue;
		};
		if (Keyword == Vertex) {
			while (*Pointer != '\n' && *Pointer != '\0') ++Pointer;
			++Pointer;
			VertexCount++; continue;
		};
		if (Keyword == Normal) {
			while (*Pointer != '\n' && *Pointer != '\0') ++Pointer;
			++Pointer;
			NormalCount++; continue;
		};
		if (Keyword == TextureCoord) {
			while (*Pointer != '\n' && *Pointer != '\0') ++Pointer;
			++Pointer;
			UVCount++;     continue;
		};
		if (Keyword == Face) {
			while (*Pointer != '\n' && *Pointer != '\0') ++Pointer;
			++Pointer;
			FaceCount++;   continue;
		};
		if (Keyword == Object) {
			while (*Pointer != '\n' && *Pointer != '\0') ++Pointer;
			if (ObjectCount++ <= 0) continue;

			ModelData.Objects.push_back(OBJObjectData());
			continue;
		}

		++Pointer;
	}
	
	ModelData.Objects.push_back(OBJObjectData());
	ModelData.ListPositions.resize(VertexCount);
	ModelData.ListNormals.resize(NormalCount);
	ModelData.ListUVs.resize(UVCount);
	ModelData.VertexIndices.reserve(FaceCount * 4);
}

bool MeshLoader::FromOBJ(FileStream * File, std::vector<MeshFaces> * Faces, std::vector<MeshVertices> * Vertices, bool hasOptimize) {
	if (File == NULL || !File->IsValid()) return false;

	OBJFileData ModelData;
	int VertexIndexCount = 0;

	{
		bool bWarned = false;
		Debug::Timer Timer;

		Debug::Log(Debug::LogNormal, L"Parsing File Model '%s'", File->GetShortPath().c_str());
		
		Timer.Start();
		String* MemoryText = new String();
		File->ReadNarrowStream(MemoryText);

		String KeyWord;
		long LineCount = 0;

		PrepareOBJData(MemoryText->c_str(), ModelData);
		ReadOBJByLine(MemoryText->c_str(), ModelData);
		delete MemoryText;

		Timer.Stop();
		VertexIndexCount = ModelData.VertexIndicesCount;
		Debug::Log(Debug::LogNormal,
			L"├> Parsed %s vertices and %s triangles in %.3fs",
			Text::FormattedUnit(VertexIndexCount, 2).c_str(),
			Text::FormattedUnit(VertexIndexCount / 3, 2).c_str(),
			Timer.GetEnlapsedSeconds()
		);
	}

	std::unordered_map<MeshVertex, unsigned> VertexToIndex;
	ModelData.VertexIndices.shrink_to_fit();
	VertexToIndex.reserve(VertexIndexCount);
	int* Indices = new int[VertexIndexCount];

	const size_t LogCountBottleNeck = 36273;
	size_t LogCount = 1; 
	int Count = 0;

	Vertices->clear();
	Faces->clear();
	size_t TotalAllocatedSize = 0;

	Debug::Timer Timer;
	Timer.Start();
	for (int ObjectCount = 0; ObjectCount < ModelData.Objects.size(); ++ObjectCount) {
		OBJObjectData* Data = &ModelData.Objects[ObjectCount];
		Vertices->push_back(MeshVertices());
		Faces->push_back(MeshFaces());
		int InitialCount = Count;
		
		for (; Count < InitialCount + Data->VertexIndicesCount; ++Count) {

			if (--LogCount <= 0) {
				float prog = Count / float(ModelData.VertexIndicesCount);
				float prog2 = (Count - InitialCount) / float(Data->VertexIndicesCount);
				LogCount = LogCountBottleNeck;
				float cur = std::ceil(prog * 12);
				float cur2 = std::ceil(prog2 * 12);
				Debug::Log(Debug::NoLog, L"\r %s : [%s%s|%s%s] %.2f%% %s vertices",
					StringToWString(Data->Name).c_str(),
					WString(int(cur2), L'%').c_str(), WString(int(12 + 1 - cur2), L' ').c_str(),
					WString(int(cur), L'#').c_str(), WString(int(12 + 1 - cur), L' ').c_str(),
					100 * prog, Text::FormattedUnit(Count, 2).c_str()
				);
			}

			MeshVertex NewVertex = {
				Data->PositionCount > 0 ?
					ModelData.ListPositions[ModelData.VertexIndices[Count][0] - 1] : 0,
				Data->NormalCount > 0 ?
					ModelData.ListNormals[ModelData.VertexIndices[Count][2] - 1] : Vector3(0.3F, 0.3F, 0.4F),
				0,
				Data->UVsCount > 0 ?
					ModelData.ListUVs[ModelData.VertexIndices[Count][1] - 1] : 0,
				Data->UVsCount > 0 ?
					ModelData.ListUVs[ModelData.VertexIndices[Count][1] - 1] : 0,
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
				Vertices->back().push_back(NewVertex);
				unsigned NewIndex = (unsigned)Vertices->back().size() - 1;
				Indices[Count] = NewIndex;
				if (hasOptimize) VertexToIndex[NewVertex] = NewIndex;
			}

			if ((Count + 1) % 3 == 0) {
				Faces->back().push_back({ Indices[Count - 2], Indices[Count - 1], Indices[Count] });
				// _LOG(LogDebug, L"Face {%d, %d, %d}",
				// 	Faces->back()[0], Faces->back()[1], Faces->back()[2]
				// );
			}
		}

		Debug::Log(Debug::NoLog, L"\r");
		Debug::Log(
			Debug::LogNormal,
			L"├> Parsed %s	vertices in %s	at [%d]'%s'",
			Text::FormattedUnit(Data->VertexIndicesCount, 2).c_str(),
			Text::FormattedData(sizeof(IntVector3) * Faces->back().size() + sizeof(MeshVertex) * Vertices->back().size(), 2).c_str(),
			Vertices->size(),
			StringToWString(Data->Name).c_str()
		);

		TotalAllocatedSize += sizeof(IntVector3) * Faces->back().size() + sizeof(MeshVertex) * Vertices->back().size();
	}

	Debug::Log(Debug::NoLog, L"\r");
	Debug::Log(Debug::LogNormal, L"├> [%s] 100.00%% %s vertices", WString(25, L'#').c_str(), Text::FormattedUnit(VertexIndexCount, 2).c_str());

	Timer.Stop();
	Debug::Log(Debug::LogNormal, L"└> Allocated %s in %.2fs", Text::FormattedData(TotalAllocatedSize, 2).c_str(), Timer.GetEnlapsedSeconds());

	return true;
}
