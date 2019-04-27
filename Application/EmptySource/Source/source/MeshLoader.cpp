
#include <cctype>
#include <cstdio>

#include "../include/Core.h"
#include "../include/Utility/Timer.h"
#include "../include/Math/CoreMath.h"
#include "../include/MeshLoader.h"
#include "../include/FileStream.h"

// --- Visual Studio
#ifdef _MSC_VER 
#define strtok_r strtok_s
#if (_MSC_VER >= 1310) 
#pragma warning( disable : 4996 ) /*VS does not like fopen, but fopen_s is not standard C so unusable here*/
#endif
#endif

// Original crack_atof version is at http://crackprogramming.blogspot.sg/2012/10/implement-atof.html
// But it cannot convert floating point with high +/- exponent.
// The version below by Tian Bo fixes that problem and improves performance by 10%
// http://coliru.stacked-crooked.com/a/2e28f0d71f47ca5e
inline float fast_strtof(const char* String, char** Character) {

	*Character = (char*)String;
	if (!*Character || !**Character)
		return 0;

	int Sign = 1;
	float IntPart = 0.0;
	float FractionPart = 0.0;
	bool hasFraction = false;
	bool hasExpo = false;

	// Take care of +/- sign
	if (**Character == '-') { ++*Character; Sign = -1; }
	else if (**Character == '+') ++*Character;

	while (**Character != '\0' && **Character != ',' && **Character != ' ') {
		if (**Character >= '0' && **Character <= '9')
			IntPart = IntPart * 10 + (**Character - '0');
		else if (**Character == '.') {
			hasFraction = true;
			++*Character; break;
		}
		else if (**Character == 'e') {
			hasExpo = true;
			++*Character; break;
		}
		else
			return Sign * IntPart;

		++*Character;
	}

	if (hasFraction) {
		float FractionExponential = 0.1F;

		while (**Character != '\0' && **Character != ',' && **Character != ' ') {
			if (**Character >= '0' && **Character <= '9') {
				FractionPart += FractionExponential * (**Character - '0');
				FractionExponential *= 0.1F;
			}
			else if (**Character == 'e') {
				hasExpo = true;
				++*Character;
				break;
			}
			else {
				return Sign * (IntPart + FractionPart);
			}
			++*Character;
		}
	}

	float ExponentPart = 1.0F;
	if ((**Character != '\0' && **Character != ',' && **Character != ' ') && hasExpo) {
		int ExponentSign = 1;
		if (**Character == '-') {
			ExponentSign = -1;
			++*Character;
		}
		else if (**Character == '+') {
			++*Character;
		}

		int e = 0;
		while ((**Character != '\0' && **Character != ',' && **Character != ' ') && **Character >= '0' && **Character <= '9') {
			e = e * 10 + **Character - '0';
			++*Character;
		}

		ExponentPart = Math::Pow10(ExponentSign * e);
	}
	++*Character;

	return Sign * (IntPart + FractionPart) * ExponentPart;
}

bool OBJLoader::GetSimilarVertexIndex(const MeshVertex & Vertex, TDictionary<MeshVertex, unsigned>& VertexToIndex, unsigned & Result) {
	TDictionary<MeshVertex, unsigned>::iterator it = VertexToIndex.find(Vertex);
	if (it == VertexToIndex.end()) {
		return false;
	} else {
		Result = it->second;
		return true;
	}
}

void OBJLoader::ExtractVector3(const Char * Text, Vector3* Vector) {
	Char* LineState;
	Vector->x = fast_strtof(Text, &LineState);
	Vector->y = fast_strtof(LineState, &LineState);
	Vector->z = fast_strtof(LineState, &LineState);
}

void OBJLoader::ExtractVector2(const Char * Text, Vector2* Vector) {
	Char* LineState;
	Vector->x = fast_strtof(Text, &LineState);
	Vector->y = fast_strtof(LineState, &LineState);
}

void OBJLoader::ExtractIntVector3(const Char * Text, IntVector3 * Vector) {
	Char* LineState;
	Vector->x = (int)fast_strtof(Text, &LineState);
	Vector->y = (int)fast_strtof(LineState, &LineState);
	Vector->z = (int)fast_strtof(LineState, NULL);
}

void OBJLoader::ParseLine(
	const Keyword& Keyword,
	Char * Line,
	FileData& ModelData,
	int ObjectCount)
{
	bool bWarned = false;
	ObjectData* ObjectData = &ModelData.Objects[ObjectCount];

	if (Keyword == CSType && !bWarned) {
		bWarned = true;
		Debug::LogClearLine(Debug::LogWarning);
		Debug::Log(Debug::LogWarning, L"The object contains free-form geometry, this is not supported");
		return;
	}
		
	if (Keyword == Object) {
		ObjectData->Name = Line;
		return;
	}
		
	if (Keyword == Vertex) {
		ExtractVector3(Line, &ModelData.ListPositions[ModelData.PositionCount++]);
		ObjectData->PositionCount++;
		return;
	}
		
	if (Keyword == Normal) {
		ExtractVector3(Line, &ModelData.ListNormals[ModelData.NormalCount++]);
		ObjectData->NormalCount++;
		return;
	}
		
	if (Keyword == TextureCoord) {
		ExtractVector2(Line, &ModelData.ListUVs[ModelData.UVsCount++]);
		ObjectData->UVsCount++;
		return;
	}
		
	if (Keyword == Face) {
		// 0 = Vertex, 1 = TextureCoords, 2 = Normal
		IntVector3 VertexIndex = IntVector3(1);
		Char *LineState, *Token;
        Token = strtok_r(Line, " ", &LineState);

		int VertexCount = 0;

		while (Token != NULL) {
			int Empty;

			if (ObjectData->UVsCount <= 0) {
				if (ObjectData->NormalCount <= 0) {
                    Empty = sscanf(Token, "%d", &VertexIndex[0]);
				} else {
                    Empty = sscanf(Token, "%d//%d", &VertexIndex[0], &VertexIndex[2]);
				}
			} else if (ObjectData->NormalCount <= 0) {
				if (ObjectData->UVsCount <= 0) {
                    Empty = sscanf(Token, "%d", &VertexIndex[0]);
                } else {
                    Empty = sscanf(Token, "%d/%d", &VertexIndex[0], &VertexIndex[1]);
                }
            } else {
                Empty = sscanf(Token, "%d/%d/%d",
                    &VertexIndex[0],
                    &VertexIndex[1],
                    &VertexIndex[2]
                );
			}
            Token = strtok_r(NULL, " ", &LineState);

			if (VertexIndex[0] < 0 && !bWarned) {
				bWarned = true;
				Debug::LogClearLine(Debug::LogWarning);
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
				ObjectData->VertexIndicesCount += 3;
				ModelData.VertexIndicesCount += 3;
			} else {
				ModelData.VertexIndices.push_back(VertexIndex);
				ObjectData->VertexIndicesCount++;
				ModelData.VertexIndicesCount++;
			}
		}

		if (VertexCount > 4 && !bWarned) {
			bWarned = true;
			Debug::LogClearLine(Debug::LogWarning);
			Debug::Log(Debug::LogWarning, L"The model has n-gons, this may lead to unwanted geometry");
		}

		return;
	}
}

OBJLoader::Keyword OBJLoader::GetKeyword(const Char * Word) {
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

size_t OBJLoader::ReadByLine(
	const Char * InFile,
	FileData& ModelData)
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
		if (std::isspace(*(Pointer++))) {
			Char* Key = (Char*)&InFile[LastSplitPosition++];
			LastSplitPosition = CharacterCount;

			while (CharacterCount <= MaxCharacterCount) {
				CharacterCount++;
				if (*(Pointer++) == '\n') {
					size_t LineSize = CharacterCount - LastSplitPosition;
					Char* Line = (Char*)&InFile[LastSplitPosition];
					Line[LineSize - 1] = '\0';
					
					Keyword LineKey = GetKeyword(Key);
					if (LineKey == Object) {
						++ObjectCount;
					}

					ParseLine(
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
			float cur = std::ceil(Progress * 12);
			Debug::LogUnadorned(Debug::LogInfo, L"\r [%ls%ls] %.2f%% %ls lines",
				WString(int(cur), L'#').c_str(), WString(int(25 + 1 - cur), L' ').c_str(),
				100 * Progress, Text::FormatUnit(LineCount, 2).c_str()
			);
		}
		
		if (CharacterCount == MaxCharacterCount) {
			Debug::LogClearLine(Debug::LogInfo);
		}
	}

	return LineCount;
}

void OBJLoader::PrepareData(const Char * InFile, FileData& ModelData) {
	const Char* Pointer = InFile;
	int VertexCount = 0;
	int NormalCount = 0;
	int UVCount = 0;
	int FaceCount = 0;
	int ObjectCount = 0;
	Keyword Keyword;

	while (*Pointer != '\0') {
		Keyword = GetKeyword(Pointer);
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

			ModelData.Objects.push_back(ObjectData());
			continue;
		}

		++Pointer;
	}
	
	ModelData.Objects.push_back(ObjectData());
	ModelData.ListPositions.resize(VertexCount);
	ModelData.ListNormals.resize(NormalCount);
	ModelData.ListUVs.resize(UVCount);
	ModelData.VertexIndices.reserve(FaceCount * 4);
}

bool OBJLoader::Load(FileStream * File, std::vector<MeshFaces> * Faces, std::vector<MeshVertices> * Vertices, bool hasOptimize) {
	if (File == NULL || !File->IsValid()) return false;

	FileData ModelData;
	int VertexIndexCount = 0;

	// --- Read File
	{
		Debug::Timer Timer;

		Debug::Log(Debug::LogInfo, L"Reading File Model '%ls'", File->GetShortPath().c_str());
		
		Timer.Start();
		String* MemoryText = new String();
		File->ReadNarrowStream(MemoryText);

		PrepareData(MemoryText->c_str(), ModelData);
		size_t LineCount = ReadByLine(MemoryText->c_str(), ModelData);
		delete MemoryText;

		Timer.Stop();
		VertexIndexCount = ModelData.VertexIndicesCount;
		Debug::Log(Debug::LogInfo,
			L"├> Readed %ls lines with %ls vertices and %ls triangles in %.3fs",
			Text::FormatUnit(LineCount, 0).c_str(),
			Text::FormatUnit(VertexIndexCount, 2).c_str(),
			Text::FormatUnit(VertexIndexCount / 3, 2).c_str(),
			Timer.GetEnlapsedSeconds()
		);
	}

	TDictionary<MeshVertex, unsigned> VertexToIndex;
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
		ObjectData* Data = &ModelData.Objects[ObjectCount];
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
				Debug::LogUnadorned(Debug::LogInfo, L"\r %ls : [%ls%ls|%ls%ls] %.2f%% %ls vertices",
					StringToWString(Data->Name).c_str(),
					WString(int(cur2), L'%').c_str(), WString(int(12 + 1 - cur2), L' ').c_str(),
					WString(int(cur), L'#').c_str(), WString(int(12 + 1 - cur), L' ').c_str(),
					100 * prog, Text::FormatUnit(Count, 2).c_str()
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

			if (bFoundIndex) { 
				// --- A similar vertex is already in the VBO, use it instead !
				Indices[Count] = Index;
			} else { 
				// --- If not, it needs to be added in the output data.
				Vertices->back().push_back(NewVertex);
				unsigned NewIndex = (unsigned)Vertices->back().size() - 1;
				Indices[Count] = NewIndex;
				if (hasOptimize) VertexToIndex[NewVertex] = NewIndex;
			}

			if ((Count + 1) % 3 == 0) {
				Faces->back().push_back({ Indices[Count - 2], Indices[Count - 1], Indices[Count] });
			}
		}

		Debug::LogClearLine(Debug::LogInfo);
		Debug::Log(
			Debug::LogInfo,
			L"├> Parsed %ls	vertices in %ls	at [%d]'%ls'",
			Text::FormatUnit(Data->VertexIndicesCount, 2).c_str(),
			Text::FormatData(sizeof(IntVector3) * Faces->back().size() + sizeof(MeshVertex) * Vertices->back().size(), 2).c_str(),
			Vertices->size(),
			StringToWString(Data->Name).c_str()
		);

		TotalAllocatedSize += sizeof(IntVector3) * Faces->back().size() + sizeof(MeshVertex) * Vertices->back().size();
		OBJLoader::ComputeTangents(Faces->back(), Vertices->back());
	}

	Debug::LogClearLine(Debug::LogInfo);
	Debug::Log(Debug::LogInfo, L"├> [%ls] 100.00%% %ls vertices", WString(25, L'#').c_str(), Text::FormatUnit(VertexIndexCount, 2).c_str());
	if (hasOptimize) {
		Debug::Log(
			Debug::LogInfo, L"├> Vertex optimization from %ls to %ls (%.2f%%)",
			Text::FormatUnit(VertexIndexCount, 2).c_str(), 
			Text::FormatUnit(VertexToIndex.size(), 2).c_str(),
			(float(VertexToIndex.size()) - VertexIndexCount) / VertexIndexCount * 100
		);
	}

	Timer.Stop();
	Debug::Log(Debug::LogInfo, L"└> Allocated %ls in %.2fs", Text::FormatData(TotalAllocatedSize, 2).c_str(), Timer.GetEnlapsedSeconds());

	return true;
}

void OBJLoader::ComputeTangents(const MeshFaces & Faces, MeshVertices & Vertices) {

	const int Size = (int)Faces.size();

	// For each triangle, compute the edge (DeltaPos) and the DeltaUV
	for (int i = 0; i < Size; ++i) {
		const Vector3 & VertexA = Vertices[Faces[i][0]].Position;
		const Vector3 & VertexB = Vertices[Faces[i][1]].Position;
		const Vector3 & VertexC = Vertices[Faces[i][2]].Position;

		const Vector2 & UVA = Vertices[Faces[i][0]].UV0;
		const Vector2 & UVB = Vertices[Faces[i][1]].UV0;
		const Vector2 & UVC = Vertices[Faces[i][2]].UV0;

		// --- Edges of the triangle : position delta
		const Vector3 Edge1 = VertexB - VertexA;
		const Vector3 Edge2 = VertexC - VertexA;

		// --- UV delta
		const Vector2 DeltaUV1 = UVB - UVA;
		const Vector2 DeltaUV2 = UVC - UVA;

		// --- We can now use our formula to compute the tangent :
		float r = 1.F / (DeltaUV1.x * DeltaUV2.y - DeltaUV1.y * DeltaUV2.x);
		      r = std::isfinite(r) ? r : 0;
		
		Vector3 Tangent;
		Tangent.x = r * (DeltaUV2.y * Edge1.x - DeltaUV1.y * Edge2.x);
		Tangent.y = r * (DeltaUV2.y * Edge1.y - DeltaUV1.y * Edge2.y);
		Tangent.z = r * (DeltaUV2.y * Edge1.z - DeltaUV1.y * Edge2.z);
		Tangent.Normalize();

		Vertices[Faces[i][0]].Tangent = Tangent;
		Vertices[Faces[i][1]].Tangent = Tangent;
		Vertices[Faces[i][2]].Tangent = Tangent;

		// Same thing for binormals
		// binormals[i * 3 + 0] = tan2[i].x; binormals[i * 3 + 1] = tan2[i].y; binormals[i * 3 + 2] = tan2[i].z;
	}
}
