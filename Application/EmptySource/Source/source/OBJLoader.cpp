
#include <cctype>
#include <cstdio>

#include "../include/Core.h"
#include "../include/Utility/Timer.h"
#include "../include/FileStream.h"
#include "../include/OBJLoader.h"


// --- Visual Studio
#ifdef _MSC_VER 
#define strtok_r strtok_s
#if (_MSC_VER >= 1310) 
/*VS does not like fopen, but fopen_s is not standard C so unusable here*/
#pragma warning( disable : 4996 ) 
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
	else if (**Character == '+')++*Character;

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
	}
	else {
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

void OBJLoader::ParseVertexPositions(FileData & Data) {
	for (TArray<const Char *>::const_iterator Line = Data.LinePositions.begin(); Line != Data.LinePositions.end(); ++Line) {
		const size_t Pos = Line - Data.LinePositions.begin();
		ExtractVector3((*Line) + 1, &Data.ListPositions[Pos]);
	}
}

void OBJLoader::ParseVertexNormals(FileData & Data) {
	for (TArray<const Char *>::const_iterator Line = Data.LineNormals.begin(); Line != Data.LineNormals.end(); ++Line) {
		const size_t Pos = Line - Data.LineNormals.begin();
		ExtractVector3((*Line) + 1, &Data.ListNormals[Pos]);
	}
}

void OBJLoader::ParseVertexUVs(FileData & Data) {
	for (TArray<const Char *>::const_iterator Line = Data.LineUVs.begin(); Line != Data.LineUVs.end(); ++Line) {
		const size_t Pos = Line - Data.LineUVs.begin();
		ExtractVector2((*Line) + 1, &Data.ListUVs[Pos]);
	}
}

void OBJLoader::ParseFaces(FileData & Data) {
	// 0 = Vertex, 1 = TextureCoords, 2 = Normal
	IntVector3 VertexIndex = IntVector3(-1);
	bool bWarned = false;
	auto CurrentObject = Data.Objects.begin();
	int VertexCount = 0;
	const Char * LineChar;
	for (TArray<const Char *>::const_iterator Line = Data.LineVertexIndices.begin(); Line != Data.LineVertexIndices.end(); ++Line) {
		if ((CurrentObject + 1) != Data.Objects.end() && Line - Data.LineVertexIndices.begin() == (CurrentObject + 1)->VertexIndicesPos)
			++CurrentObject;
		VertexIndex[0] = VertexIndex[1] = VertexIndex[2] = -1;
		VertexCount = 0;
		LineChar = *Line;
		while (*LineChar != '\0' && *LineChar != '\n') {
			VertexIndex[0] = atoi(++LineChar);
			while (*LineChar != '/' && *LineChar != ' ') ++LineChar;
			if (*LineChar != ' ') {
				VertexIndex[1] = atoi(++LineChar);
				while (*LineChar != '/' && *LineChar != ' ') ++LineChar;
				if (*LineChar != ' ') {
					VertexIndex[2] = atoi(++LineChar);
					while (!isspace(*++LineChar));
				}
			}
			++VertexCount;

			if (VertexCount == 4) {
				Data.VertexIndices.push_back(Data.VertexIndices[Data.VertexIndices.size() - 3]);
				Data.VertexIndices.push_back(Data.VertexIndices[Data.VertexIndices.size() - 2]);
				Data.VertexIndices.push_back(VertexIndex);
				CurrentObject->VertexIndicesCount += 3;
			}
			else {
				Data.VertexIndices.push_back(VertexIndex);
				CurrentObject->VertexIndicesCount++;
			}

			if (VertexCount > 4 && !bWarned) {
				bWarned = true;
				Debug::Log(Debug::LogWarning, L"The model has n-gons, this may lead to unwanted geometry");
			}
		}
	}
}

OBJLoader::Keyword OBJLoader::GetKeyword(const Char * Word) {
	if (Word + 1 == NULL || Word + 2 == NULL) return Undefined;

	if (Word[0] == 'f') return Face;
	if (Word[0] == 'v') {
		if (Word[1] == ' ') return Vertex;
		if (Word[1] == 'n' && Word[2] == ' ') return Normal;
		if (Word[1] == 't' && Word[2] == ' ') return TextureCoord;
	}
	if (Word[0] == '#' && Word[1] == ' ') return Comment;
	if (Word[0] == 'o' && Word[1] == ' ') return Object;
	if (Word[0] == 'g' && Word[1] == ' ') return Group;
	if (Word[0] == 'c') return CSType;

	return Undefined;
}

void OBJLoader::PrepareData(const Char * InFile, FileData& ModelData) {
	const Char* Pointer = InFile;
	int VertexCount = 0;
	int NormalCount = 0;
	int UVCount = 0;
	int FaceCount = 0;
	Keyword Keyword;

	while (*Pointer != '\0') {
		Keyword = GetKeyword(Pointer);
		if (Keyword == Undefined) {
			while (*Pointer != '\n' && *Pointer != '\0') ++Pointer;
			++Pointer;
			continue;
		};
		if (Keyword == Comment) {
			while (*Pointer != '\n' && *Pointer != '\0') ++Pointer;
			++Pointer;
			continue;
		};
		if (Keyword == Vertex) {
			ModelData.LinePositions.push_back(++Pointer);
			while (*Pointer != '\n' && *Pointer != '\0') ++Pointer;
			++Pointer;
			VertexCount++; continue;
		};
		if (Keyword == Normal) {
			ModelData.LineNormals.push_back(Pointer += 2);
			while (*Pointer != '\n' && *Pointer != '\0') ++Pointer;
			++Pointer;
			NormalCount++; continue;
		};
		if (Keyword == TextureCoord) {
			ModelData.LineUVs.push_back(Pointer += 2);
			while (*Pointer != '\n' && *Pointer != '\0') ++Pointer;
			++Pointer;
			UVCount++;     continue;
		};
		if (Keyword == Face) {
			if (ModelData.Objects.size() == 0) {
				ModelData.Objects.push_back(ObjectData());
				if (ModelData.Groups.size() > 0) {
					ModelData.Objects.back().Name += ModelData.Groups.back();
				}
			}
			ModelData.LineVertexIndices.push_back(++Pointer);
			while (*Pointer != '\n' && *Pointer != '\0') ++Pointer;
			++Pointer;
			FaceCount++;   continue;
		};
		if (Keyword == Group) {
			ModelData.Groups.push_back(String());
			++Pointer;
			while (*(Pointer + 1) != '\n' && *(Pointer + 1) != '\0') {
				++Pointer;
				ModelData.Groups.back() += *Pointer;
			}
			continue;
		}
		if (Keyword == Object) {
			ModelData.Objects.push_back(ObjectData());
			ModelData.Objects.back().VertexIndicesPos = (int)ModelData.LineVertexIndices.size();
			if (ModelData.Groups.size() > 0) {
				ModelData.Objects.back().Name += ModelData.Groups.back() + ".";
			}
			++Pointer;
			while (*(Pointer+1) != '\n' && *(Pointer+1) != '\0') {
				++Pointer;
				ModelData.Objects.back().Name += *Pointer;
			}
			continue;
		}

		++Pointer;
	}

	ModelData.ListPositions.resize(VertexCount);
	ModelData.ListNormals.resize(NormalCount);
	ModelData.ListUVs.resize(UVCount);
	ModelData.VertexIndices.reserve(FaceCount * 4);
}

bool OBJLoader::Load(FileStream * File, MeshLoader::FileData & OutData, bool hasOptimize) {
	if (File == NULL || !File->IsValid()) return false;

	FileData ModelData;

	// --- Read File
	{
		Debug::Timer Timer;

		Timer.Start();
		String* MemoryText = new String();
		File->ReadNarrowStream(MemoryText);

		PrepareData(MemoryText->c_str(), ModelData);
		ParseFaces(ModelData);
		ParseVertexPositions(ModelData);
		ParseVertexNormals(ModelData);
		ParseVertexUVs(ModelData);

		ModelData.LineNormals.clear();
		ModelData.LinePositions.clear();
		ModelData.LineUVs.clear();
		ModelData.LineVertexIndices.clear();

		delete MemoryText;

		Timer.Stop();
		Debug::Log(Debug::LogInfo,
			L"├> Parsed %ls vertices and %ls triangles in %.3fs",
			Text::FormatUnit(ModelData.VertexIndices.size(), 2).c_str(),
			Text::FormatUnit(ModelData.VertexIndices.size() / 3, 2).c_str(),
			Timer.GetEnlapsedSeconds()
		);
	}

	int * Indices = new int[ModelData.VertexIndices.size()];
	int Count = 0;

	size_t TotalAllocatedSize = 0;
	size_t TotalUniqueVertices = 0;

	Debug::Timer Timer;
	Timer.Start();
	for (int ObjectCount = 0; ObjectCount < ModelData.Objects.size(); ++ObjectCount) {
		ObjectData & Data = ModelData.Objects[ObjectCount];

		TDictionary<MeshVertex, unsigned> VertexToIndex;
		VertexToIndex.reserve(Data.VertexIndicesCount);

		OutData.Meshes.push_back(MeshData());
		MeshData* OutMesh = &OutData.Meshes.back();
		OutMesh->Name = StringToWString(Data.Name);

		int InitialCount = Count;
		for (; Count < InitialCount + Data.VertexIndicesCount; ++Count) {
			MeshVertex NewVertex = {
				ModelData.VertexIndices[Count][0] >= 0 ?
					ModelData.ListPositions[ModelData.VertexIndices[Count][0] - 1] : 0,
				ModelData.VertexIndices[Count][2] >= 0 ?
					ModelData.ListNormals[ModelData.VertexIndices[Count][2] - 1] : Vector3(0.F, 1.F, 0.F),
				0,
				ModelData.VertexIndices[Count][1] >= 0 ?
					ModelData.ListUVs[ModelData.VertexIndices[Count][1] - 1] : 0,
				ModelData.VertexIndices[Count][1] >= 0 ?
					ModelData.ListUVs[ModelData.VertexIndices[Count][1] - 1] : 0,
				Vector4(1.F)
			};
			Data.Bounding.Add(NewVertex.Position);

			unsigned Index = Count;
			bool bFoundIndex = false;
			if (hasOptimize) {
				bFoundIndex = GetSimilarVertexIndex(NewVertex, VertexToIndex, Index);
			}

			if (bFoundIndex) {
				// --- A similar vertex is already in the VBO, use it instead !
				Indices[Count] = Index;
			}
			else {
				// --- If not, it needs to be added in the output data.
				OutMesh->Vertices.push_back(NewVertex);
				unsigned NewIndex = (unsigned)OutMesh->Vertices.size() - 1;
				Indices[Count] = NewIndex;
				if (hasOptimize) VertexToIndex[NewVertex] = NewIndex;
				TotalUniqueVertices++;
			}

			if ((Count + 1) % 3 == 0) {
				OutMesh->Faces.push_back({ Indices[Count - 2], Indices[Count - 1], Indices[Count] });
			}
		}

		if (ModelData.VertexIndices[Count-1][1] >= 0) OutMesh->TextureCoordsCount = 1;
		if (ModelData.VertexIndices[Count-1][2] >= 0) OutMesh->hasNormals = true;
		OutMesh->ComputeTangents();
		OutMesh->Bounding = Data.Bounding;
		OutMesh->hasBoundingBox = true;

#ifdef _DEBUG
		// Debug::LogClearLine(Debug::LogInfo);
		Debug::Log(
			Debug::LogInfo,
			L"├> Parsed %ls	vertices in %ls	at [%d]'%ls'",
			Text::FormatUnit(Data.VertexIndicesCount, 2).c_str(),
			Text::FormatData(sizeof(IntVector3) * OutMesh->Faces.size() + sizeof(MeshVertex) * OutMesh->Vertices.size(), 2).c_str(),
			OutData.Meshes.size(),
			OutMesh->Name.c_str()
		);
#endif // _DEBUG

		TotalAllocatedSize += sizeof(IntVector3) * OutMesh->Faces.size() + sizeof(MeshVertex) * OutMesh->Vertices.size();
	}

	ModelData.ListNormals.clear();
	ModelData.ListPositions.clear();
	ModelData.ListUVs.clear();
	ModelData.VertexIndices.clear();

	delete[] Indices;

	Timer.Stop();
	Debug::Log(Debug::LogInfo, L"└> Allocated %ls in %.2fs", Text::FormatData(TotalAllocatedSize, 2).c_str(), Timer.GetEnlapsedSeconds());

	return true;
}
