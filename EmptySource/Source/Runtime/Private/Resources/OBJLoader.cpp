
#include <cctype>
#include <cstdio>

#include "CoreMinimal.h"
#include "Utility/Timer.h"
#include "Files/FileStream.h"
#include "Resources/OBJLoader.h"


// --- Visual Studio
#ifdef _MSC_VER 
#define strtok_r strtok_s
#if (_MSC_VER >= 1310) 
/*VS does not like fopen, but fopen_s is not standard C so unusable here*/
#pragma warning( disable : 4996 ) 
#endif
#endif

namespace EmptySource {

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

	void OBJLoader::ExtractVector3(const NChar * Text, Vector3* Vector) {
		NChar* LineState;
		Vector->x = fast_strtof(Text, &LineState);
		Vector->y = fast_strtof(LineState, &LineState);
		Vector->z = fast_strtof(LineState, &LineState);
	}

	void OBJLoader::ExtractVector2(const NChar * Text, Vector2* Vector) {
		NChar* LineState;
		Vector->x = fast_strtof(Text, &LineState);
		Vector->y = fast_strtof(LineState, &LineState);
	}

	void OBJLoader::ExtractIntVector3(const NChar * Text, IntVector3 * Vector) {
		NChar* LineState;
		Vector->x = (int)fast_strtof(Text, &LineState);
		Vector->y = (int)fast_strtof(LineState, &LineState);
		Vector->z = (int)fast_strtof(LineState, NULL);
	}

	void OBJLoader::ParseVertexPositions(ExtractedData & Data) {
		for (TArray<const NChar *>::const_iterator Line = Data.LinePositions.begin(); Line != Data.LinePositions.end(); ++Line) {
			const size_t Pos = Line - Data.LinePositions.begin();
			ExtractVector3((*Line) + 1, &Data.ListPositions[Pos]);
		}
	}

	void OBJLoader::ParseVertexNormals(ExtractedData & Data) {
		for (TArray<const NChar *>::const_iterator Line = Data.LineNormals.begin(); Line != Data.LineNormals.end(); ++Line) {
			const size_t Pos = Line - Data.LineNormals.begin();
			ExtractVector3((*Line) + 1, &Data.ListNormals[Pos]);
		}
	}

	void OBJLoader::ParseVertexUVs(ExtractedData & Data) {
		for (TArray<const NChar *>::const_iterator Line = Data.LineUVs.begin(); Line != Data.LineUVs.end(); ++Line) {
			const size_t Pos = Line - Data.LineUVs.begin();
			ExtractVector2((*Line) + 1, &Data.ListUVs[Pos]);
		}
	}

	void OBJLoader::ParseFaces(ExtractedData & Data) {
		if (Data.Objects.empty()) return;

		// 0 = Vertex, 1 = TextureCoords, 2 = Normal
		IntVector3 VertexIndex = IntVector3(-1);
		bool bWarned = false;
		auto CurrentObject = Data.Objects.begin();
		auto CurrentObjectMaterial = CurrentObject->Materials.begin();
		int VertexCount = 0;
		const NChar * LineChar;
		for (TArray<const NChar *>::const_iterator Line = Data.LineVertexIndices.begin(); Line != Data.LineVertexIndices.end(); ++Line) {
			if ((CurrentObject + 1) != Data.Objects.end() &&
				Line - Data.LineVertexIndices.begin() == (CurrentObject + 1)->VertexIndicesPos) {
				++CurrentObject;
				CurrentObjectMaterial = CurrentObject->Materials.begin();
			}

			if ((CurrentObjectMaterial + 1) != CurrentObject->Materials.end() &&
				Line - Data.LineVertexIndices.begin() == (CurrentObjectMaterial + 1)->VertexIndicesPos) {
				CurrentObjectMaterial++;
			}

			VertexIndex[0] = VertexIndex[1] = VertexIndex[2] = -1;
			VertexCount = 0;
			LineChar = *Line;
			while (*LineChar != '\0' && *LineChar != '\n') {
				VertexIndex[0] = atoi(++LineChar);
				while (*LineChar != '/' && !isspace(*LineChar)) ++LineChar;
				if (*LineChar == '/') {
					VertexIndex[1] = atoi(++LineChar);
					while (*LineChar != '/' && !isspace(*LineChar)) ++LineChar;
					if (*LineChar == '/') {
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
					CurrentObjectMaterial->VertexIndicesCount += 3;
				}
				else {
					Data.VertexIndices.push_back(VertexIndex);
					CurrentObject->VertexIndicesCount++;
					CurrentObjectMaterial->VertexIndicesCount++;
				}

				if (VertexCount > 4 && !bWarned) {
					bWarned = true;
					LOG_CORE_WARN(L"The model has n-gons, this may lead to unwanted geometry");
				}
			}
		}
	}

	OBJLoader::Keyword OBJLoader::GetKeyword(const NChar * Word) {
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
		if (Word[0] == 'u' && Word[1] == 's' && Word[2] == 'e') return Material;
		if (Word[0] == 'c') return CSType;

		return Undefined;
	}

	void OBJLoader::PrepareData(const NChar * InFile, ExtractedData& ModelData) {
		const NChar* Pointer = InFile;
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

				if (ModelData.Objects.back().Materials.empty()) {
					ModelData.Objects.back().Materials.push_back(ObjectData::Subdivision());
					ModelData.Objects.back().Materials.back().Name = "default";
				}

				ModelData.LineVertexIndices.push_back(++Pointer);
				while (*Pointer != '\n' && *Pointer != '\0') ++Pointer;
				++Pointer;
				FaceCount++;   continue;
			};
			if (Keyword == Group) {
				ModelData.Groups.push_back(NString());
				++Pointer;
				while (*(Pointer + 1) != '\n' && *(Pointer + 1) != '\0') {
					ModelData.Groups.back() += *++Pointer;
				}
				continue;
			}
			if (Keyword == Material) {
				if (ModelData.Objects.size() == 0) {
					ModelData.Objects.push_back(ObjectData());
				}
				Pointer += 6;
				NString Name;
				while (*(Pointer + 1) != '\n' && *(Pointer + 1) != '\0') { Name += *++Pointer; }
				ModelData.Objects.back().Materials.push_back(ObjectData::Subdivision());
				ModelData.Objects.back().Materials.back().Name = Name;
				ModelData.Objects.back().Materials.back().VertexIndicesPos = (int)ModelData.LineVertexIndices.size();
				continue;
			}
			if (Keyword == Object) {
				ModelData.Objects.push_back(ObjectData());
				ModelData.Objects.back().VertexIndicesPos = (int)ModelData.LineVertexIndices.size();
				if (ModelData.Groups.size() > 0) {
					ModelData.Objects.back().Name += ModelData.Groups.back() + ".";
				}
				++Pointer;
				while (*(Pointer + 1) != '\n' && *(Pointer + 1) != '\0') {
					ModelData.Objects.back().Name += *++Pointer;
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

	bool OBJLoader::Load(MeshLoader::FileData & FileData) {
		if (FileData.File == NULL || !FileData.File->IsValid()) return false;

		ExtractedData ModelData;

		// --- Read File
		{
			Debug::Timer Timer;

			Timer.Start();
			NString* MemoryText = new NString();
			FileData.File->ReadNarrowStream(MemoryText);

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
			LOG_CORE_INFO(
				L"├> Parsed {0} vertices and {1} triangles in {2:.3f}s",
				Text::FormatUnit(ModelData.VertexIndices.size(), 2),
				Text::FormatUnit(ModelData.VertexIndices.size() / 3, 2),
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

			FileData.Meshes.push_back(MeshData());
			MeshData* OutMesh = &FileData.Meshes.back();
			OutMesh->Name = Text::NarrowToWide(Data.Name);

			for (int MaterialCount = 0; MaterialCount < Data.Materials.size(); ++MaterialCount) {
				ObjectData::Subdivision & MaterialIndex = Data.Materials[MaterialCount];
				OutMesh->Materials.insert({ MaterialIndex.Name, (int)OutMesh->Materials.size() });
				OutMesh->MaterialSubdivisions.insert({ MaterialCount, MeshFaces() });

				int InitialCount = Count;
				for (; Count < InitialCount + MaterialIndex.VertexIndicesCount; ++Count) {
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
					if (FileData.Optimize) {
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
						if (FileData.Optimize) VertexToIndex[NewVertex] = NewIndex;
						TotalUniqueVertices++;
					}

					if ((Count + 1) % 3 == 0) {
						OutMesh->Faces.push_back({ Indices[Count - 2], Indices[Count - 1], Indices[Count] });
						OutMesh->MaterialSubdivisions[MaterialCount].push_back({ Indices[Count - 2], Indices[Count - 1], Indices[Count] });
					}
				}
			}

			if (ModelData.VertexIndices[Count - 1][1] >= 0) OutMesh->TextureCoordsCount = 1;
			if (ModelData.VertexIndices[Count - 1][2] >= 0) OutMesh->hasNormals = true;
			OutMesh->ComputeTangents();
			OutMesh->Bounding = Data.Bounding;
			OutMesh->hasBoundingBox = true;

#ifdef ES_DEBUG
			LOG_CORE_DEBUG(
				L"├> Parsed {0}	vertices in {1}	at [{2:d}]'{3}'",
				Text::FormatUnit(Data.VertexIndicesCount, 2),
				Text::FormatData(sizeof(IntVector3) * OutMesh->Faces.size() + sizeof(MeshVertex) * OutMesh->Vertices.size(), 2),
				FileData.Meshes.size(),
				OutMesh->Name
			);
#endif // ES_DEBUG

			TotalAllocatedSize += sizeof(IntVector3) * OutMesh->Faces.size() + sizeof(MeshVertex) * OutMesh->Vertices.size();
		}

		ModelData.ListNormals.clear();
		ModelData.ListPositions.clear();
		ModelData.ListUVs.clear();
		ModelData.VertexIndices.clear();

		delete[] Indices;

		Timer.Stop();
		LOG_CORE_INFO(L"└> Allocated {0} in {1:.2f}s", Text::FormatData(TotalAllocatedSize, 2), Timer.GetEnlapsedSeconds());

		return true;
	}

}