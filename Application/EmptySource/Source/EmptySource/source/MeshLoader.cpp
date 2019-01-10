#include <ctime>
#include <cstdio>
#include <string>

#include "..\include\Core.h"
#include "..\include\Math\Math.h"
#include "..\include\MeshLoader.h"
#include "..\include\FileStream.h"

bool MeshLoader::GetSimilarVertexIndex(const MeshVertex & Vertex, std::unordered_map<MeshVertex, unsigned>& VertexToIndex, unsigned & Result) {
	std::unordered_map<MeshVertex, unsigned>::iterator it = VertexToIndex.find(Vertex);
	if (it == VertexToIndex.end()) {
		return false;
	} else {
		Result = it->second;
		return true;
	}
}

void MeshLoader::ExtractVector3(Char * Text, Vector3* Vector) {
	Char* LineState;
	Vector->x = std::strtof(Text, &LineState);
	Vector->y = std::strtof(LineState, &LineState);
	Vector->z = std::strtof(LineState, NULL);
}

void MeshLoader::ExtractVector2(Char * Text, Vector2* Vector) {
	Char* LineState;
	Vector->x = std::strtof(Text, &LineState);
	Vector->y = std::strtof(LineState, &LineState);
}

void MeshLoader::ExtractIntVector3(Char * Text, IntVector3 * Vector) {
	Char* LineState;
	Vector->x = (int)std::strtof(Text, &LineState);
	Vector->y = (int)std::strtof(LineState, &LineState);
	Vector->z = (int)std::strtof(LineState, NULL);
}

bool MeshLoader::FromOBJ(FileStream * File, MeshFaces * Faces, MeshVertices * Vertices, bool hasOptimize) {
	if (File == NULL || !File->IsValid()) return false;

	std::vector<IntVector3> VertexIndices;
	MeshVector3D ListPositions;
	MeshVector3D ListNormals;
	MeshUVs ListUVs;

	{
		bool bWarned = false;
		clock_t StartTime = clock();
		Debug::Log(Debug::LogNormal, L"Parsing Model '%s'", File->GetShortPath().c_str());
		
		String* MemoryText = new String();
		File->ReadNarrowStream(MemoryText);
		std::istringstream InFile;
		InFile = std::istringstream(*MemoryText);
		delete MemoryText;

		String KeyWord;
		long LineCount = 0;

		while (InFile >> KeyWord) {

			Char* Line = new Char[250];
			InFile.getline(Line, (unsigned)250);
			LineCount++;

			long Progress = long(InFile.tellg());
			float prog = Progress / float(File->GetLenght());
			if ((LineCount % 8273) <= 0) {
				float cur = std::ceil(prog * 25);
				Debug::Log(Debug::NoLog, L"\r [%s%s] %.2f%% %d lines", WString(int(cur), L'#').c_str(), WString(int(25 + 1 - cur), L' ').c_str(), 100 * prog, LineCount);
			}

			if (prog == 1) {
				Debug::Log(Debug::NoLog, L"\r");
				Debug::Log(Debug::LogNormal, L"├> [%s] %.2f%% %d lines", WString(25, L'#').c_str(), 100 * prog, LineCount);
			}

			if (KeyWord == "cstype" && !bWarned) {
				bWarned = true;
				Debug::Log(Debug::NoLog, L"\r");
				Debug::Log(Debug::LogWarning, L"The model %s contains free-form geometry, this is not supported", File->GetShortPath().c_str());
				delete[] Line;
				continue;
			}
			
			if (KeyWord == "o") {
				// _LOG(LogDebug, L"Name%s", Line);
				delete[] Line;
				continue;
			}
			
			if (KeyWord == "v") {
				Vector3 Vert;
				ExtractVector3(Line, &Vert);
				ListPositions.push_back(Vert);
				delete[] Line;
				continue;
				// _LOG(LogDebug, L"Vertex %s", Vertx.ToString().c_str());
			}
			
			if (KeyWord == "vn") {
				Vector3 Normal;
				ExtractVector3(Line, &Normal);
				ListNormals.push_back(Normal);
				delete[] Line;
				continue;
				// _LOG(LogDebug, L"Normal %s", Normal.ToString().c_str());
			}
			
			if (KeyWord == "vt") {
				Vector2 TexCoords;
				ExtractVector2(Line, &TexCoords);
				ListUVs.push_back(TexCoords);
				delete[] Line;
				continue;
				// _LOG(LogDebug, L"UV %s", TexCoords.ToString().c_str());
			}
			
			if (KeyWord == "f") {
				// 0 = Vertex, 1 = TextureCoords, 2 = Normal
				IntVector3 VertexIndex = IntVector3(1);
				Char *LineState, *Token;
				Token = strtok_s(Line, " ", &LineState);
				int VertexCount = 0;
			
				while (Token != NULL) {
					int Empty;
			
					if (ListUVs.size() <= 0) {
						if (ListNormals.size() <= 0) {
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
					} else if (ListNormals.size() <= 0) {
						if (ListUVs.size() <= 0) {
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
						Debug::Log(Debug::LogWarning, L"The model %s contains negative references, this is not supported", File->GetShortPath().c_str());
						continue;
					}
			
					if (Empty < 0) {
						continue;
					}
			
					VertexCount++;
					if (VertexCount == 4) {
						VertexIndices.push_back(VertexIndices[VertexIndices.size() - 3]);
						VertexIndices.push_back(VertexIndices[VertexIndices.size() - 2]);
						VertexIndices.push_back(VertexIndex);
					} else {
						VertexIndices.push_back(VertexIndex);
					}
				}
			
				if (VertexCount > 4 && !bWarned) {
					bWarned = true;
					Debug::Log(Debug::NoLog, L"\r");
					Debug::Log(Debug::LogWarning, L"The model %s has n-gons and this is no supported yet", File->GetShortPath().c_str());
				}
			
				delete[] Line;
				continue;
			}
		}

		clock_t EndTime = clock();
		float TotalTime = float(EndTime - StartTime) / CLOCKS_PER_SEC;
		Debug::Log(Debug::LogNormal, L"├> Parsed %d vertices and %d triangles in %.3fs", VertexIndices.size(), VertexIndices.size() / 3, TotalTime);
	}

	std::unordered_map<MeshVertex, unsigned> VertexToIndex;
	VertexToIndex.reserve(VertexIndices.size());
	int* Indices = new int[VertexIndices.size()];
	Faces->reserve(VertexIndices.size() / 3);
	Vertices->reserve(VertexIndices.size());

	clock_t StartTime = clock();
	for (unsigned int Count = 0; Count < VertexIndices.size(); Count++) {

		float prog = Count / float(VertexIndices.size());
		if ((Count % 14271) <= 0) {
			float cur = std::ceil(prog * 25);
			Debug::Log(Debug::NoLog, L"\r [%s%s] %.2f%% %d vertices", WString(int(cur), L'#').c_str(), WString(int(25 + 1 - cur), L' ').c_str(), 100 * prog, Count);
		}

		MeshVertex NewVertex = {
			ListPositions.size() > 0 ?
				ListPositions[VertexIndices[Count][0] - 1] : 0,
			ListNormals.size() > 0 ?
				NewVertex.Normal = ListNormals[VertexIndices[Count][2] - 1] : Vector3(0.3F, 0.3F, 0.4F),
			0,
			ListUVs.size() > 0 ?
				ListUVs[VertexIndices[Count][1] - 1] : 0,
			ListUVs.size() > 0 ?
				ListUVs[VertexIndices[Count][1] - 1] : 0,
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
	Debug::Log(Debug::LogNormal, L"├> [%s] 100.00%% %d vertices", WString(25, L'#').c_str(), VertexIndices.size());

	clock_t EndTime = clock();
	float TotalTime = float(EndTime - StartTime) / CLOCKS_PER_SEC;
	size_t AllocatedSize = sizeof(IntVector3) * Faces->size() + sizeof(MeshVertex) * Vertices->size();
	Debug::Log(Debug::LogNormal, L"└> Allocated %.2fKB in %.2fs", AllocatedSize / 1024.F, TotalTime);

	return true;
}
