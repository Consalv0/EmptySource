#include <ctime>
#include <cstdio>

#include "..\include\Core.h"
#include "..\include\Math\Math.h"
#include "..\include\MeshLoader.h"
#include "..\include\FileStream.h"

bool MeshLoader::GetSimilarVertexIndex(MeshVertex & Vertex, std::map<MeshVertex, unsigned>& VertexToIndex, unsigned & Result) {
	std::map<MeshVertex, unsigned>::iterator it = VertexToIndex.find(Vertex);
	if (it == VertexToIndex.end()) {
		return false;
	} else {
		Result = it->second;
		return true;
	}
}

bool MeshLoader::FromOBJ(FileStream * File, MeshFaces * Faces, MeshVertices * Vertices) {
	if (File == NULL || !File->IsValid()) return false;

	std::vector<IntVector3> VertexIndices;
	MeshVector3D ListPositions;
	MeshVector3D ListNormals;
	MeshUVs ListUVs;

	_LOG(Log, L"Parsing Model '%s'", File->GetShortPath().c_str());

	WString KeyWord;
	WChar* Line;

	clock_t StartTime = clock();
	while ((*File) >> KeyWord) {

		Line = File->GetLine((unsigned)256);

		if (KeyWord == L"o") {
			// _LOG(LogDebug, L"Name%s", Line);
		}

		if (KeyWord == L"v") {
			Vector3 Vertx = Vector3();
			swscanf_s(Line, L"%f %f %f", &Vertx.x, &Vertx.y, &Vertx.z);
			ListPositions.push_back(Vertx);
			// _LOG(LogDebug, L"Vertex %s", Vertx.ToString().c_str());
		}

		if (KeyWord == L"vn") {
			Vector3 Normal = Vector3();
			swscanf_s(Line, L"%f %f %f", &Normal.x, &Normal.y, &Normal.z);
			ListNormals.push_back(Normal);
			// _LOG(LogDebug, L"Normal %s", Normal.ToString().c_str());
		}

		if (KeyWord == L"vt") {
			Vector2 TexCoords = Vector2();
			swscanf_s(Line, L"%f %f", &TexCoords.x, &TexCoords.y);
			ListUVs.push_back(TexCoords);
			// _LOG(LogDebug, L"UV %s", TexCoords.ToString().c_str());
		}

		if (KeyWord == L"f") {
			// 0 = Vertex, 1 = TextureCoords, 2 = Normal
			IntVector3 VertexIndex;
			WChar *LineState;
			int VertexCount = 0;

			for (Line; ; Line = NULL) {
				WChar* Token = wcstok_s(Line, L" ", &LineState);
				if (Token == NULL)
					break;

				int Empty = swscanf_s(
					Token, L"%d/%d/%d",
					&VertexIndex[0],
					&VertexIndex[1],
					&VertexIndex[2]
				);

				if (Empty < 0) {
					continue;
				}

				VertexCount++;
				VertexIndices.push_back(VertexIndex);
			}

			if (VertexCount > 3) {
				_LOG(LogError, L"The model %s has four or more points per face and this is no implmented yet", File->GetPath().c_str());
				return 0;
			}
		}
	}

	clock_t EndTime = clock();
	float TotalTime = float(EndTime - StartTime) / CLOCKS_PER_SEC;
	_LOG(Log, L"|> Parsed %d vertices in %.3fs", VertexIndices.size(), TotalTime);

	std::map<MeshVertex, unsigned> VertexToIndex;
	std::vector<int> Indices;

	StartTime = clock();
	for (unsigned int Count = 0; Count < VertexIndices.size(); Count++) {

		MeshVertex NewVertex;
		NewVertex.Position = ListPositions[VertexIndices[Count][0] - 1];
		NewVertex.UV0 = NewVertex.UV1 = ListUVs[VertexIndices[Count][1] - 1];
		NewVertex.Normal = ListNormals[VertexIndices[Count][2] - 1];
		NewVertex.Color = Vector4(1);
		NewVertex.Tangent = Vector3();

		unsigned Index;
		bool bFoundIndex = GetSimilarVertexIndex(NewVertex, VertexToIndex, Index);

		if (bFoundIndex) { // A similar vertex is already in the VBO, use it instead !
			Indices.push_back(Index);
		} else { // If not, it needs to be added in the output data.
			Vertices->push_back(NewVertex);
			unsigned NewIndex = (unsigned)Vertices->size() - 1;
			Indices.push_back(NewIndex);
			VertexToIndex[NewVertex] = NewIndex;
		}

		if (Indices.size() % 3 == 0) {

			Faces->push_back({ Indices[Count - 2], Indices[Count - 1], Indices[Count] });

			// _LOG(LogDebug, L"Face {%d, %d, %d}",
			// 	Faces->back()[0], Faces->back()[1], Faces->back()[2]
			// );
		}
	}
	EndTime = clock();
	TotalTime = float(EndTime - StartTime) / CLOCKS_PER_SEC;
	size_t AllocatedSize = sizeof(IntVector3) * Faces->size() + sizeof(MeshVertex) * Vertices->size();
	_LOG(Log, L"└> Allocated %.2fKB in %.2fs", AllocatedSize / 1024.F, TotalTime);

	return true;
}
