﻿
#include <fbxsdk.h>
#include "../include/FBXLoader.h"
#include "../include/Utility/Timer.h"

FbxManager * FBXLoader::gSdkManager = NULL;

void FBXLoader::InitializeSdkManager() {
	// Create the FBX SDK memory manager object.
	// The SDK Manager allocates and frees memory
	// for almost all the classes in the SDK.
	gSdkManager = FbxManager::Create();

	// create an IOSettings object
	FbxIOSettings * IOS = FbxIOSettings::Create(gSdkManager, IOSROOT);
	gSdkManager->SetIOSettings(IOS);

}


bool FBXLoader::LoadScene(FbxScene * pScene, FileStream * File) {
	int FileMajor, FileMinor, FileRevision;
	int SDKMajor, SDKMinor, SDKRevision;
	// int i, lAnimStackCount;
	bool lStatus;
	char lPassword[1024];

	// --- Get the version number of the FBX files generated by the
	// --- Version of FBX SDK that you are using.
	FbxManager::GetFileFormatVersion(SDKMajor, SDKMinor, SDKRevision);

	// --- Create an importer.
	FbxImporter* Importer = FbxImporter::Create(gSdkManager, "");

	// --- Initialize the importer by providing a filename.
	const bool ImportStatus = Importer->Initialize(WStringToString(File->GetPath()).c_str(), -1, gSdkManager->GetIOSettings());

	// --- Get the version number of the FBX file format.
	Importer->GetFileVersion(FileMajor, FileMinor, FileRevision);

	// --- Problem with the file to be imported
	if (!ImportStatus) {
		FbxString Error = Importer->GetStatus().GetErrorString();
		Debug::Log(Debug::LogError, L"Import failed, error returned : %s", CharToWString(Error.Buffer()).c_str());

		if (Importer->GetStatus().GetCode() == FbxStatus::eInvalidFileVersion) {
			Debug::Log(Debug::LogError, L"├> FBX SDK version number: %d.%d.%d",
				SDKMajor, SDKMinor, SDKRevision);
			Debug::Log(Debug::LogError, L"└> FBX version number: %d.%d.%d",
				FileMajor, FileMinor, FileRevision);
		}

		return false;
	}

	if (Importer->IsFBX()) {
		Debug::Log(Debug::LogInfo, L"├> FBX version number: %d.%d.%d",
			FileMajor, FileMinor, FileRevision);

		// // In FBX, a scene can have one or more "animation stack". An animation stack is a
		// // container for animation data.
		// // You can access a file's animation stack information without
		// // the overhead of loading the entire file into the scene.
		// 
		// UI_Printf("Animation Stack Information");
		// 
		// lAnimStackCount = lImporter->GetAnimStackCount();
		// 
		// UI_Printf("    Number of animation stacks: %d", lAnimStackCount);
		// UI_Printf("    Active animation stack: \"%s\"",
		// 	lImporter->GetActiveAnimStackName());
		// 
		// for (i = 0; i < lAnimStackCount; i++)
		// {
		// 	FbxTakeInfo* lTakeInfo = lImporter->GetTakeInfo(i);
		// 
		// 	UI_Printf("    Animation Stack %d", i);
		// 	UI_Printf("         Name: \"%s\"", lTakeInfo->mName.Buffer());
		// 	UI_Printf("         Description: \"%s\"",
		// 		lTakeInfo->mDescription.Buffer());
		// 
		// 	// Change the value of the import name if the animation stack should
		// 	// be imported under a different name.
		// 	UI_Printf("         Import Name: \"%s\"", lTakeInfo->mImportName.Buffer());
		// 
		// 	// Set the value of the import state to false
		// 	// if the animation stack should be not be imported.
		// 	UI_Printf("         Import State: %s", lTakeInfo->mSelect ? "true" : "false");
		// }

		// Import options determine what kind of data is to be imported.
		// The default is true, but here we set the options explictly.

		(*(gSdkManager->GetIOSettings())).SetBoolProp(IMP_FBX_MATERIAL, false);
		(*(gSdkManager->GetIOSettings())).SetBoolProp(IMP_FBX_TEXTURE, false);
		(*(gSdkManager->GetIOSettings())).SetBoolProp(IMP_FBX_LINK, true);
		(*(gSdkManager->GetIOSettings())).SetBoolProp(IMP_FBX_SHAPE, true);
		(*(gSdkManager->GetIOSettings())).SetBoolProp(IMP_FBX_GOBO, true);
		(*(gSdkManager->GetIOSettings())).SetBoolProp(IMP_FBX_ANIMATION, true);
		(*(gSdkManager->GetIOSettings())).SetBoolProp(IMP_FBX_GLOBAL_SETTINGS, true);
	}

	// Import the scene.
	lStatus = Importer->Import(pScene);

	if (lStatus == false &&     // The import file may have a password
		Importer->GetStatus().GetCode() == FbxStatus::ePasswordError)
	{
		Debug::LogUnadorned(Debug::LogInfo, L"Please enter password: ");

		lPassword[0] = '\0';

		FBXSDK_CRT_SECURE_NO_WARNING_BEGIN
			scanf("%s", lPassword);
		FBXSDK_CRT_SECURE_NO_WARNING_END
			FbxString lString(lPassword);

		(*(gSdkManager->GetIOSettings())).SetStringProp(IMP_FBX_PASSWORD, lString);
		(*(gSdkManager->GetIOSettings())).SetBoolProp(IMP_FBX_PASSWORD_ENABLE, true);


		lStatus = Importer->Import(pScene);

		if (lStatus == false && Importer->GetStatus().GetCode() == FbxStatus::ePasswordError) {
			Debug::Log(Debug::LogError, L"└> Incorrect password: file not imported.");
		}
	}

	// Destroy the importer
	Importer->Destroy();

	return lStatus;
}

void FBXLoader::ExtractVertexData(FbxMesh * pMesh, TArray<IntVector3>& Faces, TArray<MeshVertex>& Vertices, Box3D & BoundingBox) {
	int PolygonCount = pMesh->GetPolygonCount();
	FbxVector4 * ControlPoints = pMesh->GetControlPoints();
	int PolygonIndex; int PolygonVertexIndex;
	bool bComputeTangents = true;

	int VertexIndex = 0;
	for (PolygonIndex = 0; PolygonIndex < PolygonCount; ++PolygonIndex) {

		int PolygonVertexSize = pMesh->GetPolygonSize(PolygonIndex);

		MeshVertex Vertex;

		for (PolygonVertexIndex = 0; PolygonVertexIndex < PolygonVertexSize; ++PolygonVertexIndex) {
			int ControlPointIndex = pMesh->GetPolygonVertex(PolygonIndex, PolygonVertexIndex);

			Vertex.Position.x = (float)ControlPoints[ControlPointIndex][0];
			Vertex.Position.y = (float)ControlPoints[ControlPointIndex][1];
			Vertex.Position.z = (float)ControlPoints[ControlPointIndex][2];

			BoundingBox.Add(Vertex.Position);

			ExtractTextureCoords(pMesh, Vertex, ControlPointIndex, PolygonIndex, PolygonVertexIndex);
			ExtractNormal(pMesh, Vertex, ControlPointIndex, VertexIndex);
			ExtractVertexColor(pMesh, Vertex, ControlPointIndex, VertexIndex);
			bComputeTangents = !ExtractTangent(pMesh, Vertex, ControlPointIndex, VertexIndex);

			VertexIndex++;
			Vertices.push_back(Vertex);
		}
		Faces.push_back(IntVector3(VertexIndex - 3, VertexIndex - 2, VertexIndex - 1));
	}

	if (bComputeTangents)
		ComputeTangents(Faces, Vertices);
}

void FBXLoader::ExtractTextureCoords(
	class FbxMesh * pMesh, MeshVertex & Vertex,
	const int & ControlPointIndex, const int & PolygonIndex, const int & PolygonVertexIndex) 
{
	for (int ElementUVIndex = 0; ElementUVIndex < Math::Clamp(pMesh->GetElementUVCount(), 0, 2); ++ElementUVIndex) {
		FbxGeometryElementUV * ElementUV = pMesh->GetElementUV(ElementUVIndex);
		FbxVector2 UV;

		switch (ElementUV->GetMappingMode()) {
		case FbxGeometryElement::eByControlPoint:
			switch (ElementUV->GetReferenceMode()) {
			case FbxGeometryElement::eDirect:
				UV = ElementUV->GetDirectArray().GetAt(ControlPointIndex);

				switch (ElementUVIndex) {
				case 0:
					Vertex.UV0.u = (float)UV[0];
					Vertex.UV0.v = (float)UV[1];
					if (pMesh->GetElementUVCount() == 1) {
						Vertex.UV1.u = (float)UV[0];
						Vertex.UV1.v = (float)UV[1];
					}
					break;
				case 1:
					Vertex.UV1.u = (float)UV[0];
					Vertex.UV1.v = (float)UV[1];
					break;
				}
			case FbxGeometryElement::eIndexToDirect: {
				int ID = ElementUV->GetIndexArray().GetAt(ControlPointIndex);
				UV = ElementUV->GetDirectArray().GetAt(ID);

				switch (ElementUVIndex) {
				case 0:
					Vertex.UV0.u = (float)UV[0];
					Vertex.UV0.v = (float)UV[1];
					if (pMesh->GetElementUVCount() == 1) {
						Vertex.UV1.u = (float)UV[0];
						Vertex.UV1.v = (float)UV[1];
					}
					break;
				case 1:
					Vertex.UV1.u = (float)UV[0];
					Vertex.UV1.v = (float)UV[1];
					break;
				}
			} break;
			default:
				break;
			}
			break;

		case FbxGeometryElement::eByPolygonVertex: {
			int TextureUVIndex = pMesh->GetTextureUVIndex(PolygonIndex, PolygonVertexIndex);
			switch (ElementUV->GetReferenceMode()) {
			case FbxGeometryElement::eDirect:
			case FbxGeometryElement::eIndexToDirect: {
				UV = ElementUV->GetDirectArray().GetAt(TextureUVIndex);

				switch (ElementUVIndex) {
				case 0:
					Vertex.UV0.u = (float)UV[0];
					Vertex.UV0.v = (float)UV[1];
					if (pMesh->GetElementUVCount() == 1) {
						Vertex.UV1.u = (float)UV[0];
						Vertex.UV1.v = (float)UV[1];
					}
					break;
				case 1:
					Vertex.UV1.u = (float)UV[0];
					Vertex.UV1.v = (float)UV[1];
					break;
				}
			} break;
			default:
				break;
			}
		} break;
		default:
			break;
		}
	}
}

void FBXLoader::ExtractNormal(
	FbxMesh * pMesh, MeshVertex & Vertex,
	const int & ControlPointIndex, const int & VertexIndex) 
{
	FbxVector4 Normal;
	int ElementNormalCount = pMesh->GetElementNormalCount();

	if (ElementNormalCount > 0) {
		FbxGeometryElementNormal* ElementNormal = pMesh->GetElementNormal(0);
		if (ElementNormal->GetMappingMode() == FbxGeometryElement::eByControlPoint) {
			switch (ElementNormal->GetReferenceMode()) {
			case FbxGeometryElement::eDirect: {
				Normal = ElementNormal->GetDirectArray().GetAt(ControlPointIndex);

				Vertex.Normal.x = (float)Normal[0];
				Vertex.Normal.y = (float)Normal[1];
				Vertex.Normal.z = (float)Normal[2];
			} break;
			case FbxGeometryElement::eIndexToDirect: {
				int ID = ElementNormal->GetIndexArray().GetAt(ControlPointIndex);
				Normal = ElementNormal->GetDirectArray().GetAt(ID);

				Vertex.Normal.x = (float)Normal[0];
				Vertex.Normal.y = (float)Normal[1];
				Vertex.Normal.z = (float)Normal[2];
			} break;
			default:
				break;
			}
		}
		else if (ElementNormal->GetMappingMode() == FbxGeometryElement::eByPolygonVertex) {
			switch (ElementNormal->GetReferenceMode()) {
			case FbxGeometryElement::eDirect: {
				Normal = ElementNormal->GetDirectArray().GetAt(VertexIndex);

				Vertex.Normal.x = (float)Normal[0];
				Vertex.Normal.y = (float)Normal[1];
				Vertex.Normal.z = (float)Normal[2];
			} break;
			case FbxGeometryElement::eIndexToDirect: {
				int ID = ElementNormal->GetIndexArray().GetAt(VertexIndex);
				Normal = ElementNormal->GetDirectArray().GetAt(ID);

				Vertex.Normal.x = (float)Normal[0];
				Vertex.Normal.y = (float)Normal[1];
				Vertex.Normal.z = (float)Normal[2];
			} break;
			default:
				break;
			}
		}
	}
}

void FBXLoader::ExtractVertexColor(
	FbxMesh * pMesh, MeshVertex & Vertex,
	const int & ControlPointIndex, const int & VertexIndex)
{
	FbxColor Color;
	int ElementColorCount = pMesh->GetElementVertexColorCount();

	if (ElementColorCount > 0) {
		FbxGeometryElementVertexColor* ElementColor = pMesh->GetElementVertexColor(0);
		if (ElementColor->GetMappingMode() == FbxGeometryElement::eByControlPoint) {
			switch (ElementColor->GetReferenceMode()) {
			case FbxGeometryElement::eDirect: {
				Color = ElementColor->GetDirectArray().GetAt(ControlPointIndex);

				Vertex.Color.r = (float)Color.mRed;
				Vertex.Color.g = (float)Color.mGreen;
				Vertex.Color.b = (float)Color.mBlue;
				Vertex.Color.a = (float)Color.mAlpha;
			} break;
			case FbxGeometryElement::eIndexToDirect: {
				int ID = ElementColor->GetIndexArray().GetAt(ControlPointIndex);
				Color = ElementColor->GetDirectArray().GetAt(ID);

				Vertex.Color.r = (float)Color.mRed;
				Vertex.Color.g = (float)Color.mGreen;
				Vertex.Color.b = (float)Color.mBlue;
				Vertex.Color.a = (float)Color.mAlpha;
			} break;
			default:
				break;
			}
		}
		else if (ElementColor->GetMappingMode() == FbxGeometryElement::eByPolygonVertex) {
			switch (ElementColor->GetReferenceMode()) {
			case FbxGeometryElement::eDirect: {
				Color = ElementColor->GetDirectArray().GetAt(VertexIndex);

				Vertex.Color.r = (float)Color.mRed;
				Vertex.Color.g = (float)Color.mGreen;
				Vertex.Color.b = (float)Color.mBlue;
				Vertex.Color.a = (float)Color.mAlpha;
			} break;
			case FbxGeometryElement::eIndexToDirect: {
				int ID = ElementColor->GetIndexArray().GetAt(VertexIndex);
				Color = ElementColor->GetDirectArray().GetAt(ID);

				Vertex.Color.r = (float)Color.mRed;
				Vertex.Color.g = (float)Color.mGreen;
				Vertex.Color.b = (float)Color.mBlue;
				Vertex.Color.a = (float)Color.mAlpha;
			} break;
			default:
				break;
			}
		}
	}
}

bool FBXLoader::ExtractTangent(FbxMesh * pMesh, MeshVertex & Vertex, const int & ControlPointIndex, const int & VertexIndex) {
	FbxVector4 Tangent;
	int ElementTangentCount = pMesh->GetElementTangentCount();

	if (ElementTangentCount > 0) {
		FbxGeometryElementTangent* ElementTangent = pMesh->GetElementTangent(0);
		if (ElementTangent->GetMappingMode() == FbxGeometryElement::eByControlPoint) {
			switch (ElementTangent->GetReferenceMode()) {
			case FbxGeometryElement::eDirect: {
				Tangent = ElementTangent->GetDirectArray().GetAt(ControlPointIndex);

				Vertex.Tangent.x = (float)Tangent[0];
				Vertex.Tangent.y = (float)Tangent[1];
				Vertex.Tangent.z = (float)Tangent[2];
				return true;
			} break;
			case FbxGeometryElement::eIndexToDirect: {
				int ID = ElementTangent->GetIndexArray().GetAt(ControlPointIndex);
				Tangent = ElementTangent->GetDirectArray().GetAt(ID);

				Vertex.Tangent.x = (float)Tangent[0];
				Vertex.Tangent.y = (float)Tangent[1];
				Vertex.Tangent.z = (float)Tangent[2];
				return true;
			} break;
			default:
				break;
			}
		}
		else if (ElementTangent->GetMappingMode() == FbxGeometryElement::eByPolygonVertex) {
			switch (ElementTangent->GetReferenceMode()) {
			case FbxGeometryElement::eDirect: {
				Tangent = ElementTangent->GetDirectArray().GetAt(VertexIndex);

				Vertex.Tangent.x = (float)Tangent[0];
				Vertex.Tangent.y = (float)Tangent[1];
				Vertex.Tangent.z = (float)Tangent[2];
				return true;
			} break;
			case FbxGeometryElement::eIndexToDirect: {
				int ID = ElementTangent->GetIndexArray().GetAt(VertexIndex);
				Tangent = ElementTangent->GetDirectArray().GetAt(ID);

				Vertex.Tangent.x = (float)Tangent[0];
				Vertex.Tangent.y = (float)Tangent[1];
				Vertex.Tangent.z = (float)Tangent[2];
				return true;
			} break;
			default:
				break;
			}
		}
	}
	return false;
}

void FBXLoader::ComputeTangents(const MeshFaces & Faces, MeshVertices & Vertices) {

	const int Size = (int)Faces.size();

	// --- For each triangle, compute the edge (DeltaPos) and the DeltaUV
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
	}
}

bool FBXLoader::Load(FileStream * File, TArray<MeshFaces>* Faces, TArray<MeshVertices>* Vertices, TArray<Box3D>* BoundingBoxes, bool Optimize) {
	if (gSdkManager == NULL)
		InitializeSdkManager();

	Debug::Timer Timer;
	Timer.Start();
	
	FbxScene* Scene = FbxScene::Create(gSdkManager, WStringToString(File->GetShortPath()).c_str());

	bool bStatus = LoadScene(Scene, File);
	if (bStatus == false) return false;

	FbxAxisSystem::OpenGL.ConvertScene(Scene);
	FbxSystemUnit::m.ConvertScene(Scene);

	FbxGeometryConverter GeomConverter(gSdkManager);
	GeomConverter.Triangulate(Scene, true);
	const int NodeCount = Scene->GetSrcObjectCount<FbxNode>();
	size_t TotalAllocatedSize = 0;

	Timer.Stop();
	Debug::Log(Debug::LogInfo,
		L"├> Readed and parsed in %.3fs",
		Timer.GetEnlapsedSeconds()
	);

	Timer.Start();
	for (int NodeIndex = 0; NodeIndex < NodeCount; NodeIndex++) {
		FbxNode * Node = Scene->GetSrcObject<FbxNode>(NodeIndex);
		FbxMesh* lMesh = Node->GetMesh();
		if (lMesh) {
			Vertices->push_back(MeshVertices());
			Faces->push_back(MeshFaces());
			BoundingBoxes->push_back(BoundingBox3D());
			ExtractVertexData(lMesh, Faces->back(), Vertices->back(), BoundingBoxes->back());
#ifdef _DEBUG
			Debug::Log(
				Debug::LogInfo,
				L"├> Parsed %ls	vertices in %ls	at [%d]'%ls'",
				Text::FormatUnit(Vertices->back().size(), 2).c_str(),
				Text::FormatData(sizeof(IntVector3) * Faces->back().size() + sizeof(MeshVertex) * Vertices->back().size(), 2).c_str(),
				Vertices->size(),
				StringToWString(Node->GetName()).c_str()
			);
#endif
			TotalAllocatedSize += sizeof(IntVector3) * Faces->back().size() + sizeof(MeshVertex) * Vertices->back().size();
		}
	}

	Timer.Stop();
	Debug::Log(Debug::LogInfo, L"└> Allocated %ls in %.2fs", Text::FormatData(TotalAllocatedSize, 2).c_str(), Timer.GetEnlapsedSeconds());

	return bStatus;
}
