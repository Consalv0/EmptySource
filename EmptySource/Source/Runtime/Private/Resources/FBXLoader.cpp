﻿
#include "CoreMinimal.h"
#include <fbxsdk.h>
#include "Resources/ModelParser.h"
#include "Resources/FBXLoader.h"
#include "Resources/ModelResource.h"

#include "Utility/TextFormatting.h"

namespace ESource {

	FbxManager * FBXLoader::gSdkManager = NULL;

	bool FBXLoader::InitializeSdkManager() {
		// Create the FBX SDK memory manager object.
		// The SDK Manager allocates and frees memory
		// for almost all the classes in the SDK.
		gSdkManager = FbxManager::Create();
		if (gSdkManager == NULL)
			return false;

		FbxIOSettings * IOS = FbxIOSettings::Create(gSdkManager, IOSROOT);
		IOS->SetBoolProp(IMP_FBX_MATERIAL, true);
		IOS->SetBoolProp(IMP_FBX_TEXTURE, false);
		gSdkManager->SetIOSettings(IOS);

		return true;
	}

	bool FBXLoader::LoadScene(FbxScene * pScene, const FileStream * File) {
		int FileMajor, FileMinor, FileRevision;
		int SDKMajor, SDKMinor, SDKRevision;
		// int i, lAnimStackCount;
		bool lStatus;

		// --- Get the version number of the FBX files generated by the
		// --- Version of FBX SDK that you are using.
		FbxManager::GetFileFormatVersion(SDKMajor, SDKMinor, SDKRevision);

		// --- Create an importer.
		FbxImporter* Importer = FbxImporter::Create(gSdkManager, "");

		// --- Initialize the importer by providing a filename.
		const bool ImportStatus = Importer->Initialize(Text::WideToNarrow(File->GetPath()).c_str(), -1, gSdkManager->GetIOSettings());

		// --- Get the version number of the FBX file format.
		Importer->GetFileVersion(FileMajor, FileMinor, FileRevision);

		// --- Problem with the file to be imported
		if (!ImportStatus) {
			FbxString Error = Importer->GetStatus().GetErrorString();
			LOG_CORE_ERROR(L"Import failed, error returned : {}", Text::NarrowToWide(Error.Buffer()));

			if (Importer->GetStatus().GetCode() == FbxStatus::eInvalidFileVersion) {
				LOG_CORE_INFO(L"├> FBX SDK version number: {0:d}.{1:d}.{2:d}", SDKMajor, SDKMinor, SDKRevision);
				LOG_CORE_INFO(L"└> FBX version number: {0:d}.{1:d}.{2:d}", FileMajor, FileMinor, FileRevision);
			}

			return false;
		}

		if (Importer->IsFBX()) {
			LOG_CORE_INFO(L"├> FBX version number: {0:d}.{1:d}.{2:d}", FileMajor, FileMinor, FileRevision);

			(*(gSdkManager->GetIOSettings())).SetBoolProp(IMP_FBX_MATERIAL, false);
			(*(gSdkManager->GetIOSettings())).SetBoolProp(IMP_FBX_TEXTURE, false);
			(*(gSdkManager->GetIOSettings())).SetBoolProp(IMP_FBX_LINK, true);
			(*(gSdkManager->GetIOSettings())).SetBoolProp(IMP_FBX_GOBO, true);
			(*(gSdkManager->GetIOSettings())).SetBoolProp(IMP_FBX_ANIMATION, true);
			(*(gSdkManager->GetIOSettings())).SetBoolProp(IMP_FBX_GLOBAL_SETTINGS, true);
		}

		// Import the scene.
		lStatus = Importer->Import(pScene);

		// The import file may have a password
		if (lStatus == false && Importer->GetStatus().GetCode() == FbxStatus::ePasswordError) {
			LOG_CORE_ERROR(L"File not imported, protected by password");
		}

		// Destroy the importer
		Importer->Destroy();

		return lStatus;
	}

	bool FBXLoader::LoadAnimationStack(FbxScene * pScene, const FileStream * File) {
		int FileMajor, FileMinor, FileRevision;
		int SDKMajor, SDKMinor, SDKRevision;
		// int i, lAnimStackCount;
		bool lStatus;
		return false; // ---------------------------------------------------------------------------------------------------------------------------------

		// --- Get the version number of the FBX files generated by the
		// --- Version of FBX SDK that you are using.
		FbxManager::GetFileFormatVersion(SDKMajor, SDKMinor, SDKRevision);

		// --- Create an importer.
		FbxImporter* Importer = FbxImporter::Create(gSdkManager, "");

		// --- Initialize the importer by providing a filename.
		const bool ImportStatus = Importer->Initialize(Text::WideToNarrow(File->GetPath()).c_str(), -1, gSdkManager->GetIOSettings());

		// --- Get the version number of the FBX file format.
		Importer->GetFileVersion(FileMajor, FileMinor, FileRevision);

		// --- Problem with the file to be imported
		if (!ImportStatus) {
			FbxString Error = Importer->GetStatus().GetErrorString();
			LOG_CORE_ERROR(L"Import failed, error returned : {}", Text::NarrowToWide(Error.Buffer()));

			if (Importer->GetStatus().GetCode() == FbxStatus::eInvalidFileVersion) {
				LOG_CORE_INFO(L"├> FBX SDK version number: {0:d}.{1:d}.{2:d}", SDKMajor, SDKMinor, SDKRevision);
				LOG_CORE_INFO(L"└> FBX version number: {0:d}.{1:d}.{2:d}", FileMajor, FileMinor, FileRevision);
			}

			return false;
		}


		if (Importer->IsFBX()) {
			LOG_CORE_INFO(L"├> FBX version number: {0:d}.{1:d}.{2:d}", FileMajor, FileMinor, FileRevision);

			// In FBX, a scene can have one or more "animation stack". An animation stack is a
			// container for animation data.
			// You can access a file's animation stack information without
			// the overhead of loading the entire file into the scene.
			
			int AnimStackCount = Importer->GetAnimStackCount();
			
			LOG_CORE_DEBUG("    Number of animation stacks: {}", AnimStackCount);
			LOG_CORE_DEBUG("    Active animation stack: \"{}\"",
				Importer->GetActiveAnimStackName());
			
			for (int i = 0; i < AnimStackCount; i++)
			{
				FbxTakeInfo* lTakeInfo = Importer->GetTakeInfo(i);
			
				LOG_CORE_INFO("    Animation Stack {}", i);
				LOG_CORE_INFO("         Name: \"{}\"", lTakeInfo->mName.Buffer());
				LOG_CORE_INFO("         Description: \"{}\"",
					lTakeInfo->mDescription.Buffer());
			
				// Change the value of the import name if the animation stack should
				// be imported under a different name.
				LOG_CORE_INFO("         Import Name: \"{}\"", lTakeInfo->mImportName.Buffer());
			
				// Set the value of the import state to false
				// if the animation stack should be not be imported.
				LOG_CORE_INFO("         Import State: {}", lTakeInfo->mSelect ? "true" : "false");
			}

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

		// The import file may have a password
		if (lStatus == false && Importer->GetStatus().GetCode() == FbxStatus::ePasswordError) {
			LOG_CORE_ERROR(L"File not imported, protected by password");
		}

		// Destroy the importer
		Importer->Destroy();

		return lStatus;
	}

	void FBXLoader::ExtractVertexData(FbxMesh * pMesh, MeshData & OutData) {
		int PolygonCount = pMesh->GetPolygonCount();
		FbxVector4 * ControlPoints = pMesh->GetControlPoints();
		int PolygonIndex; int PolygonVertexIndex;
		int MaterialIndex;
		bool bWarned = false;

		OutData.UVChannels = Math::Clamp(pMesh->GetElementUVCount(), 0, 2);
		OutData.hasNormals = pMesh->GetElementNormalCount() != 0;
		OutData.hasVertexColor = pMesh->GetElementVertexColorCount() != 0;

		int VertexIndex = 0;
		for (PolygonIndex = 0; PolygonIndex < PolygonCount; ++PolygonIndex) {

			int PolygonVertexSize = pMesh->GetPolygonSize(PolygonIndex);

			StaticVertex Vertex;

			for (PolygonVertexIndex = 0; PolygonVertexIndex < PolygonVertexSize; ++PolygonVertexIndex) {
				int ControlPointIndex = pMesh->GetPolygonVertex(PolygonIndex, PolygonVertexIndex);

				Vertex.Position.x = (float)ControlPoints[ControlPointIndex][0];
				Vertex.Position.y = (float)ControlPoints[ControlPointIndex][1];
				Vertex.Position.z = (float)ControlPoints[ControlPointIndex][2];

				OutData.Bounding.Add(Vertex.Position);

				ExtractTextureCoords(pMesh, Vertex, ControlPointIndex, PolygonIndex, PolygonVertexIndex);
				ExtractNormal(pMesh, Vertex, ControlPointIndex, VertexIndex);
				ExtractVertexColor(pMesh, Vertex, ControlPointIndex, VertexIndex);
				OutData.hasTangents = ExtractTangent(pMesh, Vertex, ControlPointIndex, VertexIndex);

				VertexIndex++;
				OutData.StaticVertices.push_back(Vertex);
			}
			MaterialIndex = ExtractMaterialIndex(pMesh, PolygonIndex);

			if (PolygonVertexSize < 4) {
				OutData.Faces.push_back(IntVector3(VertexIndex - 3, VertexIndex - 2, VertexIndex - 1));
				// if (OutData.Subdivisions.find(MaterialIndex) != OutData.Subdivisions.end())
					// OutData.Subdivisions[MaterialIndex].push_back(IntVector3(VertexIndex - 3, VertexIndex - 2, VertexIndex - 1));
			}
			else {
				OutData.Faces.push_back(IntVector3(VertexIndex - 3, VertexIndex - 2, VertexIndex - 1));
				OutData.Faces.push_back(IntVector3(VertexIndex - 4, VertexIndex - 3, VertexIndex - 1));
				// if (OutData.Subdivisions.find(MaterialIndex) != OutData.Subdivisions.end()) {
				// 	OutData.Subdivisions[MaterialIndex].push_back(IntVector3(VertexIndex - 3, VertexIndex - 2, VertexIndex - 1));
				// 	OutData.Subdivisions[MaterialIndex].push_back(IntVector3(VertexIndex - 4, VertexIndex - 3, VertexIndex - 1));
				// }
			}

			if (PolygonVertexSize > 4 && !bWarned) {
				bWarned = true;
				LOG_CORE_WARN(L"The model has n-gons, this may lead to unwanted geometry");
			}
		}

		OutData.ComputeTangents();
	}

	void FBXLoader::ExtractTextureCoords(
		class FbxMesh * pMesh, StaticVertex & Vertex,
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
		FbxMesh * pMesh, StaticVertex & Vertex,
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
		FbxMesh * pMesh, StaticVertex & Vertex,
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
		} else {
			Vertex.Color = { 1.F };
		}
	}

	int FBXLoader::ExtractMaterialIndex(FbxMesh * pMesh, const int & PolygonIndex) {
		int MaterialIndex = pMesh->GetElementMaterialCount();

		if (MaterialIndex > 0) {
			FbxGeometryElementMaterial* ElementMaterial = pMesh->GetElementMaterial();
			if (ElementMaterial->GetMappingMode() == FbxGeometryElement::eAllSame) {
				return ElementMaterial->GetIndexArray().GetAt(0);
			}
			else {
				return ElementMaterial->GetIndexArray().GetAt(PolygonIndex);
			}
		}

		return -1;
	}

	bool FBXLoader::ExtractTangent(FbxMesh * pMesh, StaticVertex & Vertex, const int & ControlPointIndex, const int & VertexIndex) {
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

	void FBXLoader::ExtractNodeTransform(FbxNode * pNode, class FbxScene * pScene, Transform & Transformation) {
		// FBX Doxumentation ---- T * Roff * Rp * Rpre * R * Rpost - 1 * Rp - 1 * Soff * Sp * S * Sp - 1;
		FbxVector4 DScaling = { 1.0, 1.0, 1.0, 1.0 };
		FbxVector4 DZero = { 0.0, 0.0, 0.0, 1.0 };
		FbxAMatrix OS = FbxAMatrix(DZero, DZero, pNode->GeometricScaling.Get());
		FbxAMatrix OR = FbxAMatrix(DZero, pNode->GeometricRotation.Get(), DScaling);
		FbxAMatrix OT = FbxAMatrix(pNode->GeometricTranslation.Get(), DZero, DScaling);
		FbxAMatrix S  = FbxAMatrix(DZero, DZero, pNode->LclScaling.Get());
		FbxAMatrix R  = FbxAMatrix(DZero, pNode->LclRotation.Get(), DScaling);
		FbxAMatrix T  = FbxAMatrix(pNode->LclTranslation.Get(), DZero, DScaling);

		FbxAMatrix ResultSRT = T * R * S * OT * OR * OS;

		FbxDouble3 RScaling = ResultSRT.GetS();
		FbxDouble3 RRotation = ResultSRT.GetR();
		FbxDouble3 RTranslation = ResultSRT.GetT();


		// FbxAMatrix GMatrix = pNode->EvaluateGlobalTransform();
		// FbxAMatrix GIMatrix = GMatrix.Inverse();
		// FbxVector4 RotationA = pNode->LclRotation.Get();
		// FbxVector4 RotationB = GMatrix.GetR();
		// FbxVector4 RotationC = pNode->GetPreRotation(FbxNode::eSourcePivot);
		// FbxDouble3 Rotation = (
		// 	FbxAMatrix(FbxVector4(), RotationA, FbxVector4({ 1.0, 1.0, 1.0, 1.0 })) * FbxAMatrix(FbxVector4(), RotationC, FbxVector4({ 1.0, 1.0, 1.0, 1.0 }))
		// ).GetR();
		// FbxVector4 ScaleA = (pNode->GetParent() == NULL || pNode->GetParent()->GetParent() == NULL) ? GMatrix.GetS() : FbxVector4(1.0, 1.0, 1.0);
		// FbxDouble3 Scale = (FbxVector4)pNode->LclScaling.Get() / ScaleA;
		// FbxDouble3 TranslationA = GMatrix.GetT();
		// FbxDouble3 TranslationB = pNode->LclTranslation.Get();
		// FbxDouble3 TranslationC = pNode->GetGeometricTranslation(FbxNode::eSourcePivot);

		Transformation.Position[0] = (float)RTranslation[0];
		Transformation.Position[1] = (float)RTranslation[1];
		Transformation.Position[2] = (float)RTranslation[2];
		Transformation.Rotation = Quaternion::EulerAngles({ 
			(float)(RRotation[0]),
			(float)(RRotation[1]), 
			(float)(RRotation[2])
		});
		Transformation.Scale[0] = (float)RScaling[0];
		Transformation.Scale[1] = (float)RScaling[1];
		Transformation.Scale[2] = (float)RScaling[2];
	}

	bool FBXLoader::LoadModel(ModelParser::ModelDataInfo & Info, const ModelParser::ParsingOptions & Options) {
		if (gSdkManager == NULL)
			return false;

		Timestamp Timer;
		Timer.Begin();

		FbxScene* Scene = FbxScene::Create(gSdkManager, Text::WideToNarrow(Options.File->GetShortPath()).c_str());

		bool bStatus = LoadScene(Scene, Options.File);
		if (bStatus == false) return false;

		FbxAxisSystem::OpenGL.ConvertScene(Scene);
		FbxSystemUnit SceneSystemUnit = Scene->GetGlobalSettings().GetSystemUnit();
		if (SceneSystemUnit != FbxSystemUnit::m) {
			FbxSystemUnit::m.ConvertScene(Scene);
		}

		// FbxGeometryConverter GeomConverter(gSdkManager);
		// GeomConverter.Triangulate(Scene, true);
		const int NodeCount = Scene->GetSrcObjectCount<FbxNode>();
		size_t TotalAllocatedSize = 0;
		TDictionary<FbxUInt64, size_t> NodeMap;
		for (int NodeIndex = 0; NodeIndex < NodeCount; NodeIndex++) {
			FbxNode * Node = Scene->GetSrcObject<FbxNode>(NodeIndex);
			// NodeMap.emplace(Node->GetUniqueID(), Info.ModelNodes.size());
			// Info.ModelNodes.push_back(ModelNode(Node->GetName()));
		}

		Timer.Stop();
		LOG_CORE_INFO(L"├> Readed and parsed in {:0.3f}ms", Timer.GetDeltaTime<Time::Mili>());

		Timer.Begin();
		for (int NodeIndex = 0; NodeIndex < NodeCount; NodeIndex++) {
			FbxNode * Node = Scene->GetSrcObject<FbxNode>(NodeIndex);
			FbxMesh * lMesh = Node->GetMesh();
			// ModelNode& CurrentSceneNode = Info.ModelNodes[NodeMap[Node->GetUniqueID()]];
			// ExtractNodeTransform(Node, Scene, CurrentSceneNode.LocalTransform);
			// if (Node->GetParent() != NULL)
			// 	CurrentSceneNode.ParentIndex = NodeMap[Node->GetParent()->GetUniqueID()];
			// int ChildCount = Node->GetChildCount();
			// for (int ChildIndex = 0; ChildIndex < ChildCount; ++ChildIndex) {
			// 	CurrentSceneNode.ChildrenIndices.push_back(NodeMap[Node->GetChild(ChildIndex)->GetUniqueID()]);
			// }
			// if (lMesh) {
			// 	CurrentSceneNode.bHasMesh = true;
			// 	CurrentSceneNode.MeshKey = Info.Meshes.size();
			// 	Info.Meshes.push_back(MeshData());
			// 	MeshData & CurrentMeshData = Info.Meshes.back();
			// 	CurrentMeshData.Name = lMesh->GetName();
			// 	if (CurrentMeshData.Name.size() == 0) CurrentMeshData.Name = CurrentSceneNode.Name;
			// 	const int MaterialCount = Node->GetMaterialCount();
			// 	for (int MaterialIndex = 0; MaterialIndex < MaterialCount; ++MaterialIndex) {
			// 		// CurrentMeshData.Materials.insert({ MaterialIndex, Node->GetMaterial(MaterialIndex)->GetName() });
			// 		// CurrentMeshData.Subdivisions.insert({ MaterialIndex, MeshFaces() });
			// 	}
			// 	ExtractVertexData(lMesh, CurrentMeshData);
			// 
#ifdef ES_DE// BUG
			// 	LOG_CORE_DEBUG(L"├> Parsed {0}	vertices in {1}	at [{2:d}]'{3}'",
			// 		Text::FormatUnit(CurrentMeshData.StaticVertices.size(), 2),
			// 		Text::FormatData(sizeof(IntVector3) * CurrentMeshData.Faces.size() + sizeof(StaticVertex) * CurrentMeshData.StaticVertices.size(), 2),
			// 		Info.Meshes.size(),
			// 		Text::NarrowToWide(CurrentSceneNode.Name)
			// 	);
#endif		// 
			// 	TotalAllocatedSize += sizeof(IntVector3) * CurrentMeshData.Faces.size() + sizeof(StaticVertex) * CurrentMeshData.StaticVertices.size();
			// }
		}

		Timer.Stop();
		LOG_CORE_INFO(L"└> Allocated {0} in {1:.2f}ms", Text::FormatData(TotalAllocatedSize, 2), Timer.GetDeltaTime<Time::Mili>());

		return bStatus;
	}

}