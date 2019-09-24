
#include "CoreMinimal.h"
#include "Core/EmptySource.h"
#include "Core/CoreTime.h"
#include "Core/Window.h"
#include "Core/SpaceLayer.h"
#include "Core/GameObject.h"
#include "Core/Transform.h"
#include "Core/Input.h"

#include "Math/CoreMath.h"
#include "Math/Physics.h"

#include "Utility/TextFormattingMath.h"
#if defined(ES_PLATFORM_WINDOWS) & defined(ES_PLATFORM_CUDA)
#include "CUDA/CoreCUDA.h"
#endif

#include "Rendering/Mesh.h"
#include "Rendering/MeshPrimitives.h"

#include "Rendering/RenderingDefinitions.h"
#include "Rendering/RenderTarget.h"
#include "Rendering/RenderStage.h"
#include "Rendering/RenderPipeline.h"
#include "Rendering/Shader.h"
#include "Rendering/Texture.h"
#include "Rendering/RenderTarget.h"
#include "Rendering/Material.h"

#include "Audio/AudioDevice.h"

#include "Files/FileManager.h"

#define RESOURCES_ADD_SHADERSTAGE
#define RESOURCES_ADD_SHADERPROGRAM
#include "Resources/ResourceManager.h"
#include "Resources/MeshManager.h"
#include "Resources/MaterialManager.h"
#include "Resources/ImageConversion.h"
#include "Resources/ShaderManager.h"
#include "Resources/TextureManager.h"
#include "Resources/AudioManager.h"

#include "Components/ComponentRenderable.h"

#include "Fonts/Font.h"
#include "Fonts/Text2DGenerator.h"

#include "Events/Property.h"

#include "../External/IMGUI/imgui.h"
#include "../External/SDL2/include/SDL_keycode.h"

#include "../Public/SandboxSpaceLayer.h"

using namespace EmptySource;

class SandboxLayer : public Layer {
private:
	Font FontFace;
	Text2DGenerator TextGenerator;
	Bitmap<UCharRed> FontAtlas;

	// --- Perpective matrix (ProjectionMatrix)
	VertexBufferPtr ModelMatrixBuffer;

	// TArray<MeshPtr> SceneModels;
	// TArray<MeshPtr> LightModels;
	MeshPtr SelectedMesh;
	WString SelectedMeshName;
	
	// --- Camera rotation, position Matrix
	float ViewSpeed = 3;
	Vector3 ViewOrientation;
	Matrix4x4 ViewMatrix; 
	Quaternion CameraRotation;
	Quaternion LastCameraRotation;
	Vector2 LastCursorPosition;
	Vector2 CursorPosition;

	Material UnlitMaterial = Material(L"UnlitMaterial");
	Material UnlitMaterialWire = Material(L"UnlitMaterialWire");
	Material RenderTextureMaterial = Material(L"RenderTextureMaterial");
	MaterialPtr RenderTextMaterial = std::make_shared<Material>(L"RenderTextMaterial");
	MaterialPtr RenderCubemapMaterial = std::make_shared<Material>(L"RenderCubemapMaterial");
	Material IntegrateBRDFMaterial = Material(L"IntegrateBRDFMaterial");
	Material HDRClampingMaterial = Material(L"HDRClampingMaterial");

	float SkyboxRoughness = 1.F;
	float MaterialMetalness = 1.F;
	float MaterialRoughness = 1.F;
	float LightIntencity = 20.F;

	TArray<Transform> Transforms; 
	Matrix4x4 TransformMat;
	Matrix4x4 InverseTransform; 
	Vector3 CameraRayDirection;
	
	TArray<int> ElementsIntersected;
	float MultiuseValue = 1;
	static const int TextCount = 4;
	float FontSize = 14;
	float FontBoldness = 0.55F;
	WString RenderingText[TextCount];
	Mesh DynamicMesh;
	Point2 TextPivot;

	Texture2DPtr EquirectangularTextureHDR;
	Texture2DPtr FontMap;
	CubemapPtr CubemapTexture;

	Transform TestArrowTransform;
	Vector3 TestArrowDirection = 0;
	float TestSphereVelocity = .5F;

	bool bRandomArray = false;
	const void* curandomStateArray = 0;

protected:

	void SetSceneSkybox(const WString & Path) {
		Bitmap<FloatRGB> Equirectangular;
		ImageConversion::LoadFromFile(Equirectangular, FileManager::GetFile(Path));

		Texture2DPtr EquirectangularTexture = Texture2D::Create(L"EquirectangularTexture",
			IntVector2(Equirectangular.GetWidth(), Equirectangular.GetHeight()),
			CF_RGB32F, FM_MinMagLinear, SAM_Repeat, CF_RGB32F,
			Equirectangular.PointerToValue()
		);
		EquirectangularTextureHDR = Texture2D::Create(L"EquirectangularTextureHDR",
			IntVector2(Equirectangular.GetWidth(), Equirectangular.GetHeight()), CF_RGB32F, FM_MinMagLinear, SAM_Repeat
		);
		if (TextureManager::GetInstance().GetTexture(L"EquirectangularTextureHDR") != NULL) {
			TextureManager::GetInstance().FreeTexture(L"EquirectangularTextureHDR");
		}
		TextureManager::GetInstance().AddTexture(L"EquirectangularTextureHDR", EquirectangularTextureHDR);

		{
			RenderTargetPtr Renderer = RenderTarget::Create();
			EquirectangularTextureHDR->Bind();
			HDRClampingMaterial.Use();
			HDRClampingMaterial.SetMatrix4x4Array("_ProjectionMatrix", Matrix4x4().PointerToValue());
			HDRClampingMaterial.SetTexture2D("_EquirectangularMap", EquirectangularTexture, 0);
			MeshPrimitives::Quad.BindSubdivisionVertexArray(0);
			Matrix4x4 QuadPosition = Matrix4x4::Translation({ 0, 0, 0 });
			HDRClampingMaterial.SetAttribMatrix4x4Array(
				"_iModelMatrix", 1, QuadPosition.PointerToValue(), ModelMatrixBuffer
			);

			Renderer->BindTexture2D(EquirectangularTextureHDR);
			Renderer->Clear();
			MeshPrimitives::Quad.DrawSubdivisionInstanciated(1, 0);
			EquirectangularTextureHDR->GenerateMipMaps();
		}

		Material EquirectangularToCubemapMaterial = Material(L"EquirectangularToCubemapMaterial");
		EquirectangularToCubemapMaterial.SetShaderProgram(ShaderManager::GetInstance().GetProgram(L"EquirectangularToCubemap"));
		EquirectangularToCubemapMaterial.CullMode = CM_None;
		EquirectangularToCubemapMaterial.CullMode = CM_ClockWise;

		CubemapTexture = Cubemap::Create(L"CubemapTexture", Equirectangular.GetHeight() / 2, CF_RGB16F, FM_MinMagLinear, SAM_Clamp);
		CubemapTexture->ConvertFromHDREquirectangular(EquirectangularTextureHDR, &EquirectangularToCubemapMaterial, true);
		if (TextureManager::GetInstance().GetTexture(L"CubemapTexture") != NULL) {
			TextureManager::GetInstance().FreeTexture(L"CubemapTexture");
		}
		TextureManager::GetInstance().AddTexture(L"CubemapTexture", CubemapTexture);
	}

	virtual void OnAttach() override {

		AudioManager::GetInstance().LoadAudioFromFile(L"6503.wav", L"Resources/Sounds/6503.wav");
		AudioManager::GetInstance().LoadAudioFromFile(L"Hololooo.wav", L"Resources/Sounds/Hololooo.wav");

		TextureManager& TextureMng = TextureManager::GetInstance();
		TextureMng.LoadImageFromFile(L"WhiteTexture",   CF_RGB,  FM_MinMagNearest, SAM_Repeat, true, true, L"Resources/Textures/White.jpg");
		TextureMng.LoadImageFromFile(L"BlackTexture",   CF_RGB,  FM_MinMagNearest, SAM_Repeat, true, true, L"Resources/Textures/Black.jpg");
		TextureMng.LoadImageFromFile(L"NormalTexture",  CF_RGBA, FM_MinMagNearest, SAM_Repeat, true, true, L"Resources/Textures/Normal.jpg");
		TextureMng.LoadImageFromFile(L"FlowMapTexture", CF_RGB,  FM_MinMagLinear,  SAM_Repeat, true, true, L"Resources/Textures/FlowMap.jpg");

		ShaderManager& ShaderMng = ShaderManager::GetInstance();
		ShaderMng.LoadResourcesFromFile(L"Resources/Resources.yaml");
		ShaderMng.CreateProgram(L"PBRShader", L"Resources/Shaders/PBR.shader");

		TextureMng.LoadImageFromFile(L"Sponza/CulumnAAlbedoTexture",      CF_RGBA, FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Column_a_diffuse.tga");
		TextureMng.LoadImageFromFile(L"Sponza/CulumnARoughnessTexture",   CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Column_a_roughness.tga");
		TextureMng.LoadImageFromFile(L"Sponza/CulumnANormalTexture",      CF_RGB,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Column_a_normal.tga");
		TextureMng.LoadImageFromFile(L"Sponza/CulumnBAlbedoTexture",      CF_RGBA, FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Column_b_diffuse.tga");
		TextureMng.LoadImageFromFile(L"Sponza/CulumnBRoughnessTexture",   CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Column_b_roughness.tga");
		TextureMng.LoadImageFromFile(L"Sponza/CulumnBNormalTexture",      CF_RGB,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Column_b_normal.tga");
		TextureMng.LoadImageFromFile(L"Sponza/CulumnCAlbedoTexture",      CF_RGBA, FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Column_c_diffuse.tga");
		TextureMng.LoadImageFromFile(L"Sponza/CulumnCNormalTexture",      CF_RGB,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Column_c_normal.tga");
		TextureMng.LoadImageFromFile(L"Sponza/CulumnCRoughnessTexture",   CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Column_c_roughness.tga");
		TextureMng.LoadImageFromFile(L"Sponza/BricksAAlbedoTexture",      CF_RGBA, FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Bricks_a_Albedo.tga");
		TextureMng.LoadImageFromFile(L"Sponza/BricksARoughnessTexture",   CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Bricks_a_Roughness.tga");
		TextureMng.LoadImageFromFile(L"Sponza/BricksANormalTexture",      CF_RGB,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Bricks_a_Normal.tga");
		TextureMng.LoadImageFromFile(L"Sponza/CeilingAlbedoTexture",      CF_RGBA, FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Ceiling_diffuse.tga");
		TextureMng.LoadImageFromFile(L"Sponza/CeilingRoughnessTexture",   CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Ceiling_Roughness.tga");
		TextureMng.LoadImageFromFile(L"Sponza/CeilingNormalTexture",      CF_RGB,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Ceiling_Normal.tga");
		TextureMng.LoadImageFromFile(L"Sponza/FloorAlbedoTexture",        CF_RGBA, FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Floor_diffuse.tga");
		TextureMng.LoadImageFromFile(L"Sponza/FloorRoughnessTexture",     CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Floor_roughness.tga");
		TextureMng.LoadImageFromFile(L"Sponza/FloorNormalTexture",        CF_RGB,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Floor_normal.tga");
		TextureMng.LoadImageFromFile(L"Sponza/RoofAlbedoTexture",         CF_RGBA, FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Roof_diffuse.tga");
		TextureMng.LoadImageFromFile(L"Sponza/RoofRoughnessTexture",      CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Roof_roughness.tga");
		TextureMng.LoadImageFromFile(L"Sponza/RoofNormalTexture",         CF_RGB,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Roof_normal.tga");
		TextureMng.LoadImageFromFile(L"Sponza/ArchAlbedoTexture",         CF_RGBA, FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Arch_diffuse.tga");
		TextureMng.LoadImageFromFile(L"Sponza/ArchRoughnessTexture",      CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Arch_roughness.tga");
		TextureMng.LoadImageFromFile(L"Sponza/ArchNormalTexture",         CF_RGB,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Arch_normal.tga");
		TextureMng.LoadImageFromFile(L"Sponza/ThornAlbedoTexture",        CF_RGBA, FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Thorn_diffuse.tga");
		TextureMng.LoadImageFromFile(L"Sponza/ThornRoughnessTexture",     CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Thorn_roughness.tga");
		TextureMng.LoadImageFromFile(L"Sponza/ThornNormalTexture",        CF_RGB,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Thorn_normal.tga");
		TextureMng.LoadImageFromFile(L"Sponza/VaseAlbedoTexture",         CF_RGBA, FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Vase_diffuse.tga");
		TextureMng.LoadImageFromFile(L"Sponza/VaseRoughnessTexture",      CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Vase_roughness.tga");
		TextureMng.LoadImageFromFile(L"Sponza/VaseNormalTexture",         CF_RGB,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Vase_normal.tga");
		TextureMng.LoadImageFromFile(L"Sponza/VaseRoundAlbedoTexture",    CF_RGBA, FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/VaseRound_diffuse.tga");
		TextureMng.LoadImageFromFile(L"Sponza/VaseRoundRoughnessTexture", CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/VaseRound_roughness.tga");
		TextureMng.LoadImageFromFile(L"Sponza/VaseRoundNormalTexture",    CF_RGB,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/VaseRound_normal.tga");
		TextureMng.LoadImageFromFile(L"Sponza/VaseHangAlbedoTexture",     CF_RGBA, FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/VaseHanging_diffuse.tga");
		TextureMng.LoadImageFromFile(L"Sponza/VaseHangRoughnessTexture",  CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/VaseHanging_roughness.tga");
		TextureMng.LoadImageFromFile(L"Sponza/VaseHangNormalTexture",     CF_RGB,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/VaseHanging_normal.tga");
		TextureMng.LoadImageFromFile(L"Sponza/VasePlantAlbedoTexture",    CF_RGBA, FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/VasePlant_diffuse.tga");
		TextureMng.LoadImageFromFile(L"Sponza/VasePlantRoughnessTexture", CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/VasePlant_roughness.tga");
		TextureMng.LoadImageFromFile(L"Sponza/VasePlantNormalTexture",    CF_RGB,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/VasePlant_normal.tga");
		TextureMng.LoadImageFromFile(L"Sponza/DetailsAlbedoTexture",      CF_RGBA, FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Details_diffuse.tga");
		TextureMng.LoadImageFromFile(L"Sponza/DetailsRoughnessTexture",   CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Details_roughness.tga");
		TextureMng.LoadImageFromFile(L"Sponza/DetailsMetallicTexture",    CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Details_metallic.tga");
		TextureMng.LoadImageFromFile(L"Sponza/DetailsNormalTexture",      CF_RGB,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Details_normal.tga");
		TextureMng.LoadImageFromFile(L"Sponza/CurtainBlueAlbedoTexture",  CF_RGBA, FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Curtain_Blue_diffuse.tga");
		TextureMng.LoadImageFromFile(L"Sponza/CurtainGreenAlbedoTexture", CF_RGBA, FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Curtain_Green_diffuse.tga");
		TextureMng.LoadImageFromFile(L"Sponza/CurtainRedAlbedoTexture",   CF_RGBA, FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Curtain_Red_diffuse.tga");
		TextureMng.LoadImageFromFile(L"Sponza/CurtainRoughnessTexture",   CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Curtain_roughness.tga");
		TextureMng.LoadImageFromFile(L"Sponza/CurtainMetallicTexture",    CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Curtain_metallic.tga");
		TextureMng.LoadImageFromFile(L"Sponza/CurtainNormalTexture",      CF_RGB,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Sponza_Curtain_Red_normal.tga");
		TextureMng.LoadImageFromFile(L"Sponza/LionAlbedoTexture",         CF_RGBA, FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Lion_Albedo.tga");
		TextureMng.LoadImageFromFile(L"Sponza/LionNormalTexture",         CF_RGB,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Lion_Normal.tga");
		TextureMng.LoadImageFromFile(L"Sponza/LionRoughnessTexture",      CF_RGB,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SponzaPBR/Lion_Roughness.tga");
		TextureMng.LoadImageFromFile(L"EscafandraMV1971AlbedoTexture",    CF_RGBA, FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/EscafandraMV1971_BaseColor.png");
		TextureMng.LoadImageFromFile(L"EscafandraMV1971MetallicTexture",  CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/EscafandraMV1971_Metallic.png");
		TextureMng.LoadImageFromFile(L"EscafandraMV1971RoughnessTexture", CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/EscafandraMV1971_Roughness.png");
		TextureMng.LoadImageFromFile(L"EscafandraMV1971NormalTexture",    CF_RGB,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/EscafandraMV1971_Normal.png");
		TextureMng.LoadImageFromFile(L"PirateBarrelAlbedoTexture",        CF_RGBA, FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/PirateProps_Barrel_Texture_Color.png");
		TextureMng.LoadImageFromFile(L"PirateBarrelMetallicTexture",      CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/PirateProps_Barrel_Texture_Metal.png");
		TextureMng.LoadImageFromFile(L"PirateBarrelRoughnessTexture",     CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/PirateProps_Barrel_Texture_Roughness.png");
		TextureMng.LoadImageFromFile(L"PirateBarrelNormalTexture",        CF_RGB,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/PirateProps_Barrel_Texture_Normal.png");
		TextureMng.LoadImageFromFile(L"FlamerGunAlbedoTexture",           CF_RGBA, FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/Flamer_DefaultMaterial_albedo.jpg");
		TextureMng.LoadImageFromFile(L"FlamerGunMetallicTexture",         CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/Flamer_DefaultMaterial_metallic.jpg");
		TextureMng.LoadImageFromFile(L"FlamerGunRoughnessTexture",        CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/Flamer_DefaultMaterial_roughness.jpg");
		TextureMng.LoadImageFromFile(L"FlamerGunNormalTexture",           CF_RGB,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/Flamer_DefaultMaterial_normal.jpeg");
		TextureMng.LoadImageFromFile(L"SiFi/CeilingTileAlbedoTexture",    CF_RGBA, FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SiFi/T_CeilingTile_COL.tga");
		TextureMng.LoadImageFromFile(L"SiFi/CeilingTileMetallicTexture",  CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SiFi/T_CeilingTile_MET.tga");
		TextureMng.LoadImageFromFile(L"SiFi/CeilingTileRoughnessTexture", CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SiFi/T_CeilingTile_RGH.tga");
		TextureMng.LoadImageFromFile(L"SiFi/CeilingTileNormalTexture",    CF_RGB,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SiFi/T_CeilingTile_NRM.tga");
		TextureMng.LoadImageFromFile(L"SiFi/DoorPanelAlbedoTexture",      CF_RGBA, FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SiFi/T_DoorPanel_COL.tga");
		TextureMng.LoadImageFromFile(L"SiFi/DoorPanelMetallicTexture",    CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SiFi/T_DoorPanel_MET.tga");
		TextureMng.LoadImageFromFile(L"SiFi/DoorPanelRoughnessTexture",   CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SiFi/T_DoorPanel_RGH.tga");
		TextureMng.LoadImageFromFile(L"SiFi/DoorPanelNormalTexture",      CF_RGB,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SiFi/T_DoorPanel_NRM.tga");
		TextureMng.LoadImageFromFile(L"SiFi/GroundTileAlbedoTexture",     CF_RGBA, FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SiFi/T_GroundTile_COL.tga");
		TextureMng.LoadImageFromFile(L"SiFi/GroundTileMetallicTexture",   CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SiFi/T_GroundTile_MET.tga");
		TextureMng.LoadImageFromFile(L"SiFi/GroundTileRoughnessTexture",  CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SiFi/T_GroundTile_RGH.tga");
		TextureMng.LoadImageFromFile(L"SiFi/GroundTileNormalTexture",     CF_RGB,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/SiFi/T_GroundTile_NRM.tga");
		TextureMng.LoadImageFromFile(L"PonyCarExteriorAlbedoTexture",     CF_RGBA, FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/Body_dDo_d_orange.jpeg");
		TextureMng.LoadImageFromFile(L"PonyCarExteriorMetallicTexture",   CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/Body_dDo_s.jpeg");
		TextureMng.LoadImageFromFile(L"PonyCarExteriorRoughnessTexture",  CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/Body_dDo_g.jpg");
		TextureMng.LoadImageFromFile(L"PonyCarExteriorNormalTexture",     CF_RGB,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/Body_dDo_n.jpg");
		TextureMng.LoadImageFromFile(L"PonyCarInteriorAlbedoTexture",     CF_RGBA, FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/Interior_dDo_d_black.jpeg");
		TextureMng.LoadImageFromFile(L"PonyCarInteriorMetallicTexture",   CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/Interior_dDo_s.jpeg");
		TextureMng.LoadImageFromFile(L"PonyCarInteriorRoughnessTexture",  CF_Red,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/Interior_dDo_g.jpg");
		TextureMng.LoadImageFromFile(L"PonyCarInteriorNormalTexture",     CF_RGB,  FM_MinMagLinear, SAM_Repeat, true, true, L"Resources/Textures/Interior_dDo_n.jpg");
	}

	virtual void OnImGuiRender() override {
		static TexturePtr TextureSample = Texture2D::Create(L"TextureSample", IntVector2(1024, 1024), CF_RGBA, FM_MinMagLinear, SAM_Repeat);

		TArray<WString> TextureNameList = TextureManager::GetInstance().GetResourceNames();
		TArray<NString> NarrowTextureNameList(TextureNameList.size());
		for (int i = 0; i < NarrowTextureNameList.size(); ++i)
			NarrowTextureNameList[i] = Text::WideToNarrow((TextureNameList)[i]);

		const NChar* SkyBoxes[]{
			"Resources/Textures/Arches_E_PineTree_3k.hdr",
			"Resources/Textures/doge2.hdr",
			"Resources/Textures/ennis.hdr",
			"Resources/Textures/Factory_Catwalk_2k.hdr",
			"Resources/Textures/flower_road_2k.hdr",
			"Resources/Textures/grace-new.hdr",
			"Resources/Textures/Milkyway.hdr",
			"Resources/Textures/OutdoorResidentialRiverwalkAfternoon.hdr",
			"Resources/Textures/spruit_sunrise_2k.hdr",
			"Resources/Textures/studio_small_03_2k.hdr",
			"Resources/Textures/tucker_wreck_2k.hdr"
		};

		static int CurrentTexture = 0;
		static int CurrentSkybox = 0;
		static float SampleLevel = 0.F;
		static float Gamma = 2.2F;
		static bool ColorFilter[4] = {true, true, true, true};
		int bMonochrome = (ColorFilter[0] + ColorFilter[1] + ColorFilter[2] + ColorFilter[3]) == 1;

		TexturePtr SelectedTexture = TextureManager::GetInstance().GetTexture(TextureNameList[CurrentTexture]);
		if (SelectedTexture) {
			int bCubemap;
			if (!(bCubemap = SelectedTexture->GetDimension() == ETextureDimension::Cubemap)) {
				RenderTargetPtr Renderer = RenderTarget::Create();
				RenderTextureMaterial.Use();
				RenderTextureMaterial.SetFloat1Array("_Gamma", &Gamma);
				RenderTextureMaterial.SetInt1Array("_Monochrome", &bMonochrome);
				RenderTextureMaterial.SetFloat4Array("_ColorFilter",
					Vector4(ColorFilter[0] ? 1.F : 0.F, ColorFilter[1] ? 1.F : 0.F, ColorFilter[2] ? 1.F : 0.F, ColorFilter[3] ? 1.F : 0.F)
					.PointerToValue()
				);
				RenderTextureMaterial.SetMatrix4x4Array("_ProjectionMatrix", Matrix4x4().PointerToValue());
				RenderTextureMaterial.SetInt1Array("_IsCubemap", &bCubemap);
				SelectedTexture->Bind();
				RenderTextureMaterial.SetTexture2D("_MainTexture", SelectedTexture, 0);
				RenderTextureMaterial.SetTextureCubemap("_MainTextureCube", SelectedTexture, 1);
				float LODLevel = SampleLevel * (float)SelectedTexture->GetMipMapCount();
				RenderTextureMaterial.SetFloat1Array("_Lod", &LODLevel);

				Renderer->Bind();
				MeshPrimitives::Quad.BindSubdivisionVertexArray(0);
				Matrix4x4 QuadPosition = Matrix4x4::Scaling({ 1, -1, 1 });
				RenderTextureMaterial.SetAttribMatrix4x4Array(
					"_iModelMatrix", 1, QuadPosition.PointerToValue(), ModelMatrixBuffer
				);

				Renderer->BindTexture2D(TextureSample);
				Renderer->Clear();
				MeshPrimitives::Quad.DrawSubdivisionInstanciated(1, 0);
			}
			if (bCubemap = SelectedTexture->GetDimension() == ETextureDimension::Cubemap) {
				RenderTargetPtr Renderer = RenderTarget::Create();
				RenderTextureMaterial.Use();
				RenderTextureMaterial.SetFloat1Array("_Gamma", &Gamma);
				RenderTextureMaterial.SetInt1Array("_Monochrome", &bMonochrome);
				RenderTextureMaterial.SetFloat4Array("_ColorFilter",
					Vector4(ColorFilter[0] ? 1.F : 0.F, ColorFilter[1] ? 1.F : 0.F, ColorFilter[2] ? 1.F : 0.F, ColorFilter[3] ? 1.F : 0.F)
					.PointerToValue()
				);
				RenderTextureMaterial.SetMatrix4x4Array("_ProjectionMatrix", Matrix4x4().PointerToValue());
				RenderTextureMaterial.SetInt1Array("_IsCubemap", &bCubemap);
				SelectedTexture->Bind();
				RenderTextureMaterial.SetTexture2D("_MainTexture", SelectedTexture, 0);
				RenderTextureMaterial.SetTextureCubemap("_MainTextureCube", SelectedTexture, 1);
				float LODLevel = SampleLevel * (float)SelectedTexture->GetMipMapCount();
				RenderTextureMaterial.SetFloat1Array("_Lod", &LODLevel);

				Renderer->Bind();
				MeshPrimitives::Quad.BindSubdivisionVertexArray(0);
				Matrix4x4 QuadPosition = Matrix4x4::Scaling({ 1, -1, 1 });
				RenderTextureMaterial.SetAttribMatrix4x4Array(
					"_iModelMatrix", 1, QuadPosition.PointerToValue(), ModelMatrixBuffer
				);

				Renderer->BindTexture2D(TextureSample);
				Renderer->Clear();
				MeshPrimitives::Quad.DrawSubdivisionInstanciated(1, 0);
			}
		}

		ImGui::Begin("Meshes", 0, ImVec2(250, 300)); 
		{
			TArray<WString> MeshResourcesList = MeshManager::GetInstance().GetResourceNames();
			if (MeshResourcesList.size() > 0) {
				TArray<NString> NarrowMeshResourcesList(MeshResourcesList.size());
				for (int i = 0; i < NarrowMeshResourcesList.size(); ++i)
					NarrowMeshResourcesList[i] = Text::WideToNarrow((MeshResourcesList)[i]);

				static int Selection = 0;
				ImGui::ListBox("Mesh List", &Selection, [](void * Data, int indx, const char ** outText) -> bool {
					TArray<NString>* Items = (TArray<NString> *)Data;
					if (outText) *outText = (*Items)[indx].c_str();
					return true;
				}, &NarrowMeshResourcesList, (int)NarrowMeshResourcesList.size());
				ImGui::Text("Selected Mesh: %s", NarrowMeshResourcesList[Selection].c_str());
				MeshPtr SelectedMesh = MeshManager::GetInstance().GetMesh(MeshResourcesList[Selection]);
				if (SelectedMesh) {
					ImGui::Text("Triangle count: %d", SelectedMesh->GetMeshData().Faces.size());
					ImGui::Text("Vertices count: %d", SelectedMesh->GetMeshData().Vertices.size());
					ImGui::Text("Tangents: %s", SelectedMesh->GetMeshData().hasTangents ? "true" : "false");
					ImGui::Text("Normals: %s", SelectedMesh->GetMeshData().hasNormals ? "true" : "false");
					ImGui::Text("UVs: %d", SelectedMesh->GetMeshData().TextureCoordsCount);
					ImGui::Text("Vertex Color: %s", SelectedMesh->GetMeshData().hasVertexColor ? "true" : "false");
					ImGui::InputFloat3("##BBox0", (float *)&SelectedMesh->GetMeshData().Bounding.xMin, 10, ImGuiInputTextFlags_ReadOnly);
					ImGui::InputFloat3("##BBox1", (float *)&SelectedMesh->GetMeshData().Bounding.yMin, 10, ImGuiInputTextFlags_ReadOnly);
					ImGui::TextUnformatted("Materials:");
					for (auto KeyValue : SelectedMesh->GetMeshData().Materials) {
						ImGui::BulletText("%s : %d", KeyValue.second.c_str(), KeyValue.first);
					}
				}
			}
		}
		ImGui::End();

		ImGui::Begin("Frame Rate Graph", 0, ImVec2(250, 300));
		{
			static unsigned char FrameIndex = 0;
			static float FrameRateHist[255];
			for (unsigned int i = 1; i < 255; i++) {
				FrameRateHist[i - 1] = FrameRateHist[i];
			}
			FrameRateHist[254] = (float)Time::GetDeltaTime<Time::Mili>();
			ImGui::PushItemWidth(-1); ImGui::PlotLines("##FrameRateHistory",
				FrameRateHist, 255, NULL, 0, 0.F, 60.F, ImVec2(0, 120)); ImGui::NextColumn();
		}
		ImGui::End();

		ImGui::Begin("Shaders", 0, ImVec2(250, 300)); 
		{
			TArray<IName> ShaderNameList = ShaderManager::GetInstance().GetResourceShaderNames();
			TArray<NString> NarrowShaderNameList(ShaderNameList.size());
			for (int i = 0; i < ShaderNameList.size(); ++i)
				NarrowShaderNameList[i] = (ShaderNameList)[i].GetNarrowDisplayName();

			if (ShaderNameList.size() > 0) {
				static int Selection = 0;
				ImGui::ListBox("Shader List", &Selection, [](void * Data, int indx, const char ** outText) -> bool {
					TArray<NString>* Items = (TArray<NString> *)Data;
					if (outText) *outText = (*Items)[indx].c_str();
					return true;
				}, &NarrowShaderNameList, (int)NarrowShaderNameList.size());
			
				RShaderPtr SelectedShader = ShaderManager::GetInstance().GetProgram(ShaderNameList[Selection]);
				if (SelectedShader) {
					ImGui::Text("Selected Shader: %s", NarrowShaderNameList[Selection].c_str());
					if (ImGui::Button("Reload Shader")) {
						SelectedShader->Reload();
					}
					ImGui::Text("Shader Code:");
					ImGui::BeginChild("ScrollingRegion", ImVec2(0, 0), true, ImGuiWindowFlags_HorizontalScrollbar);
					ImGui::TextUnformatted(SelectedShader->GetSourceCode().c_str());
					ImGui::EndChild();
				}
			}
		}
		ImGui::End();
		
		ImGui::Begin("Materials", 0, ImVec2(250, 300));
		{
			static NChar Text[100];
			ImGui::InputText("##MaterialName", Text, 100);
			ImGui::SameLine();
			if (ImGui::Button("Create New Material")) {
				if (strlen(Text) > 0) {
					MaterialPtr NewMaterial = std::make_shared<Material>(Text::NarrowToWide(NString(Text)));
					MaterialManager::GetInstance().AddMaterial(NewMaterial);
				}
				Text[0] = '\0';
			}

			TArray<IName> MaterialNameList = MaterialManager::GetInstance().GetResourceNames();
			TArray<NString> NarrowMaterialNameList(MaterialNameList.size());
			for (int i = 0; i < MaterialNameList.size(); ++i)
				NarrowMaterialNameList[i] = (MaterialNameList)[i].GetNarrowDisplayName();

			TArray<IName> ShaderNameList = ShaderManager::GetInstance().GetResourceShaderNames();
			TArray<NString> NarrowShaderNameList(ShaderNameList.size());
			for (int i = 0; i < ShaderNameList.size(); ++i)
				NarrowShaderNameList[i] = (ShaderNameList)[i].GetNarrowDisplayName();

			if (MaterialNameList.size() > 0) {
				static int Selection = 0;
				ImGui::ListBox("Material List", &Selection, [](void * Data, int indx, const char ** outText) -> bool {
					TArray<NString>* Items = (TArray<NString> *)Data;
					if (outText) *outText = (*Items)[indx].c_str();
					return true;
				}, &NarrowMaterialNameList, (int)NarrowMaterialNameList.size());
				ImGui::Text("Selected Material: %s", NarrowMaterialNameList[Selection].c_str());
				MaterialPtr SelectedMaterial = MaterialManager::GetInstance().GetMaterial(MaterialNameList[Selection]);
				if (SelectedMaterial) {
					ImGui::Columns(2);
					ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 2));
					ImGui::Separator();

					ImGui::AlignTextToFramePadding(); ImGui::Text("Shader"); ImGui::NextColumn();
					int ShaderSelection = SelectedMaterial->GetShaderProgram() ? 0 : -1;
					if (ShaderSelection == 0) {
						for (int i = 0; i < NarrowShaderNameList.size(); i++) {
							if (SelectedMaterial->GetShaderProgram()->GetName() == ShaderNameList[i]) {
								ShaderSelection = i; break;
							}
						}
					}
					ImGui::PushItemWidth(-1); 
					if (ImGui::Combo(("##Shader" + 
						(SelectedMaterial->GetShaderProgram() ? SelectedMaterial->GetShaderProgram()->GetName().GetNarrowInstanceName() : "0")).c_str(),
						&ShaderSelection, [](void * Data, int indx, const char ** outText) -> bool {
						TArray<NString>* Items = (TArray<NString> *)Data;
						if (outText) *outText = (*Items)[indx].c_str();
						return true;
					}, &NarrowShaderNameList, (int)NarrowShaderNameList.size())) {
						if (ShaderSelection >= 0 && ShaderSelection < ShaderNameList.size())
							SelectedMaterial->SetShaderProgram(ShaderManager::GetInstance().GetProgram(ShaderNameList[ShaderSelection]));
					}
					ImGui::NextColumn();

					ImGui::AlignTextToFramePadding(); ImGui::Text("Render Priority"); ImGui::NextColumn();
					ImGui::PushItemWidth(-1); ImGui::DragScalar("##RPriority", ImGuiDataType_U32, &SelectedMaterial->RenderPriority, 100.F);
					ImGui::NextColumn();

					ImGui::AlignTextToFramePadding(); ImGui::Text("Write Depth"); ImGui::NextColumn();
					ImGui::PushItemWidth(-1); ImGui::Checkbox("##bDepthTest", &SelectedMaterial->bWriteDepth);
					ImGui::NextColumn();

					ImGui::AlignTextToFramePadding(); ImGui::Text("Depth Function"); ImGui::NextColumn();
					ImGui::PushItemWidth(-1); ImGui::Combo("##DepthFunction", (int *)&SelectedMaterial->DepthFunction, 
						"Never\0Less\0Equal\0LessEqual\0Greater\0NotEqual\0GreaterEqual\0Always\0");
					ImGui::NextColumn();

					ImGui::AlignTextToFramePadding(); ImGui::Text("Cull Mode"); ImGui::NextColumn();
					ImGui::PushItemWidth(-1); ImGui::Combo("##CullMode", (int *)&SelectedMaterial->CullMode,
						"None\0ClockWise\0CounterClockWise\0");
					ImGui::NextColumn();

					ImGui::AlignTextToFramePadding(); ImGui::Text("Fill Mode"); ImGui::NextColumn();
					ImGui::PushItemWidth(-1); ImGui::Combo("##FillMode", (int *)&SelectedMaterial->FillMode,
						"Point\0Wireframe\0Solid\0");
					ImGui::NextColumn();

					for (auto & KeyValue : SelectedMaterial->GetVariables()) {
						int i = 0;
						if (KeyValue.IsInternal()) continue;
						switch (KeyValue.Value.GetType()) {
						case EShaderPropertyType::Matrix4x4Array:
							ImGui::AlignTextToFramePadding(); ImGui::Text("%s", KeyValue.Name.c_str()); ImGui::NextColumn();
							for (auto& Value : KeyValue.Value.Matrix4x4Array) {
								if (ImGui::TreeNode((std::to_string(i++) + "##" + KeyValue.Name).c_str())) {
									ImGui::PushItemWidth(-1); ImGui::DragFloat4("##MatA", (float *)&Value[0], 1.F, -MathConstants::BigNumber, MathConstants::BigNumber);
									ImGui::PushItemWidth(-1); ImGui::DragFloat4("##MatA", (float *)&Value[1], 1.F, -MathConstants::BigNumber, MathConstants::BigNumber);
									ImGui::PushItemWidth(-1); ImGui::DragFloat4("##MatA", (float *)&Value[2], 1.F, -MathConstants::BigNumber, MathConstants::BigNumber);
									ImGui::PushItemWidth(-1); ImGui::DragFloat4("##MatA", (float *)&Value[3], 1.F, -MathConstants::BigNumber, MathConstants::BigNumber);
									ImGui::TreePop();
								}
							}
							ImGui::NextColumn();
							break;
						case EShaderPropertyType::Matrix4x4:
							ImGui::AlignTextToFramePadding(); ImGui::Text("%s", KeyValue.Name.c_str()); ImGui::NextColumn();
							if (ImGui::TreeNode((std::to_string(i++) + "##" + KeyValue.Name).c_str())) {
								ImGui::PushItemWidth(-1); ImGui::DragFloat4("##Mat0", (float *)&KeyValue.Value.Mat4x4[0], 1.F, -MathConstants::BigNumber, MathConstants::BigNumber);
								ImGui::PushItemWidth(-1); ImGui::DragFloat4("##Mat1", (float *)&KeyValue.Value.Mat4x4[1], 1.F, -MathConstants::BigNumber, MathConstants::BigNumber);
								ImGui::PushItemWidth(-1); ImGui::DragFloat4("##Mat2", (float *)&KeyValue.Value.Mat4x4[2], 1.F, -MathConstants::BigNumber, MathConstants::BigNumber);
								ImGui::PushItemWidth(-1); ImGui::DragFloat4("##Mat3", (float *)&KeyValue.Value.Mat4x4[3], 1.F, -MathConstants::BigNumber, MathConstants::BigNumber);
								ImGui::TreePop();
							}
							ImGui::NextColumn();
							break;
						case EShaderPropertyType::FloatArray:
							ImGui::AlignTextToFramePadding(); ImGui::Text(KeyValue.Name.c_str()); ImGui::NextColumn();
							for (auto& Value : KeyValue.Value.FloatArray) {
								if (ImGui::TreeNode((std::to_string(i++) + "##" + KeyValue.Name).c_str())) {
									ImGui::PushItemWidth(-1); ImGui::DragFloat("##FloatA", &Value, .01F, -MathConstants::BigNumber, MathConstants::BigNumber);
									ImGui::TreePop();
								}
							}
							ImGui::NextColumn();
							break;
						case EShaderPropertyType::Float:
							ImGui::AlignTextToFramePadding(); ImGui::Text(KeyValue.Name.c_str()); ImGui::NextColumn();
							ImGui::PushItemWidth(-1); ImGui::DragFloat(("##" + KeyValue.Name).c_str(), &KeyValue.Value.Float, .01F, -MathConstants::BigNumber, MathConstants::BigNumber);
							ImGui::NextColumn();
							break;
						case EShaderPropertyType::Float2DArray:
							ImGui::AlignTextToFramePadding(); ImGui::Text(KeyValue.Name.c_str()); ImGui::NextColumn();
							for (auto& Value : KeyValue.Value.Float2DArray) {
								if (ImGui::TreeNode((std::to_string(i++) + "##" + KeyValue.Name).c_str())) {
									ImGui::PushItemWidth(-1); ImGui::DragFloat2("##Float2DA", &Value[0], .1F, -MathConstants::BigNumber, MathConstants::BigNumber);
									ImGui::TreePop();
								}
							}
							ImGui::NextColumn();
							break;
						case EShaderPropertyType::Float2D:
							ImGui::AlignTextToFramePadding(); ImGui::Text(KeyValue.Name.c_str()); ImGui::NextColumn();
							ImGui::PushItemWidth(-1); ImGui::DragFloat2(("##" + KeyValue.Name).c_str(), &KeyValue.Value.Float2D[0], .1F, -MathConstants::BigNumber, MathConstants::BigNumber);
							ImGui::NextColumn();
							break;
						case EShaderPropertyType::Float3DArray:
							ImGui::AlignTextToFramePadding(); ImGui::Text(KeyValue.Name.c_str()); ImGui::NextColumn();
							for (auto& Value : KeyValue.Value.Float3DArray) {
								if (ImGui::TreeNode((std::to_string(i++) + "##" + KeyValue.Name).c_str())) {
									ImGui::PushItemWidth(-1);
									if (KeyValue.IsColor())
										ImGui::ColorEdit3("##Float3DAC", &Value[0], ImGuiColorEditFlags_Float);
									else
										ImGui::DragFloat3("##Float3DA", &Value[0], .1F, -MathConstants::BigNumber, MathConstants::BigNumber);
									ImGui::TreePop();
								}
							}
							ImGui::NextColumn();
							break;
						case EShaderPropertyType::Float3D:
							ImGui::AlignTextToFramePadding(); ImGui::Text(KeyValue.Name.c_str()); ImGui::NextColumn();
							ImGui::PushItemWidth(-1);
							if (KeyValue.IsColor())
								ImGui::ColorEdit3(("##" + KeyValue.Name).c_str(), &KeyValue.Value.Float3D[0], ImGuiColorEditFlags_Float);
							else
								ImGui::DragFloat3(("##" + KeyValue.Name).c_str(), &KeyValue.Value.Float3D[0], .1F, -MathConstants::BigNumber, MathConstants::BigNumber);
							ImGui::NextColumn();
							break;
						case EShaderPropertyType::Float4DArray:
							ImGui::AlignTextToFramePadding(); ImGui::Text(KeyValue.Name.c_str()); ImGui::NextColumn();
							for (auto& Value : KeyValue.Value.Float4DArray) {
								if (ImGui::TreeNode((std::to_string(i++) + "##" + KeyValue.Name).c_str())) {
									ImGui::PushItemWidth(-1);
									if (KeyValue.IsColor())
										ImGui::ColorEdit4("##Float4DAC", &Value[0], ImGuiColorEditFlags_Float);
									else
										ImGui::DragFloat4("##Float4DA", &Value[0], .1F, -MathConstants::BigNumber, MathConstants::BigNumber);
									ImGui::TreePop();
								}
							}
							ImGui::NextColumn();
							break;
						case EShaderPropertyType::Float4D:
							ImGui::AlignTextToFramePadding(); ImGui::Text(KeyValue.Name.c_str()); ImGui::NextColumn();
							ImGui::PushItemWidth(-1);
							if (KeyValue.IsColor())
								ImGui::ColorEdit4(("##" + KeyValue.Name).c_str(), &KeyValue.Value.Float4D[0], ImGuiColorEditFlags_Float);
							else
								ImGui::DragFloat4(("##" + KeyValue.Name).c_str(), &KeyValue.Value.Float4D[0], .1F, -MathConstants::BigNumber, MathConstants::BigNumber);
							ImGui::NextColumn();
							break;
						case EShaderPropertyType::IntArray:
							ImGui::AlignTextToFramePadding(); ImGui::Text(KeyValue.Name.c_str()); ImGui::NextColumn();
							for (auto& Value : KeyValue.Value.IntArray) {
								if (ImGui::TreeNode((std::to_string(i++) + "##" + KeyValue.Name).c_str())) {
									ImGui::PushItemWidth(-1); ImGui::DragInt("##IntA", &Value, 1, (int)-MathConstants::BigNumber, (int)MathConstants::BigNumber);
									ImGui::TreePop();
								}
							}
							ImGui::NextColumn();
							break;
						case EShaderPropertyType::Int:
							ImGui::AlignTextToFramePadding(); ImGui::Text(KeyValue.Name.c_str()); ImGui::NextColumn();
							ImGui::PushItemWidth(-1); ImGui::DragInt(("##" + KeyValue.Name).c_str(), &KeyValue.Value.Int, 1, (int)-MathConstants::BigNumber, (int)MathConstants::BigNumber);
							ImGui::NextColumn();
							break;
						case EShaderPropertyType::Cubemap:
						case EShaderPropertyType::Texture2D:
							ImGui::AlignTextToFramePadding(); ImGui::Text(KeyValue.Name.c_str()); ImGui::NextColumn();
							i = KeyValue.Value.Texture ? 0 : -1;
							if (i == 0) {
								for (int j = 0; j < NarrowTextureNameList.size(); j++) {
									if (KeyValue.Value.Texture->GetName() == TextureNameList[j]) {
										i = j; break;
									}
								}
							}
							ImGui::PushItemWidth(-1); 
							if (ImGui::Combo(("##Texture" + KeyValue.Name).c_str(), &i, [](void * Data, int indx, const char ** outText) -> bool {
								TArray<NString>* Items = (TArray<NString> *)Data;
								if (outText) *outText = (*Items)[indx].c_str();
								return true;
							}, &NarrowTextureNameList, (int)NarrowTextureNameList.size())) {
								if (i >= 0 && i < TextureNameList.size())
									KeyValue.Value.Texture = TextureManager::GetInstance().GetTexture(TextureNameList[i]);
							}
							ImGui::NextColumn();
							break;
						case EShaderPropertyType::None:
						default:
							ImGui::AlignTextToFramePadding(); ImGui::Text("%s[%d]", KeyValue.Name.c_str(), (int)KeyValue.Value.GetType()); ImGui::NextColumn();
							ImGui::NextColumn();
							break;
						}
					}
					ImGui::PopStyleVar();
					ImGui::Separator();
					ImGui::Columns(1);
				}
			}
		}
		ImGui::End();

		ImGui::Begin("Scene Settings");
		{
			ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 2));
			ImGui::Columns(2);
			ImGui::Separator();

			ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Skybox Roughness"); ImGui::NextColumn();
			ImGui::PushItemWidth(-1); ImGui::SliderFloat("##Skybox Roughness", &SkyboxRoughness, 0.F, 1.F); ImGui::NextColumn();
			ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Skybox Texture"); ImGui::NextColumn();
			ImGui::PushItemWidth(-1); if (ImGui::Combo("##Skybox Texture", &CurrentSkybox, SkyBoxes, IM_ARRAYSIZE(SkyBoxes))) {
				SetSceneSkybox(Text::NarrowToWide(SkyBoxes[CurrentSkybox]));
			} ImGui::NextColumn();
			ImGui::PushItemWidth(0);

			RenderStage * ActiveStage = Application::GetInstance()->GetRenderPipeline().GetActiveStage();
			if (ActiveStage != NULL) {
				ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Light[0].Position"); ImGui::NextColumn();
				ImGui::PushItemWidth(-1); ImGui::DragFloat3("##Light[0].Position", &ActiveStage->Scene.Lights[0].Position[0]); ImGui::NextColumn();
				ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Light[0].Color"); ImGui::NextColumn();
				ImGui::PushItemWidth(-1); ImGui::ColorEdit3("##Light[0].Color", &ActiveStage->Scene.Lights[0].Color[0]); ImGui::NextColumn();
				ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Light[0].Intensity"); ImGui::NextColumn();
				ImGui::PushItemWidth(-1); ImGui::DragFloat("##Light[0].Intensity", &ActiveStage->Scene.Lights[0].Intensity); ImGui::NextColumn();
				ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Light[1].Position"); ImGui::NextColumn();
				ImGui::PushItemWidth(-1); ImGui::DragFloat3("##Light[1].Position", &ActiveStage->Scene.Lights[1].Position[0]); ImGui::NextColumn();
				ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Light[1].Color"); ImGui::NextColumn();
				ImGui::PushItemWidth(-1); ImGui::ColorEdit3("##Light[1].Color", &ActiveStage->Scene.Lights[1].Color[0]); ImGui::NextColumn();
				ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Light[1].Intensity"); ImGui::NextColumn();
				ImGui::PushItemWidth(-1); ImGui::DragFloat("##Light[1].Intensity", &ActiveStage->Scene.Lights[1].Intensity); ImGui::NextColumn();
			}

			ImGui::Columns(1);
			ImGui::Separator();
			ImGui::PopStyleVar();
		}
		ImGui::End();

		char ProgressText[30]; 
		ImGui::Begin("Audio Settings");
		{
			{
				static TArray<float> AudioChannel1(1);
				static TArray<float> AudioChannel2(1);
				{
					AudioChannel1.resize(32768 / (2 * 4) / 2);
					AudioChannel2.resize(32768 / (2 * 4) / 2);
					auto& Device = Application::GetInstance()->GetAudioDevice();
					auto Duration = (Time::Micro::ReturnType)(((32768 * 8u / (Device.SampleSize() * Device.GetChannelCount())) / (float)Device.GetFrecuency()) * Time::Second::GetSizeInMicro());
					unsigned long long Delta = Time::GetEpochTime<Time::Micro>() - Device.LastAudioUpdate;
					float * BufferPtr = (float *)&(Application::GetInstance()->GetAudioDevice().CurrentSample[0]);
					for (unsigned int i = 0; i < 32768 / (2 * 4) / 2; ++i) {
						for (unsigned int j = i * (2 * 4); j < (i + 1) * (2 * 4) && j < 32768; j += (2 * 4)) {
							AudioChannel1[i] = *BufferPtr;
							AudioChannel2[i] = *BufferPtr++;
							BufferPtr++;
						}
					}
				}

				ImGui::PushItemWidth(-1); ImGui::PlotLines("##Audio", &AudioChannel1[0], 32768 / (2 * 4) / 2, NULL, 0, -1.F, 1.F, ImVec2(0, 125));
				ImGui::PushItemWidth(-1); ImGui::PlotLines("##Audio", &AudioChannel2[0], 32768 / (2 * 4) / 2, NULL, 0, -1.F, 1.F, ImVec2(0, 125));
			}
			for (auto KeyValue : Application::GetInstance()->GetAudioDevice()) {
				AudioDevice::SamplePlayInfo * Info = KeyValue.second;

				if (ImGui::TreeNode(("Audio" + std::to_string(Info->Identifier)).c_str())) {
					ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 2));
					ImGui::Columns(2);
					ImGui::Separator();

					ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Volume"); ImGui::NextColumn();
					ImGui::PushItemWidth(-1); ImGui::SliderFloat("##Audio Volume", &Info->Volume, 0.F, 1.F); ImGui::NextColumn();
					ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Loop"); ImGui::NextColumn();
					ImGui::PushItemWidth(-1); ImGui::Checkbox("##Audio Loop", &Info->bLoop); ImGui::NextColumn();
					ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Paused"); ImGui::NextColumn();
					ImGui::PushItemWidth(-1); ImGui::Checkbox("##Audio Pused", &Info->bPause); ImGui::NextColumn();
					ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Progress"); ImGui::NextColumn();
					sprintf(ProgressText, "%.2f", Info->Sample->GetDurationAt<Time::Second>(Info->Pos));
					ImGui::ProgressBar((float)Info->Pos / Info->Sample->GetBufferLength(), ImVec2(-1.F, 0.F), ProgressText); ImGui::NextColumn();
					ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Channel 1"); ImGui::NextColumn();

					unsigned int AudioPlotDetail = 0;
					TArray<float> AudioChannel1(1);
					TArray<float> AudioChannel2(1);
					if (AudioPlotDetail != (unsigned int)(Info->Sample->GetBufferLength() / ((ImGui::GetColumnWidth() - 12) * 4))) {
						AudioPlotDetail = (unsigned int)(Info->Sample->GetBufferLength() / ((ImGui::GetColumnWidth() - 12) * 4));
						AudioChannel1.resize(Info->Sample->GetBufferLength() / ((2 * 4) * AudioPlotDetail) * 2);
						AudioChannel2.resize(Info->Sample->GetBufferLength() / ((2 * 4) * AudioPlotDetail) * 2);
						float * BufferPtr = (float *)Info->Sample->GetBufferAt(0);
						float * BufferEndPtr = (float *)Info->Sample->GetBufferAt(Info->Sample->GetBufferLength() - 4);
						for (unsigned int i = 0; i < Info->Sample->GetBufferLength() / ((2 * 4) * AudioPlotDetail); ++i) {
							AudioChannel1[i * 2] = -MathConstants::BigNumber;
							AudioChannel1[i * 2 + 1] = MathConstants::BigNumber;
							for (unsigned int j = i * (2 * 4) * AudioPlotDetail; j < (i + 1) * ((2 * 4) * AudioPlotDetail) && j < Info->Sample->GetBufferLength(); j += (2 * 4)) {
								AudioChannel1[i * 2] = Math::Max(AudioChannel1[i * 2], *BufferPtr);
								AudioChannel1[i * 2 + 1] = Math::Min(AudioChannel1[i * 2 + 1], *BufferPtr);
								BufferPtr += 2;
							}
						}
						BufferPtr = (float *)Info->Sample->GetBufferAt(4);
						for (unsigned int i = 0; i < Info->Sample->GetBufferLength() / ((2 * 4) * AudioPlotDetail); ++i) {
							AudioChannel2[i * 2] = -MathConstants::BigNumber;
							AudioChannel2[i * 2 + 1] = MathConstants::BigNumber;
							for (unsigned int j = i * (2 * 4) * AudioPlotDetail; j < (i + 1) * ((2 * 4) * AudioPlotDetail) && j < Info->Sample->GetBufferLength(); j += (2 * 4)) {
								AudioChannel2[i * 2] = Math::Max(AudioChannel2[i * 2], *BufferPtr);
								AudioChannel2[i * 2 + 1] = Math::Min(AudioChannel2[i * 2 + 1], *BufferPtr);
								BufferPtr += 2;
							}
						}
					}

					ImGui::PushItemWidth(-1); ImGui::PlotLines("##Audio Channel 1",
						&AudioChannel1[0], (int)AudioChannel1.size(), NULL, 0, -1.F, 1.F, ImVec2(0, 40)); ImGui::NextColumn();
					ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Channel 2"); ImGui::NextColumn();
					ImGui::PushItemWidth(-1); ImGui::PlotLines("##Audio Channel 2",
						&AudioChannel2[0], (int)AudioChannel2.size(), NULL, 0, -1.F, 1.F, ImVec2(0, 40)); ImGui::NextColumn();

					ImGui::Columns(1);
					ImGui::Separator();
					ImGui::PopStyleVar();
					ImGui::TreePop();
				}
			}
		}
		ImGui::End();

		ImGui::Begin("Textures");
		{
			ImGuiIO& IO = ImGui::GetIO();
			ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 2));
			ImGui::Columns(2);
			ImGui::Separator();

			ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Texture"); ImGui::NextColumn();
			ImGui::PushItemWidth(-1); ImGui::Combo("##Texture", &CurrentTexture, [](void * Data, int indx, const char ** outText) -> bool {
				TArray<NString>* Items = (TArray<NString> *)Data;
				if (outText) *outText = (*Items)[indx].c_str();
				return true;
			}, &NarrowTextureNameList, (int)NarrowTextureNameList.size()); ImGui::NextColumn();
			if (SelectedTexture) {
				ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("LOD Level"); ImGui::NextColumn();
				ImGui::PushItemWidth(-1); ImGui::SliderFloat("##LOD Level", &SampleLevel, 0.0F, 1.0F, "%.3f"); ImGui::NextColumn();
			}
			ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Gamma"); ImGui::NextColumn();
			ImGui::PushItemWidth(-1); ImGui::SliderFloat("##Gamma", &Gamma, 1.0F, 4.0F, "%.3f"); ImGui::NextColumn();

			ImGui::Columns(1);
			ImGui::Separator();
			ImGui::PopStyleVar();

			ImGui::Checkbox("##RedFilter", &ColorFilter[0]); ImGui::SameLine();
			ImGui::ColorButton("RedFilter##RefColor", ImColor(ColorFilter[0] ? 1.F : 0.F, 0.F, 0.F, 1.F));
			ImGui::SameLine(); ImGui::Checkbox("##GreenFilter", &ColorFilter[1]); ImGui::SameLine();
			ImGui::ColorButton("GreenFilter##RefColor", ImColor(0.F, ColorFilter[1] ? 1.F : 0.F, 0.F, 1.F));
			ImGui::SameLine(); ImGui::Checkbox("##BlueFilter", &ColorFilter[2]); ImGui::SameLine();
			ImGui::ColorButton("BlueFilter##RefColor", ImColor(0.F, 0.F, ColorFilter[2] ? 1.F : 0.F, 1.F));
			ImGui::SameLine(); ImGui::Checkbox("##AlphaFilter", &ColorFilter[3]); ImGui::SameLine();
			ImGui::ColorButton("AlphaFilter##RefColor", ImColor(1.F, 1.F, 1.F, ColorFilter[3] ? 1.F : 0.F), ImGuiColorEditFlags_AlphaPreview);
			if (SelectedTexture) {
				ImVec2 ImageSize;
				ImVec2 MPos = ImGui::GetCursorScreenPos();
				if (SelectedTexture->GetDimension() == ETextureDimension::Texture2D) {
					ImageSize.x = Math::Min(
						ImGui::GetWindowWidth(), (ImGui::GetWindowHeight() - ImGui::GetCursorPosY())
						* std::dynamic_pointer_cast<Texture2D>(SelectedTexture)->GetAspectRatio()
					);
					ImageSize.x -= ImGui::GetStyle().ItemSpacing.y * 4.0F;
					ImageSize.y = ImageSize.x / std::dynamic_pointer_cast<Texture2D>(SelectedTexture)->GetAspectRatio();
				}
				else {
					ImageSize.x = Math::Min(ImGui::GetWindowWidth(), (ImGui::GetWindowHeight() - ImGui::GetCursorPosY()) * 2.F);
					ImageSize.x -= ImGui::GetStyle().ItemSpacing.y * 4.0F;
					ImageSize.y = ImageSize.x / 2.F;
				}
				ImGui::Image((void *)TextureSample->GetTextureObject(), ImageSize);
				if (ImGui::IsItemHovered()) {
					ImGui::BeginTooltip();
					float RegionSize = 32.0f;
					float RegionX = IO.MousePos.x - MPos.x - RegionSize * 0.5F;
					RegionX < 0.F ? RegionX = 0.F : (RegionX > ImageSize.x - RegionSize) ? RegionX = ImageSize.x - RegionSize : RegionX = RegionX;
					float RegionY = IO.MousePos.y - MPos.y - RegionSize * 0.5F;
					RegionY < 0.F ? RegionY = 0.F : (RegionY > ImageSize.y - RegionSize) ? RegionY = ImageSize.y - RegionSize : RegionY = RegionY;
					ImGui::Text("Min: (%.2f, %.2f)", RegionX, RegionY);
					ImGui::Text("Max: (%.2f, %.2f)", RegionX + RegionSize, RegionY + RegionSize);
					ImVec2 UV0 = ImVec2((RegionX) / ImageSize.x, (RegionY) / ImageSize.y);
					ImVec2 UV1 = ImVec2((RegionX + RegionSize) / ImageSize.x, (RegionY + RegionSize) / ImageSize.y);
					ImGui::Image((void *)TextureSample->GetTextureObject(), ImVec2(140.F, 140.F), UV0, UV1, ImVec4(1.F, 1.F, 1.F, 1.F), ImVec4(1.F, 1.F, 1.F, .5F));
					ImGui::EndTooltip();
				}
			}
		}
		ImGui::End();
	}

	virtual void OnAwake() override {
		LOG_DEBUG(L"{0}", FileManager::GetAppDirectory());

		Application::GetInstance()->GetRenderPipeline().CreateStage<RenderStage>(L"TestStage");

		Application::GetInstance()->GetAudioDevice().AddSample(AudioManager::GetInstance().GetAudioSample(L"6503.wav"), 0.255F, false, true);
		Application::GetInstance()->GetAudioDevice().AddSample(AudioManager::GetInstance().GetAudioSample(L"Hololooo.wav"), 0.255F, false, true);

		ShaderManager& ShaderMng = ShaderManager::GetInstance();
		RShaderPtr EquiToCubemapShader = ShaderMng.GetProgram(L"EquirectangularToCubemap");
		RShaderPtr HDRClampingShader   = ShaderMng.GetProgram(L"HDRClampingShader");
		RShaderPtr BRDFShader          = ShaderMng.GetProgram(L"BRDFShader");
		RShaderPtr UnlitShader         = ShaderMng.GetProgram(L"UnLitShader");
		RShaderPtr RenderTextureShader = ShaderMng.GetProgram(L"RenderTextureShader");
		RShaderPtr IntegrateBRDFShader = ShaderMng.GetProgram(L"IntegrateBRDFShader");
		RShaderPtr RenderTextShader    = ShaderMng.GetProgram(L"RenderTextShader");
		RShaderPtr RenderCubemapShader = ShaderMng.GetProgram(L"RenderCubemapShader");

		FontFace.Initialize(FileManager::GetFile(L"Resources/Fonts/ArialUnicode.ttf"));

		TextGenerator.TextFont = &FontFace;
		TextGenerator.GlyphHeight = 45;
		TextGenerator.AtlasSize = 1024;
		TextGenerator.PixelRange = 1.5F;
		TextGenerator.Pivot = 0;

		TextGenerator.PrepareCharacters(0ul, 255ul);
		TextGenerator.GenerateGlyphAtlas(FontAtlas);
		FontMap = Texture2D::Create(
			L"FontMap",
			IntVector2(TextGenerator.AtlasSize),
			CF_Red,
			FM_MinMagLinear,
			SAM_Border,
			CF_Red,
			FontAtlas.PointerToValue()
		);
		FontMap->GenerateMipMaps();

		UnlitMaterial.SetShaderProgram(UnlitShader);

		UnlitMaterialWire.SetShaderProgram(UnlitShader);
		UnlitMaterialWire.FillMode = FM_Wireframe;
		UnlitMaterialWire.CullMode = CM_None;

		RenderTextureMaterial.DepthFunction = DF_Always;
		RenderTextureMaterial.CullMode = CM_None;
		RenderTextureMaterial.SetShaderProgram(RenderTextureShader);

		RenderTextMaterial->DepthFunction = DF_Always;
		RenderTextMaterial->CullMode = CM_None;
		RenderTextMaterial->SetShaderProgram(RenderTextShader);
		MaterialManager::GetInstance().AddMaterial(RenderTextMaterial);

		RenderCubemapMaterial->CullMode = CM_None;
		RenderCubemapMaterial->SetShaderProgram(RenderCubemapShader);
		MaterialManager::GetInstance().AddMaterial(RenderCubemapMaterial);

		IntegrateBRDFMaterial.DepthFunction = DF_Always;
		IntegrateBRDFMaterial.CullMode = CM_None;
		IntegrateBRDFMaterial.SetShaderProgram(IntegrateBRDFShader);

		HDRClampingMaterial.DepthFunction = DF_Always;
		HDRClampingMaterial.CullMode = CM_None;
		HDRClampingMaterial.SetShaderProgram(HDRClampingShader);

		MeshManager::GetInstance().LoadAsyncFromFile(L"Resources/Models/SphereUV.obj", true);
		MeshManager::GetInstance().LoadAsyncFromFile(L"Resources/Models/Arrow.fbx", false);
		MeshManager::GetInstance().LoadAsyncFromFile(L"Resources/Models/Sponza.obj", true);
		MeshManager::GetInstance().LoadAsyncFromFile(L"Resources/Models/Flamer.obj", false);
		MeshManager::GetInstance().LoadAsyncFromFile(L"Resources/Models/BigPlane.obj", true);
		MeshManager::GetInstance().LoadAsyncFromFile(L"Resources/Models/PonyCartoon.fbx", false);
		MeshManager::GetInstance().LoadAsyncFromFile(L"Resources/Models/PirateProps_Barrels.obj", false);
		MeshManager::GetInstance().LoadAsyncFromFile(L"Resources/Models/Sci_Fi_Tile_Set.obj", false);
		MeshManager::GetInstance().AddMesh(L"Quad", std::make_shared<Mesh>(Mesh(MeshPrimitives::CreateQuadMeshData(0.F, 1.F))));
		MeshManager::GetInstance().GetMesh(L"Quad")->SetUpBuffers();

		Texture2DPtr RenderedTexture = Texture2D::Create(
			L"RenderedTexture",
			IntVector2(
				Application::GetInstance()->GetWindow().GetWidth(),
				Application::GetInstance()->GetWindow().GetHeight())
			/ 2, CF_RGBA32F, FM_MinLinearMagNearest, SAM_Repeat
		);

		///////// Create Matrices Buffer //////////////
		ModelMatrixBuffer = VertexBuffer::Create(NULL, 0, EUsageMode::UM_Dynamic);

		Texture2DPtr BRDFLut = Texture2D::Create(L"BRDFLut", IntVector2(512), CF_RG16F, FM_MinMagLinear, SAM_Clamp);
		{
			RenderTargetPtr Renderer = RenderTarget::Create();
			IntegrateBRDFMaterial.Use();
			IntegrateBRDFMaterial.SetMatrix4x4Array("_ProjectionMatrix", Matrix4x4().PointerToValue());

			MeshPrimitives::Quad.BindSubdivisionVertexArray(0);
			Matrix4x4 QuadPosition = Matrix4x4::Translation({ 0, 0, 0 });
			IntegrateBRDFMaterial.SetAttribMatrix4x4Array(
				"_iModelMatrix", 1, QuadPosition.PointerToValue(), ModelMatrixBuffer
			);

			Renderer->BindTexture2D(BRDFLut);
			Renderer->Clear();
			MeshPrimitives::Quad.DrawSubdivisionInstanciated(1, 0);
		}
		TextureManager::GetInstance().AddTexture(L"BRDFLut", BRDFLut);

		SetSceneSkybox(L"Resources/Textures/Arches_E_PineTree_3k.hdr");

		Transforms.push_back(Transform());

		Application::GetInstance()->GetRenderPipeline().Initialize();
		Application::GetInstance()->GetRenderPipeline().ContextInterval(0);
	}

	virtual void OnUpdate(Timestamp Stamp) override {

		// CameraRayDirection = {
		// 	(2.F * Input::GetMouseX()) / Application::GetInstance()->GetWindow().GetWidth() - 1.F,
		// 	1.F - (2.F * Input::GetMouseY()) / Application::GetInstance()->GetWindow().GetHeight(),
		// 	-1.F,
		// };
		// CameraRayDirection = ProjectionMatrix.Inversed() * CameraRayDirection;
		// CameraRayDirection.z = -1.F;
		// CameraRayDirection = ViewMatrix.Inversed() * CameraRayDirection;
		// CameraRayDirection.Normalize();
		// 
		// ViewMatrix = Transform(EyePosition, CameraRotation).GetGLViewMatrix();
		// // ViewMatrix = Matrix4x4::Scaling(Vector3(1, 1, -1)).Inversed() * Matrix4x4::LookAt(EyePosition, EyePosition + FrameRotation * Vector3(0, 0, 1), FrameRotation * Vector3(0, 1)).Inversed();

		if (Input::IsKeyDown(SDL_SCANCODE_N)) {
			MaterialMetalness -= 1.F * Time::GetDeltaTime<Time::Second>();
			MaterialMetalness = Math::Clamp01(MaterialMetalness);
		}
		if (Input::IsKeyDown(SDL_SCANCODE_M)) {
			MaterialMetalness += 1.F * Time::GetDeltaTime<Time::Second>();
			MaterialMetalness = Math::Clamp01(MaterialMetalness);
		}
		if (Input::IsKeyDown(SDL_SCANCODE_E)) {
			MaterialRoughness -= 0.5F * Time::GetDeltaTime<Time::Second>();
			MaterialRoughness = Math::Clamp01(MaterialRoughness);
		}
		if (Input::IsKeyDown(SDL_SCANCODE_R)) {
			MaterialRoughness += 0.5F * Time::GetDeltaTime<Time::Second>();
			MaterialRoughness = Math::Clamp01(MaterialRoughness);
		}
		if (Input::IsKeyDown(SDL_SCANCODE_L)) {
			LightIntencity += LightIntencity * Time::GetDeltaTime<Time::Second>();
		}
		if (Input::IsKeyDown(SDL_SCANCODE_K)) {
			LightIntencity -= LightIntencity * Time::GetDeltaTime<Time::Second>();
		}

		if (Input::IsKeyDown(SDL_SCANCODE_RIGHT)) {
			MultiuseValue += Time::GetDeltaTime<Time::Second>() * MultiuseValue;
		}
		if (Input::IsKeyDown(SDL_SCANCODE_LEFT)) {
			MultiuseValue -= Time::GetDeltaTime<Time::Second>() * MultiuseValue;
		}

		if (Input::IsKeyDown(SDL_SCANCODE_LSHIFT)) {
			if (Input::IsKeyDown(SDL_SCANCODE_I)) {
				FontSize += Time::GetDeltaTime<Time::Second>() * FontSize;
			}

			if (Input::IsKeyDown(SDL_SCANCODE_K)) {
				FontSize -= Time::GetDeltaTime<Time::Second>() * FontSize;
			}
		}
		else {
			if (Input::IsKeyDown(SDL_SCANCODE_I)) {
				FontBoldness += Time::GetDeltaTime<Time::Second>() / 10.F;
			}

			if (Input::IsKeyDown(SDL_SCANCODE_K)) {
				FontBoldness -= Time::GetDeltaTime<Time::Second>() / 10.F;
			}
		}

		if (Input::IsKeyDown(SDL_SCANCODE_V)) {
			for (int i = 0; i < 10; i++) {
				RenderingText[1] += (unsigned long)(rand() % 0x3fff);
			}
		}

		// if (Input::IsKeyDown(SDL_SCANCODE_SPACE)) {
		// 	TestArrowTransform.Position = EyePosition;
		// 	TestArrowDirection = CameraRayDirection;
		// 	TestArrowTransform.Rotation = Quaternion::LookRotation(CameraRayDirection, Vector3(0, 1, 0));
		// }

		for (int i = 0; i < TextCount; i++) {
			if (TextGenerator.PrepareFindedCharacters(RenderingText[i]) > 0) {
				TextGenerator.GenerateGlyphAtlas(FontAtlas);
				TextureManager::GetInstance().FreeTexture(L"FontMap");
				FontMap = Texture2D::Create(
					L"FontMap",
					IntVector2(TextGenerator.AtlasSize),
					CF_Red,
					FM_MinMagLinear,
					SAM_Border,
					CF_Red,
					FontAtlas.PointerToValue()
				);
				FontMap->GenerateMipMaps();
				TextureManager::GetInstance().AddTexture(L"FontMap", FontMap);
			}
		}

		// Transforms[0].Rotation = Quaternion::AxisAngle(Vector3(0, 1, 0).Normalized(), Time::GetDeltaTime() * 0.04F) * Transforms[0].Rotation;
		TransformMat = Transforms[0].GetLocalToWorldMatrix();
		InverseTransform = Transforms[0].GetWorldToLocalMatrix();

		TestArrowTransform.Scale = 0.1F;
		TestArrowTransform.Position += TestArrowDirection * TestSphereVelocity * Time::GetDeltaTime<Time::Second>();
	}

	virtual void OnRender() override {
		Timestamp Timer;

		// --- Run the device part of the program
		if (!bRandomArray) {
			// curandomStateArray = GetRandomArray(RenderedTexture.GetDimension());
			bRandomArray = true;
		}
		bool bTestResult = false;

		// Ray TestRayArrow(TestArrowTransform.Position, TestArrowDirection);
		// if (SceneModels.size() > 100)
		// 	for (int MeshCount = (int)MeshSelector; MeshCount >= 0 && MeshCount < (int)SceneModels.size(); ++MeshCount) {
		// 		BoundingBox3D ModelSpaceAABox = SceneModels[MeshCount]->GetMeshData().Bounding.Transform(TransformMat);
		// 		TArray<RayHit> Hits;
		// 
		// 		if (Physics::RaycastAxisAlignedBox(TestRayArrow, ModelSpaceAABox)) {
		// 			RayHit Hit;
		// 			Ray ModelSpaceCameraRay(
		// 				InverseTransform.MultiplyPoint(TestArrowTransform.Position),
		// 				InverseTransform.MultiplyVector(TestArrowDirection)
		// 			);
		// 			for (MeshFaces::const_iterator Face = SceneModels[MeshCount]->GetMeshData().Faces.begin(); Face != SceneModels[MeshCount]->GetMeshData().Faces.end(); ++Face) {
		// 				if (Physics::RaycastTriangle(
		// 					Hit, ModelSpaceCameraRay,
		// 					SceneModels[MeshCount]->GetMeshData().Vertices[(*Face)[0]].Position,
		// 					SceneModels[MeshCount]->GetMeshData().Vertices[(*Face)[1]].Position,
		// 					SceneModels[MeshCount]->GetMeshData().Vertices[(*Face)[2]].Position, BaseMaterial.CullMode != CM_CounterClockWise
		// 				)) {
		// 					Hit.TriangleIndex = int(Face - SceneModels[MeshCount]->GetMeshData().Faces.begin());
		// 					Hits.push_back(Hit);
		// 				}
		// 			}
		// 
		// 			std::sort(Hits.begin(), Hits.end());
		// 
		// 			if (Hits.size() > 0 && Hits[0].bHit) {
		// 				Vector3 ArrowClosestContactPoint = TestArrowTransform.Position;
		// 				Vector3 ClosestContactPoint = TestRayArrow.PointAt(Hits[0].Stamp);
		// 
		// 				if ((ArrowClosestContactPoint - ClosestContactPoint).MagnitudeSquared() < TestArrowTransform.Scale.x * TestArrowTransform.Scale.x)
		// 				{
		// 					const IntVector3 & Face = SceneModels[MeshCount]->GetMeshData().Faces[Hits[0].TriangleIndex];
		// 					const Vector3 & N0 = SceneModels[MeshCount]->GetMeshData().Vertices[Face[0]].Normal;
		// 					const Vector3 & N1 = SceneModels[MeshCount]->GetMeshData().Vertices[Face[1]].Normal;
		// 					const Vector3 & N2 = SceneModels[MeshCount]->GetMeshData().Vertices[Face[2]].Normal;
		// 					Vector3 InterpolatedNormal =
		// 						N0 * Hits[0].BaricenterCoordinates[0] +
		// 						N1 * Hits[0].BaricenterCoordinates[1] +
		// 						N2 * Hits[0].BaricenterCoordinates[2];
		// 
		// 					Hits[0].Normal = TransformMat.Inversed().Transposed().MultiplyVector(InterpolatedNormal);
		// 					Hits[0].Normal.Normalize();
		// 					Vector3 ReflectedDirection = Vector3::Reflect(TestArrowDirection, Hits[0].Normal);
		// 					TestArrowDirection = ReflectedDirection.Normalized();
		// 					TestArrowTransform.Rotation = Quaternion::LookRotation(ReflectedDirection, Vector3(0, 1, 0));
		// 				}
		// 			}
		// 		}
		// 	}

		float SkyRoughnessTemp = (SkyboxRoughness) * (CubemapTexture->GetMipMapCount() - 4);
		RenderCubemapMaterial->SetVariables({
			{ "_Skybox", { ETextureDimension::Cubemap, CubemapTexture } },
			{ "_Lod", { float( SkyRoughnessTemp ) } }
		});
		
		float CubemapTextureMipmaps = (float)CubemapTexture->GetMipMapCount();

		// size_t TotalHitCount = 0;
		// Ray CameraRay(EyePosition, CameraRayDirection);
		// for (int MeshCount = (int)MeshSelector; MeshCount >= 0 && MeshCount < (int)SceneModels.size(); ++MeshCount) {
		// 	const MeshData & ModelData = SceneModels[MeshCount]->GetMeshData();
		// 	BoundingBox3D ModelSpaceAABox = ModelData.Bounding.Transform(TransformMat);
		// 	TArray<RayHit> Hits;
		// 
		// 	if (Physics::RaycastAxisAlignedBox(CameraRay, ModelSpaceAABox)) {
		// 		RayHit Hit;
		// 		Ray ModelSpaceCameraRay(
		// 			InverseTransform.MultiplyPoint(EyePosition),
		// 			InverseTransform.MultiplyVector(CameraRayDirection)
		// 		);
		// 		for (MeshFaces::const_iterator Face = ModelData.Faces.begin(); Face != ModelData.Faces.end(); ++Face) {
		// 			if (Physics::RaycastTriangle(
		// 				Hit, ModelSpaceCameraRay,
		// 				ModelData.Vertices[(*Face)[0]].Position,
		// 				ModelData.Vertices[(*Face)[1]].Position,
		// 				ModelData.Vertices[(*Face)[2]].Position, BaseMaterial.CullMode != CM_CounterClockWise
		// 			)) {
		// 				Hit.TriangleIndex = int(Face - ModelData.Faces.begin());
		// 				Hits.push_back(Hit);
		// 			}
		// 		}
		// 
		// 		std::sort(Hits.begin(), Hits.end());
		// 		TotalHitCount += Hits.size();
		// 
		// 		if (Hits.size() > 0 && Hits[0].bHit) {
		// 			if (LightModels.size() > 0) {
		// 				LightModels[0]->SetUpBuffers();
		// 				LightModels[0]->BindVertexArray();
		// 
		// 				IntVector3 Face = ModelData.Faces[Hits[0].TriangleIndex];
		// 				const Vector3 & N0 = ModelData.Vertices[Face[0]].Normal;
		// 				const Vector3 & N1 = ModelData.Vertices[Face[1]].Normal;
		// 				const Vector3 & N2 = ModelData.Vertices[Face[2]].Normal;
		// 				Vector3 InterpolatedNormal =
		// 					N0 * Hits[0].BaricenterCoordinates[0] +
		// 					N1 * Hits[0].BaricenterCoordinates[1] +
		// 					N2 * Hits[0].BaricenterCoordinates[2];
		// 
		// 				Hits[0].Normal = TransformMat.Inversed().Transposed().MultiplyVector(InterpolatedNormal);
		// 				Hits[0].Normal.Normalize();
		// 				Vector3 ReflectedCameraDir = Vector3::Reflect(CameraRayDirection, Hits[0].Normal);
		// 				Matrix4x4 HitMatrix[2] = {
		// 					Matrix4x4::Translation(CameraRay.PointAt(Hits[0].Stamp)) *
		// 					Matrix4x4::Rotation(Quaternion::LookRotation(ReflectedCameraDir, Vector3(0, 1, 0))) *
		// 					Matrix4x4::Scaling(0.1F),
		// 					Matrix4x4::Translation(CameraRay.PointAt(Hits[0].Stamp)) *
		// 					Matrix4x4::Rotation(Quaternion::LookRotation(Hits[0].Normal, Vector3(0, 1, 0))) *
		// 					Matrix4x4::Scaling(0.07F)
		// 				};
		// 				BaseMaterial.SetAttribMatrix4x4Array("_iModelMatrix", 2, &HitMatrix[0], ModelMatrixBuffer);
		// 
		// 				LightModels[0]->DrawInstanciated(2);
		// 				TriangleCount += LightModels[0]->GetMeshData().Faces.size() * 1;
		// 				VerticesCount += LightModels[0]->GetMeshData().Vertices.size() * 1;
		// 			}
		// 		}
		// 	}
		// 
		// 	SceneModels[MeshCount]->SetUpBuffers();
		// 	SceneModels[MeshCount]->BindVertexArray();
		// 
		// 	BaseMaterial.SetAttribMatrix4x4Array("_iModelMatrix", 1, &TransformMat, ModelMatrixBuffer);
		// 	SceneModels[MeshCount]->DrawInstanciated(1);
		// 
		// 	TriangleCount += ModelData.Faces.size() * 1;
		// 	VerticesCount += ModelData.Vertices.size() * 1;
		// }

		// SelectedMesh = MeshManager::GetInstance().GetMesh(SelectedMeshName);
		// if (SelectedMesh) {
		// 	SelectedMesh->BindVertexArray();
		// 	
		// 	BaseMaterial->SetAttribMatrix4x4Array("_iModelMatrix", 1, &TransformMat, ModelMatrixBuffer);
		// 	SelectedMesh->DrawInstanciated(1);
		// }

		// 
		// if (LightModels.size() > 0) {
		// 	LightModels[0]->SetUpBuffers();
		// 	LightModels[0]->BindVertexArray();
		// 
		// 	Matrix4x4 ModelMatrix = TestArrowTransform.GetLocalToWorldMatrix();
		// 	BaseMaterial.SetAttribMatrix4x4Array("_iModelMatrix", 1, &ModelMatrix, ModelMatrixBuffer);
		// 
		// 	LightModels[0]->DrawInstanciated(1);
		// 	TriangleCount += LightModels[0]->GetMeshData().Faces.size() * 1;
		// 	VerticesCount += LightModels[0]->GetMeshData().Vertices.size() * 1;
		// }

		ElementsIntersected.clear();
		// TArray<Matrix4x4> BBoxTransforms;
		// for (int MeshCount = (int)MeshSelector; MeshCount >= 0 && MeshCount < (int)SceneModels.size(); ++MeshCount) {
		// 	BoundingBox3D ModelSpaceAABox = SceneModels[MeshCount]->GetMeshData().Bounding.Transform(TransformMat);
		// 	if (Physics::RaycastAxisAlignedBox(CameraRay, ModelSpaceAABox)) {
		// 		ElementsIntersected.push_back(MeshCount);
		// 		BBoxTransforms.push_back(Matrix4x4::Translation(ModelSpaceAABox.GetCenter()) * Matrix4x4::Scaling(ModelSpaceAABox.GetSize()));
		// 	}
		// }
		// if (BBoxTransforms.size() > 0) {
		// 	UnlitMaterialWire.Use();
		// 
		// 	UnlitMaterialWire.SetMatrix4x4Array("_ProjectionMatrix", ProjectionMatrix.PointerToValue());
		// 	UnlitMaterialWire.SetMatrix4x4Array("_ViewMatrix", ViewMatrix.PointerToValue());
		// 	UnlitMaterialWire.SetFloat3Array("_ViewPosition", EyePosition.PointerToValue());
		// 	UnlitMaterialWire.SetFloat4Array("_Material.Color", Vector4(.7F, .2F, .07F, .3F).PointerToValue());
		// 
		// 	MeshPrimitives::Cube.BindVertexArray();
		// 	UnlitMaterialWire.SetAttribMatrix4x4Array("_iModelMatrix", (int)BBoxTransforms.size(), &BBoxTransforms[0], ModelMatrixBuffer);
		// 	MeshPrimitives::Cube.DrawInstanciated((int)BBoxTransforms.size());
		// }
		// 
		// UnlitMaterial.Use();
		// 
		// UnlitMaterial.SetMatrix4x4Array("_ProjectionMatrix", ProjectionMatrix.PointerToValue());
		// UnlitMaterial.SetMatrix4x4Array("_ViewMatrix", ViewMatrix.PointerToValue());
		// UnlitMaterial.SetFloat3Array("_ViewPosition", EyePosition.PointerToValue());
		// UnlitMaterial.SetFloat4Array("_Material.Color", (Vector4(1.F, 1.F, .9F, 1.F) * LightIntencity).PointerToValue());
		// 
		// if (LightModels.size() > 0) {
		// 	MeshPrimitives::Cube.BindVertexArray();
		// 
		// 	TArray<Matrix4x4> LightPositions;
		// 	LightPositions.push_back(Matrix4x4::Translation(LightPosition0) * Matrix4x4::Scaling(0.1F));
		// 	LightPositions.push_back(Matrix4x4::Translation(LightPosition1) * Matrix4x4::Scaling(0.1F));
		// 
		// 	UnlitMaterial.SetAttribMatrix4x4Array("_iModelMatrix", 2, &LightPositions[0], ModelMatrixBuffer);
		// 
		// 	MeshPrimitives::Cube.DrawInstanciated(2);
		// }

		Rendering::SetViewport(Box2D(0.F, 0.F, (float)Application::GetInstance()->GetWindow().GetWidth(), (float)Application::GetInstance()->GetWindow().GetHeight()));
		// --- Activate corresponding render state

		float FontScale = (FontSize / TextGenerator.GlyphHeight);
		RenderTextMaterial->SetVariables({
			{ "_MainTextureSize", { FontMap->GetSize().FloatVector3() }, SPFlags_None },
			{ "_ProjectionMatrix", { Matrix4x4::Orthographic(
				0.F, (float)Application::GetInstance()->GetWindow().GetWidth(),
				0.F, (float)Application::GetInstance()->GetWindow().GetHeight()
			) }, SPFlags_None },
			{ "_MainTexture", { ETextureDimension::Texture2D, FontMap }, SPFlags_None},
			{ "_TextSize", { FontScale }, SPFlags_None },
			{ "_TextBold", { FontBoldness }, SPFlags_None }
		});
		RenderTextMaterial->Use();

		double TimeCount = 0;
		int TotalCharacterSize = 0;
		MeshData TextMeshData;
		for (int i = 0; i < TextCount; i++) {
			Timer.Begin();
			Vector2 Pivot = TextPivot + Vector2(
				0.F, Application::GetInstance()->GetWindow().GetHeight() - (i + 1) * FontSize + FontSize / TextGenerator.GlyphHeight);

			TextGenerator.GenerateMesh(
				Box2D(0, 0, (float)Application::GetInstance()->GetWindow().GetWidth(), Pivot.y),
				FontSize, RenderingText[i], &TextMeshData.Faces, &TextMeshData.Vertices
			);
			Timer.Stop();
			TimeCount += Timer.GetDeltaTime<Time::Mili>();
			TotalCharacterSize += (int)RenderingText[i].size();
		}
		DynamicMesh.SwapMeshData(TextMeshData);
		if (DynamicMesh.SetUpBuffers()) {
			DynamicMesh.BindSubdivisionVertexArray(0);
			RenderTextMaterial->SetAttribMatrix4x4Array("_iModelMatrix", 1, Matrix4x4().PointerToValue(), ModelMatrixBuffer);
			DynamicMesh.DrawSubdivisionInstanciated(1, 0);
		}

		RenderingText[2] = Text::Formatted(
			L"└> ElementsIntersected(%d), RayHits(%d)",
			ElementsIntersected.size(),
			0 // TotalHitCount
		);

		RenderingText[0] = Text::Formatted(
			L"Character(%.2f μs, %d), Temp [%.1f°], %.1f FPS (%.2f ms), LightIntensity(%.3f), DeltaCursor(%ls)",
			TimeCount / double(TotalCharacterSize) * 1000.0,
			TotalCharacterSize,
			Application::GetInstance()->GetDeviceFunctions().GetDeviceTemperature(0),
			1.F / Time::GetAverageDelta<Time::Second>(),
			Time::GetDeltaTime<Time::Mili>(),
			LightIntencity / 10000.F + 1.F,
			Text::FormatMath(Vector3(LastCursorPosition.y - Input::GetMouseY(), -LastCursorPosition.x - -Input::GetMouseX())).c_str()
		);

		if (Input::IsKeyDown(SDL_SCANCODE_ESCAPE)) {
			Application::GetInstance()->ShouldClose();
		}

	}

	virtual void OnDetach() override { }

public:
	
	SandboxLayer() : Layer(L"SandboxApp", 2000) {}
};

class SandboxApplication : public Application {
public:
	SandboxApplication() : Application() { }

	void OnInitialize() override {
		PushLayer(new SandboxLayer());
		PushLayer(new SandboxSpaceLayer(L"Main", 2001));
	}

	~SandboxApplication() {

	}
};

EmptySource::Application * EmptySource::CreateApplication() {
	return new SandboxApplication();
}
