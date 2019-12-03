
#include "CoreMinimal.h"
#include "Core/EmptySource.h"
#include "Core/SpaceLayer.h"

#include "Math/CoreMath.h"
#include "Physics/Physics.h"

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
#include "Resources/ModelManager.h"
#include "Resources/MaterialManager.h"
#include "Resources/ImageConversion.h"
#include "Resources/ShaderManager.h"
#include "Resources/TextureManager.h"
#include "Resources/AudioManager.h"

#include "Components/ComponentRenderable.h"

#include "Physics/PhysicsWorld.h"

#include "Fonts/Font.h"
#include "Fonts/Text2DGenerator.h"

#include "Events/Property.h"

#include "../External/IMGUI/imgui.h"
#include "../External/SDL2/include/SDL_keycode.h"

#include "../Public/GameSpaceLayer.h"
#include "../Public/RenderStageFirst.h"
#include "../Public/RenderStageSecond.h"

class GameLayer : public ESource::Layer {
private:

	float SkyboxRoughness = 1.F;

	ESource::Material RenderTextureMaterial = ESource::Material(L"RenderTextureMaterial");
	ESource::Material HDRClampingMaterial = ESource::Material(L"HDRClampingMaterial");
	ESource::MaterialPtr RenderTextMaterial = std::make_shared<ESource::Material>(L"RenderTextMaterial");
	ESource::Material IntegrateBRDFMaterial = ESource::Material(L"IntegrateBRDFMaterial");

	static const int TextCount = 4;
	float FontSize = 10;
	float FontBoldness = 0.55F;
	ESource::WString RenderingText[TextCount];
	ESource::Mesh DynamicMesh;
	ESource::Point2 TextPivot;
	ESource::Font FontFace;
	ESource::Text2DGenerator TextGenerator;
	ESource::PixelMap FontAtlas;

	ESource::RTexturePtr EquirectangularTextureHDR;
	ESource::RTexturePtr FontMap;
	ESource::RTexturePtr CubemapTexture;

protected:

	void SetSceneSkybox(const WString & Path) {
		auto File = ESource::FileManager::GetFile(Path);
		if (File == NULL) return;

		ESource::RTexturePtr EquirectangularTexture = ESource::TextureManager::GetInstance().CreateTexture2D(
			L"EquirectangularTexture_" + File->GetFileName(), Path, ESource::PF_RGB32F, ESource::FM_MinMagLinear, ESource::SAM_Repeat
		); 
		EquirectangularTexture->Load();

		EquirectangularTextureHDR = ESource::TextureManager::GetInstance().CreateTexture2D(
			L"EquirectangularTextureHDR_" + File->GetFileName(), L"", ESource::PF_RGB32F, ESource::FM_MinMagLinear, ESource::SAM_Repeat,
			EquirectangularTexture->GetSize()
		);
		EquirectangularTextureHDR->Load();

		{
			static ESource::RenderTargetPtr Renderer = ESource::RenderTarget::Create();
			EquirectangularTextureHDR->GetTexture()->Bind();
			HDRClampingMaterial.Use();
			HDRClampingMaterial.SetMatrix4x4Array("_ProjectionMatrix", Matrix4x4().PointerToValue());
			HDRClampingMaterial.SetTexture2D("_EquirectangularMap", EquirectangularTexture, 0);

			ESource::MeshPrimitives::Quad.GetVertexArray()->Bind();
			Renderer->BindTexture2D((ESource::Texture2D *)EquirectangularTextureHDR->GetTexture(), EquirectangularTextureHDR->GetSize());
			ESource::Rendering::SetViewport({ 0, 0, EquirectangularTextureHDR->GetSize().X, EquirectangularTextureHDR->GetSize().Y });
			Renderer->Clear();
			ESource::Rendering::DrawIndexed(ESource::MeshPrimitives::Quad.GetVertexArray());
			EquirectangularTextureHDR->GenerateMipMaps();
			Renderer->Unbind();
		}

		ESource::Material EquirectangularToCubemapMaterial = ESource::Material(L"EquirectangularToCubemapMaterial");
		EquirectangularToCubemapMaterial.SetShaderProgram(ESource::ShaderManager::GetInstance().GetProgram(L"EquirectangularToCubemap"));
		EquirectangularToCubemapMaterial.CullMode = ESource::CM_None;
		EquirectangularToCubemapMaterial.CullMode = ESource::CM_ClockWise;

		if (ESource::TextureManager::GetInstance().GetTexture(L"CubemapTexture") == NULL) {
			CubemapTexture = ESource::TextureManager::GetInstance().CreateCubemap(L"CubemapTexture", L"",
				ESource::PF_RGB32F, ESource::FM_MinMagLinear, ESource::SAM_Clamp, EquirectangularTexture->GetSize().Y / 2);
		}
		CubemapTexture->Load();
		CubemapTexture->RenderHDREquirectangular(EquirectangularTextureHDR, &EquirectangularToCubemapMaterial, true);
		EquirectangularTextureHDR->Unload();
	}

	virtual void OnAttach() override {
		ESource::AudioManager::GetInstance().LoadAudioFromFile(L"GunShot.wav", L"Resources/Sounds/GunShot.wav");
		ESource::AudioManager::GetInstance().LoadAudioFromFile(L"Kuak.wav", L"Resources/Sounds/Kuak.wav");

		ESource::PixelMap WhiteMap = ESource::PixelMap(1, 1, 1, ESource::PF_RGB8);
		ESource::PixelMapUtility::PerPixelOperator(WhiteMap, [](unsigned char * Value, const unsigned char Channels) { Value[0] = 255; Value[1] = 255; Value[2] = 255; });
		ESource::PixelMap BlackMap = ESource::PixelMap(1, 1, 1, ESource::PF_RGB8);
		ESource::PixelMapUtility::PerPixelOperator(BlackMap, [](unsigned char * Value, const unsigned char Channels) { Value[0] = 0; Value[1] = 0; Value[2] = 0; });
		ESource::PixelMap NormlMap = ESource::PixelMap(1, 1, 1, ESource::PF_RGB8);
		ESource::PixelMapUtility::PerPixelOperator(NormlMap, [](unsigned char * Value, const unsigned char Channels) { Value[0] = 128; Value[1] = 128; Value[2] = 255; });

		ESource::TextureManager& TextureMng = ESource::TextureManager::GetInstance();
		auto & WhiteTexture = TextureMng.CreateTexture2D(L"WhiteTexture", L"", ESource::PF_RGB8, ESource::FM_MinMagNearest, ESource::SAM_Repeat);
		WhiteTexture->SetPixelData(WhiteMap);
		WhiteTexture->Load();
		auto & BlackTexture = TextureMng.CreateTexture2D(L"BlackTexture", L"", ESource::PF_RGB8, ESource::FM_MinMagNearest, ESource::SAM_Repeat);
		BlackTexture->SetPixelData(BlackMap);
		BlackTexture->Load();
		auto & NormalTexture = TextureMng.CreateTexture2D(L"NormalTexture", L"", ESource::PF_RGB8, ESource::FM_MinMagNearest, ESource::SAM_Repeat);
		NormalTexture->SetPixelData(NormlMap);
		NormalTexture->Load();

		TextureMng.LoadImageFromFile(L"CrossHead",  ESource::PF_RGBA8, ESource::FM_MinMagNearest, ESource::SAM_Clamp, false, L"Resources/Textures/CrossHead.png");
		TextureMng.LoadImageFromFile(L"UIArrow",    ESource::PF_RGBA8, ESource::FM_MinMagLinear,  ESource::SAM_Clamp, true,  L"Resources/Textures/UIArrow.png");
		TextureMng.LoadImageFromFile(L"TextShadow", ESource::PF_RGBA8, ESource::FM_MinMagNearest, ESource::SAM_Clamp, true,  L"Resources/Textures/TextShadow.png");

		TextureMng.LoadImageFromFile(L"Tiles/DesertSends_A", ESource::PF_RGBA8, ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Tiles/DesertSends_A.jpg");
		TextureMng.LoadImageFromFile(L"Tiles/DesertSends_N", ESource::PF_RGB8,  ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Tiles/DesertSends_N.jpg");
		TextureMng.LoadImageFromFile(L"Tiles/DesertSends_R", ESource::PF_R8,    ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Tiles/DesertSends_R.jpg");

		TextureMng.LoadImageFromFile(L"Tiles/GroundBricks_A",  ESource::PF_RGBA8, ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Tiles/GroundBricks_A.jpeg");
		TextureMng.LoadImageFromFile(L"Tiles/GroundBricks_N",  ESource::PF_RGB8,  ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Tiles/GroundBricks_N.png");
		TextureMng.LoadImageFromFile(L"Tiles/GroundBricks_R",  ESource::PF_R8,    ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Tiles/GroundBricks_R.jpeg");
		TextureMng.LoadImageFromFile(L"Tiles/GroundBricks_AO", ESource::PF_R8,    ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Tiles/GroundBricks_AO.jpeg");

		TextureMng.LoadImageFromFile(L"Objects/EgyptianCat_A",  ESource::PF_RGBA8, ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Objects/EgyptianCat_A.jpg");
		TextureMng.LoadImageFromFile(L"Objects/EgyptianCat_N",  ESource::PF_RGB8,  ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Objects/EgyptianCat_N.png");
		TextureMng.LoadImageFromFile(L"Objects/EgyptianCat_R",  ESource::PF_R8,    ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Objects/EgyptianCat_R.jpg");
		TextureMng.LoadImageFromFile(L"Objects/EgyptianCat_M",  ESource::PF_R8,    ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Objects/EgyptianCat_M.jpg");
		TextureMng.LoadImageFromFile(L"Objects/EgyptianCat_AO", ESource::PF_R8,    ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Objects/EgyptianCat_AO.jpg");
		
		TextureMng.LoadImageFromFile(L"Objects/FalloutCar_A",  ESource::PF_RGBA8, ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Objects/FalloutCar_A.png");
		TextureMng.LoadImageFromFile(L"Objects/FalloutCar_N",  ESource::PF_RGB8,  ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Objects/FalloutCar_N.png");
		TextureMng.LoadImageFromFile(L"Objects/FalloutCar_R",  ESource::PF_R8,    ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Objects/FalloutCar_R.png");
		TextureMng.LoadImageFromFile(L"Objects/FalloutCar_M",  ESource::PF_R8,    ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Objects/FalloutCar_M.png");
		TextureMng.LoadImageFromFile(L"Objects/FalloutCar_AO", ESource::PF_R8,    ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Objects/FalloutCar_AO.png");

		TextureMng.LoadImageFromFile(L"Objects/Backpack_A",  ESource::PF_RGBA8, ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Objects/Backpack_A.jpg");
		TextureMng.LoadImageFromFile(L"Objects/Backpack_N",  ESource::PF_RGB8,  ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Objects/Backpack_N.jpg");
		TextureMng.LoadImageFromFile(L"Objects/Backpack_R",  ESource::PF_R8,    ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Objects/Backpack_R.jpg");
		TextureMng.LoadImageFromFile(L"Objects/Backpack_M",  ESource::PF_R8,    ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Objects/Backpack_M.jpg");
		TextureMng.LoadImageFromFile(L"Objects/Backpack_AO", ESource::PF_R8,    ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Objects/Backpack_AO.jpg");

		TextureMng.LoadImageFromFile(L"Objects/FlareGun_A",  ESource::PF_RGBA8, ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Objects/FlareGun_A.png");
		TextureMng.LoadImageFromFile(L"Objects/FlareGun_N",  ESource::PF_RGB8,  ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Objects/FlareGun_N.png");
		TextureMng.LoadImageFromFile(L"Objects/FlareGun_R",  ESource::PF_R8,    ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Objects/FlareGun_R.png");
		TextureMng.LoadImageFromFile(L"Objects/FlareGun_M",  ESource::PF_R8,    ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Objects/FlareGun_M.png");
		TextureMng.LoadImageFromFile(L"Objects/FlareGun_AO", ESource::PF_R8,    ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Objects/FlareGun_AO.png");

		TextureMng.LoadImageFromFile(L"Objects/Neko_A", ESource::PF_RGBA8, ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Objects/Neko_A.png");
		TextureMng.LoadImageFromFile(L"Objects/Neko_N", ESource::PF_RGB8,  ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Objects/Neko_N.png");
		TextureMng.LoadImageFromFile(L"Objects/Neko_R", ESource::PF_R8,    ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Objects/Neko_R.png");
		TextureMng.LoadImageFromFile(L"Objects/Neko_M", ESource::PF_R8,    ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Objects/Neko_M.png");
		TextureMng.LoadImageFromFile(L"Objects/NekoEye_A", ESource::PF_RGBA8, ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Objects/NekoEye_A.png");
		TextureMng.LoadImageFromFile(L"Objects/NekoEye_N", ESource::PF_RGB8,  ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Objects/NekoEye_N.png");
		TextureMng.LoadImageFromFile(L"Objects/NekoEye_R", ESource::PF_R8,    ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Objects/NekoEye_R.png");
		TextureMng.LoadImageFromFile(L"Objects/NekoEye_M", ESource::PF_R8,    ESource::FM_MinMagLinear, ESource::SAM_Repeat, true, L"Resources/Textures/Objects/NekoEye_M.png");

		ESource::ShaderManager& ShaderMng = ESource::ShaderManager::GetInstance();
		ShaderMng.LoadResourcesFromFile(L"Resources/Resources.yaml");
		ShaderMng.CreateProgram(L"CookTorranceShader", L"Resources/Shaders/CookTorrance.shader");
		
		ESource::ModelManager& ModelMng = ESource::ModelManager::GetInstance();
		// https://sketchfab.com/3d-models/flare-gun-ca3695b7ecaf4e35a8b3f2e1ffb84c2c
		ModelMng.LoadFromFile(L"Resources/Models/FlareGun.dae", true);
		// ModelMng.CreateSubModelMesh(L"FlareGun", L"Flare_Short");
		// ModelMng.CreateSubModelMesh(L"FlareGun", L"FlareGun_Frame");
		// ModelMng.CreateSubModelMesh(L"FlareGun", L"FlareGun_Barrel");
		// ModelMng.CreateSubModelMesh(L"FlareGun", L"FlareGun_Hammer");
		// ModelMng.CreateSubModelMesh(L"FlareGun", L"FlareGun_Trigger");
		ModelMng.LoadAsyncFromFile(L"Resources/Models/Neko.obj", true);
		ModelMng.CreateSubModelMesh(L"Neko", L"Neko");
		ModelMng.CreateSubModelMesh(L"Neko", L"NekoCollision");
		ModelMng.LoadAsyncFromFile(L"Resources/Models/SphereUV.obj", true);
		ModelMng.CreateSubModelMesh(L"SphereUV", L"pSphere1");
		// https://sketchfab.com/3d-models/eastern-substances-6eae4e979bc447c99af70284bfc4065a
		// https://sketchfab.com/3d-models/low-poly-stylized-ground-314529106f6640f4b436408095b0944c
		ModelMng.LoadAsyncFromFile(L"Resources/Models/TileDesertSends.obj", true);
		ModelMng.CreateSubModelMesh(L"TileDesertSends", L"TileDesertSends");
		ModelMng.CreateSubModelMesh(L"TileDesertSends", L"TileGroundBricks");
		ModelMng.LoadAsyncFromFile(L"Resources/Models/EgyptianCat.obj", true);
		ModelMng.CreateSubModelMesh(L"EgyptianCat", L"Cat_Statue_CatStatue");
		// https://sketchfab.com/3d-models/fallout-car-2-cf54e5b166644fc7ade7bbaac502a04f
		ModelMng.LoadAsyncFromFile(L"Resources/Models/FalloutCar.fbx", true);
		ModelMng.CreateSubModelMesh(L"FalloutCar", L"default");
		// https://sketchfab.com/3d-models/a-backpack-for-an-adventure-2ad86321197a49feb54b7726743d7fd0
		ModelMng.LoadAsyncFromFile(L"Resources/Models/Backpack.dae", true);
		ModelMng.CreateSubModelMesh(L"Backpack", L"Cylinder025");
		
		ModelMng.CreateMesh(ESource::MeshPrimitives::CreateQuadMeshData(0.F, 1.F))->Load();
		ModelMng.CreateMesh(ESource::MeshPrimitives::CreateCubeMeshData(0.F, 1.F))->Load();
		
		ESource::MaterialManager& MaterialMng = ESource::MaterialManager::GetInstance();
		// MaterialMng.CreateMaterial(L"DebugMaterial", ShaderMng.GetProgram(L"UnLitShader"), true, DF_LessEqual, FM_Wireframe, CM_None, {
		// 	{ "_Material.Color", { Vector4(1.F, 0.F, 1.F, 1.F) } }
		// })->bCastShadows = false;
		MaterialMng.CreateMaterial(L"Core/ShadowDepth", ShaderMng.GetProgram(L"DepthTestShader"), true, ESource::DF_LessEqual, ESource::FM_Solid, ESource::CM_None, {}
		)->SetShaderInstancingProgram(ShaderMng.GetProgram(L"DepthTestShader#Instancing"));
		MaterialMng.CreateMaterial(L"Tiles/DesertSends", ShaderMng.GetProgram(L"CookTorranceShader"), true, ESource::DF_LessEqual, ESource::FM_Solid, ESource::CM_CounterClockWise, {
			{ "_MainTexture",      { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Tiles/DesertSends_A") } },
			{ "_NormalTexture",    { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Tiles/DesertSends_N") } },
			{ "_RoughnessTexture", { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Tiles/DesertSends_R") } },
		})->SetShaderInstancingProgram(ShaderMng.GetProgram(L"CookTorranceShader#Instancing"));
		MaterialMng.CreateMaterial(L"Tiles/GroundBricks", ShaderMng.GetProgram(L"CookTorranceShader"), true, ESource::DF_LessEqual, ESource::FM_Solid, ESource::CM_CounterClockWise, {
			{ "_MainTexture",      { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Tiles/GroundBricks_A") } },
			{ "_NormalTexture",    { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Tiles/GroundBricks_N") } },
			{ "_RoughnessTexture", { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Tiles/GroundBricks_R") } },
			{ "_AOTexture",        { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Tiles/GroundBricks_AO") } },
		})->SetShaderInstancingProgram(ShaderMng.GetProgram(L"CookTorranceShader#Instancing"));
		MaterialMng.CreateMaterial(L"Objects/EgyptianCat", ShaderMng.GetProgram(L"CookTorranceShader"), true, ESource::DF_LessEqual, ESource::FM_Solid, ESource::CM_CounterClockWise, {
			{ "_MainTexture",      { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Objects/EgyptianCat_A") } },
			{ "_NormalTexture",    { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Objects/EgyptianCat_N") } },
			{ "_RoughnessTexture", { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Objects/EgyptianCat_R") } },
			{ "_MetallicTexture",  { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Objects/EgyptianCat_M") } },
			{ "_AOTexture",        { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Objects/EgyptianCat_AO") } },
		})->SetShaderInstancingProgram(ShaderMng.GetProgram(L"CookTorranceShader#Instancing"));
		MaterialMng.CreateMaterial(L"Objects/FalloutCar", ShaderMng.GetProgram(L"CookTorranceShader"), true, ESource::DF_LessEqual, ESource::FM_Solid, ESource::CM_CounterClockWise, {
			{ "_MainTexture",      { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Objects/FalloutCar_A") } },
			{ "_NormalTexture",    { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Objects/FalloutCar_N") } },
			{ "_RoughnessTexture", { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Objects/FalloutCar_R") } },
			{ "_MetallicTexture",  { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Objects/FalloutCar_M") } },
			{ "_AOTexture",        { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Objects/FalloutCar_AO") } },
		})->SetShaderInstancingProgram(ShaderMng.GetProgram(L"CookTorranceShader#Instancing"));
		MaterialMng.CreateMaterial(L"Objects/Backpack", ShaderMng.GetProgram(L"CookTorranceShader"), true, ESource::DF_LessEqual, ESource::FM_Solid, ESource::CM_CounterClockWise, {
			{ "_MainTexture",        { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Objects/Backpack_A") } },
			{ "_NormalTexture",      { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Objects/Backpack_N") } },
			{ "_RoughnessTexture",   { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Objects/Backpack_R") } },
			{ "_MetallicTexture",    { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Objects/Backpack_M") } },
			{ "_AOTexture",          { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Objects/Backpack_AO") } },
			{ "_Material.Roughness", { 1.9F } }
		})->SetShaderInstancingProgram(ShaderMng.GetProgram(L"CookTorranceShader#Instancing"));
		MaterialMng.CreateMaterial(L"Objects/FlareGun", ShaderMng.GetProgram(L"CookTorranceShader"), true, ESource::DF_LessEqual, ESource::FM_Solid, ESource::CM_CounterClockWise, {
			{ "_MainTexture",        { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Objects/FlareGun_A") } },
			{ "_NormalTexture",      { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Objects/FlareGun_N") } },
			{ "_RoughnessTexture",   { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Objects/FlareGun_R") } },
			{ "_MetallicTexture",    { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Objects/FlareGun_M") } },
			{ "_AOTexture",          { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Objects/FlareGun_AO") } },
			{ "_Material.Roughness", { 1.0F } }
		})->SetShaderInstancingProgram(ShaderMng.GetProgram(L"CookTorranceShader#Instancing"));
		MaterialMng.CreateMaterial(L"Objects/Neko", ShaderMng.GetProgram(L"CookTorranceShader"), true, ESource::DF_LessEqual, ESource::FM_Solid, ESource::CM_CounterClockWise, {
			{ "_MainTexture",        { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Objects/Neko_A") } },
			{ "_NormalTexture",      { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Objects/Neko_N") } },
			{ "_RoughnessTexture",   { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Objects/Neko_R") } },
			{ "_MetallicTexture",    { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Objects/Neko_M") } },
			{ "_Material.Roughness", { 1.0F } }
		})->SetShaderInstancingProgram(ShaderMng.GetProgram(L"CookTorranceShader#Instancing"));
		MaterialMng.CreateMaterial(L"Objects/NekoEye", ShaderMng.GetProgram(L"CookTorranceShader"), true, ESource::DF_LessEqual, ESource::FM_Solid, ESource::CM_CounterClockWise, {
			{ "_MainTexture",        { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Objects/NekoEye_A") } },
			{ "_NormalTexture",      { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Objects/NekoEye_N") } },
			{ "_RoughnessTexture",   { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Objects/NekoEye_R") } },
			{ "_MetallicTexture",    { ESource::ETextureDimension::Texture2D, TextureMng.GetTexture(L"Objects/NekoEye_M") } },
			{ "_Material.Roughness", { 1.0F } }
		})->SetShaderInstancingProgram(ShaderMng.GetProgram(L"CookTorranceShader#Instancing"));
		ESource::MaterialPtr SkyMaterial = 
			MaterialMng.CreateMaterial(L"RenderCubemapMaterial", ShaderMng.GetProgram(L"RenderCubemapShader"), true, ESource::DF_Always, ESource::FM_Solid, ESource::CM_None, {});
		SkyMaterial->RenderPriority = 1;
		SkyMaterial->bWriteDepth = false;
	}

	virtual void OnImGuiRender() override {
		static ESource::RTexturePtr TextureSample = 
			ESource::TextureManager::GetInstance().CreateTexture2D(L"TextureSample", L"", ESource::PF_RGBA8, ESource::FM_MinMagLinear, ESource::SAM_Repeat, IntVector2(1024, 1024));
		TextureSample->Load();
		
		TArray<WString> TextureNameList = ESource::TextureManager::GetInstance().GetResourceNames();
		TArray<NString> NarrowTextureNameList(TextureNameList.size());
		for (int i = 0; i < NarrowTextureNameList.size(); ++i)
			NarrowTextureNameList[i] = ESource::Text::WideToNarrow((TextureNameList)[i]);
		
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
			"Resources/Textures/tucker_wreck_2k.hdr",
			"Resources/Textures/OpenfootageNET_field_low.hdr"
		};
		
		static int CurrentTexture = 0;
		static int CurrentSkybox = 0;
		static float SampleLevel = 0.F;
		static float Gamma = 2.2F;
		static bool ColorFilter[4] = {true, true, true, true};
		int bMonochrome = (ColorFilter[0] + ColorFilter[1] + ColorFilter[2] + ColorFilter[3]) == 1;
		
		ESource::RTexturePtr SelectedTexture = ESource::TextureManager::GetInstance().GetTexture(
			TextureNameList[Math::Clamp((unsigned long long)CurrentTexture, 0ull, TextureNameList.size() -1)]
		);
		if (SelectedTexture && SelectedTexture->GetLoadState() == ESource::LS_Loaded) {
			int bCubemap;
			if (!(bCubemap = SelectedTexture->GetDimension() == ESource::ETextureDimension::Cubemap)) {
				static ESource::RenderTargetPtr Renderer = ESource::RenderTarget::Create();
				RenderTextureMaterial.Use();
				RenderTextureMaterial.SetFloat1Array("_Gamma", &Gamma);
				RenderTextureMaterial.SetInt1Array("_Monochrome", &bMonochrome);
				RenderTextureMaterial.SetFloat4Array("_ColorFilter",
					Vector4(ColorFilter[0] ? 1.F : 0.F, ColorFilter[1] ? 1.F : 0.F, ColorFilter[2] ? 1.F : 0.F, ColorFilter[3] ? 1.F : 0.F)
					.PointerToValue()
				);
				RenderTextureMaterial.SetMatrix4x4Array("_ProjectionMatrix", Matrix4x4().PointerToValue());
				RenderTextureMaterial.SetInt1Array("_IsCubemap", &bCubemap);
				SelectedTexture->GetTexture()->Bind();
				RenderTextureMaterial.SetTexture2D("_MainTexture", SelectedTexture, 0);
				RenderTextureMaterial.SetTextureCubemap("_MainTextureCube", SelectedTexture, 1);
				float LODLevel = SampleLevel * (float)SelectedTexture->GetMipMapCount();
				RenderTextureMaterial.SetFloat1Array("_Lod", &LODLevel);
		
				Renderer->Bind();
				ESource::MeshPrimitives::Quad.GetVertexArray()->Bind();
				Matrix4x4 QuadPosition = Matrix4x4::Scaling({ 1, -1, 1 });
				RenderTextureMaterial.SetMatrix4x4Array("_ModelMatrix", QuadPosition.PointerToValue());
		
				Renderer->BindTexture2D((ESource::Texture2D *)TextureSample->GetTexture(), TextureSample->GetSize());
				ESource::Rendering::SetViewport({ 0, 0, TextureSample->GetSize().X, TextureSample->GetSize().Y });
				Renderer->Clear();
				ESource::Rendering::DrawIndexed(ESource::MeshPrimitives::Quad.GetVertexArray());
				Renderer->Unbind();
			}
			if (bCubemap) {
				static ESource::RenderTargetPtr Renderer = ESource::RenderTarget::Create();
				RenderTextureMaterial.Use();
				RenderTextureMaterial.SetFloat1Array("_Gamma", &Gamma);
				RenderTextureMaterial.SetInt1Array("_Monochrome", &bMonochrome);
				RenderTextureMaterial.SetFloat4Array("_ColorFilter",
					Vector4(ColorFilter[0] ? 1.F : 0.F, ColorFilter[1] ? 1.F : 0.F,
						ColorFilter[2] ? 1.F : 0.F, ColorFilter[3] ? 1.F : 0.F).PointerToValue()
				);
				RenderTextureMaterial.SetMatrix4x4Array("_ProjectionMatrix", Matrix4x4().PointerToValue());
				RenderTextureMaterial.SetInt1Array("_IsCubemap", &bCubemap);
				SelectedTexture->GetTexture()->Bind();
				RenderTextureMaterial.SetTexture2D("_MainTexture", SelectedTexture, 0);
				RenderTextureMaterial.SetTextureCubemap("_MainTextureCube", SelectedTexture, 1);
				float LODLevel = SampleLevel * (float)SelectedTexture->GetMipMapCount();
				RenderTextureMaterial.SetFloat1Array("_Lod", &LODLevel);
		
				Renderer->Bind();
				ESource::MeshPrimitives::Quad.GetVertexArray()->Bind();
				Matrix4x4 QuadPosition = Matrix4x4::Scaling({ 1, -1, 1 });
				RenderTextureMaterial.SetMatrix4x4Array("_ModelMatrix", QuadPosition.PointerToValue());
		
				Renderer->BindTexture2D((ESource::Texture2D *)TextureSample->GetTexture(), TextureSample->GetSize());
				ESource::Rendering::SetViewport({ 0, 0, TextureSample->GetSize().X, TextureSample->GetSize().Y });
				Renderer->Clear();
				ESource::Rendering::DrawIndexed(ESource::MeshPrimitives::Quad.GetVertexArray());
				Renderer->Unbind();
			}
		}

		ImGui::Begin("Model", 0); 
		{
			TArray<ESource::IName> ModelNameList = ESource::ModelManager::GetInstance().GetResourceModelNames();
			if (ModelNameList.size() > 0) {
				TArray<NString> NarrowModelResourcesList(ModelNameList.size());
				for (int i = 0; i < NarrowModelResourcesList.size(); ++i)
					NarrowModelResourcesList[i] = ESource::Text::WideToNarrow((ModelNameList)[i].GetDisplayName());

				static int Selection = 0;
				ImGui::ListBox("Model List", &Selection, [](void * Data, int indx, const char ** outText) -> bool {
					TArray<NString>* Items = (TArray<NString> *)Data;
					if (outText) *outText = (*Items)[indx].c_str();
					return true;
				}, &NarrowModelResourcesList, (int)NarrowModelResourcesList.size());

				ESource::RModelPtr SelectedModel = ESource::ModelManager::GetInstance().GetModel(ModelNameList[Selection]);
				ImGui::Selectable(NarrowModelResourcesList[Selection].c_str());
				if (SelectedModel != NULL && ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
					ImGui::SetDragDropPayload("ModelHierarchy", &*SelectedModel, sizeof(ESource::RModel));
					ImGui::Text(NarrowModelResourcesList[Selection].c_str());
					ImGui::EndDragDropSource();
				}

				if (SelectedModel != NULL) {
					ImGui::BulletText("AnimationTracks:");
					ImGui::Indent();
					ImGui::Columns(3);
					ImGui::TextUnformatted("Name"); ImGui::NextColumn();
					ImGui::TextUnformatted("Duration"); ImGui::NextColumn();
					ImGui::TextUnformatted("Ticks/Seconds"); ImGui::NextColumn();
					ImGui::Separator();
					for (auto & AnimIt : SelectedModel->GetAnimations()) {
						ImGui::Text("%s", AnimIt.Name.c_str()); ImGui::NextColumn();
						ImGui::Text("%f", AnimIt.Duration); ImGui::NextColumn();
						ImGui::Text("%f", AnimIt.TicksPerSecond); ImGui::NextColumn();
					}
					ImGui::Columns(1);
					ImGui::Unindent();
					ImGui::Separator();
				}

				if (SelectedModel && SelectedModel->IsValid()) {
					ImGui::BulletText("Meshes:");
					ImGui::Indent();
					for (auto & SelectedMesh : SelectedModel->GetMeshes()) {
						if (ImGui::TreeNode(ESource::Text::WideToNarrow(SelectedMesh.second->GetName().GetDisplayName()).c_str())) {
							ImGui::Text("Triangle count: %d", SelectedMesh.second->GetVertexData().Faces.size());
							ImGui::Text("Vertices count: %d", SelectedMesh.second->GetVertexData().StaticVertices.size());
							ImGui::Text("Tangents: %s", SelectedMesh.second->GetVertexData().hasTangents ? "true" : "false");
							ImGui::Text("Normals: %s", SelectedMesh.second->GetVertexData().hasNormals ? "true" : "false");
							ImGui::Text("UVs: %d", SelectedMesh.second->GetVertexData().UVChannels);
							ImGui::Text("Vertex Color: %s", SelectedMesh.second->GetVertexData().hasVertexColor ? "true" : "false");
							ImGui::InputFloat3("##BBox0", (float *)&SelectedMesh.second->GetVertexData().Bounding.MinX, 10, ImGuiInputTextFlags_ReadOnly);
							ImGui::InputFloat3("##BBox1", (float *)&SelectedMesh.second->GetVertexData().Bounding.MinY, 10, ImGuiInputTextFlags_ReadOnly);
							ImGui::TextUnformatted("Materials:");
							for (auto & KeyValue : SelectedMesh.second->GetVertexData().MaterialsMap) {
								ImGui::BulletText("%s : %d", KeyValue.second.c_str(), KeyValue.first);
							}
							ImGui::TreePop();
						}
					}
					ImGui::Unindent();
				}
			}
		}
		ImGui::End();

		ImGui::Begin("Frame Rate History", 0);
		{
			static unsigned char FrameIndex = 0;
			static float FrameRateHist[255];
			for (unsigned int i = 1; i < 255; i++) {
				FrameRateHist[i - 1] = FrameRateHist[i];
			}
			FrameRateHist[254] = (float)ESource::Time::GetDeltaTime<ESource::Time::Mili>();
			ImGui::PushItemWidth(-1); ImGui::PlotLines("##FrameRateHistory",
				FrameRateHist, 255, NULL, 0, 0.F, 60.F, ImVec2(0, ImGui::GetWindowHeight() - ImGui::GetStyle().ItemInnerSpacing.y * 4)); ImGui::NextColumn();
		}
		ImGui::End();

		ImGui::Begin("Shaders", 0); 
		{
			TArray<ESource::IName> ShaderNameList = ESource::ShaderManager::GetInstance().GetResourceShaderNames();
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
			
				ESource::RShaderPtr SelectedShader = ESource::ShaderManager::GetInstance().GetProgram(ShaderNameList[Selection]);
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
		
		ImGui::Begin("Materials", 0);
		{
			static NChar Text[100];
			ImGui::InputText("##MaterialName", Text, 100);
			ImGui::SameLine();
			if (ImGui::Button("Create New Material")) {
				if (strlen(Text) > 0) {
					ESource::MaterialPtr NewMaterial = std::make_shared<ESource::Material>(ESource::Text::NarrowToWide(NString(Text)));
					ESource::MaterialManager::GetInstance().AddMaterial(NewMaterial);
				}
				Text[0] = '\0';
			}

			TArray<ESource::IName> MaterialNameList = ESource::MaterialManager::GetInstance().GetResourceNames();
			TArray<NString> NarrowMaterialNameList(MaterialNameList.size());
			for (int i = 0; i < MaterialNameList.size(); ++i)
				NarrowMaterialNameList[i] = (MaterialNameList)[i].GetNarrowDisplayName();

			TArray<ESource::IName> ShaderNameList = ESource::ShaderManager::GetInstance().GetResourceShaderNames();
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
				ESource::MaterialPtr SelectedMaterial = ESource::MaterialManager::GetInstance().GetMaterial(MaterialNameList[Selection]);
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
							SelectedMaterial->SetShaderProgram(ESource::ShaderManager::GetInstance().GetProgram(ShaderNameList[ShaderSelection]));
					}
					ImGui::NextColumn();

					ImGui::AlignTextToFramePadding(); ImGui::Text("Render Priority"); ImGui::NextColumn();
					ImGui::PushItemWidth(-1); ImGui::DragScalar("##RPriority", ImGuiDataType_U32, &SelectedMaterial->RenderPriority, 100.F);
					ImGui::NextColumn();

					ImGui::AlignTextToFramePadding(); ImGui::Text("Cast Shadows"); ImGui::NextColumn();
					ImGui::PushItemWidth(-1); ImGui::Checkbox("##bCastShadows", &SelectedMaterial->bCastShadows);
					ImGui::NextColumn();

					ImGui::AlignTextToFramePadding(); ImGui::Text("Write Depth"); ImGui::NextColumn();
					ImGui::PushItemWidth(-1); ImGui::Checkbox("##bDepthTest", &SelectedMaterial->bWriteDepth);
					ImGui::NextColumn();

					ImGui::AlignTextToFramePadding(); ImGui::Text("Transparent"); ImGui::NextColumn();
					ImGui::PushItemWidth(-1); ImGui::Checkbox("##bTransparent", &SelectedMaterial->bTransparent);
					ImGui::NextColumn();

					ImGui::AlignTextToFramePadding(); ImGui::Text("Depth Function"); ImGui::NextColumn();
					ImGui::PushItemWidth(-1); ImGui::Combo("##DepthFunction", (int *)&SelectedMaterial->DepthFunction, 
						"Never\0Less\0Equal\0LessEqual\0Greater\0NotEqual\0GreaterEqual\0Always\0");
					ImGui::NextColumn();

					ImGui::AlignTextToFramePadding(); ImGui::Text("Stencil"); ImGui::NextColumn();
					ImGui::Combo("Function", (int *)&SelectedMaterial->StencilFunction,
						"Never\0Less\0Equal\0LessEqual\0Greater\0NotEqual\0GreaterEqual\0Always\0");
					ImGui::InputScalar("Reference", ImGuiDataType_U8,  &SelectedMaterial->StencilReference);
					ImGui::InputScalar("Mask", ImGuiDataType_U8, &SelectedMaterial->StencilMask);
					ImGui::Combo("OnlyPass", (int *)&SelectedMaterial->StencilOnlyPass,
						"Keep\0Zero\0Replace\0Increment\0IncrementLoop\0Decrement\0DecrementLoop\0Invert\0");
					ImGui::Combo("OnlyFail", (int *)&SelectedMaterial->StencilOnlyFail,
						"Keep\0Zero\0Replace\0Increment\0IncrementLoop\0Decrement\0DecrementLoop\0Invert\0");
					ImGui::Combo("PassFail", (int *)&SelectedMaterial->StencilPassFail,
						"Keep\0Zero\0Replace\0Increment\0IncrementLoop\0Decrement\0DecrementLoop\0Invert\0");
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
						case ESource::EShaderUniformType::Matrix4x4Array:
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
						case ESource::EShaderUniformType::Matrix4x4:
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
						case ESource::EShaderUniformType::FloatArray:
							ImGui::AlignTextToFramePadding(); ImGui::Text(KeyValue.Name.c_str()); ImGui::NextColumn();
							for (auto& Value : KeyValue.Value.FloatArray) {
								if (ImGui::TreeNode((std::to_string(i++) + "##" + KeyValue.Name).c_str())) {
									ImGui::PushItemWidth(-1); ImGui::DragFloat("##FloatA", &Value, .01F, -MathConstants::BigNumber, MathConstants::BigNumber);
									ImGui::TreePop();
								}
							}
							ImGui::NextColumn();
							break;
						case ESource::EShaderUniformType::Float:
							ImGui::AlignTextToFramePadding(); ImGui::Text(KeyValue.Name.c_str()); ImGui::NextColumn();
							ImGui::PushItemWidth(-1); ImGui::DragFloat(("##" + KeyValue.Name).c_str(), &KeyValue.Value.Float, .01F, -MathConstants::BigNumber, MathConstants::BigNumber);
							ImGui::NextColumn();
							break;
						case ESource::EShaderUniformType::Float2DArray:
							ImGui::AlignTextToFramePadding(); ImGui::Text(KeyValue.Name.c_str()); ImGui::NextColumn();
							for (auto& Value : KeyValue.Value.Float2DArray) {
								if (ImGui::TreeNode((std::to_string(i++) + "##" + KeyValue.Name).c_str())) {
									ImGui::PushItemWidth(-1); ImGui::DragFloat2("##Float2DA", &Value[0], .1F, -MathConstants::BigNumber, MathConstants::BigNumber);
									ImGui::TreePop();
								}
							}
							ImGui::NextColumn();
							break;
						case ESource::EShaderUniformType::Float2D:
							ImGui::AlignTextToFramePadding(); ImGui::Text(KeyValue.Name.c_str()); ImGui::NextColumn();
							ImGui::PushItemWidth(-1); ImGui::DragFloat2(("##" + KeyValue.Name).c_str(), &KeyValue.Value.Float2D[0], .1F, -MathConstants::BigNumber, MathConstants::BigNumber);
							ImGui::NextColumn();
							break;
						case ESource::EShaderUniformType::Float3DArray:
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
						case ESource::EShaderUniformType::Float3D:
							ImGui::AlignTextToFramePadding(); ImGui::Text(KeyValue.Name.c_str()); ImGui::NextColumn();
							ImGui::PushItemWidth(-1);
							if (KeyValue.IsColor())
								ImGui::ColorEdit3(("##" + KeyValue.Name).c_str(), &KeyValue.Value.Float3D[0], ImGuiColorEditFlags_Float);
							else
								ImGui::DragFloat3(("##" + KeyValue.Name).c_str(), &KeyValue.Value.Float3D[0], .1F, -MathConstants::BigNumber, MathConstants::BigNumber);
							ImGui::NextColumn();
							break;
						case ESource::EShaderUniformType::Float4DArray:
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
						case ESource::EShaderUniformType::Float4D:
							ImGui::AlignTextToFramePadding(); ImGui::Text(KeyValue.Name.c_str()); ImGui::NextColumn();
							ImGui::PushItemWidth(-1);
							if (KeyValue.IsColor())
								ImGui::ColorEdit4(("##" + KeyValue.Name).c_str(), &KeyValue.Value.Float4D[0], ImGuiColorEditFlags_Float);
							else
								ImGui::DragFloat4(("##" + KeyValue.Name).c_str(), &KeyValue.Value.Float4D[0], .1F, -MathConstants::BigNumber, MathConstants::BigNumber);
							ImGui::NextColumn();
							break;
						case ESource::EShaderUniformType::IntArray:
							ImGui::AlignTextToFramePadding(); ImGui::Text(KeyValue.Name.c_str()); ImGui::NextColumn();
							for (auto& Value : KeyValue.Value.IntArray) {
								if (ImGui::TreeNode((std::to_string(i++) + "##" + KeyValue.Name).c_str())) {
									ImGui::PushItemWidth(-1); ImGui::DragInt("##IntA", &Value, 1, INT_MIN, INT_MAX);
									ImGui::TreePop();
								}
							}
							ImGui::NextColumn();
							break;
						case ESource::EShaderUniformType::Int:
							ImGui::AlignTextToFramePadding(); ImGui::Text(KeyValue.Name.c_str()); ImGui::NextColumn();
							ImGui::PushItemWidth(-1); ImGui::DragInt(("##" + KeyValue.Name).c_str(), &KeyValue.Value.Int, 1, INT_MIN, INT_MAX);
							ImGui::NextColumn();
							break;
						case ESource::EShaderUniformType::Cubemap:
						case ESource::EShaderUniformType::Texture2D:
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
									KeyValue.Value.Texture = ESource::TextureManager::GetInstance().GetTexture(TextureNameList[i]);
							}
							ImGui::NextColumn();
							break;
						case ESource::EShaderUniformType::None:
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

			ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("MaxFrameRate"); ImGui::NextColumn();
			ImGui::PushItemWidth(-1); ImGui::DragScalar("##MaxFrameRate", ImGuiDataType_U64, &ESource::Time::MaxUpdateDeltaMicro, 1.F); ImGui::NextColumn();
			ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("MaxRenderFrameRate"); ImGui::NextColumn();
			ImGui::PushItemWidth(-1); ImGui::DragScalar("##MaxRenderFrameRate", ImGuiDataType_U64, &ESource::Time::MaxRenderDeltaMicro, 1.F); ImGui::NextColumn();
			ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Render Scale"); ImGui::NextColumn();
			static float RenderSize = 1.F;
			ImGui::DragFloat("##RenderScale", &RenderSize, 0.01F, 0.1F, 1.F); ImGui::SameLine();
			if (ImGui::Button("Apply")) { ESource::Application::GetInstance()->GetRenderPipeline().SetRenderScale(RenderSize); } ImGui::NextColumn();
			ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Gamma"); ImGui::NextColumn();
			ImGui::PushItemWidth(-1); ImGui::SliderFloat("##Gamma", &ESource::Application::GetInstance()->GetRenderPipeline().Gamma, 0.F, 4.F); ImGui::NextColumn();
			ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Exposure"); ImGui::NextColumn();
			ImGui::PushItemWidth(-1); ImGui::SliderFloat("##Exposure", &ESource::Application::GetInstance()->GetRenderPipeline().Exposure, 0.F, 10.F); ImGui::NextColumn();
			ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Skybox Roughness"); ImGui::NextColumn();
			ImGui::PushItemWidth(-1); ImGui::SliderFloat("##Skybox Roughness", &SkyboxRoughness, 0.F, 1.F); ImGui::NextColumn();
			ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Skybox Texture"); ImGui::NextColumn();
			ImGui::PushItemWidth(-1); if (ImGui::Combo("##Skybox Texture", &CurrentSkybox, SkyBoxes, IM_ARRAYSIZE(SkyBoxes))) {
				SetSceneSkybox(ESource::Text::NarrowToWide(SkyBoxes[CurrentSkybox]));
			} ImGui::NextColumn();
			ImGui::PushItemWidth(0);

			ImGui::Columns(1);
			ImGui::Separator();
			ImGui::PopStyleVar();
		}
		ImGui::End();

		char ProgressText[30]; 
		ImGui::Begin("Audio");
		{
			{
				static TArray<float> AudioChannel1(1);
				static TArray<float> AudioChannel2(1);
				{
					AudioChannel1.resize(32768 / (2 * 4) / 2);
					AudioChannel2.resize(32768 / (2 * 4) / 2);
					auto& Device = ESource::Application::GetInstance()->GetAudioDevice();
					auto Duration = (ESource::Time::Micro::ReturnType)(((32768 * 8u / (Device.SampleSize() * Device.GetChannelCount())) / (float)Device.GetFrecuency()) * ESource::Time::Second::GetSizeInMicro());
					unsigned long long Delta = ESource::Time::GetEpochTime<ESource::Time::Micro>() - Device.LastAudioUpdate;
					float * BufferPtr = (float *)&(ESource::Application::GetInstance()->GetAudioDevice().CurrentSample[0]);
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
			for (auto KeyValue : ESource::Application::GetInstance()->GetAudioDevice()) {
				ESource::AudioDevice::SamplePlayInfo * Info = KeyValue.second;

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
					sprintf(ProgressText, "%.2f", Info->Sample->GetDurationAt<ESource::Time::Second>(Info->Pos));
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
			if (SelectedTexture) {
				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 2));
				ImGui::Columns(2);
				ImGui::Separator();
			
				ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Texture"); ImGui::NextColumn();
				ImGui::PushItemWidth(-1); ImGui::Combo("##Texture", &CurrentTexture, [](void * Data, int indx, const char ** outText) -> bool {
					TArray<NString>* Items = (TArray<NString> *)Data;
					if (outText) *outText = (*Items)[indx].c_str();
					return true;
				}, &NarrowTextureNameList, (int)NarrowTextureNameList.size()); ImGui::NextColumn();
			
				ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("LOD Level"); ImGui::NextColumn();
				ImGui::PushItemWidth(-1); ImGui::SliderFloat("##LOD Level", &SampleLevel, 0.0F, 1.0F, "%.3f"); ImGui::NextColumn();
				ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Gamma"); ImGui::NextColumn();
				ImGui::PushItemWidth(-1); ImGui::SliderFloat("##Gamma", &Gamma, 0.01F, 4.0F, "%.3f"); ImGui::NextColumn();
			
				ImGui::Columns(1);
				ImGui::Separator();
				ImGui::PopStyleVar();
			
				if (ImGui::Button("Delete")) {
					ESource::TextureManager::GetInstance().FreeTexture(SelectedTexture->GetName().GetDisplayName());
					SelectedTexture = NULL;
				}
			
				if (SelectedTexture)
					if (SelectedTexture->GetLoadState() == ESource::LS_Loaded) {
						ImGui::SameLine();
						if (ImGui::Button("Reload")) 
							SelectedTexture->Reload();
						ImGui::SameLine();
						if (ImGui::Button("Unload")) SelectedTexture->Unload();
					} else if (SelectedTexture->GetLoadState() == ESource::LS_Unloaded) {
						ImGui::SameLine();
						if (ImGui::Button("Load")) SelectedTexture->Load();
					}
			}
			ImGui::Separator();
			
			if (SelectedTexture && SelectedTexture->GetLoadState() == ESource::LS_Loaded) {
				ImGui::Checkbox("##RedFilter", &ColorFilter[0]); ImGui::SameLine();
				ImGui::ColorButton("RedFilter##RefColor", ImColor(ColorFilter[0] ? 1.F : 0.F, 0.F, 0.F, 1.F));
				ImGui::SameLine(); ImGui::Checkbox("##GreenFilter", &ColorFilter[1]); ImGui::SameLine();
				ImGui::ColorButton("GreenFilter##RefColor", ImColor(0.F, ColorFilter[1] ? 1.F : 0.F, 0.F, 1.F));
				ImGui::SameLine(); ImGui::Checkbox("##BlueFilter", &ColorFilter[2]); ImGui::SameLine();
				ImGui::ColorButton("BlueFilter##RefColor", ImColor(0.F, 0.F, ColorFilter[2] ? 1.F : 0.F, 1.F));
				ImGui::SameLine(); ImGui::Checkbox("##AlphaFilter", &ColorFilter[3]); ImGui::SameLine();
				ImGui::ColorButton("AlphaFilter##RefColor", ImColor(1.F, 1.F, 1.F, ColorFilter[3] ? 1.F : 0.F), ImGuiColorEditFlags_AlphaPreview);
				ImVec2 ImageSize;
				ImVec2 MPos = ImGui::GetCursorScreenPos();
				if (SelectedTexture->GetDimension() == ESource::ETextureDimension::Texture2D) {
					ImageSize.x = Math::Min(
						ImGui::GetWindowWidth(), (ImGui::GetWindowHeight() - ImGui::GetCursorPosY())
						* SelectedTexture->GetAspectRatio()
					);
					ImageSize.x -= ImGui::GetStyle().ItemSpacing.y * 4.0F;
					ImageSize.y = ImageSize.x / SelectedTexture->GetAspectRatio();
				}
				else {
					ImageSize.x = Math::Min(ImGui::GetWindowWidth(), (ImGui::GetWindowHeight() - ImGui::GetCursorPosY()) * 2.F);
					ImageSize.x -= ImGui::GetStyle().ItemSpacing.y * 4.0F;
					ImageSize.y = ImageSize.x / 2.F;
				}
				ImGui::Image((void *)TextureSample->GetTexture()->GetNativeTexture(), ImageSize);
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
					ImGui::Image((void *)TextureSample->GetTexture()->GetNativeTexture(),
						ImVec2(140.F, 140.F), UV0, UV1, ImVec4(1.F, 1.F, 1.F, 1.F), ImVec4(1.F, 1.F, 1.F, .5F));
					ImGui::EndTooltip();
				}
			}
		}
		ImGui::End();
	}

	virtual void OnAwake() override {
		LOG_DEBUG(L"{0}", ESource::FileManager::GetAppDirectory());

		ESource::Application::GetInstance()->GetRenderPipeline().CreateStage<RenderStageSecond>(L"MainStage");
		ESource::Application::GetInstance()->GetRenderPipeline().Exposure = 3.5F;
		SkyboxRoughness = 0.225F;

		ESource::Application::GetInstance()->GetAudioDevice().AddSample(ESource::AudioManager::GetInstance().GetAudioSample(L"GunShot.wav"), 0.255F, false, true);
		ESource::Application::GetInstance()->GetAudioDevice().AddSample(ESource::AudioManager::GetInstance().GetAudioSample(L"Kuak.wav"), 0.255F, false, true);

		ESource::ShaderManager& ShaderMng = ESource::ShaderManager::GetInstance();
		ESource::RShaderPtr EquiToCubemapShader = ShaderMng.GetProgram(L"EquirectangularToCubemap");
		ESource::RShaderPtr HDRClampingShader = ShaderMng.GetProgram(L"HDRClampingShader");
		ESource::RShaderPtr BRDFShader = ShaderMng.GetProgram(L"BRDFShader");
		ESource::RShaderPtr UnlitShader = ShaderMng.GetProgram(L"UnLitShader");
		ESource::RShaderPtr RenderTextureShader = ShaderMng.GetProgram(L"RenderTextureShader");
		ESource::RShaderPtr IntegrateBRDFShader = ShaderMng.GetProgram(L"IntegrateBRDFShader");
		ESource::RShaderPtr RenderTextShader = ShaderMng.GetProgram(L"RenderTextShader");
		ESource::RShaderPtr RenderCubemapShader = ShaderMng.GetProgram(L"RenderCubemapShader");

		FontFace.Initialize(ESource::FileManager::GetFile(L"Resources/Fonts/ArialUnicode.ttf"));

		TextGenerator.TextFont = &FontFace;
		TextGenerator.GlyphHeight = 45;
		TextGenerator.AtlasSize = 1024;
		TextGenerator.PixelRange = 1.5F;
		TextGenerator.Pivot = 0;

		TextGenerator.PrepareCharacters(0u, 49u);
		TextGenerator.GenerateGlyphAtlas(FontAtlas);
		if (FontMap == NULL) {
			FontMap = ESource::TextureManager::GetInstance().CreateTexture2D(
				L"FontMap", L"", ESource::PF_R8, ESource::FM_MinMagLinear, ESource::SAM_Border, IntVector2(TextGenerator.AtlasSize)
			);
		} else {
			FontMap->Unload();
		}
		FontMap->SetPixelData(FontAtlas);
		FontMap->Load();
		FontMap->GenerateMipMaps();

		RenderTextureMaterial.DepthFunction = ESource::DF_Always;
		RenderTextureMaterial.CullMode = ESource::CM_None;
		RenderTextureMaterial.SetShaderProgram(RenderTextureShader);

		RenderTextMaterial->DepthFunction = ESource::DF_Always;
		RenderTextMaterial->CullMode = ESource::CM_None;
		RenderTextMaterial->SetShaderProgram(RenderTextShader);
		ESource::MaterialManager::GetInstance().AddMaterial(RenderTextMaterial);

		IntegrateBRDFMaterial.DepthFunction = ESource::DF_Always;
		IntegrateBRDFMaterial.CullMode = ESource::CM_None;
		IntegrateBRDFMaterial.SetShaderProgram(IntegrateBRDFShader);

		HDRClampingMaterial.DepthFunction = ESource::DF_Always;
		HDRClampingMaterial.CullMode = ESource::CM_None;
		HDRClampingMaterial.SetShaderProgram(HDRClampingShader);

		ESource::RTexturePtr BRDFLut = ESource::TextureManager::GetInstance().CreateTexture2D(
			L"BRDFLut", L"", ESource::PF_RG16F, ESource::FM_MinMagLinear, ESource::SAM_Clamp, { 512, 512 }
		);
		BRDFLut->Load();
		{
			static ESource::RenderTargetPtr Renderer = ESource::RenderTarget::Create();
			IntegrateBRDFMaterial.Use();
			IntegrateBRDFMaterial.SetMatrix4x4Array("_ProjectionMatrix", Matrix4x4().PointerToValue());

			ESource::MeshPrimitives::Quad.GetVertexArray()->Bind();
			Matrix4x4 QuadPosition = Matrix4x4::Translation({ 0, 0, 0 });
			IntegrateBRDFMaterial.SetMatrix4x4Array("_ModelMatrix", QuadPosition.PointerToValue());

			Renderer->BindTexture2D((ESource::Texture2D *)BRDFLut->GetTexture(), BRDFLut->GetSize());
			ESource::Rendering::SetViewport({ 0, 0, BRDFLut->GetSize().X, BRDFLut->GetSize().Y });
			Renderer->Clear();
			ESource::Rendering::DrawIndexed(ESource::MeshPrimitives::Quad.GetVertexArray());
			Renderer->Unbind();
		}

		SetSceneSkybox(L"Resources/Textures/Arches_E_PineTree_3k.hdr");

		ESource::Application::GetInstance()->GetRenderPipeline().Initialize();
		ESource::Application::GetInstance()->GetRenderPipeline().ContextInterval(0);
	}

	virtual void OnUpdate(ESource::Timestamp Stamp) override {

		if (ESource::Input::IsKeyPressed(ESource::EScancode::Insert)) {
			ESource::Application::GetInstance()->SetRenderImGui(!ESource::Application::GetInstance()->GetRenderImGui());
		}

		if (ESource::Input::IsKeyDown(ESource::EScancode::LeftShift)) {
			if (ESource::Input::IsKeyDown(ESource::EScancode::I)) {
				FontSize += ESource::Time::GetDeltaTime<ESource::Time::Second>() * FontSize;
			}

			if (ESource::Input::IsKeyDown(ESource::EScancode::K)) {
				FontSize -= ESource::Time::GetDeltaTime<ESource::Time::Second>() * FontSize;
			}
		}
		else {
			if (ESource::Input::IsKeyDown(ESource::EScancode::I)) {
				FontBoldness += ESource::Time::GetDeltaTime<ESource::Time::Second>() / 10.F;
			}

			if (ESource::Input::IsKeyDown(ESource::EScancode::K)) {
				FontBoldness -= ESource::Time::GetDeltaTime<ESource::Time::Second>() / 10.F;
			}
		}

		if (ESource::Input::IsKeyDown(ESource::EScancode::V)) {
			for (int i = 0; i < 10; i++) {
				RenderingText[1] += (unsigned long)(rand() % 0x3fff);
			}
		}

		for (int i = 0; i < TextCount; i++) {
			if (FontMap != NULL && TextGenerator.PrepareCharacters(RenderingText[i]) > 0) {
				TextGenerator.GenerateGlyphAtlas(FontAtlas);
				FontMap->Unload();
				FontMap->SetPixelData(FontAtlas);
				FontMap->Load();
				FontMap->GenerateMipMaps();
			}
		}

		if (ESource::Input::IsKeyPressed(ESource::EScancode::F11)) {
			auto & AppWindow = ESource::Application::GetInstance()->GetWindow();
			AppWindow.SetWindowMode(AppWindow.GetWindowMode() == ESource::WM_Windowed ? ESource::WM_FullScreen : ESource::WM_Windowed);
		}

		if (ESource::Input::IsKeyDown(ESource::EScancode::Escape)) {
			ESource::Application::GetInstance()->ShouldClose();
		}

	}

	virtual void OnPostRender() override {
		uint32_t CubemapTextureMipMapCount = CubemapTexture ? CubemapTexture->GetMipMapCount() : 0;
		float SkyRoughnessTemp = (SkyboxRoughness) * (CubemapTextureMipMapCount - 4);
		ESource::MaterialManager::GetInstance().GetMaterial(L"RenderCubemapMaterial")->SetParameters({
			{ "_Skybox", { ESource::ETextureDimension::Cubemap, CubemapTexture } },
			{ "_Lod", { float( SkyRoughnessTemp ) } }
		});

		ESource::Rendering::SetViewport(ESource::Application::GetInstance()->GetWindow().GetViewport());

		float FontScale = (FontSize / TextGenerator.GlyphHeight);
		RenderTextMaterial->SetParameters({
			{ "_MainTextureSize", { FontMap->GetSize().FloatVector3() }, ESource::SPFlags_None },
			{ "_ProjectionMatrix", { Matrix4x4::Orthographic(
				0.F, (float)ESource::Application::GetInstance()->GetWindow().GetWidth(),
				0.F, (float)ESource::Application::GetInstance()->GetWindow().GetHeight()
			) }, ESource::SPFlags_None },
			{ "_MainTexture", { ESource::ETextureDimension::Texture2D, FontMap }, ESource::SPFlags_None},
			{ "_TextSize", { FontScale }, ESource::SPFlags_None },
			{ "_TextBold", { FontBoldness }, ESource::SPFlags_None }
		});
		RenderTextMaterial->Use();

		double TimeCount = 0;
		int TotalCharacterSize = 0;
		ESource::MeshData TextMeshData;
		for (int i = 0; i < TextCount; i++) {
			Vector2 Pivot = TextPivot + Vector2(
				0.F, ESource::Application::GetInstance()->GetWindow().GetHeight() - (i + 1) * FontSize + FontSize / TextGenerator.GlyphHeight);

			TextGenerator.GenerateMesh(
				ESource::Box2D(0, 0, (float)ESource::Application::GetInstance()->GetWindow().GetWidth(), Pivot.Y),
				FontSize, true, RenderingText[i], &TextMeshData.Faces, &TextMeshData.StaticVertices
			);
			TotalCharacterSize += (int)RenderingText[i].size();
		}
		DynamicMesh.SwapMeshData(TextMeshData);
		if (DynamicMesh.SetUpBuffers()) {
			DynamicMesh.GetVertexArray()->Bind();
			RenderTextMaterial->SetMatrix4x4Array("_ModelMatrix", Matrix4x4().PointerToValue());
			ESource::Rendering::DrawIndexed(DynamicMesh.GetVertexArray());
		}

		RenderingText[0] = ESource::Text::Formatted(
			L"%ls, Temp [%.1f°], %.1f FPS (%.2f ms)",
			ESource::Text::NarrowToWide(ESource::Application::GetInstance()->GetWindow().GetContext()->GetDeviceName()).c_str(),
			ESource::Application::GetInstance()->GetDeviceFunctions().GetDeviceTemperature(0),
			1.F / ESource::Time::GetAverageDelta<ESource::Time::Second>(),
			ESource::Time::GetDeltaTime<ESource::Time::Mili>()
		);

		RenderingText[1] = ESource::Text::Formatted(
			L"JoystickDevice 0: Connected (%ls), Name(%ls), Haptics(%ls)\nJoystickDevice 1: Connected (%ls), Name(%ls), Haptics(%ls)",
			ESource::Input::IsJoystickConnected(0) ? L"T" : L"F", ESource::Input::GetJoystickState(0).Name.GetInstanceName().c_str(), ESource::Input::GetJoystickState(0).bHaptics ? L"T" : L"F",
			ESource::Input::IsJoystickConnected(1) ? L"T" : L"F", ESource::Input::GetJoystickState(1).Name.GetInstanceName().c_str(), ESource::Input::GetJoystickState(1).bHaptics ? L"T" : L"F"
		);

	}

	virtual void OnDetach() override { }

public:
	
	GameLayer() : Layer(L"SandboxApp", 2000) {}
};

class GameApplication : public ESource::Application {
public:
	GameApplication() : ESource::Application() { }

	void OnInitialize() override {
		PushLayer(new GameLayer());
		PushLayer(new GameSpaceLayer(L"Main", 1999));
	}

	~GameApplication() {

	}
};

ESource::Application * ESource::CreateApplication() {
	return new GameApplication();
}
