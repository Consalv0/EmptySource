
#include "CoreMinimal.h"
#include "Core/Application.h"
#include "Core/GameObject.h"
#include "Core/Transform.h"
#include "Core/Input.h"

#include "Resources/ModelResource.h"
#include "Rendering/MeshPrimitives.h"
#include "Resources/TextureManager.h"
#include "Resources/ShaderManager.h"

#include "../Public/GameStateComponent.h"

CGameState::CGameState(ESource::GGameObject & GameObject) : CComponent(L"GameState", GameObject), RenderingText() {
	RenderTextMaterial = ESource::MaterialManager::GetInstance().GetMaterial(ESource::IName(L"RenderTextMaterial", 0));
}

void CGameState::OnInputEvent(ESource::InputEvent & InEvent) {
	ESource::EventDispatcher<ESource::InputEvent> Dispatcher(InEvent);
	Dispatcher.Dispatch<ESource::JoystickButtonPressedEvent>([this](ESource::JoystickButtonPressedEvent & Event) {
		if (Event.GetButton() == ESource::EJoystickButton::Start && GameState == EGameState::WaitStart) {
			if (ESource::Input::IsJoystickConnected(0) && ESource::Input::IsJoystickConnected(1)) {
				GameState = EGameState::Starting;
				CountDown = 5;
			}
		}
	});
	Dispatcher.Dispatch<ESource::JoystickConnectionEvent>([this](ESource::JoystickConnectionEvent & Event) {
		if (!ESource::Input::IsJoystickConnected(0) || !ESource::Input::IsJoystickConnected(1)) {
			GameState = EGameState::WaitStart;
		}
	});
}

void CGameState::OnUpdate(const ESource::Timestamp & DeltaTime) {
	if (GameState == EGameState::WaitStart) {
		if (ESource::Input::IsJoystickConnected(0) && ESource::Input::IsJoystickConnected(1)) {
			RenderingText = L"Press Start";
			FontScaleAnimation = 1.F;
		}
		else {
			RenderingText = L"Connect another gamepad device";
			FontScaleAnimation = 0.5F;
		}
	}

	if (GameState == EGameState::Starting) {
		CountDown -= DeltaTime.GetDeltaTime<ESource::Time::Second>();
		float Mod = std::fmodf(CountDown, 1.F);
		FontScaleAnimation = Math::Map(Mod * 4.F, 0.F, 4.F, 1.F, 4.F);
		RenderingText = ESource::Text::Formatted(L"%.0f", CountDown + 0.5F);
		if (CountDown < 0) {
			ESource::Input::SendHapticImpulse(0, 0, 1.F, 200);
			ESource::Input::SendHapticImpulse(1, 0, 1.F, 200);
			GameState = EGameState::Started;
		}
	}

	if (GameState == EGameState::Started) {
		FontScaleAnimation = 1.F;
		RenderingText = L"";
	}

	if (FontMap != NULL && TextGenerator.PrepareCharacters(RenderingText) > 0) {
		TextGenerator.GenerateGlyphAtlas(FontAtlas);
		FontMap->Unload();
		FontMap->SetPixelData(FontAtlas);
		FontMap->Load();
		FontMap->GenerateMipMaps();
	}
}

void CGameState::OnAwake() {
	RenderingText = L"Press Start";
	FontFace.Initialize(ESource::FileManager::GetFile(L"Resources/Fonts/FuturaCondensed.ttf"));
	TextGenerator.TextFont = &FontFace;
	TextGenerator.GlyphHeight = 45;
	TextGenerator.AtlasSize = 1024;
	TextGenerator.PixelRange = 1.5F;
	TextGenerator.Pivot = 0;

	TextGenerator.PrepareCharacters(RenderingText);
	TextGenerator.GenerateGlyphAtlas(FontAtlas);
	if (FontMap == NULL) {
		FontMap = ESource::TextureManager::GetInstance().CreateTexture2D(
			L"TitleFontMap", L"", ESource::PF_R8, ESource::FM_MinMagLinear, ESource::SAM_Border, IntVector2(TextGenerator.AtlasSize)
		);
	}
	else {
		FontMap->Unload();
	}
	FontMap->SetPixelData(FontAtlas);
	FontMap->Load();
	FontMap->GenerateMipMaps();
}

void CGameState::OnPostRender() {
	float FontScale = (GetFontSize() / TextGenerator.GlyphHeight);
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

	ESource::MeshData TextMeshData;
	{
		Vector2 TextLenght = TextGenerator.GetLenght(GetFontSize(), RenderingText);
		ESource::Box2D Box = ESource::Box2D(
			(float)ESource::Application::GetInstance()->GetWindow().GetWidth() * 0.5F - TextLenght.X * 0.5F,
			0.F,
			(float)ESource::Application::GetInstance()->GetWindow().GetWidth(),
			(float)ESource::Application::GetInstance()->GetWindow().GetHeight() * 0.5F - TextLenght.Y * 0.35F
		);

		TextGenerator.GenerateMesh(
			Box, GetFontSize(), false, RenderingText, &TextMeshData.Faces, &TextMeshData.StaticVertices
		);
	}
	DynamicMesh.SwapMeshData(TextMeshData);
	if (DynamicMesh.SetUpBuffers()) {
		DynamicMesh.GetVertexArray()->Bind();
		RenderTextMaterial->SetMatrix4x4Array("_ModelMatrix", Matrix4x4().PointerToValue());
		ESource::Rendering::DrawIndexed(DynamicMesh.GetVertexArray());
	}
}

void CGameState::OnDelete() {
}

float CGameState::GetFontSize() {
	return FontSize * FontScaleAnimation * ESource::Application::GetInstance()->GetWindow().GetAspectRatio();
}
