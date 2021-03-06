#pragma once

#include "Components/Component.h"

#include "Fonts/Font.h"
#include "Fonts/Text2DGenerator.h"

class CGameState : public ESource::CComponent {
	IMPLEMENT_COMPONENT(CGameState)
public:
	enum class EGameState {
		WaitStart = 1,
		Starting = 2,
		Started = 3,
		Ending = 4,
		Restart = 5
	};

public:
	ESource::Material RenderTextureMaterial;
	ESource::MaterialPtr RenderTextMaterial;
	float FontSize = 100;
	float FontBoldness = 0.45F;
	ESource::Mesh DynamicMesh;
	ESource::Font FontFace;
	ESource::Text2DGenerator TextGenerator;
	ESource::PixelMap FontAtlas;
	ESource::RTexturePtr FontMap;
	EGameState GameState = EGameState::WaitStart;

	void SetPlayerWinner(int Player);

	void SetBarLenght(int Lenght);

	float GetAnimationTime();

protected:
	typedef ESource::CComponent Supper;

	CGameState(ESource::GGameObject & GameObject);

	virtual void OnInputEvent(ESource::InputEvent & InEvent) override;

	virtual void OnUpdate(const ESource::Timestamp & DeltaTime) override;

	virtual void OnAwake() override;

	virtual void OnPostRender() override;

	virtual void OnDelete() override;

private:

	float GetFontSize();

	float CountDown;
	float FontScaleAnimation;
	ESource::WString RenderingText;
};