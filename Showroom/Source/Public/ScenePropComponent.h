#pragma once

#include "Components/Component.h"

class CSceneProp : public ESource::CComponent {
	IMPLEMENT_COMPONENT(CSceneProp)
public:
	class CGameState * GameStateComponent;
	
	void ScaleDown(float ScaleFactor);

	bool bReloading;

protected:
	typedef ESource::CComponent Supper;

	ESource::Transform StartingTransform;

	CSceneProp(ESource::GGameObject & GameObject, float ScaleSpeed);

	virtual void OnUpdate(const ESource::Timestamp & DeltaTime) override;

	virtual void OnDelete() override;

private:
	float Scale;
	float ScaleSpeed;
};