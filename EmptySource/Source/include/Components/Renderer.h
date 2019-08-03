#pragma once

#include "../include/Component.h"
#include "../include/Observer.h"

class CRenderer : public CComponent {
protected:
	typedef CComponent Supper;
	friend class GGameObject;
	friend class Space;
	CRenderer(GGameObject & GameObject);

	virtual bool Initialize();

	virtual void OnDelete();

	// Maybe the material will be called here. It will use the event sistem in the App RP
	// So this component needs to be suscribed to the render event.
	void Render();

public:
	Observer TestStageObserver;

	class Mesh * Model;
};