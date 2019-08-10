
#include "CoreMinimal.h"
#include "Engine/GameObject.h"
#include "Components/Component.h"

namespace EmptySource {

	CComponent::CComponent(GGameObject & GameObject) : OObject(L"Component"), Holder(GameObject) {
	}

	CComponent::CComponent(WString Name, GGameObject & GameObject) : OObject(Name), Holder(GameObject) {
	}

	void CComponent::OnDelete() {
		Supper::OnDelete();
	}

	bool CComponent::Initialize() {
		return true;
	}

	GGameObject & CComponent::GetGameObject() const {
		return Holder;
	}

}