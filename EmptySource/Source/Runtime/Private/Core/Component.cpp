
#include "CoreMinimal.h"
#include "Core/GameObject.h"
#include "Components/Component.h"

namespace ESource {

	CComponent::CComponent(GGameObject & GameObject) : OObject(L"Component"), Holder(GameObject) {
	}

	CComponent::CComponent(const IName & Name, GGameObject & GameObject) : OObject(Name), Holder(GameObject) {
	}

	void CComponent::OnDelete() {
		Supper::OnDelete();
	}

	GGameObject & CComponent::GetGameObject() const {
		return Holder;
	}

}