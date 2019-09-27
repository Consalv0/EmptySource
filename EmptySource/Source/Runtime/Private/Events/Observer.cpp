
#include "CoreMinimal.h"
#include "Events/Observer.h"

namespace ESource {

	void Observer::Call() const
	{
		for (auto& Callback : Callbacks) {
			Callback.second();
		}
	}

	bool Observer::AddCallback(const NString & Identifier, std::function<void()> Functor) {
		if (Callbacks.find(Identifier) == Callbacks.end()) {
			Callbacks.emplace(std::pair<NString, std::function<void()>>(Identifier, Functor));
			return true;
		}
		return false;
	}

	void Observer::RemoveCallback(const NString & Identifier) {
		Callbacks.erase(Identifier);
	}

	void Observer::RemoveAllCallbacks() {
		Callbacks.clear();
	}

	Observer::~Observer() {
		Callbacks.clear();
	}

}