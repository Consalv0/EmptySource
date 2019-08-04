#include "Events/Observer.h"

namespace EmptySource {

	void Observer::Call() const
	{
		for (auto& Callback : Callbacks) {
			Callback.second();
		}
	}

	bool Observer::AddCallback(const String & Identifier, std::function<void()> Functor) {
		if (Callbacks.find(Identifier) == Callbacks.end()) {
			Callbacks.emplace(std::pair<String, std::function<void()>>(Identifier, Functor));
			return true;
		}
		return false;
	}

	void Observer::RemoveCallback(const String & Identifier) {
		Callbacks.erase(Identifier);
	}

	void Observer::RemoveAllCallbacks() {
		Callbacks.clear();
	}

	Observer::~Observer() {
		Callbacks.clear();
	}

}