#include "../include/Observer.h"

void Observer::Call() const
{
	for (auto& Callback : Callbacks) {
		Callback.second();
	}
}

int Observer::AddCallback(std::function<void()> Functor) {
	static int KeyGenerator = 0;
	Callbacks.emplace(std::pair<int, std::function<void()>>(KeyGenerator++, Functor));
	return KeyGenerator;
}

void Observer::RemoveCallback(const int & Identifier) {
	Callbacks.erase(Identifier);
}
