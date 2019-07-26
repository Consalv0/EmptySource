#pragma once

#include "../include/CoreTypes.h"
#include "../include/Observer.h"

struct Event {
protected:
	TArray<const Observer*> Observers;

public:

	virtual inline void Notify() {
		for (auto& ItObserver : Observers) {
			ItObserver->Call();
		}
	}

	virtual inline void AttachObserver(const Observer* Value) {
		Observers.push_back(Value);
	}

	virtual inline void DetachObserver(const Observer* Value) {
		Observers.erase(std::remove(Observers.begin(), Observers.end(), Value), Observers.end());
	}
};