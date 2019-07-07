#pragma once

#include "../include/CoreTypes.h"
#include "../include/Observer.h"

template <typename T>
struct Property {
	T InternalValue;
	TArray<const Observer*> Observers;

	virtual inline void NotifyChange() {
		for (auto& ItObserver : Observers) {
			ItObserver->Call();
		}
	}

public:
	Property() : InternalValue() {
	}

	Property(const T & Value) : InternalValue(Value) {
	}

	T virtual inline Get() const { return InternalValue; }
	inline operator T() const { return InternalValue; }

	T virtual inline Set(const T & Value) {
		InternalValue = Value;
		NotifyChange();
		return InternalValue;
	}
	T virtual inline operator=(const T & Value) { return Set(Value); }

	virtual inline void AttachObserver(const Observer* Value) {
		Observers.push_back(Value);
	}

	virtual inline void DetachObserver(const Observer* Value) {
		Observers.erase(std::remove(Observers.begin(), Observers.end(), Value), Observers.end());
	}
};