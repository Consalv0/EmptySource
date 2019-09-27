#pragma once

#include "CoreTypes.h"

#define IMPLEMENT_EVENT_ENUMTYPE(EnumType, Type) static EnumType GetStaticType() { return EnumType::##Type; }\
												 virtual EnumType GetEventType() const override { return GetStaticType(); }\
												 virtual const WChar* GetName() const override { return L#EnumType ## "::" ## #Type; }

#define IMPLEMENT_EVENT_CATEGORY(Category) virtual unsigned int GetCategoryFlags() const override { return Category; }

namespace ESource {

	class Event {
	public:
		virtual const WChar* GetName() const = 0;
		virtual unsigned int GetCategoryFlags() const = 0;
	};

	template<typename B>
	class EventDispatcher {
		template<typename T>
		using EventFunction = std::function<void(T&)>;
	public:
		EventDispatcher(B& BaseEvent)
			: DispatchedEvent(BaseEvent) {
		}

		template<typename T>
		bool Dispatch(EventFunction<T> Function) {
			if (DispatchedEvent.GetEventType() == T::GetStaticType()) {
				Function(*(T*)&DispatchedEvent);
				return true;
			}
			return false;
		}

	private:
		B& DispatchedEvent;
	};

}