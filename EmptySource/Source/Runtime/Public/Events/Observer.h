#pragma once

#include "Engine/CoreTypes.h"
#include "Engine/Text.h"

#include <functional>

namespace EmptySource {

	struct Observer {
	public:
		typedef Observer Supper;

		virtual void Call() const;

		bool AddCallback(const String& Identifier, std::function<void()>);

		void RemoveCallback(const String& Identifier);

		void RemoveAllCallbacks();

		virtual ~Observer();

	private:
		TDictionary<String, std::function<void()>> Callbacks;
	};

}