
#include "Utility/TextFormatting.h"

namespace EmptySource::Text {

	NString WideToNarrow(const WChar *From) {
		if (From == NULL)
			return NULL;

		std::mbstate_t State = std::mbstate_t();
		size_t SizeNeeded = std::wcsrtombs(NULL, &From, 0, &State);
		NString To = NString(SizeNeeded, '\0');
		std::wcsrtombs(&To[0], &From, SizeNeeded, &State);

		return To;
	}

	WString NarrowToWide(const NChar *From) {
		if (From == NULL)
			return NULL;

		std::mbstate_t State = std::mbstate_t();
		size_t SizeNeeded = std::mbsrtowcs(NULL, &From, 0, &State);
		WString To = WString(SizeNeeded, L'\0');
		std::mbsrtowcs(&To[0], &From, SizeNeeded, &State);

		return To;
	}

	NString WideToNarrow(const WString &From) {
		if (From.empty())
			return std::string();

		size_t SizeNeeded = std::wcstombs(NULL, &From[0], 0);
		NString To = NString(SizeNeeded, '\0');
		To.resize(std::wcstombs(&To[0], From.c_str(), SizeNeeded));

		return To;
	}

	WString NarrowToWide(const NString &From) {
		if (From.empty())
			return std::wstring();

		size_t SizeNeeded = std::mbstowcs(NULL, &From[0], 0);
		WString To = WString(SizeNeeded, '\0');
		To.resize(std::mbstowcs(&To[0], From.c_str(), SizeNeeded));

		return To;
	}

}
