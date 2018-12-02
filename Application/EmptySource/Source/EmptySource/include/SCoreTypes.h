#pragma once

#ifndef FSTRING
#define FString(STRING) std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(STRING)
#define FChar(STRING) std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(STRING).c_str()
#endif // !FSTRING

#define FORCEINLINE __forceinline	/* Force code to be inline */