#pragma once

#include "CoreTypes.h"
#include <algorithm>

namespace EmptySource {

	namespace Text {

		NString WideToNarrow(const WChar *From);

		WString NarrowToWide(const NChar *From);

		NString WideToNarrow(const WString &From);

		WString NarrowToWide(const NString &From);

		template<class T>
		inline bool CompareIgnoreCase(T A, T B) {
			transform(A.begin(), A.end(), A.begin(), toupper);
			transform(B.begin(), B.end(), B.begin(), toupper);

			return (A == B);
		}

		//* Replace part of string with another string
		template<class T, class C, class S>
		inline bool Replace(T& String, const C& From, const S& To) {
			size_t StartPos = String.find(From);

			if (StartPos == T::npos) {
				return false;
			}

			String.replace(StartPos, From.length(), To);
			return true;
		}

		//* Replace part of string to the end of the string with another string
		template<class T, class C, class S>
		inline bool ReplaceFromLast(T& String, const C& From, const S& To) {
			size_t StartPos = String.rfind(From);

			if (StartPos == T::npos) {
				return false;
			}

			String.replace(StartPos, String.length() - StartPos, To);
			return true;
		}

		//* Get the last part not containing the elements in string
		template<class T, class C>
		inline bool GetLastNotOf(const T& String, T& Residue, T& Last, const C& Elements) {
			size_t StartPos = String.find_last_not_of(Elements);

			if (StartPos == T::npos) {
				return false;
			}

			Last = String.substr(StartPos + 1);
			Residue = String.substr(0, StartPos + 1);
			return true;
		}

		template<typename ... Arguments>
		WString Formatted(const WString& Format, Arguments ... Args) {
			const WChar* FormatBuffer = Format.c_str();
			int Size = (int)sizeof(WChar) * (int)Format.size();
			std::unique_ptr<WChar[]> Buffer;

			while (true) {
				Buffer = std::make_unique<WChar[]>(Size);
				int OldSize = Size;
				Size = std::swprintf(Buffer.get(), Size, FormatBuffer, Args ...);

				if (Size < 0) {
					Size += OldSize + 10;
				}
				else {
					break;
				}
			}

			return WString(Buffer.get(), Buffer.get() + Size);
		}

		template<typename ... Arguments>
		WString Formatted(const WChar* Format, Arguments ... Args) {
			int Size = (int)std::wcslen(Format);
			std::unique_ptr<WChar[]> Buffer;

			while (true) {
				Buffer = std::make_unique<WChar[]>(Size);
				int OldSize = Size;
				Size = std::swprintf(Buffer.get(), Size, Format, Args ...);

				if (Size < 0) {
					Size += OldSize + 25;
				}
				else {
					break;
				}
			}

			return WString(Buffer.get(), Buffer.get() + Size); // We don't want the '\0' inside
		}

		template<class Num>
		inline WString FormatUnit(const Num & Number, const int & Decimals) {
			double PrecisionNumber = (double)Number;
			WString Suffix = L"";
			if (PrecisionNumber > 1e3 && PrecisionNumber <= 1e6) {
				Suffix = L'k';
				PrecisionNumber /= 1e3;
			}
			else
				if (PrecisionNumber > 1e6 && PrecisionNumber <= 1e9) {
					Suffix = L'M';
					PrecisionNumber /= 1e6;
				}
				else
					if (PrecisionNumber > 1e9 && PrecisionNumber <= 1e12) {
						Suffix = L'G';
						PrecisionNumber /= 1e9;
					}
					else
						if (PrecisionNumber > 1e12) {
							Suffix = L'T';
							PrecisionNumber /= 1e12;
						}

			if (int(PrecisionNumber) == PrecisionNumber) {
				return Formatted(L"%d%s", (int)PrecisionNumber, Suffix.c_str());
			}
			else {
				return Formatted(L"%." + std::to_wstring(Decimals) + L"f%s", PrecisionNumber, Suffix.c_str());
			}
		}

		template<class Num>
		inline WString FormatData(const Num & Number, const int & MaxDecimals) {
			double PrecisionNumber = (double)Number;
			WString Suffix = L"b";
			if (PrecisionNumber > 1 << 10 && PrecisionNumber <= 1 << 20) {
				Suffix = L"kb";
				PrecisionNumber /= 1 << 10;
			}
			else
				if (PrecisionNumber > 1 << 20 && PrecisionNumber <= 1 << 30) {
					Suffix = L"Mb";
					PrecisionNumber /= 1 << 20;
				}
				else
					if (PrecisionNumber > 1 << 30 && PrecisionNumber <= (size_t)1 << 40) {
						Suffix = L"Gb";
						PrecisionNumber /= 1 << 30;
					}
					else
						if (PrecisionNumber > (size_t)1 << 40) {
							Suffix = L"Tb";
							PrecisionNumber /= (size_t)1 << 40;
						}

			if (int(PrecisionNumber) == PrecisionNumber) {
				return Formatted(L"%d%s", (int)PrecisionNumber, Suffix.c_str());
			}
			else {
				return Formatted(L"%." + std::to_wstring(MaxDecimals) + L"f%s", PrecisionNumber, Suffix.c_str());
			}
		}
	}

}