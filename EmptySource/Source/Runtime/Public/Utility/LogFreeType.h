#pragma once

#include <ft2build.h>
#include FT_FREETYPE_H

#include "Utility/TextFormatting.h"

inline const ESource::WChar* FT_ErrorMessage(FT_Error err) {
#undef __FTERRORS_H__
#define FT_ERRORDEF( e, v, s )  case e: return ESource::Text::NarrowToWide(s).c_str();
#define FT_ERROR_START_LIST     switch (err) {
#define FT_ERROR_END_LIST       }
#include FT_ERRORS_H
	return L"Unknown error";
}
