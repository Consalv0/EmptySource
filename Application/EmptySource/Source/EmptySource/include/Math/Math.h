#pragma once

#include <math.h>

#include "Vector2.h"
#include "Vector3.h"
#include "Vector4.h"
#include "IntVector3.h"

#include "Matrix3x3.h"
#include "Matrix4x4.h"

inline double Pow10(int Number) {
	double Ret = 1.0;
	double R = 10.0;
	if (Number < 0) {
		Number = -Number;
		R = 0.1;
	}

	while (Number) {
		if (Number & 1) {
			Ret *= R;
		}
		R *= R;
		Number >>= 1;
	}
	return Ret;
}