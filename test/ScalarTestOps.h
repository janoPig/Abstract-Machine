#pragma once
#include "../examples/ScalarOps.h"

namespace ScalarTestOps
{
	using namespace AbstractVM;

	static void OpF2I(int& d, const float& a) noexcept
	{
		d = static_cast<int>(a);
	}

	static void OpI2F(float& d, const int& a) noexcept
	{
		d = static_cast<float>(a);
	}

	static void OpFAddI(float& d, const float& a, const int& b) noexcept
	{
		d = a + static_cast<float>(b);
	}

	static void OpFSplitIntFrac(int& dInt, float& dFrac, const float& a) noexcept
	{
		dInt = static_cast<int>(a);
		dFrac = a - static_cast<float>(dInt);
	}

	static constexpr OpDescriptor kBasicOps[] =
	{
		{ "Add",  &CreateOp<ScalarOps::OpAdd<float>> },
		{ "Mul",  &CreateOp<ScalarOps::OpMul<float>> },
		{ "F2I",  &CreateOp<OpF2I> },
		{ "IAdd", &CreateOp<ScalarOps::OpAdd<int>> },
	};

	static constexpr OpDescriptor kFloatOps[] =
	{
		{ "Add",  &CreateOp<ScalarOps::OpAdd<float>>                               },
		{ "Sub",  &CreateOp<ScalarOps::OpSub<float>>                               },
		{ "Mul",  &CreateOp<ScalarOps::OpMul<float>>                               },
		{ "Neg",  &CreateOp<ScalarOps::OpNeg<float>>                               },
		{ "Div",  &CreateOp<ScalarOps::OpDiv<float>, ScalarOps::ValNonZero<float>> },
	};

	static constexpr OpDescriptor kIntOps[] =
	{
		{ "Add", &CreateOp<ScalarOps::OpAdd<int>> },
		{ "Mul", &CreateOp<ScalarOps::OpMul<int>> },
	};

	static constexpr OpDescriptor kMixedOps[] =
	{
		{ "F2I",          &CreateOp<OpF2I>                        },
		{ "I2F",          &CreateOp<OpI2F>                        },
		{ "FAddI",        &CreateOp<OpFAddI>                      },
		{ "SplitIntFrac", &CreateOp<OpFSplitIntFrac, nullptr, 2> },
	};

	static constexpr OpDescriptor kEdgeCaseOps[] =
	{
		{ "Add",  &CreateOp<ScalarOps::OpAdd<float>>                               },
		{ "Mul",  &CreateOp<ScalarOps::OpMul<float>>                               },
		{ "Div",  &CreateOp<ScalarOps::OpDiv<float>, ScalarOps::ValNonZero<float>> },
		{ "Sqrt", &CreateOp<ScalarOps::OpSqrt<float>>                              },
		{ "F2I",  &CreateOp<OpF2I>                                                  },
		{ "I2F",  &CreateOp<OpI2F>                                                  },
	};
}
