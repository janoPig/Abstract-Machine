#pragma once
#include "../include/OpImpl.h"
#include "../include/AbstractVM.h"
#include "../include/InstructionSet.h"
#include <cmath>
#include <set>
#include <type_traits>
#include <algorithm>

namespace ScalarOps
{
	using AbstractVM::CreateOp;
	using InstructionSet = AbstractVM::InstructionSet;

#define OP_NAME(Name) \
		(std::is_same_v<T, float>    ? Name "F" : \
		 std::is_same_v<T, double>   ? Name "D" : \
		 std::is_same_v<T, int32_t>  ? Name "I" : \
		 std::is_same_v<T, int64_t>  ? Name "I64" : \
		 std::is_same_v<T, uint64_t> ? Name "U64" : Name "?")

#define OP_ENTRY(Name, OpFunc, ValFunc) \
		{ OP_NAME(Name), &CreateOp<OpFunc<T>, ValFunc<T>> }

#define OP_ENTRY_0(Name, OpFunc) \
		{ OP_NAME(Name), &CreateOp<OpFunc<T>> }

	// --- Validators ---

	template<typename T>
	FORCE_INLINE bool ValNonZero(const T&, const T&, const T& b) noexcept
	{
		return b != T{};
	}

	// --- Arithmetic ---

	template<typename T>
	FORCE_INLINE void OpAdd(T& d, const T& a, const T& b) noexcept
	{
		d = a + b;
	}

	template<typename T>
	FORCE_INLINE void OpSub(T& d, const T& a, const T& b) noexcept
	{
		d = a - b;
	}

	template<typename T>
	FORCE_INLINE void OpMul(T& d, const T& a, const T& b) noexcept
	{
		d = a * b;
	}

	template<typename T>
	FORCE_INLINE void OpDiv(T& d, const T& a, const T& b) noexcept
	{
		d = a / b;
	}

	template<typename T>
	FORCE_INLINE void OpNeg(T& d, const T& a) noexcept
	{
		if constexpr (std::is_signed_v<T>)
		{
			d = -a;
		}
	}

	template<typename T>
	FORCE_INLINE void OpAbs(T& d, const T& a) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			d = std::abs(a);
		}
		else if constexpr (std::is_signed_v<T>)
		{
			d = (a < 0) ? -a : a;
		}
		else
		{
			d = a;
		}
	}

	template<typename T>
	FORCE_INLINE void OpMax(T& d, const T& a, const T& b) noexcept
	{
		d = (a > b) ? a : b;
	}

	template<typename T>
	FORCE_INLINE void OpMin(T& d, const T& a, const T& b) noexcept
	{
		d = (a < b) ? a : b;
	}

	// --- Math (Floating Point) ---

#define GEN_UNARY_MATH_OP(Name, Func) \
	template<typename T> \
	FORCE_INLINE void Op##Name(T& d, const T& a) noexcept \
	{ \
		if constexpr (std::is_floating_point_v<T>) \
		{ \
			d = std::Func(a); \
		} \
	}

#define GEN_BINARY_MATH_OP(Name, Func) \
	template<typename T> \
	FORCE_INLINE void Op##Name(T& d, const T& a, const T& b) noexcept \
	{ \
		if constexpr (std::is_floating_point_v<T>) \
		{ \
			d = std::Func(a, b); \
		} \
	}

	GEN_UNARY_MATH_OP(Sin, sin)
	GEN_UNARY_MATH_OP(Cos, cos)
	GEN_UNARY_MATH_OP(Tan, tan)
	GEN_UNARY_MATH_OP(Asin, asin)
	GEN_UNARY_MATH_OP(Acos, acos)
	GEN_UNARY_MATH_OP(Atan, atan)
	GEN_BINARY_MATH_OP(Atan2, atan2)

	GEN_UNARY_MATH_OP(Exp, exp)
	GEN_UNARY_MATH_OP(Log, log)
	GEN_UNARY_MATH_OP(Log10, log10)
	GEN_BINARY_MATH_OP(Pow, pow)
	GEN_UNARY_MATH_OP(Sqrt, sqrt)
	GEN_UNARY_MATH_OP(Cbrt, cbrt)

	GEN_UNARY_MATH_OP(Floor, floor)
	GEN_UNARY_MATH_OP(Ceil, ceil)
	GEN_UNARY_MATH_OP(Round, round)
	GEN_UNARY_MATH_OP(Trunc, trunc)
	GEN_BINARY_MATH_OP(Fmod, fmod)

	// --- Bitwise (Integral) ---

	template<typename T>
	FORCE_INLINE void OpBitAnd(T& d, const T& a, const T& b) noexcept
	{
		if constexpr (std::is_integral_v<T>)
		{
			d = a & b;
		}
	}

	template<typename T>
	FORCE_INLINE void OpBitOr(T& d, const T& a, const T& b) noexcept
	{
		if constexpr (std::is_integral_v<T>)
		{
			d = a | b;
		}
	}

	template<typename T>
	FORCE_INLINE void OpBitXor(T& d, const T& a, const T& b) noexcept
	{
		if constexpr (std::is_integral_v<T>)
		{
			d = a ^ b;
		}
	}

	template<typename T>
	FORCE_INLINE void OpBitNot(T& d, const T& a) noexcept
	{
		if constexpr (std::is_integral_v<T>)
		{
			d = ~a;
		}
	}

	template<typename T>
	FORCE_INLINE void OpBitShl(T& d, const T& a, const T& b) noexcept
	{
		if constexpr (std::is_integral_v<T>)
		{
			d = a << b;
		}
	}

	template<typename T>
	FORCE_INLINE void OpBitShr(T& d, const T& a, const T& b) noexcept
	{
		if constexpr (std::is_integral_v<T>)
		{
			d = a >> b;
		}
	}

	// --- Provider ---

	template<typename T>
	struct ScalarOpProvider
	{
		[[nodiscard]] static bool AddTo(InstructionSet& iset, const AbstractVM::TypeRegister& reg, const std::set<const char*>* selection = nullptr)
		{
			static const AbstractVM::OpDescriptor baseOps[] = {
				OP_ENTRY_0("Add", OpAdd),
				OP_ENTRY_0("Sub", OpSub),
				OP_ENTRY_0("Mul", OpMul),
				OP_ENTRY_0("Max", OpMax),
				OP_ENTRY_0("Min", OpMin),
			};

			bool ret = iset.Add(baseOps, reg, selection);

			if constexpr (std::is_signed_v<T>)
			{
				static const AbstractVM::OpDescriptor signOps[] = {
					OP_ENTRY_0("Neg", OpNeg),
					OP_ENTRY_0("Abs", OpAbs),
				};
				ret &= iset.Add(signOps, reg, selection);
			}
			else if constexpr (std::is_unsigned_v<T>)
			{
				static const AbstractVM::OpDescriptor unsignOps[] = {
					OP_ENTRY_0("Abs", OpAbs),
				};
				ret &= iset.Add(unsignOps, reg, selection);
			}

			if constexpr (std::is_integral_v<T>)
			{
				static const AbstractVM::OpDescriptor bitOps[] = {
					OP_ENTRY_0("BitAnd", OpBitAnd),
					OP_ENTRY_0("BitOr",  OpBitOr),
					OP_ENTRY_0("BitXor", OpBitXor),
					OP_ENTRY_0("BitNot", OpBitNot),
					OP_ENTRY_0("BitShl", OpBitShl),
					OP_ENTRY_0("BitShr", OpBitShr),
				};
				ret &= iset.Add(bitOps, reg, selection);
			}

			if constexpr (std::is_floating_point_v<T>)
			{
				static const AbstractVM::OpDescriptor mathOps[] = {
					OP_ENTRY("Div",   OpDiv,   ValNonZero),
					OP_ENTRY_0("Sin",  OpSin),
					OP_ENTRY_0("Cos",  OpCos),
					OP_ENTRY_0("Tan",  OpTan),
					OP_ENTRY_0("Asin", OpAsin),
					OP_ENTRY_0("Acos", OpAcos),
					OP_ENTRY_0("Atan", OpAtan),
					OP_ENTRY_0("Atan2",OpAtan2),
					OP_ENTRY_0("Exp",  OpExp),
					OP_ENTRY_0("Log",  OpLog),
					OP_ENTRY_0("Log10",OpLog10),
					OP_ENTRY_0("Pow",  OpPow),
					OP_ENTRY_0("Sqrt", OpSqrt),
					OP_ENTRY_0("Cbrt", OpCbrt),
					OP_ENTRY_0("Floor",OpFloor),
					OP_ENTRY_0("Ceil", OpCeil),
					OP_ENTRY_0("Round",OpRound),
					OP_ENTRY_0("Trunc",OpTrunc),
					OP_ENTRY_0("Fmod", OpFmod),
				};
				ret &= iset.Add(mathOps, reg, selection);
			}

			return ret;
		}
	};

	template<typename... Ts>
	[[nodiscard]] static bool RegisterScalarOps(InstructionSet& iset, const AbstractVM::TypeRegister& reg, const std::set<const char*>* selection = nullptr)
	{
		bool ret = true;
		((ret &= ScalarOpProvider<Ts>::AddTo(iset, reg, selection)), ...);
		return ret;
	}
}
