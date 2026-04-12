#pragma once
#include "../include/OpImpl.h"
#include "../include/AbstractVM.h"
#include "../include/InstructionSet.h"
#include <cmath>
#include <set>
#include <type_traits>

namespace ScalarOps
{
	using AbstractVM::CreateOp;
	using InstructionSet = AbstractVM::InstructionSet;

#define OP_NAME(Name) \
		(std::is_same_v<T, float>  ? Name "F" : \
		 std::is_same_v<T, double> ? Name "D" : \
		 std::is_same_v<T, int>    ? Name "I" : Name "?")

#define OP_ENTRY(Name, OpFunc, ValFunc) \
		{ OP_NAME(Name), &CreateOp<OpFunc<T>, ValFunc<T>> }

#define OP_ENTRY_0(Name, OpFunc) \
		{ OP_NAME(Name), &CreateOp<OpFunc<T>> }

	template<typename T>
	FORCE_INLINE bool ValNonZero(const T&, const T&, const T& b) noexcept
	{
		return b != T{};
	}

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
			d = std::fabs(a);
		}
		else if constexpr (std::is_signed_v<T>)
		{
			d = std::abs(a);
		}
	}

	template<typename T>
	FORCE_INLINE void OpSqrt(T& d, const T& a) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			d = std::sqrt(a);
		}
	}

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
	struct ScalarOpProvider
	{
		[[nodiscard]] static bool AddTo(InstructionSet& iset, const AbstractVM::TypeRegister& reg, const std::set<const char*>* selection = nullptr)
		{
			static const AbstractVM::OpDescriptor baseOps[] = {
				OP_ENTRY_0("Add", OpAdd),
				OP_ENTRY_0("Sub", OpSub),
				OP_ENTRY_0("Mul", OpMul),
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

			if constexpr (std::is_integral_v<T>)
			{
				static const AbstractVM::OpDescriptor bitOps[] = {
					OP_ENTRY_0("BitAnd", OpBitAnd),
					OP_ENTRY_0("BitOr",  OpBitOr),
					OP_ENTRY_0("BitXor", OpBitXor),
					OP_ENTRY_0("BitNot", OpBitNot),
				};
				ret &= iset.Add(bitOps, reg, selection);
			}

			if constexpr (std::is_floating_point_v<T>)
			{
				static const AbstractVM::OpDescriptor floatOps[] = {
					OP_ENTRY("Div",  OpDiv,  ValNonZero),
					OP_ENTRY_0("Sqrt", OpSqrt),
				};
				ret &= iset.Add(floatOps, reg, selection);
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
