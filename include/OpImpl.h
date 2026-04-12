#pragma once
#include "TypeRegister.h"
#include "TypeErasedVM.h"

namespace AbstractVM
{
	template<typename T, size_t DstCount>
	struct FuncTraits;

	template<typename... Args, size_t DstCount>
	struct FuncTraits<void(Args...) noexcept, DstCount>
	{
		static_assert(DstCount <= sizeof...(Args), "DstCount out of range");

		using AllTuple = std::tuple<Args...>;

		template<size_t... I>
		static auto MakeDstTuple(std::index_sequence<I...>) -> std::tuple<std::tuple_element_t<I, AllTuple>...>;

		template<size_t... I>
		static auto MakeSrcTuple(std::index_sequence<I...>) -> std::tuple<std::tuple_element_t<DstCount + I, AllTuple>...>;

		using DstTuple = decltype(MakeDstTuple(std::make_index_sequence<DstCount>{}));
		using SrcTuple = decltype(MakeSrcTuple(std::make_index_sequence<sizeof...(Args) - DstCount>{}));

		static constexpr size_t DstCountV = DstCount;
		static constexpr size_t SrcCountV = sizeof...(Args) - DstCount;
	};

	// pointer forwarding
	template<typename... Args, size_t DstCount>
	struct FuncTraits<void(*)(Args...) noexcept, DstCount>
		: FuncTraits<void(Args...) noexcept, DstCount> {
	};

	template<auto Fn, size_t DstCount, auto ValFn = nullptr>
	struct OpImpl final : Op
	{
		using Traits = FuncTraits<decltype(Fn), DstCount>;

		explicit OpImpl(const TypeRegister& reg) noexcept
		{
			dstCount = static_cast<uint8_t>(Traits::DstCountV);
			srcCount = static_cast<uint8_t>(Traits::SrcCountV);

			FillTypes<typename Traits::DstTuple, typename Traits::SrcTuple>(
				reg,
				std::make_index_sequence<Traits::DstCountV>{},
				std::make_index_sequence<Traits::SrcCountV>{}
			);
		}

		void Execute(DataType* RESTRICT dst, const DataType* RESTRICT src) const noexcept override
		{
			Call<typename Traits::DstTuple, typename Traits::SrcTuple>(
				dst, src,
				std::make_index_sequence<Traits::DstCountV>{},
				std::make_index_sequence<Traits::SrcCountV>{}
			);
		}

		bool Validate(const DataType* RESTRICT dst, const DataType* RESTRICT src) const noexcept override
		{
			if constexpr (ValFn == nullptr)
			{
				return true;
			}
			else
			{
				return Check<typename Traits::DstTuple, typename Traits::SrcTuple>(
					dst, src,
					std::make_index_sequence<Traits::DstCountV>{},
					std::make_index_sequence<Traits::SrcCountV>{}
				);
			}
		}

		[[nodiscard]] bool IsValid() const noexcept
		{
			for (uint8_t i = 0; i < dstCount; i++)
			{
				assert(returnTypes[i] != InvalidTypeId && "Destination type not registered");
				if (returnTypes[i] == InvalidTypeId)
				{
					return false;
				}
			}
			for (uint8_t i = 0; i < srcCount; i++)
			{
				assert(srcTypes[i] != InvalidTypeId && "Source type not registered");
				if (srcTypes[i] == InvalidTypeId)
				{
					return false;
				}
			}
			return true;
		}

	private:
		template<typename DTpl, typename STpl, size_t... DI, size_t... SI>
		void FillTypes(const TypeRegister& reg,
			std::index_sequence<DI...>,
			std::index_sequence<SI...>) noexcept
		{
			((returnTypes[DI] = reg.GetId(GetTypeToken<std::remove_cvref_t<std::tuple_element_t<DI, DTpl>>>())), ...);
			((srcTypes[SI] = reg.GetId(GetTypeToken<std::remove_cvref_t<std::tuple_element_t<SI, STpl>>>())), ...);
		}

		template<typename DTpl, typename STpl, size_t... DI, size_t... SI>
		void Call(DataType* d, const DataType* s,
			std::index_sequence<DI...>,
			std::index_sequence<SI...>) const noexcept
		{
			Fn(
				(*reinterpret_cast<std::remove_reference_t<std::tuple_element_t<DI, DTpl>>*>(d[DI]))...,
				(*reinterpret_cast<const std::remove_reference_t<std::tuple_element_t<SI, STpl>>*>(s[SI]))...
			);
		}

		template<typename DTpl, typename STpl, size_t... DI, size_t... SI>
		bool Check(const DataType* d, const DataType* s,
			std::index_sequence<DI...>,
			std::index_sequence<SI...>) const noexcept
		{
			return ValFn(
				(*reinterpret_cast<const std::remove_reference_t<std::tuple_element_t<DI, DTpl>>*>(d[DI]))...,
				(*reinterpret_cast<const std::remove_reference_t<std::tuple_element_t<SI, STpl>>*>(s[SI]))...
			);
		}

	};

	template<auto Fn, auto ValFn = nullptr, size_t DstCount = 1>
	const Op* CreateOp(const TypeRegister& reg)
	{
		auto ret = new OpImpl<Fn, DstCount, ValFn>(reg);
		assert(ret && ret->IsValid());
		if (!ret || !ret->IsValid())
		{
			delete ret;
			return nullptr;
		}
		return ret;
	}
}
