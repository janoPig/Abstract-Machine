#pragma once

#include <type_traits>
#include <tuple>
#include <concepts>

template <size_t... Is, typename F>
constexpr void constexpr_for(std::index_sequence<Is...>, F&& f)
{
	(f(std::integral_constant<size_t, Is>{}), ...);
}

template<size_t N, typename F>
constexpr void constexpr_for(F&& f)
{
	constexpr_for(std::make_index_sequence<N>{}, std::forward<F>(f));
}

template <typename... Ts>
struct TypePack
{
	static constexpr size_t TypesCount = sizeof...(Ts);

	template <typename T>
	static constexpr size_t CountOf = (size_t(std::is_same_v<T, Ts>) + ...);

	static_assert(((CountOf<Ts> == 1) && ...), "TypePack contains duplicate types!");

	template <typename T>
	static constexpr bool HasType = (CountOf<T> > 0);

	template <size_t I>
	static constexpr bool ValidIndex = (I < TypesCount);

	template <typename T>
	static constexpr size_t IndexAt = [] {
		static_assert(HasType<T>, "Type not found in TypePack!");
		size_t idx = 0;
		(void)((std::is_same_v<T, Ts> ? true : (++idx, false)) || ...);
		return idx;
		}();

	template <size_t I>
	using TypeAt = std::tuple_element_t<I, std::tuple<Ts...>>;

	template <typename OtherPack>
	static constexpr bool Includes = []<typename... Us>(TypePack<Us...>) {
		return (HasType<Us> && ...);
	}(OtherPack{});

	template <typename F>
	static constexpr void ForEach(F&& func)
	{
		constexpr_for<TypesCount>(std::forward<F>(func));
	}
};

template <typename Sub, typename Super>
concept SubsetOf = Super::template Includes<Sub>;

template <size_t I, typename Pack>
concept IsValidIndex = Pack::template ValidIndex<I>;


#define USE_TYPE_PACK(PACK_NAME)\
	using _Pack = PACK_NAME;\
	template<size_t I>\
	using TypeAt = typename _Pack::template TypeAt<I>;\
	template<typename T>\
	static constexpr size_t IndexAt = _Pack::template IndexAt<T>;\
	template<typename T>\
	static constexpr bool HasType = _Pack::template HasType<T>;\
	static constexpr size_t TypesCount = _Pack::TypesCount;\
	template<typename F>\
	static constexpr void ForEach(F&& func)\
	{\
		_Pack::ForEach(std::forward<F>(func));\
	}
