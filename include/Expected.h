#pragma once

#ifdef __cpp_lib_expected
#include <expected>
template<typename T, typename E>
using Expected = std::expected<T, E>;
template<typename E>
using Unexpected = std::unexpected<E>;
#else
#include <variant>

template<typename E>
struct Unexpected
{
	E error;

	explicit Unexpected(E e)
		: error(std::move(e))
	{
	}
};

template<typename T, typename E>
struct Expected
{
	std::variant<T, E> data;

	Expected(T val)
		: data(std::move(val))
	{
	}

	Expected(Unexpected<E> u)
		: data(std::move(u.error))
	{
	}

	auto has_value() const noexcept
	{
		return std::holds_alternative<T>(data);
	}

	explicit operator bool() const noexcept
	{
		return has_value();
	}

	const auto& value() const noexcept
	{
		return std::get<T>(data);
	}

	const E& error() const noexcept
	{
		return std::get<E>(data);
	}

	const T* operator->() const noexcept
	{
		return &value();
	}

	const T& operator*() const noexcept
	{
		return value();
	}
};
#endif
