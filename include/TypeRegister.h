#pragma once
#include "Defs.h"
#include "TypePack.h"
#include <vector>
#include <array>
#include <cassert>

namespace AbstractVM
{
	// Zero-cost type identity — one unique pointer address per type T.
	// No RTTI, no typeid, no std::type_index required.
	using TypeToken = const void*;
	constexpr static inline uint8_t InvalidTypeId = 255;

	template<typename T>
	inline TypeToken GetTypeToken() noexcept
	{
		static const char tag = 0;
		return &tag;
	}

	template<typename T>
	struct TypeNameTrait
	{
		static constexpr const char* Get()
		{
			if constexpr (requires { T::TypeName; })
			{
				return T::TypeName;
			}
			else
			{
				return "?";
			}
		}
	};

	template<> struct TypeNameTrait<float> { static constexpr const char* Get() { return "float"; } };
	template<> struct TypeNameTrait<double> { static constexpr const char* Get() { return "double"; } };
	template<> struct TypeNameTrait<int32_t> { static constexpr const char* Get() { return "int"; } };
	template<> struct TypeNameTrait<int64_t> { static constexpr const char* Get() { return "int64"; } };
	template<> struct TypeNameTrait<uint64_t> { static constexpr const char* Get() { return "uint64"; } };

	struct RegTypeInfo
	{
		size_t size;
		size_t alignment;
		const char* name;
	};

	// Abstract base: runtime interface for type identity and layout queries.
	// Owns the TypeInfo vector — populated once by TypeRegisterT<Ts...> at construction.
	class TypeRegister
	{
	public:
		virtual uint8_t GetId(TypeToken token) const noexcept = 0;

		FORCE_INLINE const auto& GetTypeInfos() const noexcept
		{
			return m_infos;
		}

		virtual ~TypeRegister() = default;

	protected:
		std::vector<RegTypeInfo> m_infos;
	};

	// TypeRegisterT inherits TypePack<Ts...> — compile-time type metadata
	// (IndexAt, HasType, TypeAt, TypesCount) is available directly on the register.
	template<typename... Ts>
	class TypeRegisterT final : public TypeRegister, public TypePack<Ts...>
	{
	public:
		TypeRegisterT()
		{
			m_infos = { RegTypeInfo{ sizeof(Ts), alignof(Ts), TypeNameTrait<Ts>::Get() }... };
		}

		// Runtime lookup
		uint8_t GetId(TypeToken token) const noexcept override
		{
			for (uint8_t i = 0; i < static_cast<uint8_t>(m_tokens.size()); ++i)
			{
				if (m_tokens[i] == token)
				{
					return i;
				}
			}
			assert(false && "Type not registered");
			return InvalidTypeId;
		}

	private:
		const static inline std::array<TypeToken, sizeof...(Ts)> m_tokens{ GetTypeToken<Ts>()... };
	};
}
