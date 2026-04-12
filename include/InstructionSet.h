#pragma once
#include "TypeRegister.h"
#include "TypeErasedVM.h"
#include <vector>
#include <string>
#include <cassert>
#include <set>
#include <unordered_map>
#include <span>
#include <deque>

namespace AbstractVM
{
	// Entry in a static instruction table.
	// Implementors declare a static constexpr array of these.
	struct OpEntry
	{
		const char* name; // local op name, e.g. "Add"
		const Op* op; // non-owning, must outlive InstructionSet
	};

	struct OpDescriptor
	{
		const char* name;
		const Op* (*factory)(const TypeRegister&);
	};

	class InstructionSet : public OpTable
	{
		DISALLOW_COPY_MOVE_AND_ASSIGN(InstructionSet);

	public:
		static constexpr OpEntry NopEntry{ "NOP", &g_nopOp };

		InstructionSet() = default;
		~InstructionSet() override = default;

		explicit InstructionSet(const char* name)
			: m_name(name)
		{
		}

		explicit InstructionSet(const char* name, std::span<const OpEntry> table, const std::set<const char*>* selection = nullptr)
			: m_name(name)
		{
			Add(table, selection);
		}

		[[nodiscard]] bool Add(std::span<const OpDescriptor> table, const TypeRegister& reg, const std::set<const char*>* selection = nullptr, const char* prefix = nullptr)
		{
			EnsureNop();
			for (const auto& desc : table)
			{
				if (selection && !selection->contains(desc.name))
				{
					continue;
				}
				const Op* op = desc.factory(reg);
				assert(op);
				if (!op)
				{
					return false;
				}
				m_ownedOps.emplace_back(op);
				AddEntry({ desc.name, op }, prefix);
			}
			return true;
		}

		bool Lookup(const char* token, uint32_t& outOpcode) const
		{
			auto it = m_instructionsMap.find(token);
			if (it == m_instructionsMap.end())
			{
				return false;
			}
			outOpcode = it->second;
			return true;
		}

		FORCE_INLINE const char* Name() const noexcept
		{
			return m_name;
		}

		FORCE_INLINE size_t Size() const noexcept
		{
			return m_entries.size();
		}

		FORCE_INLINE const OpEntry& operator[](uint32_t opcode) const noexcept
		{
			assert(opcode < m_entries.size());
			return m_entries[opcode];
		}

		FORCE_INLINE const Op* GetOp(uint32_t opcode) const noexcept override
		{
			assert(opcode < m_entries.size()); return m_entries[opcode].op;
		}

		FORCE_INLINE const auto& Entries() const noexcept
		{
			return m_entries;
		}

	protected:
		void Add(std::span<const OpEntry> table, const std::set<const char*>* selection = nullptr, const char* prefix = nullptr)
		{
			EnsureNop();
			for (const auto& entry : table)
			{
				if (selection && !selection->contains(entry.name))
				{
					continue;
				}
				AddEntry(entry, prefix);
			}
		}

	private:
		void EnsureNop()
		{
			if (m_entries.empty())
			{
				AddEntry(NopEntry, nullptr);
			}
		}

		void AddEntry(const OpEntry& entry, const char* prefix)
		{
			auto& qualName = m_qualifiedNames.emplace_back(
				prefix ? std::string(prefix) + "." + entry.name : entry.name
			);
			const auto opcode = static_cast<uint32_t>(m_entries.size());
			m_instructionsMap.emplace(qualName, opcode);
			// Store entry with name pointing to the owned qualified string.
			m_entries.push_back({ qualName.c_str(), entry.op });
		}

		const char* m_name{};
		std::vector<OpEntry> m_entries{};
		std::vector<std::unique_ptr<const Op>> m_ownedOps{};
		std::unordered_map<std::string, uint32_t> m_instructionsMap{};
		// Owns qualified name strings (e.g. "Vec.Add") — deque ensures pointer stability on growth.
		std::deque<std::string> m_qualifiedNames{};
	};

	class CompoundInstructionSet : public InstructionSet
	{
		DISALLOW_COPY_MOVE_AND_ASSIGN(CompoundInstructionSet);

	public:
		CompoundInstructionSet() = default;

		void AddSet(const InstructionSet& set)
		{
			InstructionSet::Add(std::span<const OpEntry>(set.Entries()), nullptr, set.Name());
		}
	};

	// Generic MakeInstructionSet
	template<typename Container>
	InstructionSet MakeInstructionSet(const char* name,
		const Container& table,
		const std::set<const char*>* selection = nullptr)
	{
		return InstructionSet(name, std::span<const OpEntry>(table), selection);
	}

	// Raw array overload
	template<size_t N>
	InstructionSet MakeInstructionSet(const char* name,
		const OpEntry(&table)[N],
		const std::set<const char*>* selection = nullptr)
	{
		return InstructionSet(name, std::span<const OpEntry>(table, N), selection);
	}
}
