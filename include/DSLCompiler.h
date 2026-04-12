#pragma once

#include "InstructionSet.h"
#include "TypeRegister.h"
#include "Expected.h"
#include <string_view>
#include <string>
#include <vector>
#include <optional>
#include <format>
#include <algorithm>

namespace AbstractVM
{
	struct CompileError
	{
		size_t line;
		std::string message;
	};

	template<VMConfigSpec Cfg>
	class DslCompiler
	{
	public:
		using ProgramT = Program<Cfg>;

		DslCompiler(const InstructionSet& iset, const TypeRegister& types) noexcept
			: m_iset(iset), m_types(types)
		{
		}

		[[nodiscard]] Expected<std::monostate, CompileError> Compile(std::string_view dslCode, ProgramT& outProgram) const
		{
			outProgram.FillNop();

			std::vector<Instruction<Cfg>> collected;
			collected.reserve(outProgram.MaxSize());

			std::vector<size_t> stackPtrs(Cfg::TypesCount, 0);
			const auto& typeInfos = m_types.GetTypeInfos();

			size_t lineIndex = 0;

			while (!dslCode.empty())
			{
				lineIndex++;

				auto lineEnd = dslCode.find('\n');
				auto line = dslCode.substr(0, lineEnd);
				if (lineEnd != std::string_view::npos)
				{
					dslCode.remove_prefix(lineEnd + 1);
				}
				else
				{
					dslCode = {};
				}

				if (auto commentPos = line.find('#'); commentPos != std::string_view::npos)
				{
					line = line.substr(0, commentPos);
				}

				auto token = NextToken(line);
				if (token.empty())
				{
					continue;
				}

				std::optional<Address> lhsAddr;
				std::optional<std::string_view> lhsType;

				if (IsLhs(token))
				{
					if (!ParseAddressWithType(token, lhsAddr, lhsType))
					{
						return Unexpected(CompileError{ lineIndex, "Invalid LHS address" });
					}

					if (lhsAddr->segment != SegmentType::SEG_STACK)
					{
						return Unexpected(CompileError{ lineIndex, "LHS must be S[]" });
					}

					if (NextToken(line) != "=")
					{
						return Unexpected(CompileError{ lineIndex, "Expected '='" });
					}

					token = NextToken(line);
					if (token.empty())
					{
						return Unexpected(CompileError{ lineIndex, "Missing op name" });
					}
				}

				std::string opName{ token };
				uint32_t opcode = 0;
				if (!m_iset.Lookup(opName.c_str(), opcode))
				{
					return Unexpected(CompileError{ lineIndex, std::format("Unknown op '{}'", opName) });
				}

				const Op* op = m_iset.GetOp(opcode);
				if (!op)
				{
					return Unexpected(CompileError{ lineIndex, "Missing op metadata" });
				}

				// ---- LHS VALIDATION ----
				if (lhsAddr && op->dstCount > 0)
				{
					if (const auto expected = stackPtrs[op->returnTypes[0]]; lhsAddr->offset != expected)
					{
						return Unexpected(CompileError{ lineIndex, std::format("Invalid stack index, expected {}", expected) });
					}

					if (lhsType)
					{
						const char* expectedType = typeInfos[op->returnTypes[0]].name;
						if (!expectedType || *lhsType != expectedType)
						{
							return Unexpected(CompileError{ lineIndex, std::format("Type mismatch, expected {}", expectedType ? expectedType : "?") });
						}
					}
				}

				Instruction<Cfg> instr{};
				instr.opcode = opcode;

				// ---- SRC VALIDATION ----
				for (uint8_t i = 0; i < op->srcCount; ++i)
				{
					auto argToken = NextToken(line);
					if (argToken.empty())
					{
						return Unexpected(CompileError{ lineIndex, "Missing argument" });
					}

					std::optional<Address> addr;
					std::optional<std::string_view> argType;

					if (!ParseAddressWithType(argToken, addr, argType))
					{
						return Unexpected(CompileError{ lineIndex, "Invalid address" });
					}

					if (argType)
					{
						const char* expectedType = typeInfos[op->srcTypes[i]].name;
						if (!expectedType || *argType != expectedType)
						{
							return Unexpected(CompileError{ lineIndex, std::format("Argument type mismatch, expected {}", expectedType ? expectedType : "?") });
						}
					}

					instr.src[i] = *addr;
				}

				if (!NextToken(line).empty())
				{
					return Unexpected(CompileError{ lineIndex, "Trailing garbage" });
				}

				for (uint8_t j = 0; j < op->dstCount; ++j)
				{
					stackPtrs[op->returnTypes[j]]++;
				}

				collected.push_back(instr);
			}

			if (collected.size() > outProgram.MaxSize())
			{
				return Unexpected(CompileError{ lineIndex, "Program too large" });
			}

			const size_t tailOffset = outProgram.MaxSize() - collected.size();
			auto& instrs = outProgram.Instructions();
			std::copy(collected.begin(), collected.end(), instrs.begin() + tailOffset);

			return std::monostate{};
		}

		[[nodiscard]] std::optional<std::string> Decompile(const ProgramT& program) const
		{
			std::string result;
			result.reserve(program.MaxSize() * 24);

			std::vector<size_t> stackPtrs(Cfg::TypesCount, 0);
			const auto& typeInfos = m_types.GetTypeInfos();

			for (const auto& instr : program.Instructions())
			{
				if (instr.opcode == 0)
				{
					continue;
				}
				if (instr.opcode >= m_iset.Size())
				{
					return std::nullopt;
				}

				const OpEntry& entry = m_iset[instr.opcode];
				const Op* op = entry.op;
				if (!entry.name || !op)
				{
					return std::nullopt;
				}

				auto outIt = std::back_inserter(result);

				if (op->dstCount == 1)
				{
					uint8_t tid = op->returnTypes[0];
					const char* tname = typeInfos[tid].name;
					std::format_to(outIt, "S<{}>[{}] = ", tname ? tname : "?", stackPtrs[tid]);
				}
				else if (op->dstCount > 1)
				{
					size_t first = stackPtrs[op->returnTypes[0]];
					size_t last = stackPtrs[op->returnTypes[op->dstCount - 1]];
					std::format_to(outIt, "S[{}..{}] = ", first, last);
				}

				for (uint8_t j = 0; j < op->dstCount; ++j)
				{
					stackPtrs[op->returnTypes[j]]++;
				}

				result += entry.name;

				for (uint8_t i = 0; i < op->srcCount; ++i)
				{
					uint8_t tid = op->srcTypes[i];
					const char* tname = typeInfos[tid].name;

					std::format_to(outIt, " {}<{}>[{}]",
						SegmentChar(instr.src[i].segment),
						tname ? tname : "?",
						instr.src[i].offset);
				}

				result += '\n';
			}

			return result;
		}

	private:
		const InstructionSet& m_iset;
		const TypeRegister& m_types;

		static std::string_view NextToken(std::string_view& str) noexcept
		{
			const auto start = str.find_first_not_of(" \t\r");
			if (start == std::string_view::npos)
			{
				str = {};
				return {};
			}

			const auto end = str.find_first_of(" \t\r", start);
			auto token = str.substr(start, end - start);

			str.remove_prefix(end == std::string_view::npos ? str.size() : end);
			return token;
		}

		static constexpr bool IsLhs(std::string_view token) noexcept
		{
			if (token.size() < 3)
			{
				return false;
			}
			if (token[0] != 'S' && token[0] != 'I' && token[0] != 'C')
			{
				return false;
			}
			return token[1] == '[' || token[1] == '<';
		}

		static bool ParseAddressWithType(std::string_view token,
			std::optional<Address>& outAddr,
			std::optional<std::string_view>& outType) noexcept
		{
			outType.reset();

			if (token.size() < 4)
			{
				return false;
			}

			Address addr{};

			switch (token[0])
			{
				using enum SegmentType;
			case 'I': addr.segment = SEG_DATA; break;
			case 'C': addr.segment = SEG_CONST; break;
			case 'S': addr.segment = SEG_STACK; break;
			default: return false;
			}

			size_t pos = 1;

			if (pos < token.size() && token[pos] == '<')
			{
				auto close = token.find('>', pos);
				if (close == std::string_view::npos)
				{
					return false;
				}

				outType = token.substr(pos + 1, close - pos - 1);
				pos = close + 1;
			}

			if (pos >= token.size() || token[pos] != '[')
			{
				return false;
			}
			pos++;

			auto end = token.find(']', pos);
			if (end == std::string_view::npos)
			{
				return false;
			}

			uint16_t offset = 0;
			auto [ptr, ec] = std::from_chars(token.data() + pos, token.data() + end, offset);

			if (ec != std::errc() || ptr != token.data() + end)
			{
				return false;
			}

			addr.offset = offset;
			outAddr = addr;
			return true;
		}

		static char SegmentChar(SegmentType seg) noexcept
		{
			switch (seg)
			{
				using enum SegmentType;
			case SEG_DATA: return 'I';
			case SEG_CONST: return 'C';
			case SEG_STACK: return 'S';
			default: return '?';
			}
		}
	};
}
