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
#include <charconv>
#include <cstring>

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

			size_t lineIndex = 0;
			std::string_view remainingCode = dslCode;

			std::vector<size_t> constCounts(Cfg::TypesCount, 0);
			std::vector<size_t> stackPtrs(Cfg::TypesCount, 0);

			// ---- CONST PARSING ----
			auto constRes = ParseConstants(remainingCode, outProgram, lineIndex, constCounts);
			if (!constRes)
			{
				return Unexpected(constRes.error());
			}

			// ---- INSTRUCTION PARSING ----
			std::vector<Instruction<Cfg>> collected;
			collected.reserve(outProgram.MaxSize());

			while (!remainingCode.empty())
			{
				lineIndex++;
				auto line = ConsumeLine(remainingCode);
				
				auto instrRes = ParseInstructionLine(line, lineIndex, stackPtrs, constCounts, outProgram);
				if (!instrRes)
				{
					return Unexpected(instrRes.error());
				}

				if (instrRes->has_value())
				{
					if (collected.size() >= outProgram.MaxSize())
					{
						return Unexpected(CompileError{ lineIndex, "Program too large" });
					}
					collected.push_back(**instrRes);
				}
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

			// ---- EMIT CONST BLOCKS ----
			for (uint8_t tid = 0; tid < Cfg::TypesCount; ++tid)
			{
				const auto& info = typeInfos[tid];
				if (!info.name)
				{
					continue;
				}

				// Only emit CONST blocks for types supported by FormatValue
				std::string_view name{ info.name };
				if (name != "float" && name != "double" && name != "int" && name != "int64" && name != "uint64")
				{
					continue;
				}

				size_t count = program.Constants().Size();
				size_t lastNonZero = 0;
				bool found = false;

				auto dataPtr = program.Constants().Data(tid);
				if (dataPtr)
				{
					for (size_t i = 0; i < count; ++i)
					{
						if (IsNonZero(program.Constants().At(tid, i), info.size))
						{
							lastNonZero = i;
							found = true;
						}
					}
				}

				if (found)
				{
					result += std::format("CONST <{}> [", info.name);
					for (size_t i = 0; i <= lastNonZero; ++i)
					{
						result += FormatValue(program.Constants().At(tid, i), info);
						if (i < lastNonZero)
						{
							result += ", ";
						}
					}
					result += "]\n";
				}
			}
			if (!result.empty())
			{
				result += "\n";
			}

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
		[[nodiscard]] Expected<std::monostate, CompileError> ParseConstants(
			std::string_view& remainingCode,
			ProgramT& outProgram,
			size_t& lineIndex,
			std::vector<size_t>& constCounts) const
		{
			const auto& typeInfos = m_types.GetTypeInfos();

			while (!remainingCode.empty())
			{
				auto line = PeekLine(remainingCode);
				auto tempLine = line;
				auto token = NextToken(tempLine);

				if (token.empty() || token.starts_with('#'))
				{
					ConsumeLine(remainingCode);
					lineIndex++;
					continue;
				}

				if (token != "CONST")
				{
					break;
				}

				ConsumeLine(remainingCode);
				lineIndex++;

				auto typeToken = NextToken(tempLine);
				if (typeToken.empty() || typeToken[0] != '<' || typeToken.back() != '>')
				{
					return Unexpected(CompileError{ lineIndex, "Expected <type> after CONST" });
				}
				std::string_view typeName = typeToken.substr(1, typeToken.size() - 2);

				uint8_t tid = InvalidTypeId;
				for (uint8_t i = 0; i < typeInfos.size(); ++i)
				{
					if (typeInfos[i].name && typeName == typeInfos[i].name)
					{
						tid = i;
						break;
					}
				}

				if (tid == InvalidTypeId)
				{
					return Unexpected(CompileError{ lineIndex, std::format("Unknown type '{}'", typeName) });
				}

				auto valuesToken = NextToken(tempLine);
				if (valuesToken.empty() || valuesToken[0] != '[')
				{
					return Unexpected(CompileError{ lineIndex, "Expected '[' for constant values" });
				}

				std::string valuesStr{ valuesToken };
				while (valuesStr.back() != ']' && !tempLine.empty())
				{
					valuesStr += " ";
					valuesStr += NextToken(tempLine);
				}

				if (valuesStr.back() != ']')
				{
					return Unexpected(CompileError{ lineIndex, "Missing ']' for constant values" });
				}

				std::string_view list = std::string_view(valuesStr).substr(1, valuesStr.size() - 2);
				while (!list.empty())
				{
					auto comma = list.find(',');
					auto valStr = Trim(list.substr(0, comma));
					if (!valStr.empty())
					{
						if (constCounts[tid] >= outProgram.Constants().Size())
						{
							return Unexpected(CompileError{ lineIndex, "Too many constants" });
						}

						if (!WriteConstant(outProgram, tid, constCounts[tid], valStr, typeInfos[tid]))
						{
							return Unexpected(CompileError{ lineIndex, std::format("Failed to parse constant value '{}'", valStr) });
						}
						constCounts[tid]++;
					}

					if (comma == std::string_view::npos)
					{
						break;
					}
					list.remove_prefix(comma + 1);
				}
			}

			return std::monostate{};
		}

		[[nodiscard]] Expected<std::optional<Instruction<Cfg>>, CompileError> ParseInstructionLine(
			std::string_view line,
			size_t lineIndex,
			std::vector<size_t>& stackPtrs,
			const std::vector<size_t>& constCounts,
			const ProgramT& outProgram) const
		{
			const auto& typeInfos = m_types.GetTypeInfos();

			if (auto commentPos = line.find('#'); commentPos != std::string_view::npos)
			{
				line = line.substr(0, commentPos);
			}

			auto token = NextToken(line);
			if (token.empty())
			{
				return std::optional<Instruction<Cfg>>{ std::nullopt };
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

				if (addr->segment == SegmentType::SEG_CONST)
				{
					uint8_t tid = op->srcTypes[i];
					size_t limit = constCounts[tid] > 0 ? constCounts[tid] : outProgram.Constants().Size();
					if (addr->offset >= limit)
					{
						return Unexpected(CompileError{ lineIndex, std::format("Constant index {} out of bounds for type {}", addr->offset, typeInfos[tid].name ? typeInfos[tid].name : "?") });
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

			return std::optional<Instruction<Cfg>>{ instr };
		}

		const InstructionSet& m_iset;
		const TypeRegister& m_types;

		static std::string_view PeekLine(std::string_view str) noexcept
		{
			auto lineEnd = str.find('\n');
			return str.substr(0, lineEnd);
		}

		static std::string_view ConsumeLine(std::string_view& str) noexcept
		{
			auto lineEnd = str.find('\n');
			auto line = str.substr(0, lineEnd);
			if (lineEnd != std::string_view::npos)
			{
				str.remove_prefix(lineEnd + 1);
			}
			else
			{
				str = {};
			}
			return line;
		}

		static std::string_view Trim(std::string_view str) noexcept
		{
			const auto start = str.find_first_not_of(" \t\r");
			if (start == std::string_view::npos)
			{
				return {};
			}
			const auto end = str.find_last_not_of(" \t\r");
			return str.substr(start, end - start + 1);
		}

		bool WriteConstant(ProgramT& outProgram, uint8_t tid, size_t index, std::string_view valStr, const RegTypeInfo& info) const
		{
			void* ptr = outProgram.Constants().At(tid, index);
			if (!info.name)
			{
				return false;
			}

			std::string_view name{ info.name };
			if (name == "float")
			{
				std::string s{ valStr };
				char* end = nullptr;
				float v = std::strtof(s.c_str(), &end);
				if (end == s.c_str() + s.size())
				{
					*reinterpret_cast<float*>(ptr) = v;
					return true;
				}
			}
			else if (name == "double")
			{
				std::string s{ valStr };
				char* end = nullptr;
				double v = std::strtod(s.c_str(), &end);
				if (end == s.c_str() + s.size())
				{
					*reinterpret_cast<double*>(ptr) = v;
					return true;
				}
			}
			else if (name == "int")
			{
				int32_t v = 0;
				if (auto [p, ec] = std::from_chars(valStr.data(), valStr.data() + valStr.size(), v); ec == std::errc())
				{
					*reinterpret_cast<int32_t*>(ptr) = v;
					return true;
				}
			}
			else if (name == "int64")
			{
				int64_t v = 0;
				if (auto [p, ec] = std::from_chars(valStr.data(), valStr.data() + valStr.size(), v); ec == std::errc())
				{
					*reinterpret_cast<int64_t*>(ptr) = v;
					return true;
				}
			}
			else if (name == "uint64")
			{
				uint64_t v = 0;
				if (auto [p, ec] = std::from_chars(valStr.data(), valStr.data() + valStr.size(), v); ec == std::errc())
				{
					*reinterpret_cast<uint64_t*>(ptr) = v;
					return true;
				}
			}
			return false;
		}

		static bool IsNonZero(const void* ptr, size_t size) noexcept
		{
			const uint8_t* p = reinterpret_cast<const uint8_t*>(ptr);
			for (size_t i = 0; i < size; ++i)
			{
				if (p[i] != 0)
				{
					return true;
				}
			}
			return false;
		}

		std::string FormatValue(const void* ptr, const RegTypeInfo& info) const
		{
			std::string_view name{ info.name };
			if (name == "float")
			{
				return std::format("{}", *reinterpret_cast<const float*>(ptr));
			}
			if (name == "double")
			{
				return std::format("{}", *reinterpret_cast<const double*>(ptr));
			}
			if (name == "int")
			{
				return std::format("{}", *reinterpret_cast<const int32_t*>(ptr));
			}
			if (name == "int64")
			{
				return std::format("{}", *reinterpret_cast<const int64_t*>(ptr));
			}
			if (name == "uint64")
			{
				return std::format("{}", *reinterpret_cast<const uint64_t*>(ptr));
			}
			return "?";
		}

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
