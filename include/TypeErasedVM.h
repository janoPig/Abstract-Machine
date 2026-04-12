#pragma once

#include <vector>
#include <cassert>
#include <cstring>


namespace AbstractVM
{
#ifndef _SRC_MAX_ARITY
#define _SRC_MAX_ARITY 3
#endif

#ifndef _DST_MAX_ARITY
#define _DST_MAX_ARITY 3
#endif

	// Global arity limits
	constexpr size_t __SrcMaxArity = _SRC_MAX_ARITY;
	constexpr size_t __DstMaxArity = _DST_MAX_ARITY;

	template<typename Cfg>
	concept VMConfigSpec =
		requires {
		typename std::integral_constant<size_t, Cfg::SrcMaxArity>;
		typename std::integral_constant<size_t, Cfg::DstMaxArity>;
		typename std::integral_constant<size_t, Cfg::MaxProgramSize>;
		// declared by TypePack<Ts...>
		typename std::integral_constant<size_t, Cfg::TypesCount>;
		typename std::integral_constant<size_t, Cfg::ConstTypesCount>;
	}
	&& (Cfg::SrcMaxArity > 0 && Cfg::SrcMaxArity <= __SrcMaxArity)
		&& (Cfg::DstMaxArity > 0 && Cfg::DstMaxArity <= __DstMaxArity)
		&& (Cfg::TypesCount > 0 && Cfg::ConstTypesCount <= Cfg::TypesCount)
		&& (Cfg::MaxProgramSize > 0);

	using DataType = uint8_t* RESTRICT;

	enum class SegmentType : uint8_t
	{
		SEG_DATA,
		SEG_CONST,
		SEG_STACK,
		COUNT
	};

	constexpr auto SegmentCount = static_cast<size_t>(SegmentType::COUNT);

	struct Address
	{
		SegmentType segment;
		uint16_t    offset;

		FORCE_INLINE bool IsStack() const noexcept
		{
			return segment == SegmentType::SEG_STACK;
		}
	};

	template<VMConfigSpec Cfg>
	struct Instruction
	{
		uint32_t opcode;
		Address src[Cfg::SrcMaxArity];
	};

	template<VMConfigSpec Cfg>
	constexpr static auto NopInstruction = Instruction<Cfg>{};

	// Abstract operation base.
	// Allows define any instruction so use __DstMaxArity and __SrcMaxArity slots
	struct Op
	{
		uint8_t returnTypes[__DstMaxArity]{};
		uint8_t srcTypes[__SrcMaxArity]{};
		uint8_t dstCount{};
		uint8_t srcCount{};

		virtual bool Validate(const DataType* RESTRICT dst, const DataType* RESTRICT src) const noexcept = 0;
		virtual void Execute(DataType* RESTRICT dst, const DataType* RESTRICT src) const noexcept = 0;
		virtual ~Op() = default;
	};

	// NOP operation: opcode 0, no inputs, no outputs.
	// Must be the first entry (index 0) in every InstructionSet.
	struct NopOp : Op
	{
		NopOp() = default;
		bool Validate(const DataType* RESTRICT, const DataType* RESTRICT) const noexcept override { return true; }
		void Execute(DataType* RESTRICT, const DataType* RESTRICT) const noexcept override {}
	};

	static const inline NopOp g_nopOp{};

	class OpTable
	{
	public:
		virtual ~OpTable() = default;
		virtual const Op* GetOp(uint32_t opcode) const noexcept = 0;
	};

	struct TypeInfo
	{
		size_t size;
		size_t alignment;
	};

	// Set Used = false for unused segment (Program constants without using)
	template<size_t TypesCount>
	class Segment
	{
		DISALLOW_COPY_MOVE_AND_ASSIGN(Segment);
	public:
		Segment() = default;
		~Segment() = default;

		void Init(size_t size)
		{
			m_size = size;
		}

		FORCE_INLINE auto Size() const noexcept
		{
			return m_size;
		}

		FORCE_INLINE auto Data(size_t typeId) noexcept
		{
			assert(m_size);
			assert(typeId < TypesCount);
			return m_segments[typeId];
		}

		FORCE_INLINE auto Data(size_t typeId) const noexcept
		{
			assert(m_size);
			assert(typeId < TypesCount);
			return m_segments[typeId];
		}

		FORCE_INLINE auto At(size_t typeId, size_t idx) noexcept
		{
			assert(m_size);
			assert(typeId < TypesCount);
			assert(idx < m_size);
			return Data(typeId) + idx * m_typeInfo[typeId].size;
		}

		FORCE_INLINE auto At(size_t typeId, size_t idx) const noexcept
		{
			assert(m_size);
			assert(typeId < TypesCount);
			assert(idx < m_size);
			return Data(typeId) + idx * m_typeInfo[typeId].size;
		}

	protected:
		template<size_t I>
		auto& DataRef()
		{
			static_assert(I < TypesCount);
			return m_segments[I];
		}

		template<size_t I>
		auto& TypeRef()
		{
			static_assert(I < TypesCount);
			return m_typeInfo[I];
		}

	private:
		size_t m_size{};
		NO_UNIQUE_ADDRESS std::array<DataType, TypesCount> m_segments{};
		NO_UNIQUE_ADDRESS std::array<TypeInfo, TypesCount> m_typeInfo{};
	};

	template<size_t TypesCount>
	class Stack
	{
		DISALLOW_COPY_MOVE_AND_ASSIGN(Stack);
	public:
		using SegmentT = Segment<TypesCount>;

		Stack() = delete;

		explicit Stack(size_t capacity)
			: m_capacity(capacity)
		{
		}

		FORCE_INLINE void Reset()
		{
			std::memset(m_pointer.data(), 0, m_pointer.size() * sizeof(size_t));
		}

		FORCE_INLINE auto& GetSegment() noexcept
		{
			return m_segment;
		}

		FORCE_INLINE auto Size() const noexcept
		{
			return m_segment.Size();
		}

		FORCE_INLINE auto Capacity() const noexcept
		{
			return m_capacity;
		}

		FORCE_INLINE auto Pointer(size_t idx) const noexcept
		{
			return m_pointer.at(idx);
		}

		FORCE_INLINE DataType PushNext(uint8_t type)
		{
			assert(type < m_pointer.size());
			assert(m_pointer[type] < Size() && "Stack overflow");
			return m_segment.At(type, m_pointer[type]++);
		}

		template<size_t S>
		FORCE_INLINE void GetTop(std::array<uint8_t *, S>& val)
		{
			static_assert(S == TypesCount);
			for (size_t i = 0; i < m_pointer.size(); i++)
			{
				val[i] = m_pointer[i] > 0 ? m_segment.At(i, m_pointer[i] - 1) : nullptr;
			}
		}

		FORCE_INLINE size_t GetLastOffset(uint8_t type) const
		{
			assert(type < m_pointer.size());
			assert(m_pointer[type] > 0 && "No element pushed yet");
			return m_pointer[type] - 1;
		}

	private:
		size_t m_capacity;
		Segment<TypesCount> m_segment{};
		std::array<size_t, TypesCount> m_pointer{};
	};

	template<VMConfigSpec Cfg>
	class Program
	{
	public:
		static constexpr auto UseConst = Cfg::ConstTypesCount > 0;

		Program() = default;

		// Fill all instruction slots with NOP (opcode 0).
		FORCE_INLINE void FillNop()
		{
			std::ranges::fill(instructions, NopInstruction<Cfg>);
		}

		FORCE_INLINE auto& Instructions()
		{
			return instructions;
		}

		FORCE_INLINE const auto& Instructions() const
		{
			return instructions;
		}

		FORCE_INLINE auto& Constants()
		{
			return constants;
		}

		FORCE_INLINE const auto& Constants() const
		{
			return constants;
		}

		FORCE_INLINE size_t MaxSize() const noexcept
		{
			return instructions.size();
		}

	private:
		template<typename T>
		static constexpr size_t prog_size()
		{
			if constexpr (requires { T::MaxProgramSize; })
				return T::MaxProgramSize;
			else
				return 1; // Fallback
		}

		std::array<Instruction<Cfg>, Cfg::MaxProgramSize> instructions{};
		Segment<Cfg::TypesCount> constants{};
	};

	using RunMode = uint32_t;
	static constexpr RunMode RmNone = 0;
	static constexpr RunMode RmFull = 1;
	static constexpr RunMode RmAnalyze = 2;
	static constexpr RunMode RmOptimized = 4;
	static constexpr RunMode RmValidate = 8;
	static constexpr RunMode RmInvalidMask = ~((RmValidate << 1) - 1);

	constexpr static bool ValidRunMode(RunMode mode) noexcept
	{
		if (!mode || (mode & RmInvalidMask))
		{
			return false;
		}
		if (mode & RmFull)
		{
			return (mode & RmOptimized) == RmNone;
		}
		if (mode & RmAnalyze)
		{
			return false;
		}
		return (mode & RmOptimized) || (mode & RmValidate);
	}

	template<VMConfigSpec Cfg>
	class Processor
	{
		DISALLOW_COPY_MOVE_AND_ASSIGN(Processor);
		
	public:
		using StackT = Stack<Cfg::TypesCount>;
		using SegmentT = StackT::SegmentT;
		using ProgramT = Program<Cfg>;

		explicit Processor(const OpTable& ops)
			: m_opTable(ops)
		{
		}

		template<RunMode mode>
		bool Execute(
			const ProgramT& program,
			const SegmentT& input,
			StackT& stack,
			std::vector<std::vector<int32_t>>* producers = nullptr,
			const std::vector<bool>* usedMask  = nullptr
		) const
		{
			static_assert(ValidRunMode(mode));
			const SegmentT* segments[SegmentCount] = { &input, &program.Constants(), &stack.GetSegment()};
			DataType dstPtrs[Cfg::DstMaxArity];
			DataType srcPtrs[Cfg::SrcMaxArity];

			const auto& instructions = program.Instructions();
			for (size_t i = 0; i < instructions.size(); ++i)
			{
				const auto& instr = instructions[i];

				if (!instr.opcode)
				{
					continue;
				}

				const auto op = m_opTable.GetOp(instr.opcode);
				assert(op && op->dstCount <= Cfg::DstMaxArity && op->srcCount <= Cfg::SrcMaxArity);

				for (uint8_t j = 0; j < op->dstCount; j++)
				{
					const auto type = op->returnTypes[j];
					dstPtrs[j] = stack.PushNext(type);

					if constexpr (mode & RmAnalyze)
					{
						(*producers)[type][stack.GetLastOffset(type)] = static_cast<int32_t>(i);
					}
				}

				if constexpr (mode & RmOptimized)
				{
					if (!(*usedMask)[i])
					{
						continue;
					}
				}

				if constexpr (!(mode & RmAnalyze))
				{
					for (uint8_t j = 0; j < op->srcCount; j++)
					{
						const Address& addr = instr.src[j];
						srcPtrs[j] = segments[static_cast<size_t>(addr.segment)]->At(op->srcTypes[j], addr.offset);
						//[j] = (*segments[static_cast<size_t>(addr.segment)])[op->srcTypes[j]].GetAt(addr.offset);
					}
					if constexpr (mode & RmValidate)
					{
						if (!op->Validate(dstPtrs, srcPtrs))
						{
							return false;
						}
					}
					else
					{
						op->Execute(dstPtrs, srcPtrs);
					}
				}
			}
			return true;
		}

	private:
		const OpTable& m_opTable;
	};

	template<VMConfigSpec Cfg>
	class Machine
	{
		DISALLOW_COPY_MOVE_AND_ASSIGN(Machine);
	public:
		using ProcessorT = Processor<Cfg>;
		using ProgramT = ProcessorT::ProgramT;
		using StackT = ProcessorT::StackT;
		using SegmentT = ProcessorT::SegmentT;

		explicit Machine(const OpTable& ops)
			: m_opTable(ops)
			, m_processor(ops)
			, m_stack(Cfg::MaxProgramSize * Cfg::DstMaxArity)
		{
			m_producers.assign(Cfg::TypesCount, std::vector<int32_t>(Cfg::MaxProgramSize * Cfg::DstMaxArity, -1));
			m_usedMask.assign(Cfg::MaxProgramSize, false);
			m_worklist.reserve(Cfg::MaxProgramSize * Cfg::SrcMaxArity);
		}

		void OptimizeProgram(const ProgramT& program)
		{
			std::fill(m_usedMask.begin(), m_usedMask.end(), false);
			for (auto& pVec : m_producers)
			{
				std::fill(pVec.begin(), pVec.end(), -1);
			}
			m_stack.Reset();

			// Input is unused in RmAnalyze mode (src resolution is skipped).
			m_processor.template Execute<RmFull | RmAnalyze>(program, {}, m_stack, &m_producers);

			for (size_t type = 0; type < Cfg::TypesCount; type++)
			{
				if (m_stack.Pointer(type) > 0)
				{
					m_worklist.push_back({ static_cast<uint8_t>(type), static_cast<uint16_t>(m_stack.Pointer(type) - 1) });
				}
			}

			while (!m_worklist.empty())
			{
				const auto [type, offset] = m_worklist.back();
				m_worklist.pop_back();

				const auto pIdx = m_producers[type][offset];
				if (pIdx != -1 && !m_usedMask[pIdx])
				{
					m_usedMask[pIdx] = true;
					const auto& instr = program.Instructions()[pIdx];
					const auto op = m_opTable.GetOp(instr.opcode);
					assert(op);
					for (uint8_t j = 0; j < op->srcCount; j++)
					{
						if (instr.src[j].IsStack())
						{
							m_worklist.push_back({ op->srcTypes[j], instr.src[j].offset });
						}
					}
				}
			}
		}

		void Run(const ProgramT& program, const SegmentT& input, bool optimized = false)
		{
			m_stack.Reset();
			if (optimized)
			{
				m_processor.template Execute<RmOptimized>(program, input, m_stack, nullptr, &m_usedMask);
			}
			else
			{
				m_processor.template Execute<RmFull>(program, input, m_stack);
			}
		}

		bool Validate(const ProgramT& program, const SegmentT& input, bool optimized = false)
		{
			m_stack.Reset();
			if (optimized)
			{
				return m_processor.template Execute<RmOptimized | RmValidate>(program, input, m_stack, nullptr, &m_usedMask);
			}
			else
			{
				return m_processor.template Execute<RmFull | RmValidate>(program, input, m_stack);
			}
		}

		template<size_t S>
		FORCE_INLINE void GetResult(std::array<uint8_t*, S>& val)
		{
			m_stack.GetTop(val);
		}

		FORCE_INLINE auto& GetStack()
		{
			return m_stack;
		}

	private:
		const OpTable& m_opTable;
		ProcessorT m_processor;
		StackT m_stack;

		std::vector<std::vector<int32_t>> m_producers;
		std::vector<bool> m_usedMask;
		// worklist entries are always stack references: (type, offset)
		std::vector<std::pair<uint8_t, uint16_t>> m_worklist;
	};
}
