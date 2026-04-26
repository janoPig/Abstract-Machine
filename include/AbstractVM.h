#pragma once
#include "InstructionSet.h"
#include "TypePack.h"
#include <tuple>
#include <utility>
#include <array>

namespace AbstractVM
{
	template<typename Cfg>
	concept VMConfig =
		requires {
		typename std::integral_constant<size_t, Cfg::SrcMaxArity>;
		typename std::integral_constant<size_t, Cfg::DstMaxArity>;
		typename std::integral_constant<size_t, Cfg::MaxProgramSize>;
		typename std::integral_constant<size_t, Cfg::MaxConstantsCount>;
	}
	&& (Cfg::SrcMaxArity > 0 && Cfg::SrcMaxArity <= __SrcMaxArity)
		&& (Cfg::DstMaxArity > 0 && Cfg::DstMaxArity <= __DstMaxArity);

	template<typename BaseCfg, typename Types, typename ConstTypes>
	struct VMConfigImpl;

	template<typename BaseCfg, typename... Ts, typename... Cs>
	struct VMConfigImpl<BaseCfg, TypePack<Ts...>, TypePack<Cs...>> : BaseCfg
	{
		static constexpr size_t TypesCount = sizeof...(Ts);
		static constexpr size_t ConstTypesCount = sizeof...(Cs);
	};

	// Base class for objects that support dynamic size
	class VectorLike
	{
	public:
		virtual void Init(size_t capacity) = 0;
		virtual ~VectorLike() = default;
	};

	template<typename T>
	consteval TypeInfo MakeTypeInfo()
	{
		return TypeInfo{ sizeof(T), alignof(T) };
	}

	template<typename T, typename U>
	class SegmentImpl;

	template<typename... Ts, typename... Us>
	class SegmentImpl<TypePack<Ts...>, TypePack<Us...>> : private Segment<sizeof...(Ts)>, public TypePack<Ts...>
	{
		DISALLOW_COPY_MOVE_AND_ASSIGN(SegmentImpl);
		using Base = Segment<sizeof...(Ts)>;
		USE_TYPE_PACK(TypePack<Ts...>)

	public:
		FORCE_INLINE operator const Base&() const noexcept { return *reinterpret_cast<const Base*>(this); }
		FORCE_INLINE operator Base&() noexcept { return *reinterpret_cast<Base*>(this); }
		explicit SegmentImpl(size_t count, size_t vectorCapacity = 1)
			: Base()
		{
			Init(count, vectorCapacity);
		}

		~SegmentImpl()
		{
			Destroy();
		}

		template<size_t I>
		auto GetData() noexcept
		{
			return reinterpret_cast<TypeAt<I>*>(Base::Data(I));
		}

		template<size_t I>
		auto GetData() const noexcept
		{
			return reinterpret_cast<const TypeAt<I>*>(Base::Data(I));
		}

		template<typename T>
		auto GetData() noexcept
		{
			return reinterpret_cast<T*>(Base::Data(IndexAt<T>));
		}

		template<typename T>
		auto GetData() const noexcept
		{
			return reinterpret_cast<const T*>(Base::Data(IndexAt<T>));
		}

		void Init(size_t count, size_t vectorCapacity = 1)
		{
			Base::Init(count);
			ForEach([this, count, vectorCapacity](auto idx) {
				constexpr size_t index = idx;
				using T = TypeAt<index>;
				Base::template TypeRef<index>() = MakeTypeInfo<T>();
				Base::template DataRef<index>() = (DataType)new T[count]();
				if constexpr (std::is_base_of_v<VectorLike, T>)
				{
					auto ptr = GetData<index>();
					for (size_t i = 0; i < count; i++)
					{
						ptr[i].Init(vectorCapacity);
					}
				}
				});
		}

		void Destroy()
		{
			ForEach([this](auto idx) {
				constexpr size_t index = idx;
				using T = TypeAt<index>;
				auto& dataRef = Base::template DataRef<index>();
				auto tptr = reinterpret_cast<T*>(dataRef);
				assert(tptr);
				delete[] tptr;
				dataRef = (DataType)nullptr;
				});
		}
	};

	template<typename T>
	using DataSegmentImpl = SegmentImpl<T, T>;

	template<typename T, typename U = TypePack<>>
	using ConstSegmentImpl = SegmentImpl<T, U>;

	template<VMConfig Cfg, typename T, typename C>
	class ProgramImpl;

	template<VMConfig Cfg, typename... Ts, typename... Cs>
	class ProgramImpl<Cfg, TypePack<Ts...>, TypePack<Cs...>> : private Program<VMConfigImpl<Cfg, TypePack<Ts...>, TypePack<Cs...>>>
	{
		using Config = VMConfigImpl<Cfg, TypePack<Ts...>, TypePack<Cs...>>;
		using Base = Program<Config>;

	public:
		FORCE_INLINE operator const Base&() const noexcept { return *reinterpret_cast<const Base*>(this); }
		FORCE_INLINE operator Base&() noexcept { return *reinterpret_cast<Base*>(this); }
		static constexpr auto TypesCount = sizeof...(Ts);
		using ConstT = ConstSegmentImpl<TypePack<Ts...>, TypePack<Cs...>>;

		explicit ProgramImpl(size_t vectorCapacity = 1)
			: Base()
		{
			if constexpr (Cfg::MaxConstantsCount > 0)
			{
				((ConstT&)Base::Constants()).Init(Cfg::MaxConstantsCount, vectorCapacity);
			}
		}

		auto& Constants() noexcept
		{
			return (ConstT&)Base::Constants();
		}

		const auto& Constants() const noexcept
		{
			return (ConstT&)Base::Constants();
		}

		FORCE_INLINE ~ProgramImpl()
		{
			((ConstT&)Constants()).Destroy();
		}

		template<size_t I>
		auto GetConst() noexcept
		{
			return Constants().template GetData<I>();
		}

		template<size_t I>
		auto GetConst() const noexcept
		{
			return Constants().template GetData<I>();
		}

		FORCE_INLINE void FillNop()
		{
			Base::FillNop();
		}

		FORCE_INLINE auto& Instructions()
		{
			return Base::Instructions();
		}
	};

	template<VMConfig Cfg, typename... Ts>
	class StackImpl : private Stack<sizeof...(Ts)>
	{
		DISALLOW_COPY_MOVE_AND_ASSIGN(StackImpl);
		using Base = Stack<sizeof...(Ts)>;
		using SegmentT = DataSegmentImpl<TypePack<Ts...>>;
	public:
		FORCE_INLINE operator const Base&() const noexcept { return *reinterpret_cast<const Base*>(this); }
		FORCE_INLINE operator Base&() noexcept { return *reinterpret_cast<Base*>(this); }
		void Init(size_t stackSize, size_t vectorCapacity = 1)
		{
			((SegmentT&)Base::GetSegment()).Init(stackSize, vectorCapacity);
		}

		void Destroy()
		{
			((SegmentT&)Base::GetSegment()).Destroy();
		}
	};

	// Ensures TypeRegisterT and InstructionSet are constructed before Machine.
	template<typename... Ts>
	struct MachineImplBase
	{
		TypeRegisterT<Ts...> typeReg{};
		InstructionSet       iset;

		explicit MachineImplBase(const char* name)
			: iset(name)
		{
		}
	};

	template<VMConfig Cfg, typename T, typename C = TypePack<>>
	class MachineImpl;

	template<VMConfig Cfg, typename... Ts, typename...Cs>
	class MachineImpl<Cfg, TypePack<Ts...>, TypePack<Cs...>> : private MachineImplBase<Ts...>, private Machine<VMConfigImpl<Cfg, TypePack<Ts...>, TypePack<Cs...>>>
	{
		DISALLOW_COPY_MOVE_AND_ASSIGN(MachineImpl);

	public:
		using Config = VMConfigImpl<Cfg, TypePack<Ts...>, TypePack<Cs...>>;
		using Types = TypePack<Ts...>;
		using ConstTypes = TypePack<Cs...>;
		using Base = MachineImplBase<Ts...>;
		using MachineBase = Machine<Config>;
		using MachineBase::GetStack;

	public:
		using BaseProgramT = Program<Config>;
		using ProgramT = ProgramImpl<Cfg, Types, ConstTypes>;
		using InputT = DataSegmentImpl<Types>;
		using StackT = StackImpl<Cfg, Ts...>;
		using BaseSegmentT = Segment<Config::TypesCount>;
		using ResultT = std::tuple<const Ts* RESTRICT...>;

		explicit MachineImpl(const char* name, size_t vectorCapacity = 1)
			: Base(name)
			, MachineBase(Base::iset)
		{
			InitStack(vectorCapacity);
		}

		~MachineImpl()
		{
			DestroyStack();
		}

		// Add ops from a descriptor table — registry is supplied automatically.
		[[nodiscard]] bool AddInstructions(std::span<const OpDescriptor> table, const std::set<const char*>* selection = nullptr, const char* prefix = nullptr)
		{
			return Base::iset.Add(table, Base::typeReg, selection, prefix);
		}

		// Returns the owned InstructionSet, e.g. for DslCompiler construction.
		FORCE_INLINE const auto& GetInstructionSet() const noexcept
		{
			return Base::iset;
		}

		FORCE_INLINE auto& GetInstructionSet() noexcept
		{
			return Base::iset;
		}

		// Returns the owned type register, e.g. for CreateOp call sites.
		FORCE_INLINE const auto& GetTypeReg() const noexcept
		{
			return Base::typeReg;
		}

		void OptimizeProgram(const ProgramT& program)
		{
			MachineBase::OptimizeProgram((const BaseProgramT &)program);
		}

		void Run(const ProgramT& program, const InputT& input, bool optimized = false)
		{
			MachineBase::Run((const BaseProgramT&)program, (const BaseSegmentT&)input, optimized);
		}

		bool Validate(const ProgramT& program, const InputT& input, bool optimized = false)
		{
			return MachineBase::Validate((const BaseProgramT&)program, (const BaseSegmentT&)input, optimized);
		}

		// Returns typed pointers to the top of each stack segment.
		FORCE_INLINE auto GetResult()
		{
			ResultT result;
			GetResult(result);
			return result;
		}

		FORCE_INLINE void GetResult(ResultT& result)
		{
			std::array<uint8_t*, sizeof...(Ts)> ptrs{};
			MachineBase::GetResult(ptrs);

			[&] <size_t... Is>(std::index_sequence<Is...>)
			{
				((std::get<Is>(result) = reinterpret_cast<const Ts * RESTRICT>(ptrs[Is])), ...);
			}(std::index_sequence_for<Ts...>{});
		}

	private:
		void InitStack(size_t vectorCapacity = 1)
		{
			const auto maxObjs = GetStack().Capacity();
			((StackT&)GetStack()).Init(maxObjs, vectorCapacity);
		}

		void DestroyStack()
		{
			((StackT&)GetStack()).Destroy();
		}
	};
}
