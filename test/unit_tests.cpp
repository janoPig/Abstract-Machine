#include "../gtest/GTestLite.h"
#include "../include/OpImpl.h"
#include "../include/AbstractVM.h"
#include <vector>

using namespace AbstractVM;

TEST(MultiSegmentTest, DefaultInit)
{
	DataSegmentImpl<TypePack<int, float>> ms(5);
	EXPECT_TRUE(ms.GetData<0>() != nullptr);
	EXPECT_TRUE(ms.GetData<1>() != nullptr);
	EXPECT_EQ(ms.GetData<0>()[0], int{});
	EXPECT_EQ(ms.GetData<1>()[0], float{});
}

struct DestroyCounter
{
	static int alive;
	DestroyCounter() { alive++; }
	DestroyCounter(const DestroyCounter&) { alive++; }
	~DestroyCounter() { alive--; }
};
int DestroyCounter::alive = 0;

TEST(MultiSegmentTest, DestructorCalled)
{
	DestroyCounter::alive = 0;
	{
		DataSegmentImpl<TypePack<DestroyCounter, int>> ms(3);
		EXPECT_EQ(DestroyCounter::alive, 3);
	}
	EXPECT_EQ(DestroyCounter::alive, 0);
}

struct TestConfig
{
	static constexpr size_t DstMaxArity = 1;
	static constexpr size_t SrcMaxArity = 2;
	static constexpr size_t MaxProgramSize = 32;
	static constexpr size_t TypesCount = 2;
	static constexpr size_t ConstTypesCount = 2;
};

TEST(ProgramTest, InstructionsWritable)
{
	Program<TestConfig> prog{};
	auto& instrs = prog.Instructions();
	instrs[0] = { 1, { {SegmentType::SEG_CONST, 0} } };

	EXPECT_EQ(instrs.size(), 32);
	EXPECT_EQ(instrs[0].opcode, 1);
}

static void OpAdd(float& d, const float& a, const float& b) noexcept { d = a + b; }
static void OpMul(float& d, const float& a, const float& b) noexcept { d = a * b; }
static void OpF2I(int& d, const float& a)                   noexcept { d = static_cast<int>(a); }
static void OpIAdd(int& d, const int& a, const int& b)      noexcept { d = a + b; }

static constexpr OpDescriptor g_opLibrary[] =
{
	{ "Add",  &CreateOp<OpAdd> },
	{ "Mul",  &CreateOp<OpMul> },
	{ "F2I",  &CreateOp<OpF2I> },
	{ "IAdd", &CreateOp<OpIAdd> },
};

TEST(OpImplTest, ExecuteSingleDst)
{
	std::set<const char*> sel = { "Add" };
	MachineImpl<TestConfig, TypePack<float, int>, TypePack<float, int>> vm("Test", 1);
	vm.AddInstructions(g_opLibrary, &sel);

	uint32_t addOpId{};
	ASSERT_TRUE(vm.GetInstructionSet().Lookup("Add", addOpId));
	const Op* op = vm.GetInstructionSet().GetOp(addOpId);

	float dst = 0, a = 3.0f, b = 4.0f;
	DataType d[1] = { (uint8_t*)&dst };
	DataType s[2] = { (uint8_t*)&a, (uint8_t*)&b };
	op->Execute(d, s);
	EXPECT_NEAR(dst, 7.0f, 1e-6f);
}

struct E2EFixture
{
	static constexpr uint32_t OP_NOP  = 0;
	static constexpr uint32_t OP_FADD = 1;
	static constexpr uint32_t OP_F2I  = 3;
	static constexpr uint32_t OP_IADD = 4;

	using Types = TypePack<float, int>;
	using MachineT = MachineImpl<TestConfig, Types, Types>;
	using ProgramT = MachineT::ProgramT;

	MachineT vm{ "E2E" };
	ProgramT prog{ 4 };
	MachineT::InputT input{ 1 };

	E2EFixture()
	{
		vm.AddInstructions(g_opLibrary);

		prog.GetConst<0>()[0] = 3.0f;
		prog.GetConst<0>()[1] = 7.0f;
		prog.GetConst<1>()[0] = 10;

		input.GetData<0>()[0] = 2.0f;

		prog.FillNop();
		auto& I = prog.Instructions();
		// I[0]: float[0] = float_const[0] + float_input[0]  =>  3+2 = 5
		I[0] = { OP_FADD, {{ SegmentType::SEG_CONST, 0 }, { SegmentType::SEG_DATA,  0 }} };
		// I[1]: float[1] = float_stack[0] + float_const[1]  =>  5+7 = 12 -- unused (dead)
		I[1] = { OP_FADD, {{ SegmentType::SEG_STACK, 0 }, { SegmentType::SEG_CONST, 1 }} };
		// I[2]: int[0]   = F2I(float_stack[1])              =>  int(12) = 12 -- unused (dead)
		I[2] = { OP_F2I,  {{ SegmentType::SEG_STACK, 1 }} };
		// I[3]: int[1]   = int_stack[0] + int_const[0]      =>  12+10 = 22
		I[3] = { OP_IADD, {{ SegmentType::SEG_STACK, 0 }, { SegmentType::SEG_CONST, 0 }} };
		// I[4]: float[2] = float_const[0] + float_const[0]  =>  3+3 = 6
		I[4] = { OP_FADD, {{ SegmentType::SEG_CONST, 0 }, { SegmentType::SEG_CONST, 0 }} };
	}

	~E2EFixture() = default;
};

TEST(MachineTest, NormalRun)
{
	E2EFixture f;
	f.vm.Run(f.prog, f.input, false);

	const auto [pFloat, pInt] = f.vm.GetResult();
	ASSERT_TRUE(pFloat != nullptr);
	ASSERT_TRUE(pInt   != nullptr);
	EXPECT_NEAR(*pFloat, 6.0f, 1e-6f);
	EXPECT_EQ(*pInt, 22);
}

TEST(MachineTest, OptimizedRun_DeadCodeElimination)
{
	E2EFixture f;
	f.vm.OptimizeProgram(f.prog);
	f.vm.Run(f.prog, f.input, true);

	const auto [pFloat, pInt] = f.vm.GetResult();
	ASSERT_TRUE(pFloat != nullptr);
	ASSERT_TRUE(pInt   != nullptr);
	EXPECT_NEAR(*pFloat, 6.0f, 1e-6f);
	EXPECT_EQ(*pInt, 22);
}

TEST(MachineTest, MultipleRuns_ProduceConsistentResults)
{
	E2EFixture f;
	for (int run = 0; run < 3; run++)
	{
		f.vm.Run(f.prog, f.input, false);
		const auto [pFloat, pInt] = f.vm.GetResult();
		EXPECT_EQ(*pInt, 22);
	}
}
