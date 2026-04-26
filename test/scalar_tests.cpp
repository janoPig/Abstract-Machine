#include "../gtest/GTestLite.h"
#include "../examples/ScalarOps.h"
#include "TestDslUtils.h"
#include "../include/DSLCompiler.h"

using namespace ScalarOps;
using namespace AbstractVM;

static TypeRegisterT<float, int> g_reg;

class ScalarDirectOpTest : public ::GTestLite::Test
{
protected:
	void RunExecute(const Op* op, std::vector<void*> dst_ptrs, std::vector<void*> src_ptrs)
	{
		op->Execute((DataType*)dst_ptrs.data(), (const DataType*)src_ptrs.data());
	}

	bool RunValidate(const Op* op, std::vector<void*> dst_ptrs, std::vector<void*> src_ptrs)
	{
		return op->Validate((DataType*)dst_ptrs.data(), (const DataType*)src_ptrs.data());
	}
};

TEST_F(ScalarDirectOpTest, AddFloat_ComputesExpectedValue)
{
	const Op* op = CreateOp<OpAdd<float>>(g_reg);
	float dst{};
	float a = 1.5f;
	float b = 2.25f;

	RunExecute(op, { &dst }, { &a, &b });
	EXPECT_NEAR(dst, 3.75f, 1e-6f);

	delete op;
}

TEST_F(ScalarDirectOpTest, DivFloat_ZeroRejectedByValidator)
{
	const Op* op = CreateOp<OpDiv<float>, ValNonZero<float>>(g_reg);
	float dst{};
	float a = 6.0f;
	float b = 0.0f;

	EXPECT_FALSE(RunValidate(op, { &dst }, { &a, &b }));
	delete op;
}

TEST_F(ScalarDirectOpTest, AbsInt_UsesSignedInstruction)
{
	const Op* op = CreateOp<OpAbs<int>>(g_reg);
	int dst{};
	int a = -17;

	RunExecute(op, { &dst }, { &a });
	EXPECT_EQ(dst, 17);

	delete op;
}

TEST_F(ScalarDirectOpTest, BitXorInt_ComputesExpectedValue)
{
	const Op* op = CreateOp<OpBitXor<int>>(g_reg);
	int dst{};
	int a = 0b1100;
	int b = 0b1010;

	RunExecute(op, { &dst }, { &a, &b });
	EXPECT_EQ(dst, 0b0110);

	delete op;
}

class ScalarMachineIntegrationTest : public ::GTestLite::Test
{
protected:
	struct TestConfig
	{
		static constexpr size_t DstMaxArity = 1;
		static constexpr size_t SrcMaxArity = 2;
		static constexpr size_t MaxProgramSize = 32;
		static constexpr size_t MaxConstantsCount = 8;
	};

	using MachineT = MachineImpl<TestConfig, TypePack<float, int>, TypePack<float, int>>;
	using ProgramT = MachineT::ProgramT;

	void SetUp() override
	{
		const bool registered = RegisterScalarOps<float, int>(vm.GetInstructionSet(), vm.GetTypeReg());
		ASSERT_TRUE(registered);
		compiler = new DslCompiler<MachineT::Config>(vm.GetInstructionSet(), vm.GetTypeReg());
	}

	void TearDown() override
	{
		if (program)
		{
			delete program;
		}
		delete compiler;
	}

	ProgramT* MakeProgram(const char* dsl)
	{
		auto* prog = new ProgramT();
		TestDslUtils::CompileOrFail(*compiler, dsl, *prog);
		return prog;
	}

	MachineT vm{ "Scalar", 1 };
	DslCompiler<MachineT::Config>* compiler = nullptr;
	ProgramT* program = nullptr;
};

TEST_F(ScalarMachineIntegrationTest, FloatChain_ExecutesThroughDsl)
{
	program = MakeProgram(
		"S[0] = AddF I[0] I[1]\n"
		"S[1] = MulF S[0] C[0]\n"
	);

	program->GetConst<0>()[0] = 4.0f;

	MachineT::InputT input(2);
	input.GetData<float>()[0] = 1.5f;
	input.GetData<float>()[1] = 2.0f;

	vm.Run(*program, input);

	const auto [pf, pi] = vm.GetResult();
	ASSERT_TRUE(pf != nullptr);
	EXPECT_NEAR(*pf, 14.0f, 1e-6f);
	EXPECT_TRUE(pi == nullptr);
}

TEST_F(ScalarMachineIntegrationTest, IntBitwiseChain_ExecutesThroughDsl)
{
	program = MakeProgram(
		"S[0] = BitAndI I[0] C[0]\n"
		"S[1] = BitXorI S[0] I[1]\n"
	);

	program->GetConst<1>()[0] = 0b0110;

	MachineT::InputT input(2);
	input.GetData<int>()[0] = 0b1110;
	input.GetData<int>()[1] = 0b0011;

	vm.Run(*program, input);

	const auto [pf, pi] = vm.GetResult();
	EXPECT_TRUE(pf == nullptr);
	ASSERT_TRUE(pi != nullptr);
	EXPECT_EQ(*pi, 0b0101);
}

TEST_F(ScalarMachineIntegrationTest, FloatDivision_ValidationFailsOnZeroConstant)
{
	program = MakeProgram(
		"S[0] = DivF I[0] C[0]\n"
	);

	program->GetConst<0>()[0] = 0.0f;

	MachineT::InputT input(1);
	input.GetData<float>()[0] = 9.0f;

	EXPECT_FALSE(vm.Validate(*program, input));
}
