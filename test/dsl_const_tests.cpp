#include "../gtest/GTestLite.h"
#include "../include/DSLCompiler.h"
#include "ScalarTestOps.h"
#include "TestDslUtils.h"

using namespace AbstractVM;

namespace
{
	struct TestConfig
	{
		static constexpr size_t DstMaxArity = 1;
		static constexpr size_t SrcMaxArity = 2;
		static constexpr size_t MaxProgramSize = 32;
		static constexpr size_t MaxConstantsCount = 8;
	};
	using MachineT = MachineImpl<TestConfig, TypePack<float, int>, TypePack<float, int>>;
	using DslCompilerT = DslCompiler<MachineT::Config>;
	using ProgramT = MachineT::ProgramT;
	using DSLProgramT = DslCompilerT::ProgramT;

	TEST(DslConstTests, CompileWithConstants)
	{

		MachineT machine("TestMachine");
		ASSERT_TRUE(machine.AddInstructions(ScalarTestOps::kBasicOps));

		DslCompilerT compiler(machine.GetInstructionSet(), machine.GetTypeReg());

		const char* dsl = R"(
CONST <float> [1.5, 2.5]
CONST <int> [10, 20]
S<float>[0] = Add C<float>[0] C<float>[1]
S<int>[0] = IAdd C<int>[0] C<int>[1]
)";

		ProgramT program{ 4 };
		auto result = compiler.Compile(dsl, (DSLProgramT&)program);
		ASSERT_TRUE(result) << result.error().message;

		// Verify constants in program
		EXPECT_NEAR(*program.template GetConst<0>(), 1.5f, 1e-6f);
		EXPECT_NEAR(program.template GetConst<0>()[1], 2.5f, 1e-6f);
		EXPECT_EQ(*program.template GetConst<1>(), 10);
		EXPECT_EQ(program.template GetConst<1>()[1], 20);
	}

	TEST(DslConstTests, RoundtripConsistency)
	{
		MachineT machine("TestMachine");
		ASSERT_TRUE(machine.AddInstructions(ScalarTestOps::kBasicOps));
		DslCompilerT compiler(machine.GetInstructionSet(), machine.GetTypeReg());

		const char* dsl = R"(CONST <float> [1.5, 2.5]
CONST <int> [10, 20]

S<float>[0] = Add C<float>[0] C<float>[1]
S<int>[0] = IAdd C<int>[0] C<int>[1]
)";

		TestDslUtils::AssertRoundtrip<MachineT>(compiler, dsl);
	}

	TEST(DslConstTests, ErrorHandlingOutOfBounds)
	{
		MachineT machine("TestMachine");
		ASSERT_TRUE(machine.AddInstructions(ScalarTestOps::kBasicOps));
		DslCompilerT compiler(machine.GetInstructionSet(), machine.GetTypeReg());

		const char* dsl = R"(
CONST <float> [1.5]
S<float>[0] = Add C<float>[0] C<float>[1]
)";

		ProgramT program(10);
		auto result = compiler.Compile(dsl, (DSLProgramT&)program);
		ASSERT_FALSE(result);
		EXPECT_TRUE(result.error().message.find("out of bounds") != std::string::npos);
	}

	TEST(DslConstTests, ErrorHandlingInvalidType)
	{
		MachineT machine("TestMachine");
		ASSERT_TRUE(machine.AddInstructions(ScalarTestOps::kBasicOps));
		DslCompilerT compiler(machine.GetInstructionSet(), machine.GetTypeReg());

		const char* dsl = R"(
CONST <double> [1.5]
S<float>[0] = Add C<float>[0] C<float>[0]
)";

		ProgramT program(10);
		auto result = compiler.Compile(dsl, (DSLProgramT&)program);
		ASSERT_FALSE(result);
		EXPECT_TRUE(result.error().message.find("Unknown type 'double'") != std::string::npos);
	}
}
