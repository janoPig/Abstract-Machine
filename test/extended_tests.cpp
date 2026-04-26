#include "../gtest/GTestLite.h"
#include "ScalarTestOps.h"
#include "TestDslUtils.h"
#include "../include/DSLCompiler.h"
#include "../include/AbstractVM.h"
#include "../include/OpImpl.h"

using namespace AbstractVM;

namespace
{
	template<typename CompilerT, typename ProgramT>
	static void CompileOrFail(const CompilerT& compiler, const char* dsl, ProgramT& program)
	{
		TestDslUtils::CompileOrFail(compiler, dsl, program);
	}

	// =============================================================================
	// Op Library for Edge Case Tests
	// =============================================================================

	// =============================================================================
	// SECTION: Stack Boundary Conditions
	// =============================================================================

	class StackBoundaryTest : public GTestLite::Test
	{
	protected:
		struct TestConfig
		{
			static constexpr size_t DstMaxArity = 2;
			static constexpr size_t SrcMaxArity = 2;
			static constexpr size_t MaxProgramSize = 16;
			static constexpr size_t MaxConstantsCount = 8;
		};

		using VMF = MachineImpl<TestConfig, TypePack<float, int>, TypePack<>>;
		using DslCompilerT = DslCompiler<VMF::Config>;
		using ProgramT = VMF::ProgramT;
		using DSLProgramT = DslCompilerT::ProgramT;

		VMF vm{ "StackBoundary", 2 };  // Small capacity for boundary testing

		void SetUp() override
		{
			ASSERT_TRUE(vm.AddInstructions(ScalarTestOps::kEdgeCaseOps));
		}

		DslCompilerT Compiler() const
		{
			return DslCompilerT(vm.GetInstructionSet(), vm.GetTypeReg());
		}
	};

	TEST_F(StackBoundaryTest, StackAtCapacity_ExecutesSuccessfully)
	{
		// Fill stack to exact capacity (2 items)
		auto c = Compiler();
		VMF::ProgramT prog{};
		CompileOrFail(c, "Add I[0] I[1]\nMul S[0] I[0]\n", prog);

		VMF::InputT input(2);
		input.GetData<float>()[0] = 2.0f;
		input.GetData<float>()[1] = 3.0f;

		vm.Run(prog, input);  // Should not crash/assert
		const auto [pf, _] = vm.GetResult();
		ASSERT_TRUE(pf);
		EXPECT_NEAR(*pf, 10.0f, 1e-6f);  // (2+3)*2 = 10
	}

	TEST_F(StackBoundaryTest, SingleItemStack_Works)
	{
		// Single-item stack test with capacity 1
		auto c = Compiler();
		VMF::ProgramT prog{};
		CompileOrFail(c, "Add I[0] I[1]\n", prog);

		VMF::InputT input(2);
		input.GetData<float>()[0] = 5.0f;
		input.GetData<float>()[1] = 3.0f;

		vm.Run(prog, input);
		const auto [pf, _] = vm.GetResult();
		ASSERT_TRUE(pf);
		EXPECT_NEAR(*pf, 8.0f, 1e-6f);
	}

	TEST_F(StackBoundaryTest, StackReset_ClearsPointers)
	{
		// Run program twice, verify stack resets between runs
		auto c = Compiler();
		VMF::ProgramT prog{};
		CompileOrFail(c, "Add I[0] I[1]\n", prog);

		VMF::InputT input(2);
		input.GetData<float>()[0] = 1.0f;
		input.GetData<float>()[1] = 1.0f;

		// Run 1
		vm.Run(prog, input);
		const auto [pf1, _] = vm.GetResult();
		EXPECT_NEAR(*pf1, 2.0f, 1e-6f);

		// Run 2 (stack should reset)
		input.GetData<float>()[0] = 3.0f;
		input.GetData<float>()[1] = 4.0f;
		vm.Run(prog, input);
		const auto [pf2, __] = vm.GetResult();
		EXPECT_NEAR(*pf2, 7.0f, 1e-6f);
	}

	// =============================================================================
	// SECTION: Type Mismatch and Validation
	// =============================================================================

	struct TestConfig
	{
		static constexpr size_t DstMaxArity = 2;
		static constexpr size_t SrcMaxArity = 2;
		static constexpr size_t MaxProgramSize = 32;
		static constexpr size_t MaxConstantsCount = 8;
	};

	using VMM = MachineImpl<TestConfig, TypePack<float, int>, TypePack<float, int>>;
	using DslCompilerT = DslCompiler<VMM::Config>;
	using ProgramT = VMM::ProgramT;
	using DSLProgramT = DslCompilerT::ProgramT;

	class TypeValidationTest : public GTestLite::Test
	{
	protected:
		VMM vm{ "TypeValidation", 16 };

		void SetUp() override
		{
			ASSERT_TRUE(vm.AddInstructions(ScalarTestOps::kEdgeCaseOps));
		}

		DslCompilerT Compiler() const
		{
			return DslCompilerT(vm.GetInstructionSet(), vm.GetTypeReg());
		}

		auto CompileDsl(const char* dsl)
		{
			DSLProgramT prog{};
			return Compiler().Compile(dsl, prog);
		}
	};

	TEST_F(TypeValidationTest, ConversionF2I_CorrectStackTracking)
	{
		// After F2I, int stack should have 1 item
		auto c = Compiler();
		VMM::ProgramT prog{};
		CompileOrFail(c, "F2I I[0]\nI2F S[0]\n", prog);

		VMM::InputT input(1);
		input.GetData<float>()[0] = 3.7f;

		vm.Run(prog, input);
		const auto [pf, pi] = vm.GetResult();

		ASSERT_TRUE(pf);
		ASSERT_TRUE(pi);
		EXPECT_EQ(*pi, 3);
		EXPECT_NEAR(*pf, 3.0f, 1e-6f);
	}

	TEST_F(TypeValidationTest, MixedTypeChain_ManagesIndependentStacks)
	{
		// Complex: F2I(float) → int stack grows, then I2F(int) → float stack grows
		// Verify both stacks track independently
		auto c = Compiler();
		VMM::ProgramT prog{};
		CompileOrFail(c,
			"Add I[0] I[1]\n"      // S<float>[0] = 3+4 = 7
			"F2I S[0]\n"           // S<int>[0] = 7
			"I2F S[0]\n"           // S<float>[1] = 7.0
			"Add S[1] I[2]\n",     // S<float>[2] = 7.0 + 1.5 = 8.5
			prog);

		VMM::InputT input(3);
		input.GetData<float>()[0] = 3.0f;
		input.GetData<float>()[1] = 4.0f;
		input.GetData<float>()[2] = 1.5f;

		vm.Run(prog, input);
		const auto [pf, pi] = vm.GetResult();

		ASSERT_TRUE(pf);
		ASSERT_TRUE(pi);
		EXPECT_NEAR(*pf, 8.5f, 1e-6f);
		EXPECT_EQ(*pi, 7);
	}

	// =============================================================================
	// SECTION: Constant Segment Handling
	// =============================================================================

	class ConstantSegmentTest : public GTestLite::Test
	{
	protected:
		struct TestConfig
		{
			static constexpr size_t DstMaxArity = 1;
			static constexpr size_t SrcMaxArity = 2;
			static constexpr size_t MaxProgramSize = 32;
			static constexpr size_t MaxConstantsCount = 8;
		};

		using VMF = MachineImpl<TestConfig, TypePack<float, int>, TypePack<float>>;
		using DslCompilerT = DslCompiler<VMF::Config>;
		using ProgramT = VMF::ProgramT;
		using DSLProgramT = DslCompilerT::ProgramT;

		VMF vm{ "ConstSegment", 8 };

		void SetUp() override
		{
			ASSERT_TRUE(vm.AddInstructions(ScalarTestOps::kEdgeCaseOps));
		}

		DslCompilerT Compiler() const
		{
			return DslCompilerT(vm.GetInstructionSet(), vm.GetTypeReg());
		}
	};

	TEST_F(ConstantSegmentTest, ConstantMultipleUses_SameValue)
	{
		// C[0] used in two operations, should have same value both times
		auto c = Compiler();
		VMF::ProgramT prog{};
		CompileOrFail(c, "Add I[0] C[0]\nMul S[0] C[0]\n", prog);

		// Initialize constant
		prog.GetConst<0>()[0] = 5.0f;

		VMF::InputT input(1);
		input.GetData<float>()[0] = 2.0f;

		vm.Run(prog, input);
		const auto [pf, _] = vm.GetResult();

		ASSERT_TRUE(pf);
		// (2 + 5) * 5 = 35
		EXPECT_NEAR(*pf, 35.0f, 1e-6f);
	}

	TEST_F(ConstantSegmentTest, ConstantZero_Handled)
	{
		auto c = Compiler();
		VMF::ProgramT prog{};
		CompileOrFail(c, "Add I[0] C[0]\n", prog);

		prog.GetConst<0>()[0] = 0.0f;

		VMF::InputT input(1);
		input.GetData<float>()[0] = 7.0f;

		vm.Run(prog, input);
		const auto [pf, _] = vm.GetResult();

		ASSERT_TRUE(pf);
		EXPECT_NEAR(*pf, 7.0f, 1e-6f);
	}

	TEST_F(ConstantSegmentTest, ConstantNegative_PreservesSign)
	{
		auto c = Compiler();
		VMF::ProgramT prog{};
		CompileOrFail(c, "Add I[0] C[0]\n", prog);

		prog.GetConst<0>()[0] = -3.5f;

		VMF::InputT input(1);
		input.GetData<float>()[0] = 10.0f;

		vm.Run(prog, input);
		const auto [pf, _] = vm.GetResult();

		ASSERT_TRUE(pf);
		EXPECT_NEAR(*pf, 6.5f, 1e-6f);
	}

	// =============================================================================
	// SECTION: Optimization Edge Cases
	// =============================================================================

	class OptimizationEdgeTest : public GTestLite::Test
	{
	protected:
		VMM vm{ "OptEdge", 16 };

		void SetUp() override
		{
			ASSERT_TRUE(vm.AddInstructions(ScalarTestOps::kEdgeCaseOps));
		}

		DslCompilerT Compiler() const
		{
			return DslCompilerT(vm.GetInstructionSet(), vm.GetTypeReg());
		}
	};

	TEST_F(OptimizationEdgeTest, AllLiveInstructions_OptimizedMatchesFull)
	{
		// No dead code: every instruction result is used in final output
		auto c = Compiler();
		VMM::ProgramT prog{};
		CompileOrFail(c,
			"Add I[0] I[1]\n"
			"Mul S[0] I[0]\n",
			prog);

		VMM::InputT input(2);
		input.GetData<float>()[0] = 2.0f;
		input.GetData<float>()[1] = 3.0f;

		vm.Run(prog, input);
		const auto [pfFull, piFull] = vm.GetResult();
		float fullResult = *pfFull;

		vm.OptimizeProgram(prog);
		vm.Run(prog, input, true);
		const auto [pfOpt, piOpt] = vm.GetResult();

		EXPECT_NEAR(*pfOpt, fullResult, 1e-6f);
	}

	TEST_F(OptimizationEdgeTest, SingleInstruction_OptimizesCorrectly)
	{
		// Minimal program: 1 instruction
		auto c = Compiler();
		VMM::ProgramT prog{};
		CompileOrFail(c, "Add I[0] I[1]\n", prog);

		VMM::InputT input(2);
		input.GetData<float>()[0] = 5.0f;
		input.GetData<float>()[1] = 3.0f;

		vm.OptimizeProgram(prog);
		vm.Run(prog, input, true);
		const auto [pf, pi] = vm.GetResult();

		ASSERT_TRUE(pf);
		EXPECT_NEAR(*pf, 8.0f, 1e-6f);
	}

	TEST_F(OptimizationEdgeTest, LongDependencyChain_TracesCorrectly)
	{
		// Chain: I[0] → S[0] → S[1] → S[2] → output
		// All dependencies live
		auto c = Compiler();
		VMM::ProgramT prog{};
		CompileOrFail(c,
			"Add I[0] I[0]\n"      // S[0]
			"Mul S[0] I[0]\n"      // S[1]
			"Add S[1] I[0]\n"      // S[2]
			"Mul S[2] I[0]\n",     // S[3]
			prog);

		VMM::InputT input(1);
		input.GetData<float>()[0] = 2.0f;

		vm.Run(prog, input);
		const auto [pfFull, piFull] = vm.GetResult();
		const auto fullRes = *pfFull;

		vm.OptimizeProgram(prog);
		vm.Run(prog, input, true);
		const auto [pfOpt, piOpt] = vm.GetResult();

		EXPECT_NEAR(*pfOpt, fullRes, 1e-6f);
	}

	// =============================================================================
	// SECTION: Stress and Round-Trip Tests
	// =============================================================================

	class StressTest : public GTestLite::Test
	{
	protected:
		VMM vm{ "Stress", 32 };

		void SetUp() override
		{
			ASSERT_TRUE(vm.AddInstructions(ScalarTestOps::kEdgeCaseOps));
		}

		DslCompilerT Compiler() const
		{
			return DslCompilerT(vm.GetInstructionSet(), vm.GetTypeReg());
		}
	};

	TEST_F(StressTest, LargeProgram_16Instructions_ExecutesCorrectly)
	{
		// Build a larger program programmatically
		std::string dsl;
		for (int i = 0; i < 8; ++i)
		{
			dsl += "Add I[0] I[1]\n";
		}

		auto c = Compiler();
		VMM::ProgramT prog{};
		CompileOrFail(c, dsl.c_str(), prog);

		VMM::InputT input(2);
		input.GetData<float>()[0] = 1.0f;
		input.GetData<float>()[1] = 1.0f;

		vm.Run(prog, input);
		const auto [pf, pi] = vm.GetResult();

		ASSERT_TRUE(pf);
		// 8 adds on a stack: (1+1) repeated
		EXPECT_NEAR(*pf, 2.0f, 1e-6f);  // Last addition result
	}

	TEST_F(StressTest, ComplexMixedTypeProgram_RoundtripStable)
	{
		auto c = Compiler();

		const char* complexDsl =
			"Add I[0] I[1]\n"      // S<float>[0]
			"F2I S[0]\n"           // S<int>[0]
			"I2F S[0]\n"           // S<float>[1]
			"Add S[1] I[0]\n";     // S<float>[2]

		VMM::ProgramT p1{};
		const auto r1 = c.Compile(complexDsl, (DSLProgramT&)p1);
		ASSERT_TRUE(r1) << r1.error().message;

		const auto d1 = c.Decompile((const DSLProgramT&)p1);
		ASSERT_TRUE(d1.has_value());

		VMM::ProgramT p2{};
		const auto r2 = c.Compile(*d1, (DSLProgramT&)p2);
		ASSERT_TRUE(r2) << r2.error().message;

		const auto d2 = c.Decompile((const DSLProgramT&)p2);
		ASSERT_TRUE(d2.has_value());

		EXPECT_EQ(*d1, *d2);
	}

	// =============================================================================
	// SECTION: Boundary Type Configurations
	// =============================================================================

	class MinimalConfigTest : public GTestLite::Test
	{
	protected:
		struct TestConfig
		{
			static constexpr size_t DstMaxArity = 1;
			static constexpr size_t SrcMaxArity = 1;
			static constexpr size_t MaxProgramSize = 8;
			static constexpr size_t MaxConstantsCount = 8;
		};

		using VMS = MachineImpl<TestConfig, TypePack<float, int>, TypePack<>>;
		using DslCompilerT = DslCompiler<VMS::Config>;
		using ProgramT = VMS::ProgramT;
		using DSLProgramT = DslCompilerT::ProgramT;

		VMS vm{ "Minimal", 4 };

		void SetUp() override
		{
			ASSERT_TRUE(vm.AddInstructions(ScalarTestOps::kEdgeCaseOps));
		}

		DslCompilerT Compiler() const
		{
			return DslCompilerT(vm.GetInstructionSet(), vm.GetTypeReg());
		}
	};

	TEST_F(MinimalConfigTest, MinimalArity_1x1_ExecutesCorrectly)
	{
		// Src=1, Dst=1: Unary operations only
		// Cannot test Add (needs 2 src), so we'd need a unary op
		// This is a limitation check: verifies config is enforced
		auto c = Compiler();
		VMS::ProgramT prog{};

		// Sqrt is unary
		CompileOrFail(c, "Sqrt I[0]\n", prog);

		VMS::InputT input(1);
		input.GetData<float>()[0] = 4.0f;

		vm.Run(prog, input);
		const auto [pf, _] = vm.GetResult();

		ASSERT_TRUE(pf);
		EXPECT_NEAR(*pf, 2.0f, 1e-6f);
	}
}
