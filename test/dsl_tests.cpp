#include "../gtest/GTestLite.h"
#include "ScalarTestOps.h"
#include "TestDslUtils.h"
#include "../include/DSLCompiler.h"
#include "../include/AbstractVM.h"
#include "../include/OpImpl.h"

using namespace AbstractVM;

static constexpr auto& g_floatOps = ScalarTestOps::kFloatOps;
static constexpr auto& g_intOps = ScalarTestOps::kIntOps;
static constexpr auto& g_mixedOps = ScalarTestOps::kMixedOps;

// =============================================================================
// Shared type register — float=0, int=1
// =============================================================================

static TypeRegisterT<float, int> g_reg;

template<typename CompilerT, typename ProgramT>
static void CompileOrFail(const CompilerT& compiler, const char* dsl, ProgramT& program)
{
	TestDslUtils::CompileOrFail(compiler, dsl, program);
}

template<VMConfigSpec Cfg>
static void AssertRoundtrip(const DslCompiler<Cfg>& compiler, const char* dsl)
{
	TestDslUtils::AssertRoundtrip(compiler, dsl);
}



// =============================================================================
// SECTION 1 — Single-type float VM: compile / decompile / execute basics
// =============================================================================

class FloatVmTest : public GTestLite::Test
{
protected:
	struct TestConfig
	{
		static constexpr size_t DstMaxArity = 3;
		static constexpr size_t SrcMaxArity = 3;
		static constexpr size_t MaxProgramSize = 32;
	};

	using VMF = MachineImpl<TestConfig, TypePack<float>, TypePack<float>>;
	using DslCompilerT = DslCompiler<VMF::Config>;
	using ProgramT = VMF::ProgramT;
	using DSLProgramT = DslCompilerT::ProgramT;

	VMF vm{ "Float", 16 };

	void SetUp() override
	{
		ASSERT_TRUE(vm.AddInstructions(g_floatOps));
	}

	DslCompilerT Compiler() const
	{
		return DslCompilerT(vm.GetInstructionSet(), vm.GetTypeReg());
	}
};

TEST_F(FloatVmTest, Compile_EmptySource_AllNops)
{
	auto c = Compiler();
	VMF::ProgramT prog(4, 0);
	CompileOrFail(c, "", (ProgramT&)prog);
	for (const auto& instr : prog.Instructions())
		EXPECT_EQ(instr.opcode, 0u);
}

TEST_F(FloatVmTest, Compile_SingleOp_WithoutLhs)
{
	auto c = Compiler();
	VMF::ProgramT prog(4, 0);
	CompileOrFail(c, "Add I[0] I[1]\n", (ProgramT&)prog);
	// instruction placed at tail
	const auto& instr = prog.Instructions();
	EXPECT_NE(instr[instr.size()-1].opcode, 0u);
}

TEST_F(FloatVmTest, Compile_SingleOp_WithLhs)
{
	auto c = Compiler();
	VMF::ProgramT prog(4, 0);
	CompileOrFail(c, "S[0] = Add I[0] I[1]\n", (ProgramT&)prog);
	const auto& instr = prog.Instructions();
	EXPECT_NE(instr[instr.size() - 1].opcode, 0u);
}

TEST_F(FloatVmTest, Decompile_SingleOp_ExactFormat)
{
	auto c = Compiler();
	VMF::ProgramT prog(4, 0);
	CompileOrFail(c, "Add I[0] I[1]\n", prog);

	auto out = c.Decompile((const DSLProgramT&)prog);
	ASSERT_TRUE(out.has_value());
	EXPECT_EQ(*out, "S<float>[0] = Add I<float>[0] I<float>[1]\n");
}

TEST_F(FloatVmTest, Decompile_ConstSegment_ExactFormat)
{
	auto c = Compiler();
	VMF::ProgramT prog(4, 0);
	CompileOrFail(c, "Add C[0] I[1]\n", prog);

	auto out = c.Decompile((const DSLProgramT&)prog);
	ASSERT_TRUE(out.has_value());
	EXPECT_EQ(*out, "S<float>[0] = Add C<float>[0] I<float>[1]\n");
}

TEST_F(FloatVmTest, Decompile_StackSrc_ExactFormat)
{
	auto c = Compiler();
	VMF::ProgramT prog(8, 0);
	CompileOrFail(c, "Add I[0] I[1]\nMul S[0] I[2]\n", prog);

	auto out = c.Decompile((const DSLProgramT&)prog);
	ASSERT_TRUE(out.has_value());
	EXPECT_EQ(*out,
		"S<float>[0] = Add I<float>[0] I<float>[1]\n"
		"S<float>[1] = Mul S<float>[0] I<float>[2]\n");
}

TEST_F(FloatVmTest, Roundtrip_ChainOfOps)
{
	auto c = Compiler();
	AssertRoundtrip(c,
		"Add I[0] I[1]\n"
		"Mul S[0] C[0]\n"
		"Sub S[1] I[2]\n"
		);
}

TEST_F(FloatVmTest, Execute_AddThenMul)
{
	// S[0] = I[0] + I[1] = 3+4 = 7
	// S[1] = S[0] * I[0] = 7*3 = 21
	auto c = Compiler();
	VMF::ProgramT prog(8, 0);
	CompileOrFail(c, "Add I[0] I[1]\nMul S[0] I[0]\n", prog);

	VMF::InputT input(2);
	input.GetData<float>()[0] = 3.0f;
	input.GetData<float>()[1] = 4.0f;

	vm.Run(prog, input);
	const auto [pf] = vm.GetResult();
	ASSERT_TRUE(pf);
	EXPECT_NEAR(*pf, 21.0f, 1e-6f);
}

TEST_F(FloatVmTest, Execute_UnaryNeg)
{
	auto c = Compiler();
	VMF::ProgramT prog(4, 0);
	CompileOrFail(c, "Neg I[0]\n", prog);

	VMF::InputT input(1);
	input.GetData<float>()[0] = 5.0f;

	vm.Run(prog, input);
	const auto [pf] = vm.GetResult();
	ASSERT_TRUE(pf);
	EXPECT_NEAR(*pf, -5.0f, 1e-6f);
}

TEST_F(FloatVmTest, Execute_Comment_IsIgnored)
{
	auto c = Compiler();
	VMF::ProgramT prog(4, 0);
	CompileOrFail(c, "# full comment line\nAdd I[0] I[1] # inline comment\n", prog);

	VMF::InputT input(2);
	input.GetData<float>()[0] = 1.0f;
	input.GetData<float>()[1] = 2.0f;

	vm.Run(prog, input);
	const auto [pf] = vm.GetResult();
	ASSERT_TRUE(pf);
	EXPECT_NEAR(*pf, 3.0f, 1e-6f);
}

TEST_F(FloatVmTest, Execute_Validate_DivByZero_Fails)
{
	auto c = Compiler();
	VMF::ProgramT prog(4, 0);
	CompileOrFail(c, "Div I[0] I[1]\n", prog);

	VMF::InputT input(2);
	input.GetData<float>()[0] = 1.0f;
	input.GetData<float>()[1] = 0.0f;   // denominator zero

	EXPECT_FALSE(vm.Validate(prog, input));
}

TEST_F(FloatVmTest, Execute_Validate_DivNonZero_Passes)
{
	auto c = Compiler();
	VMF::ProgramT prog(4, 0);
	CompileOrFail(c, "Div I[0] I[1]\n", prog);

	VMF::InputT input(2);
	input.GetData<float>()[0] = 6.0f;
	input.GetData<float>()[1] = 2.0f;

	EXPECT_TRUE(vm.Validate(prog, input));

	vm.Run(prog, input);
	const auto [pf] = vm.GetResult();
	ASSERT_TRUE(pf);
	EXPECT_NEAR(*pf, 3.0f, 1e-6f);
}

// =============================================================================
// SECTION 2 — Compile errors
// =============================================================================

class CompileErrorTest : public GTestLite::Test
{
protected:
	struct TestConfig
	{
		static constexpr size_t DstMaxArity = 3;
		static constexpr size_t SrcMaxArity = 3;
		static constexpr size_t MaxProgramSize = 2;
	};

	using VMM = MachineImpl<TestConfig, TypePack<float, int>, TypePack<float, int>>;      // multi-type float+int
	using DslCompilerT = DslCompiler<VMM::Config>;
	using ProgramT = VMM::ProgramT;
	using DSLProgramT = DslCompilerT::ProgramT;

	VMM vm{ "Mixed", 16 };

	void SetUp() override
	{
		ASSERT_TRUE(vm.AddInstructions(g_floatOps));
		ASSERT_TRUE(vm.AddInstructions(g_intOps, nullptr, "Int"));
		ASSERT_TRUE(vm.AddInstructions(g_mixedOps, nullptr, "X"));
	}

	DslCompilerT Compiler() const
	{
		return DslCompilerT(vm.GetInstructionSet(), vm.GetTypeReg());
	}

	auto CompileDsl(const char* dsl)
	{
		//ProgramT prog(8, 0);
		DSLProgramT prog{};
		return Compiler().Compile(dsl, prog);
	}
};

TEST_F(CompileErrorTest, UnknownOp_ReturnsError)
{
	auto res = CompileDsl("FooBar I[0] I[1]\n");
	ASSERT_FALSE(res);
}

TEST_F(CompileErrorTest, MissingArgument_ReturnsError)
{
	// Add expects 2 src, only 1 provided
	auto res = CompileDsl("Add I[0]\n");
	ASSERT_FALSE(res);
}

TEST_F(CompileErrorTest, TrailingGarbage_ReturnsError)
{
	auto res = CompileDsl("Add I[0] I[1] I[2]\n");
	ASSERT_FALSE(res);
}

TEST_F(CompileErrorTest, LhsMustBeStack_ReturnsError)
{
	auto res = CompileDsl("I[0] = Add I[0] I[1]\n");
	ASSERT_FALSE(res);
}

TEST_F(CompileErrorTest, MissingEquals_ReturnsError)
{
	auto res = CompileDsl("S[0] Add I[0] I[1]\n");
	ASSERT_FALSE(res);
}

TEST_F(CompileErrorTest, WrongStackIndex_ReturnsError)
{
	// First result must be S[0], not S[5]
	auto res = CompileDsl("S[5] = Add I[0] I[1]\n");
	ASSERT_FALSE(res);
}

TEST_F(CompileErrorTest, LhsTypeMismatch_ReturnsError)
{
	// Add returns float, but annotated as int
	auto res = CompileDsl("S<int>[0] = Add I[0] I[1]\n");
	ASSERT_FALSE(res);
}

TEST_F(CompileErrorTest, SrcTypeMismatch_ReturnsError)
{
	// Add(float,float) but src annotated as int
	auto res = CompileDsl("Add I<int>[0] I[1]\n");
	ASSERT_FALSE(res);
}

TEST_F(CompileErrorTest, ProgramTooLarge_ReturnsError)
{
	auto res = CompileDsl("Add I[0] I[1]\nAdd I[0] I[1]\nAdd I[0] I[1]\n");
	ASSERT_FALSE(res);
}

TEST_F(CompileErrorTest, ErrorReportsCorrectLine)
{
	// Error is on line 2
	auto res = CompileDsl("Add I[0] I[1]\nFooBar I[0]\n");
	ASSERT_FALSE(res);
	EXPECT_EQ(res.error().line, 2u);
}

TEST_F(CompileErrorTest, InvalidAddress_ReturnsError)
{
	// 'X[0]' is not a valid segment prefix (only I/C/S)
	auto res = CompileDsl("Add X[0] I[1]\n");
	ASSERT_FALSE(res);
}

// =============================================================================
// SECTION 3 — Multi-type VM: float + int
// =============================================================================

class MultiTypeTest : public GTestLite::Test
{
protected:
	struct TestConfig
	{
		static constexpr size_t DstMaxArity = 3;
		static constexpr size_t SrcMaxArity = 3;
		static constexpr size_t MaxProgramSize = 32;
	};

	using VMM = MachineImpl<TestConfig, TypePack<float, int>, TypePack<float, int>>;      // multi-type float+int
	using DslCompilerT = DslCompiler<VMM::Config>;
	using ProgramT = VMM::ProgramT;
	using DSLProgramT = DslCompilerT::ProgramT;
	// =============================================================================
	// Helper: compile DSL into a fresh ProgramImpl, assert success
	// =============================================================================
	VMM vm{ "Multi", 16 };

	void SetUp() override
	{
		ASSERT_TRUE(vm.AddInstructions(g_floatOps));
		ASSERT_TRUE(vm.AddInstructions(g_intOps,   nullptr, "Int"));
		ASSERT_TRUE(vm.AddInstructions(g_mixedOps, nullptr, "X"));
	}

	DslCompilerT Compiler() const
	{
		return DslCompilerT(vm.GetInstructionSet(), vm.GetTypeReg());
	}
};

TEST_F(MultiTypeTest, Execute_F2I_Conversion)
{
	// S<int>[0] = X.F2I(I<float>[0])
	auto c = Compiler();
	VMM::ProgramT prog(4, 0);
	CompileOrFail(c, "X.F2I I[0]\n", prog);

	VMM::InputT input(1);
	input.GetData<float>()[0] = 3.7f;

	vm.Run(prog, input);
	const auto [pf, pi] = vm.GetResult();
	ASSERT_TRUE(pi);
	EXPECT_EQ(*pi, 3);
}

TEST_F(MultiTypeTest, Execute_I2F_Conversion)
{
	auto c = Compiler();
	VMM::ProgramT prog(4, 0);
	CompileOrFail(c, "X.I2F I[0]\n", prog);

	VMM::InputT input(1);
	input.GetData<int>()[0] = 7;

	vm.Run(prog, input);
	const auto [pf, pi] = vm.GetResult();
	ASSERT_TRUE(pf);
	EXPECT_NEAR(*pf, 7.0f, 1e-6f);
}

TEST_F(MultiTypeTest, Execute_FAddI_MixedSrc)
{
	// S<float>[0] = I<float>[0] + (float)I<int>[0]
	auto c = Compiler();
	VMM::ProgramT prog(4, 0);
	CompileOrFail(c, "X.FAddI I[0] I[0]\n", prog);

	VMM::InputT input(1);
	input.GetData<float>()[0] = 1.5f;
	input.GetData<int>()[0]   = 3;

	vm.Run(prog, input);
	const auto [pf, pi] = vm.GetResult();
	ASSERT_TRUE(pf);
	EXPECT_NEAR(*pf, 4.5f, 1e-6f);
}

TEST_F(MultiTypeTest, Execute_FloatChain_ThenF2I)
{
	// S<float>[0] = I[0] + I[1]
	// S<int>[0]   = X.F2I(S<float>[0])
	auto c = Compiler();
	VMM::ProgramT prog(8, 0);
	CompileOrFail(c, "Add I[0] I[1]\nX.F2I S[0]\n", prog);

	VMM::InputT input(2);
	input.GetData<float>()[0] = 2.9f;
	input.GetData<float>()[1] = 4.0f;

	vm.Run(prog, input);
	const auto [pf, pi] = vm.GetResult();
	ASSERT_TRUE(pi);
	EXPECT_EQ(*pi, 6);
}

TEST_F(MultiTypeTest, Execute_IntOps_WithPrefix)
{
	// Int.Add then Int.Mul
	auto c = Compiler();
	VMM::ProgramT prog(8, 0);
	CompileOrFail(c, "Int.Add I[0] I[1]\nInt.Mul S[0] I[0]\n", prog);

	VMM::InputT input(2);
	input.GetData<int>()[0] = 3;
	input.GetData<int>()[1] = 4;

	vm.Run(prog, input);
	const auto [pf, pi] = vm.GetResult();
	ASSERT_TRUE(pi);
	EXPECT_EQ(*pi, 21);   // (3+4)*3
}

TEST_F(MultiTypeTest, Decompile_MultiType_CorrectTypeAnnotations)
{
	auto c = Compiler();
	VMM::ProgramT prog(8, 0);
	CompileOrFail(c, "Add I[0] I[1]\nX.F2I S[0]\n", prog);

	auto out = c.Decompile((const DSLProgramT&)prog);
	ASSERT_TRUE(out.has_value());
	EXPECT_EQ(*out,
		"S<float>[0] = Add I<float>[0] I<float>[1]\n"
		"S<int>[0] = X.F2I S<float>[0]\n");
}

TEST_F(MultiTypeTest, Roundtrip_MultiType_ChainWithConversions)
{
	auto c = Compiler();
	AssertRoundtrip(c,
		"Add I[0] I[1]\n"
		"X.F2I S[0]\n"
		"Int.Add S[0] I[0]\n"
		);
}

TEST_F(MultiTypeTest, TypeAnnotation_LhsCorrectType_Accepted)
{
	auto c = Compiler();
	VMM::ProgramT prog(4, 0);
	// S<float>[0] is correct for Add(float,float)
	EXPECT_TRUE(c.Compile("S<float>[0] = Add I[0] I[1]\n", (DSLProgramT&)prog));
}

TEST_F(MultiTypeTest, TypeAnnotation_SrcCorrectType_Accepted)
{
	auto c = Compiler();
	VMM::ProgramT prog(4, 0);
	EXPECT_TRUE(c.Compile("Add I<float>[0] I<float>[1]\n", (DSLProgramT&)prog));
}

// =============================================================================
// SECTION 4 — Multi-dst ops (DstCount=2, mixed types per dst)
// =============================================================================

class MultiDstTest : public GTestLite::Test
{
protected:
	struct TestConfig
	{
		static constexpr size_t DstMaxArity = 3;
		static constexpr size_t SrcMaxArity = 3;
		static constexpr size_t MaxProgramSize = 32;
	};

	using VMM = MachineImpl<TestConfig, TypePack<float, int>, TypePack<float, int>>;      // multi-type float+int
	using DslCompilerT = DslCompiler<VMM::Config>;
	using ProgramT = VMM::ProgramT;
	using DSLProgramT = DslCompilerT::ProgramT;

	VMM vm{ "MultiDst", 16 };

	void SetUp() override
	{
		ASSERT_TRUE(vm.AddInstructions(g_mixedOps, nullptr, "X"));
		ASSERT_TRUE(vm.AddInstructions(g_intOps,   nullptr, "Int"));
	}

	DslCompilerT Compiler() const
	{
		return DslCompilerT(vm.GetInstructionSet(), vm.GetTypeReg());
	}
};

TEST_F(MultiDstTest, Execute_SplitIntFrac_TwoDsts)
{
	// X.SplitIntFrac produces S<int>[0] and S<float>[0] from one input
	auto c = Compiler();
	VMM::ProgramT prog(4, 0);
	CompileOrFail(c, "X.SplitIntFrac I[0]\n", prog);

	VMM::InputT input(1);
	input.GetData<float>()[0] = 3.75f;

	vm.Run(prog, input);
	const auto [pf, pi] = vm.GetResult();
	ASSERT_TRUE(pf);
	ASSERT_TRUE(pi);
	EXPECT_EQ(*pi, 3);
	EXPECT_NEAR(*pf, 0.75f, 1e-5f);
}

TEST_F(MultiDstTest, Execute_SplitThenUseBothDsts)
{
	// S<int>[0], S<float>[0] = X.SplitIntFrac(I<float>[0])
	// S<int>[1]              = Int.Add(S<int>[0], S<int>[0])
	auto c = Compiler();
	VMM::ProgramT prog(8, 0);
	CompileOrFail(c,
		"X.SplitIntFrac I[0]\n"
		"Int.Add S[0] S[0]\n",
		prog);

	VMM::InputT input(1);
	input.GetData<float>()[0] = 5.5f;

	vm.Run(prog, input);
	const auto [pf, pi] = vm.GetResult();
	ASSERT_TRUE(pi);
	EXPECT_EQ(*pi, 10);   // floor(5.5)=5, 5+5=10
}

TEST_F(MultiDstTest, Decompile_MultiDst_ContainsBothTypes)
{
	// Decompile should emit S[x..y] = for dstCount>1
	auto c = Compiler();
	VMM::ProgramT prog(4, 0);
	CompileOrFail(c, "X.SplitIntFrac I[0]\n", prog);

	auto out = c.Decompile((const DSLProgramT&)prog);
	ASSERT_TRUE(out.has_value());
	// output must reference the op name
	EXPECT_TRUE(out->find("X.SplitIntFrac") != std::string::npos);
	// and the src input
	EXPECT_TRUE(out->find("I<float>[0]") != std::string::npos);
}

// NOTE: multi-dst decompile emits "S[x..y] = " which is intentionally not
// re-parseable by Compile (no LHS type info available for heterogeneous dsts).
// Roundtrip is therefore only verified for single-dst ops.

// =============================================================================
// SECTION 5 — CompoundInstructionSet (prefix-qualified lookup)
// =============================================================================

class CompoundISetTest : public GTestLite::Test
{
protected:
	struct TestConfig
	{
		static constexpr size_t DstMaxArity = 3;
		static constexpr size_t SrcMaxArity = 3;
		static constexpr size_t MaxProgramSize = 32;
	};

	using VMM = MachineImpl<TestConfig, TypePack<float, int>, TypePack<float, int>>;
	using DslCompilerT = DslCompiler<VMM::Config>;
	using ProgramT = VMM::ProgramT;
	using DSLProgramT = DslCompilerT::ProgramT;

	// Two independent single-type ISets merged into one compound set.
	InstructionSet  m_floatIset{ "Float" };
	InstructionSet  m_intIset{ "Int" };
	CompoundInstructionSet m_compound;
	TypeRegisterT<float, int> m_reg;

	void SetUp() override
	{
		ASSERT_TRUE(m_floatIset.Add(g_floatOps, m_reg));
		ASSERT_TRUE(m_intIset.Add(g_intOps,   m_reg));
		m_compound.AddSet(m_floatIset);
		m_compound.AddSet(m_intIset);
	}

	DslCompilerT Compiler() const
	{
		return DslCompilerT(m_compound, m_reg);
	}
};

TEST_F(CompoundISetTest, Lookup_FloatPrefix_Found)
{
	uint32_t opcode{};
	EXPECT_TRUE(m_compound.Lookup("Float.Add", opcode));
}

TEST_F(CompoundISetTest, Lookup_IntPrefix_Found)
{
	uint32_t opcode{};
	EXPECT_TRUE(m_compound.Lookup("Int.Add", opcode));
}

TEST_F(CompoundISetTest, Lookup_Unprefixed_NotFound)
{
	uint32_t opcode{};
	// Without prefix, names collide and are not registered bare
	EXPECT_FALSE(m_compound.Lookup("Add", opcode));
}

TEST_F(CompoundISetTest, Lookup_WrongPrefix_NotFound)
{
	uint32_t opcode{};
	EXPECT_FALSE(m_compound.Lookup("Int.Sub", opcode));
}

TEST_F(CompoundISetTest, Compile_FloatAndIntPrefixed_Succeeds)
{
	auto c = Compiler();
	DSLProgramT prog{};
	EXPECT_TRUE(c.Compile("Float.Add I[0] I[1]\nInt.Add I[0] I[1]\n", prog));
}

TEST_F(CompoundISetTest, Compile_PrefixedOps_ExecuteCorrectly)
{
	// Use VM that mirrors the compound set
	VMM vm2{ "Compound", 16 };
	ASSERT_TRUE(vm2.AddInstructions(g_floatOps, nullptr, "Float"));
	ASSERT_TRUE(vm2.AddInstructions(g_intOps,   nullptr, "Int"));

	DslCompilerT c(vm2.GetInstructionSet(), vm2.GetTypeReg());
	VMM::ProgramT prog(8, 0);
	CompileOrFail(c, "Float.Add I[0] I[1]\nInt.Mul I[0] I[1]\n", prog);

	VMM::InputT input(2);
	input.GetData<float>()[0] = 1.0f;
	input.GetData<float>()[1] = 2.0f;
	input.GetData<int>()[0]   = 3;
	input.GetData<int>()[1]   = 4;

	vm2.Run(prog, input);
	const auto [pf, pi] = vm2.GetResult();
	ASSERT_TRUE(pf);
	ASSERT_TRUE(pi);
	EXPECT_NEAR(*pf, 3.0f, 1e-6f);
	EXPECT_EQ(*pi, 12);
}

TEST_F(CompoundISetTest, Roundtrip_PrefixedCompound_Stable)
{
	VMM vm2{ "Compound", 16 };
	ASSERT_TRUE(vm2.AddInstructions(g_floatOps, nullptr, "Float"));
	ASSERT_TRUE(vm2.AddInstructions(g_intOps,   nullptr, "Int"));

	DslCompilerT c(vm2.GetInstructionSet(), vm2.GetTypeReg());
	AssertRoundtrip(c, "Float.Add I[0] I[1]\nInt.Mul I[0] I[1]\n");
}

// =============================================================================
// SECTION 6 — Optimized (dead-code eliminated) run matches full run
// =============================================================================

class OptimizedRunTest : public GTestLite::Test
{
protected:
	struct TestConfig
	{
		static constexpr size_t DstMaxArity = 3;
		static constexpr size_t SrcMaxArity = 3;
		static constexpr size_t MaxProgramSize = 32;
	};

	using VMM = MachineImpl<TestConfig, TypePack<float, int>, TypePack<float, int>>;
	using DslCompilerT = DslCompiler<VMM::Config>;
	using ProgramT = VMM::ProgramT;
	using DSLProgramT = DslCompilerT::ProgramT;

	VMM vm{ "Opt", 16 };

	void SetUp() override
	{
		ASSERT_TRUE(vm.AddInstructions(g_floatOps));
		ASSERT_TRUE(vm.AddInstructions(g_mixedOps, nullptr, "X"));
		ASSERT_TRUE(vm.AddInstructions(g_intOps,   nullptr, "Int"));
	}

	DslCompilerT Compiler() const
	{
		return DslCompilerT(vm.GetInstructionSet(), vm.GetTypeReg());
	}
};

TEST_F(OptimizedRunTest, DeadFloat_OptimizedMatchesFull)
{
	// S<float>[1] is never consumed — dead
	auto c = Compiler();
	VMM::ProgramT prog(8, 0);
	CompileOrFail(c,
		"Add I[0] I[1]\n"   // S<float>[0] — live (used below)
		"Add I[0] I[0]\n"   // S<float>[1] — dead
		"X.F2I S[0]\n",     // S<int>[0]   — live output
		prog);

	VMM::InputT input(2);
	input.GetData<float>()[0] = 2.0f;
	input.GetData<float>()[1] = 3.0f;

	vm.Run(prog, input);
	const auto [pfFull, piFull] = vm.GetResult();
	const int expectedInt = *piFull;

	vm.OptimizeProgram(prog);
	vm.Run(prog, input, true);
	const auto [pfOpt, piOpt] = vm.GetResult();
	ASSERT_TRUE(piOpt);
	EXPECT_EQ(*piOpt, expectedInt);
}

TEST_F(OptimizedRunTest, AllLive_OptimizedMatchesFull)
{
	auto c = Compiler();
	VMM::ProgramT prog(8, 0);
	CompileOrFail(c,
		"Add I[0] I[1]\n"
		"X.F2I S[0]\n"
		"Int.Add S[0] I[0]\n",
		prog);

	VMM::InputT input(2);
	input.GetData<float>()[0] = 1.5f;
	input.GetData<float>()[1] = 2.5f;
	input.GetData<int>()[0]   = 10;

	vm.Run(prog, input);
	const auto [pfFull, piFull] = vm.GetResult();
	const int expInt = *piFull;

	vm.OptimizeProgram(prog);
	vm.Run(prog, input, true);
	const auto [pfOpt, piOpt] = vm.GetResult();
	ASSERT_TRUE(piOpt);
	EXPECT_EQ(*piOpt, expInt);
}

// =============================================================================
// SECTION 7 — Property: random valid DSL always survives roundtrip
// =============================================================================

class RoundtripPropertyTest : public GTestLite::Test
{
protected:
	struct TestConfig
	{
		static constexpr size_t DstMaxArity = 3;
		static constexpr size_t SrcMaxArity = 3;
		static constexpr size_t MaxProgramSize = 32;
	};

	using VMM = MachineImpl<TestConfig, TypePack<float, int>, TypePack<float, int>>;
	using DslCompilerT = DslCompiler<VMM::Config>;
	using ProgramT = VMM::ProgramT;
	using DSLProgramT = DslCompilerT::ProgramT;

	VMM vm{ "Prop", 32 };

	void SetUp() override
	{
		ASSERT_TRUE(vm.AddInstructions(g_floatOps));
		ASSERT_TRUE(vm.AddInstructions(g_intOps,   nullptr, "Int"));
		ASSERT_TRUE(vm.AddInstructions(g_mixedOps, nullptr, "X"));
	}
};

TEST_F(RoundtripPropertyTest, RandomMultiTypePrograms_RoundtripStable)
{
	DslCompilerT c(vm.GetInstructionSet(), vm.GetTypeReg());

	// op name + arity (all take 2 src except Neg=1, X.F2I=1, X.I2F=1)
	struct OpSpec { const char* name; int srcCount; };
	static constexpr OpSpec ops[] = {
		{ "Add",      2 }, { "Sub",      2 }, { "Mul",      2 },
		{ "Int.Add",  2 }, { "Int.Mul",  2 },
		{ "X.F2I",    1 }, { "X.I2F",   1 }, { "X.FAddI",  2 },
	};

	// stack pointer per type: [0]=float, [1]=int
	constexpr int ITER = 300;
	srand(42);

	for (int it = 0; it < ITER; ++it)
	{
		std::string src;
		int stackPtrs[2] = { 0, 0 };

		const int instrCount = 1 + rand() % 8;
		for (int i = 0; i < instrCount; ++i)
		{
			const auto& op = ops[rand() % (int)(sizeof(ops) / sizeof(ops[0]))];

			src += op.name;

			// generate src addresses (any segment, offset within pushed range)
			for (int s = 0; s < op.srcCount; ++s)
			{
				int seg = rand() % 3;   // 0=I,1=C,2=S
				const char segChar = (seg == 0) ? 'I' : (seg == 1) ? 'C' : 'S';
				int maxIdx = (seg == 2) ? (stackPtrs[0] > 0 ? stackPtrs[0] : 1) : 4;
				int idx = rand() % maxIdx;
				src += ' ';
				src += segChar;
				src += '[';
				src += std::to_string(idx);
				src += ']';
			}
			src += '\n';

			// track how many values each op pushes onto float/int stack
			// (we don't know exact types here, just skip — invalid ones will be rejected)
			stackPtrs[0]++;   // conservative: assume worst case for S[] references
		}

		DSLProgramT p1{};
		auto r1 = c.Compile(src, p1);
		if (!r1) continue;   // invalid programs are expected — skip

		auto d1 = c.Decompile(p1);
		ASSERT_TRUE(d1.has_value());

		DSLProgramT p2{};
		auto r2 = c.Compile(*d1, p2);
		ASSERT_TRUE(r2) << "Decompiled output failed to recompile:\n" << *d1;

		auto d2 = c.Decompile(p2);
		ASSERT_TRUE(d2.has_value());

		EXPECT_EQ(*d1, *d2) << "Roundtrip mismatch on iteration " << it;
	}
}
