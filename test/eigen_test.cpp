#include "../gtest/GTestLite.h"
#include "../examples/EigenOps.h"
#include "../include/DSLCompiler.h"

using namespace EigenOps;
using namespace AbstractVM;

// Shared registry — EMatF=0, EVecF=1, float=2.
static AbstractVM::TypeRegisterT<EMatF, EVecF, float> g_reg;

template<typename... Args>
bool fill(EMatF& A, int r, int c, Args... args)
{
	constexpr size_t N = sizeof...(Args);
	if (static_cast<size_t>(r * c) != N)
	{
		return false;
	}
	A.resize(r, c);
	auto m = A.view();
	std::array<float, N> data = { static_cast<float>(args)... };
	int idx = 0;
	for (int i = 0; i < r; ++i)
	{
		for (int j = 0; j < c; ++j)
		{
			m(i, j) = data[idx++];
		}
	}
	return true;
}

template<typename... Args>
bool fill(EVecF& V, int size, Args... args)
{
	constexpr size_t N = sizeof...(Args);
	if (static_cast<size_t>(size) != N)
	{
		return false;
	}
	V.resize(size);
	auto v = V.view();
	std::array<float, N> data = { static_cast<float>(args)... };
	for (size_t i = 0; i < N; ++i)
	{
		v(i) = data[i];
	}
	return true;
}

// =============================================================================
// Per-op unit tests — direct Execute/Validate without Machine
// =============================================================================

class EigenMachineTest : public ::GTestLite::Test
{
protected:
	void SetUp() override
	{
		A.Init(100);
		B.Init(100);
		C.Init(100);
		V1.Init(100);
		V2.Init(100);
	}

	EMatF A;
	EMatF B;
	EMatF C;
	EVecF V1;
	EVecF V2;
	float s_res{};

	void RunExecute(const Op* op, std::vector<void*> dst_ptrs, std::vector<void*> src_ptrs)
	{
		op->Execute((DataType*)dst_ptrs.data(), (const DataType*)src_ptrs.data());
	}

	bool RunValidate(const Op* op, std::vector<void*> dst_ptrs, std::vector<void*> src_ptrs)
	{
		return op->Validate((DataType*)dst_ptrs.data(), (const DataType*)src_ptrs.data());
	}
};

TEST_F(EigenMachineTest, MatMul_CorrectCalculation)
{
	ASSERT_TRUE(fill(A, 2, 2, 1.0f, 2.0f, 3.0f, 4.0f));
	ASSERT_TRUE(fill(B, 2, 2, 5.0f, 6.0f, 7.0f, 8.0f));

	const auto op = CreateOp<OpMatMul<float>, ValMatMul<float>>(g_reg);
	ASSERT_TRUE(RunValidate(op, { &C }, { &A, &B }));
	RunExecute(op, { &C }, { &A, &B });

	EMatF Expected; Expected.Init(100);
	fill(Expected, 2, 2, 19.0f, 22.0f, 43.0f, 50.0f);
	EXPECT_EQ(C, Expected);

	delete op;
}

TEST_F(EigenMachineTest, MatMul_InvalidDimensions_Fails)
{
	const Op* op = CreateOp<OpMatMul<float>, ValMatMul<float>>(g_reg);
	ASSERT_TRUE(fill(A, 2, 3, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f));
	ASSERT_TRUE(fill(B, 2, 2, 0.0f, 0.0f, 0.0f, 0.0f));
	EXPECT_FALSE(RunValidate(op, { &C }, { &A, &B }));
	delete op;
}

TEST_F(EigenMachineTest, MatMul_CapacityExceeded_Fails)
{
	const Op* op = CreateOp<OpMatMul<float>, ValMatMul<float>>(g_reg);
	EMatF Tmp{};
	Tmp.Init(4);
	ASSERT_TRUE(fill(A, 3, 3, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f));
	ASSERT_TRUE(fill(B, 3, 3, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f));
	EXPECT_FALSE(RunValidate(op, { &Tmp }, { &A, &B }));
	delete op;
}

TEST_F(EigenMachineTest, MatAdd_CorrectCalculation)
{
	const Op* op = CreateOp<OpMatAdd<float>, ValMatSameShape<float>>(g_reg);
	ASSERT_TRUE(fill(A, 2, 2, 1.0f, 2.0f, 3.0f, 4.0f));
	ASSERT_TRUE(fill(B, 2, 2, 10.0f, 20.0f, 30.0f, 40.0f));

	ASSERT_TRUE(RunValidate(op, { &C }, { &A, &B }));
	RunExecute(op, { &C }, { &A, &B });

	EMatF Expected; Expected.Init(100);
	fill(Expected, 2, 2, 11.0f, 22.0f, 33.0f, 44.0f);
	EXPECT_EQ(C, Expected);

	delete op;
}

TEST_F(EigenMachineTest, MatAdd_ShapeMismatch_Fails)
{
	const Op* op = CreateOp<OpMatAdd<float>, ValMatSameShape<float>>(g_reg);
	ASSERT_TRUE(fill(A, 2, 2, 0.0f, 0.0f, 0.0f, 0.0f));
	ASSERT_TRUE(fill(B, 2, 3, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f));
	EXPECT_FALSE(RunValidate(op, { &C }, { &A, &B }));
	delete op;
}

TEST_F(EigenMachineTest, MatScale_CorrectCalculation)
{
	const Op* op = CreateOp<OpMatMulScalar<float>, ValMatScalar<float>>(g_reg);
	ASSERT_TRUE(fill(A, 2, 2, 1.0f, 2.0f, 3.0f, 4.0f));
	float s = 3.0f;

	ASSERT_TRUE(RunValidate(op, { &C }, { &A, &s }));
	RunExecute(op, { &C }, { &A, &s });

	EMatF Expected; Expected.Init(100);
	fill(Expected, 2, 2, 3.0f, 6.0f, 9.0f, 12.0f);
	EXPECT_EQ(C, Expected);

	delete op;
}

TEST_F(EigenMachineTest, MatVecMul_CorrectCalculation)
{
	const Op* op = CreateOp<OpMatVecMul<float>, ValMatVec<float>>(g_reg);
	ASSERT_TRUE(fill(A, 2, 3, 1.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f));
	ASSERT_TRUE(fill(V1, 3, 5.0f, 3.0f, 1.0f));

	ASSERT_TRUE(RunValidate(op, { &V2 }, { &A, &V1 }));
	RunExecute(op, { &V2 }, { &A, &V1 });

	EVecF Expected; Expected.Init(100);
	fill(Expected, 2, 5.0f, 6.0f);
	EXPECT_EQ(V2, Expected);

	delete op;
}

TEST_F(EigenMachineTest, MatVecMul_DimMismatch_Fails)
{
	const Op* op = CreateOp<OpMatVecMul<float>, ValMatVec<float>>(g_reg);
	ASSERT_TRUE(fill(A, 2, 3, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f));
	ASSERT_TRUE(fill(V1, 2, 0.0f, 0.0f));
	EXPECT_FALSE(RunValidate(op, { &V2 }, { &A, &V1 }));
	delete op;
}

TEST_F(EigenMachineTest, Relu_ZerosNegativeValues)
{
	const Op* op = CreateOp<OpMatRelu<float>, ValMatUnary<float>>(g_reg);
	ASSERT_TRUE(fill(A, 2, 2, -10.0f, 5.0f, 0.0f, -1.0f));

	ASSERT_TRUE(RunValidate(op, { &C }, { &A }));
	RunExecute(op, { &C }, { &A });

	EMatF Expected; Expected.Init(100);
	fill(Expected, 2, 2, 0.0f, 5.0f, 0.0f, 0.0f);
	EXPECT_EQ(C, Expected);

	delete op;
}

TEST_F(EigenMachineTest, VecDot_CorrectCalculation)
{
	const Op* op = CreateOp<OpVecDot<float>, ValVecDot<float>>(g_reg);
	ASSERT_TRUE(fill(V1, 3, 1.0f, 0.0f, 3.0f));
	ASSERT_TRUE(fill(V2, 3, 4.0f, 5.0f, 2.0f));

	ASSERT_TRUE(RunValidate(op, { &s_res }, { &V1, &V2 }));
	RunExecute(op, { &s_res }, { &V1, &V2 });
	EXPECT_NEAR(s_res, 10.0f, 1e-6f);

	delete op;
}

TEST_F(EigenMachineTest, VecDot_SizeMismatch_Fails)
{
	const Op* op = CreateOp<OpVecDot<float>, ValVecDot<float>>(g_reg);
	ASSERT_TRUE(fill(V1, 3, 0.0f, 0.0f, 0.0f));
	ASSERT_TRUE(fill(V2, 4, 0.0f, 0.0f, 0.0f, 0.0f));
	EXPECT_FALSE(RunValidate(op, { &s_res }, { &V1, &V2 }));
	delete op;
}

TEST_F(EigenMachineTest, Transpose_SwapsDimensions)
{
	const Op* op = CreateOp<OpMatTranspose<float>, ValMatUnary<float>>(g_reg);
	ASSERT_TRUE(fill(A, 3, 2, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f));

	ASSERT_TRUE(RunValidate(op, { &C }, { &A }));
	RunExecute(op, { &C }, { &A });

	EMatF Expected; Expected.Init(100);
	fill(Expected, 2, 3, 1.0f, 3.0f, 5.0f, 2.0f, 4.0f, 6.0f);
	EXPECT_EQ(C, Expected);

	delete op;
}

// =============================================================================
// Machine integration fixture — full instruction set, DSL-based program setup
// =============================================================================

class MachineIntegrationTest : public ::GTestLite::Test
{
	struct TestConfig
	{
		static constexpr size_t DstMaxArity = 1;
		static constexpr size_t SrcMaxArity = 2;
		static constexpr size_t MaxProgramSize = 32;
		static constexpr size_t TypesCount = 3;
		static constexpr size_t ConstTypesCount = 0;
	};
protected:
	using MachineT = MachineImpl<TestConfig, TypePack<EMatF, EVecF, float>>;
	using ProgramT = MachineT::ProgramT;

	void SetUp() override
	{
		ASSERT_TRUE(RegisterAllOps<float>(vm.GetInstructionSet(), vm.GetTypeReg()));
		compiler = new DslCompiler<TestConfig>(vm.GetInstructionSet(), g_reg);
	}

	void TearDown() override
	{
		if (program)
		{
			delete program;
		}
		delete compiler;
	}

	// Compile DSL into a fresh Program. Constants segment has constSlots slots.
	ProgramT* MakeProgram(const char* dsl)
	{
		auto* prog = new ProgramT(16, 100);
		ASSERT_TRUE(compiler->Compile(dsl, *(Program<TestConfig> *)prog));
		return prog;
	}

	InstructionSet iset{};
	MachineT vm{ "LinAlg", 100 };
	DslCompiler<TestConfig>* compiler = nullptr;
	ProgramT* program = nullptr;
};

TEST_F(MachineIntegrationTest, MatMul_ThenMatAdd_WithConstant)
{
	// S[0] = A * B
	// S[1] = S[0] + C[0]   (C[0] is all-ones)
	program = MakeProgram(
		"S[0] = MatMulF I[0] I[1]\n"
		"S[1] = MatAddF S[0] C[0]\n"
	);

	fill(program->GetConst<0>()[0], 2, 2, 1.0f, 1.0f, 1.0f, 1.0f);

	MachineT::InputT input(2, 100);
	fill(input.GetData<EMatF>()[0], 2, 2, 1.0f, 2.0f, 3.0f, 4.0f);
	fill(input.GetData<EMatF>()[1], 2, 2, 5.0f, 6.0f, 7.0f, 8.0f);

	vm.Run(*program, input);

	const auto res = vm.GetResult();
	const EMatF* r = std::get<0>(res);
	ASSERT_TRUE(r != nullptr);

	EMatF Expected; Expected.Init(100);
	fill(Expected, 2, 2, 20.0f, 23.0f, 44.0f, 51.0f);
	EXPECT_EQ(*r, Expected);
}

TEST_F(MachineIntegrationTest, MatMul_ThenMatAdd_Validation_IncompatibleShapes)
{
	program = MakeProgram(
		"S[0] = MatMulF I[0] I[1]\n"
	);

	MachineT::InputT input(2, 100);
	fill(input.GetData<EMatF>()[0], 2, 2, 1.0f, 2.0f, 3.0f, 4.0f);

	// Replace second input with a 3x3 — incompatible with 2x2
	fill(input.GetData<EMatF>()[1], 3, 3, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);

	EXPECT_FALSE(vm.Validate(*program, input));
}

TEST_F(MachineIntegrationTest, Relu_AfterMatMul)
{
	// S[0] = A * B,  S[1] = relu(S[0])
	// Use a matrix with negative products.
	program = MakeProgram(
		"S[0] = MatMulF I[0] I[1]\n"
		"S[1] = MatReluF   S[0]\n"
	);

	MachineT::InputT input(2, 100);
	// A = [[1,-1],[-1,1]], B = I => A*B = A, relu => [[1,0],[0,1]]
	fill(input.GetData<EMatF>()[0], 2, 2, 1.0f, -1.0f, -1.0f, 1.0f);
	fill(input.GetData<EMatF>()[1], 2, 2, 1.0f, 0.0f, 0.0f, 1.0f);

	vm.Run(*program, input);

	const auto res = vm.GetResult();
	const EMatF* r = std::get<0>(res);
	ASSERT_TRUE(r != nullptr);

	EMatF Expected; Expected.Init(100);
	fill(Expected, 2, 2, 1.0f, 0.0f, 0.0f, 1.0f);
	EXPECT_EQ(*r, Expected);
}

TEST_F(MachineIntegrationTest, MatScale_ThenTranspose)
{
	// S[0] = A * 2.0,  S[1] = S[0]^T
	program = MakeProgram(
		"S[0] = MatMulScalarF I[0] C[0]\n"
		"S[1] = MatTransposeF S[0]\n"
	);

	// C[0] is float = 2.0f — stored in segment index 2 (float)
	program->GetConst<2>()[0] = 2.0f;

	MachineT::InputT input(1, 100);
	fill(input.GetData<EMatF>()[0], 2, 3, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);

	vm.Run(*program, input);

	const auto res = vm.GetResult();
	const EMatF* r = std::get<0>(res);
	ASSERT_TRUE(r != nullptr);

	// A*2 = [[2,4,6],[8,10,12]], transposed = [[2,8],[4,10],[6,12]]
	EMatF Expected; Expected.Init(100);
	fill(Expected, 3, 2, 2.0f, 8.0f, 4.0f, 10.0f, 6.0f, 12.0f);
	EXPECT_EQ(*r, Expected);
}

TEST_F(MachineIntegrationTest, DeadCodeElimination_OptimizedMatchesFull)
{
	// I[4] is dead — never reaches any output.
	program = MakeProgram(
		"S[0] = MatMulF I[0] I[1]\n"    // used
		"S[1] = MatAddF I[0] I[0]\n"    // dead — result never read
		"S[2] = MatAddF S[0] I[0]\n" // used
	);

	MachineT::InputT input(2, 100);
	fill(input.GetData<EMatF>()[0], 2, 2, 1.0f, 0.0f, 0.0f, 2.0f);
	fill(input.GetData<EMatF>()[1], 2, 2, 3.0f, 0.0f, 0.0f, 4.0f);

	// Full run
	vm.Run(*program, input);
	const auto resFull = vm.GetResult();
	const EMatF& full = *std::get<0>(resFull);

	// Optimized run
	vm.OptimizeProgram(*program);
	vm.Run(*program, input, true);
	const auto resOpt = vm.GetResult();
	const EMatF* opt = std::get<0>(resOpt);
	ASSERT_TRUE(opt != nullptr);

	EXPECT_EQ(full, *opt);
}
