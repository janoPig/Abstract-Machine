#pragma once
#include "../gtest/GTestLite.h"
#include "../include/DSLCompiler.h"

namespace TestDslUtils
{
	template<typename CompilerT, typename ProgramT>
	void CompileOrFail(const CompilerT& compiler, const char* dsl, ProgramT& program)
	{
		using DslProgramT = typename CompilerT::ProgramT;
		const auto result = compiler.Compile(dsl, (DslProgramT&)program);
		ASSERT_TRUE(result) << result.error().message;
	}

	template<AbstractVM::VMConfigSpec Cfg>
	void AssertRoundtrip(const AbstractVM::DslCompiler<Cfg>& compiler, const char* dsl)
	{
		using ProgramT = AbstractVM::Program<Cfg>;
		ProgramT first{};
		const auto firstCompile = compiler.Compile(dsl, first);
		ASSERT_TRUE(firstCompile) << firstCompile.error().message;

		const auto firstDsl = compiler.Decompile(first);
		ASSERT_TRUE(firstDsl);

		ProgramT second{};
		const auto secondCompile = compiler.Compile(*firstDsl, second);
		ASSERT_TRUE(secondCompile) << secondCompile.error().message;

		const auto secondDsl = compiler.Decompile(second);
		ASSERT_TRUE(secondDsl);
		EXPECT_EQ(*firstDsl, *secondDsl);

	}
}
