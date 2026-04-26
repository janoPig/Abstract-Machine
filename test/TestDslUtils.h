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

	template<typename MachineT>
	void AssertRoundtrip(const AbstractVM::DslCompiler<typename MachineT::Config>& compiler, const char* dsl)
	{
		using ProgramT = typename MachineT::ProgramT;
		using DslT = AbstractVM::DslCompiler<typename MachineT::Config>;
		using DslProgramT = typename DslT::ProgramT;
		ProgramT first{};
		const auto firstCompile = compiler.Compile(dsl, (DslProgramT&)first);
		ASSERT_TRUE(firstCompile) << firstCompile.error().message;

		const auto firstDsl = compiler.Decompile((const DslProgramT&)first);
		ASSERT_TRUE(firstDsl);

		ProgramT second{};
		const auto secondCompile = compiler.Compile(*firstDsl, (DslProgramT&)second);
		ASSERT_TRUE(secondCompile) << secondCompile.error().message;

		const auto secondDsl = compiler.Decompile((const DslProgramT&)second);
		ASSERT_TRUE(secondDsl);
		EXPECT_EQ(*firstDsl, *secondDsl);

	}
}
