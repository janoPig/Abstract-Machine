# [WIP] Abstract Virtual Machine (AVM)

> [!WARNING]
> This project is a **prototype under construction**. Architectural details are subject to change, and the current version is optimized for development flexibility and evaluation.

A lightweight, header-only, embeddable virtual machine for executing statically-typed instruction programs in C++20.

---

## Features

- **Header-only** — drop `include/` into your project, no linking required
- **Strongly typed** — all operand types are resolved at compile time via `TypePack<Ts...>`
- **Zero RTTI** — type identity uses unique static addresses, no `typeid` or `std::type_index`
- **Multiple execution modes** — full, validated, optimized (dead-code elimination)
- **Text DSL** — compile/decompile programs from/to a human-readable instruction format
- **Extensible** — define custom ops as plain C++ functions; `CreateOp<Fn>` wraps them automatically
- **Example implementation** — `EigenOps` provides a example of linear algebra instruction set via [Eigen 5.0](https://eigen.tuxfamily.org)

---

## Requirements

| Requirement | Version |
|---|---|
| C++ Standard | C++20 |
| CMake | ≥ 3.25 |
| Compiler | MSVC 19.3+, GCC 13+, Clang 16+ |
| Optional dependency | Eigen 5.0 (for `EigenOps`) |

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                              USER CODE                                 │
└────────────────┬───────────────────────────────────────────┬───────────┘
                 │                                           │
  ┌──────────────▼──────────────┐             ┌──────────────▼───────────┐
  │         DSL Script          │             │ MachineImpl<Cfg, Ts, Cs> │
  └──────────────┬──────────────┘             │ (Typed Frontend)         │
                 │                            └──────┬────────────┬──────┘
  ┌──────────────▼──────────────┐                    │            │
  │         DslCompiler         │                    │ owns       │ owns
  │ (Uses TypeReg & InstrSet)   │             ┌──────▼─────┐┌─────▼──────┐
  └──────────────┬──────────────┘             │TypeRegister││Instruction │
                 │ generates                  │<Ts...>     ││Set         │
  ┌──────────────▼──────────────┐             └────────────┘└─────┬──────┘
  │         ProgramImpl         │                                 │
  │ (Typed Program + Constants) │                                 │
  └──────────────┬──────────────┘                                 │
                 │ inherits                                       │
  ┌──────────────▼──────────────┐             ┌───────────────────▼──────┐
  │           Program           │◄────────────┤      Machine<Cfg>        │
  │ (Instruction[], Segment)    │    runs     │  (Type-Erased Backend)   │
  └─────────────────────────────┘             └──────┬────────────┬──────┘
                                                     │            │
                                                owns │            │ owns
                                              ┌──────▼─────┐┌─────▼──────┐
                                              │ Processor  ││ Stack<N>   │
                                              │ <Cfg>      ││ (Segments) │
                                              └────────────┘└────────────┘
```

## Memory Model

Every `SegmentImpl` maintains **one heap-allocated typed array per type** in the `TypePack`. Three segment kinds are used at runtime:

| Segment | Symbol | Description |
|---|---|---|
| Input (data) | `I[n]` | Read-only runtime inputs provided per execution |
| Constants | `C[n]` | Mutable values embedded in the `Program`, written once before execution |
| Stack | `S[n]` | Output slots; grows monotonically as instructions execute |

Addresses encode segment type + element offset: `I[0]`, `C<float>[1]`, `S[3]`.  
The optional type annotation (`<float>`) is validated by `DslCompiler` against the op's declared signature.

---

## DSL Syntax & Programming

The VM uses a text-based DSL for program definition. Each instruction consists of an operation name followed by destination and source addresses.

### Address Format
Addresses are encoded as `Segment<Type>[Offset]`:
* `I<T>[n]`: Input segment (Read-only data provided at runtime).
* `C<T>[n]`: Constant segment (Read-only data baked into the program).
* `S<T>[n]`: Stack segment (Read-write temporary storage).

### Example Script
```text
# Matrix multiplication of input and constant, stored on stack
MatMulF S<EMatF>[0] I<EMatF>[0] C<EMatF>[0]

# Add another matrix from input to the result
MatAddF S<EMatF>[1] S<EMatF>[0] I<EMatF>[1]

# Final result is stored in S<EMatF>[1]
```

---

## Type System

Types are identified by their **position** in the user-supplied `TypePack<Ts...>`. This index serves as both the segment index in `SegmentImpl` and the `type` field in `Address`.

```cpp
using namespace AbstractVM;

using MyTypes = TypePack<float, int>;
// float → index 0,  int → index 1

MachineImpl<MyCfg, MyTypes, MyTypes> vm("MyVM");
```

Type tokens are unique static addresses — no RTTI, no hash maps:

```cpp
template<typename T>
inline TypeToken GetTypeToken() noexcept {
    static const char tag = 0;
    return &tag;   // one unique pointer per T
}
```

Custom types can expose a `static constexpr const char* TypeName` for use in DSL diagnostics. Some primitives (`float`, `int`, `double`, `size_t`) are pre-registered (TBD).

---

## Configuration

Each machine is parameterized by a `VMConfig` struct:

```cpp
struct MyConfig {
    static constexpr size_t SrcMaxArity   = 2;   // max inputs per instruction
    static constexpr size_t DstMaxArity   = 1;   // max outputs per instruction
    static constexpr size_t MaxProgramSize = 64;  // fixed instruction slot count
    static constexpr size_t TypesCount    = 2;   // must equal sizeof...(Ts)
    static constexpr size_t ConstTypesCount = 2; // ≤ TypesCount, 0 = no constants
};
```

`VMConfig` is enforced as a C++20 concept; mismatches are caught at compile time.

---

## Defining Operations

Operations are plain C++ functions. The first `DstCount` arguments are outputs (passed by non-const reference); the remaining arguments are inputs (const reference):

```cpp
// 1 output, 2 inputs — DstCount = 1 (default)
static void OpAdd(float& dst, const float& a, const float& b) noexcept {
    dst = a + b;
}

// Optional validation function — same signature, returns bool
static bool ValAdd(const float&, const float&, const float&) noexcept {
    return true;
}
```

Wrap the function into an `Op` object with `CreateOp`:

```cpp
// CreateOp<ExecFn, ValFn = nullptr, DstCount = 1>
static constexpr OpDescriptor g_ops[] = {
    { "Add", &CreateOp<OpAdd, ValAdd> },
    { "Mul", &CreateOp<OpMul> },
};

vm.AddInstructions(g_ops);
```

`OpImpl` uses `FuncTraits` to deduce argument counts and types automatically — no manual registration of arity or type indices.

---

## Execution Modes

| Mode flag | Behaviour |
|---|---|
| `RmFull` | Execute all non-NOP instructions in order |
| `RmFull \| RmAnalyze` | Record the producer instruction index for every stack slot (used by the optimizer) |
| `RmOptimized` | Skip instructions not marked as live by `OptimizeProgram()` |
| `RmValidate` | Call `Op::Validate()` instead of `Op::Execute()`; returns `false` on first failure |
| `RmOptimized \| RmValidate` | Combined optimized + validated pass |

```cpp
vm.Run(prog, input);                    // RmFull
vm.OptimizeProgram(prog);               // analyze + mark live instructions
vm.Run(prog, input, /*optimized=*/true); // RmOptimized (skip dead code)
vm.Validate(prog, input);               // RmFull | RmValidate
```

The optimizer performs a backward **dead-code elimination** pass: starting from stack tops, it traces producers through `RmAnalyze` metadata and marks only reachable instructions.

---

## Example EigenOps — Linear Algebra Backend

`examples/EigenOps.h` provides a ready-to-use instruction set built on [Eigen 5.0](https://eigen.tuxfamily.org). It defines `EMat<T>` (matrix) and `EVec<T>` (column vector) as `VectorLike` types that pre-allocate capacity at machine init time — no per-instruction heap allocation during execution.

**Type aliases:** `EMatF`, `EMatD`, `EMatI32`, `EVecF`, `EVecD`, …

**Supported scalar element types:** `float`, `double`, `int32_t`, `int64_t`, `uint32_t`, `uint64_t`, `BitVector` (64-bit packed).

**Instruction categories:**

| Category | Examples |
|---|---|
| Linear algebra | `MatMulF`, `MatVecMulF`, `MatTransposeF`, `MatInverseF`, `MatDetF`, `MatQRF` |
| Arithmetic | `MatAddF`, `VecSubF`, `MatElemMulF`, `VecDotF`, `MatMulScalarF` |
| Activation / math | `MatReluF`, `VecAbsF`, `MatSinF`, `VecExpF`, `MatSqrtF` |
| Bitwise (integer) | `MatBitAndI32`, `VecBitOrU64`, `MatBitXorI64` |

Register ops with `RegisterAllOps<float, int32_t>(iset, reg)`.

---

## License

MIT — see [LICENSE](LICENSE).
