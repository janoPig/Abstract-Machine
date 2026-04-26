# [WIP] Abstract Virtual Machine (AVM)

> [!WARNING]
> This project is a **prototype under construction**. Architectural details are subject to change.

A lightweight, header-only, embeddable virtual machine for executing statically-typed instruction programs in C++20.

---

## Features

- **Header-only** — drop `include/` into your project, no linking required
- **Strongly typed** — all operand types are resolved at compile time via `TypePack<Ts...>`
- **Zero RTTI** — type identity uses unique static addresses, no `typeid` or `std::type_index`
- **Multiple execution modes** — full, validated, optimized (dead-code elimination)
- **Text DSL** — compile/decompile programs from/to a human-readable format

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

---

## Memory Model

Every `SegmentImpl` maintains **one heap-allocated typed array per type** in the `TypePack`. Three segment kinds are used at runtime:

| Segment | Symbol | Description |
|---|---|---|
| Input (data) | `I[n]` | Read-only runtime inputs provided per execution |
| Constants | `C[n]` | Mutable values embedded in the `Program`, can be initialized via `CONST` block in DSL |
| Stack | `S[n]` | Output slots; grows monotonically as instructions execute |

Addresses encode segment type + element offset: `I[0]`, `C<float>[1]`, `S[3]`.  
The optional type annotation (`<float>`) is validated by `DslCompiler` against the op's declared signature.

---

## DSL Syntax & Programming

The VM uses a text-based DSL for program definition. Each instruction consists of an operation name followed by destination and source addresses.

### Address Format
Addresses are encoded as `Segment<Type>[Offset]`:
* `I<T>[n]`: Input segment (Read-only data provided at runtime).
* `C<T>[n]`: Constant segment (Pre-defined data baked into the program).
* `S<T>[n]`: Stack segment (Read-write temporary storage).

### Constant Block
You can define scalar constants (float, int) at the beginning of the program:
```text
CONST <float> [0.5, -1.0, 3.14]
CONST <int> [42, 1337]
```
These values are mapped to `C<float>[0..2]` and `C<int>[0..1]` respectively.

### Complex Example: Self-Attention (Transformer)
The following script demonstrates a functional **Self-Attention** forward pass using the `EigenOps` example backend.

```text
# Step 0: Define constants
CONST <float> [0.7071]                        # 1/sqrt(dk) where dk=2

# Step 1: Project input X to Query, Key, and Value matrices
# X: [SeqLen x Dim], Wq/Wk/Wv: [Dim x Dim]
S<EMatF>[0] = MatMulF I<EMatF>[0] I<EMatF>[1]    # Q = X * Wq
S<EMatF>[1] = MatMulF I<EMatF>[0] I<EMatF>[2]    # K = X * Wk
S<EMatF>[2] = MatMulF I<EMatF>[0] I<EMatF>[3]    # V = X * Wv

# Step 2: Calculate Attention Scores (Q * K^T)
S<EMatF>[3] = MatTransposeF S<EMatF>[1]          # K^T
S<EMatF>[4] = MatMulF S<EMatF>[0] S<EMatF>[3]    # Scores = Q * K^T

# Step 3: Scale scores by 1/sqrt(dk)
S<EMatF>[5] = MatMulScalarF S<EMatF>[4] C<float>[0]

# Step 4: Apply Softmax to get Attention Weights
S<EMatF>[6] = MatSoftmaxF S<EMatF>[5]            # Weights = Softmax(Scaled)

# Step 5: Final weighted sum (Weights * V)
S<EMatF>[7] = MatMulF S<EMatF>[6] S<EMatF>[2]    # Output [SeqLen x Dim]
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

Custom types can expose a `static constexpr const char* TypeName` for use in DSL diagnostics.

---

## Configuration

Each machine is parameterized by a `VMConfig` struct:

```cpp
struct MyConfig {
    static constexpr size_t SrcMaxArity   = 4;   // max inputs per instruction
    static constexpr size_t DstMaxArity   = 2;   // max outputs per instruction
    static constexpr size_t MaxProgramSize = 128; // fixed instruction slot count
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
static bool ValAdd(const float&, const float&, const float& b) noexcept {
    return b != 0;
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

`OpImpl` uses `FuncTraits` to deduce argument counts and types automatically.

---

## Execution Modes

| Mode flag | Behaviour |
|---|---|
| `RmFull` | Execute all non-NOP instructions in order |
| `RmFull \| RmAnalyze` | Record the producer instruction index for every stack slot (used by the optimizer) |
| `RmOptimized` | Skip instructions not marked as live by `OptimizeProgram()` |
| `RmValidate` | Call `Op::Validate()` instead of `Op::Execute()`; returns `false` on first failure |

```cpp
vm.Run(prog, input);                    // RmFull
vm.OptimizeProgram(prog);               // analyze + mark live instructions
vm.Run(prog, input, /*optimized=*/true); // RmOptimized (skip dead code)
vm.Validate(prog, input);               // RmFull | RmValidate
```

The optimizer performs a backward **dead-code elimination** pass: starting from stack tops, it traces producers through `RmAnalyze` metadata and marks only reachable instructions.

---

## Example Backends

The repository includes sample instruction sets to demonstrate extensibility. These are reference implementations and not part of the core VM logic.

### EigenOps (Linear Algebra)
Built on [Eigen 5.0](https://eigen.tuxfamily.org). Provides high-performance matrix and vector operations (`EMat<T>`, `EVec<T>`) including BLAS, decompositions, and ML activations (ReLU, Sigmoid, Softmax).

Register all ops: `RegisterAllOps<float>(iset, reg);`

### ScalarOps (Primitive Arithmetic)
Standard operations for `float`, `double`, `int32`, `int64`, and `uint64`. Covers full `<cmath>` library and bitwise logic.

Register ops: `RegisterScalarOps<float, int32_t>(iset, reg);`

---

## License

MIT — see [LICENSE](LICENSE).
