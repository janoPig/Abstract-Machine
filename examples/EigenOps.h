#pragma once
#include "../include/Utils.h"
#include "../include/OpImpl.h"
#include "../include/AbstractVM.h"
#include "../include/InstructionSet.h"
#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/LU>
#include <set>
#include <type_traits>
#include <cstdlib>

namespace EigenOps
{
	using AbstractVM::CreateOp;
	using VectorLike = AbstractVM::VectorLike;
	using InstructionSet = AbstractVM::InstructionSet;

	// 1. Type Traits & Macros

	struct BitVector {};

	template<typename T> struct EigenTypeTrait
	{
		using type = T;
	};

	template<> struct EigenTypeTrait<BitVector>
	{
		using type = uint64_t;
	};

	template<typename T> using EigenT = typename EigenTypeTrait<T>::type;

#define DECLARE_EIGEN_NAME(Prefix) \
		static constexpr const char* TypeName = \
			std::is_same_v<T, float>     ? Prefix "F" : \
			std::is_same_v<T, double>    ? Prefix "D" : \
			std::is_same_v<T, int32_t>   ? Prefix "I32" : \
			std::is_same_v<T, int64_t>   ? Prefix "I64" : \
			std::is_same_v<T, uint32_t>  ? Prefix "U32" : \
			std::is_same_v<T, uint64_t>  ? Prefix "U64" : \
			std::is_same_v<T, BitVector> ? Prefix "B" : Prefix "?";

#define OP_NAME(Name) \
		(std::is_same_v<T, float>     ? Name "F" : \
		 std::is_same_v<T, double>    ? Name "D" : \
		 std::is_same_v<T, int32_t>   ? Name "I32" : \
		 std::is_same_v<T, int64_t>   ? Name "I64" : \
		 std::is_same_v<T, uint32_t>  ? Name "U32" : \
		 std::is_same_v<T, uint64_t>  ? Name "U64" : \
		 std::is_same_v<T, BitVector> ? Name "B" : Name "?")

#define OP_ENTRY(Name, OpFunc, ValFunc) \
		{ OP_NAME(Name), &CreateOp<OpFunc<T>, ValFunc<T>> }

#define OP_ENTRY_N(Name, OpFunc, ValFunc, N) \
		{ OP_NAME(Name), &CreateOp<OpFunc<T>, ValFunc<T>, N> }

	// 2. Core Structures

	template<typename T>
	struct EBase : public VectorLike
	{
		DISALLOW_COPY_MOVE_AND_ASSIGN(EBase);
		using ET = EigenT<T>;

	public:
		EBase() = default;

		~EBase() override
		{
			if (m_data)
			{
				AlignedFree(m_data);
				m_data = nullptr;
			}
		}

		FORCE_INLINE void Init(size_t capacity) override
		{
			m_capacity = capacity;
			// For BitVector, we allocate uint64_t blocks (64 bits per block)
			size_t allocElements = std::is_same_v<T, BitVector> ? ((capacity + 63) / 64) : capacity;
			m_data = static_cast<ET*>(AlignedAlloc(allocElements * sizeof(ET), 32));
			m_rows = 0; m_cols = 0;
		}

		// NOTE: This operator is intended strictly for unit tests.
		// It provides value-based comparison with proper handling per underlying type T.
		bool operator==(const EBase& other) const noexcept
		{
			if (m_rows != other.m_rows || m_cols != other.m_cols)
			{
				return false;
			}
			if (m_rows == 0)
			{
				return true;
			}
			if constexpr (std::is_floating_point_v<T>)
			{ // Use Eigen approximate comparison for floating point types
				return view().isApprox(other.view());
			}
			else
			{ // Exact comparison for all non-floating types
				return (view().array() == other.view().array()).all();
			}
		}

		FORCE_INLINE auto size() const noexcept
		{
			return m_rows * m_cols;
		}

		FORCE_INLINE auto capacity() const noexcept
		{
			return m_capacity;
		}

		FORCE_INLINE auto rows() const noexcept
		{
			return m_rows;
		}

		FORCE_INLINE auto cols() const noexcept
		{
			return m_cols;
		}

		FORCE_INLINE void resize(auto r, auto c) noexcept
		{
			m_rows = (Eigen::Index)r;
			m_cols = (Eigen::Index)c;
		}

		FORCE_INLINE void resize(auto s) noexcept
		{
			m_rows = (Eigen::Index)s;
			m_cols = 1;
		}

		// Map mapped_rows to 64-bit blocks for BitVector, otherwise return real count
		FORCE_INLINE Eigen::Index mapped_rows() const noexcept
		{
			return std::is_same_v<T, BitVector> ? ((m_rows + 63) / 64) : m_rows;
		}

		using MatrixMap = Eigen::Map<Eigen::Matrix<ET, Eigen::Dynamic, Eigen::Dynamic>>;
		using ConstMatrixMap = Eigen::Map<const Eigen::Matrix<ET, Eigen::Dynamic, Eigen::Dynamic>>;
		using VectorMap = Eigen::Map<Eigen::Matrix<ET, Eigen::Dynamic, 1>>;
		using ConstVectorMap = Eigen::Map<const Eigen::Matrix<ET, Eigen::Dynamic, 1>>;

		FORCE_INLINE MatrixMap view() noexcept
		{
			return MatrixMap(m_data, mapped_rows(), m_cols);
		}

		FORCE_INLINE ConstMatrixMap view() const noexcept
		{
			return ConstMatrixMap(m_data, mapped_rows(), m_cols);
		}

		FORCE_INLINE VectorMap v_view() noexcept
		{
			return VectorMap(m_data, mapped_rows());
		}

		FORCE_INLINE ConstVectorMap v_view() const noexcept
		{
			return ConstVectorMap(m_data, mapped_rows());
		}

	protected:
		ET* m_data{};
		size_t m_capacity{};
		Eigen::Index m_rows{};
		Eigen::Index m_cols{};
	};

	template<typename T> struct EVec : public EBase<T>
	{
		DECLARE_EIGEN_NAME("EVec")
	};

	template<typename T> struct EMat : public EBase<T>
	{
		DECLARE_EIGEN_NAME("EMat")
	};

	// 3. Validation Functions (Specific Types)

	// Matrix Validations

	template<typename T> FORCE_INLINE bool ValMatSameShape(const EMat<T>& c, const EMat<T>& a, const EMat<T>& b) noexcept
	{
		return (a.rows() == b.rows() && a.cols() == b.cols()) && ((size_t)a.size() <= c.capacity());
	}

	template<typename T> FORCE_INLINE bool ValMatUnary(const EMat<T>& c, const EMat<T>& a) noexcept
	{
		return (size_t)a.size() <= c.capacity();
	}

	template<typename T> FORCE_INLINE bool ValMatScalar(const EMat<T>& c, const EMat<T>& a, const EigenT<T>&) noexcept
	{
		return (size_t)a.size() <= c.capacity();
	}

	// Vector Validations

	template<typename T> FORCE_INLINE bool ValVecSameShape(const EVec<T>& c, const EVec<T>& a, const EVec<T>& b) noexcept
	{
		return (a.rows() == b.rows()) && ((size_t)a.size() <= c.capacity());
	}

	template<typename T> FORCE_INLINE bool ValVecUnary(const EVec<T>& c, const EVec<T>& a) noexcept
	{
		return (size_t)a.size() <= c.capacity();
	}

	template<typename T> FORCE_INLINE bool ValVecScalar(const EVec<T>& c, const EVec<T>& a, const EigenT<T>&) noexcept {
		return (size_t)a.size() <= c.capacity();
	}

	// Specialized Linear Algebra Validations

	template<typename T> FORCE_INLINE bool ValMatMul(const EMat<T>& c, const EMat<T>& a, const EMat<T>& b) noexcept
	{
		return (a.cols() == b.rows()) && ((size_t)(a.rows() * b.cols()) <= c.capacity());
	}

	template<typename T> FORCE_INLINE bool ValMatVec(const EVec<T>& c, const EMat<T>& a, const EVec<T>& b) noexcept
	{
		return (a.cols() == b.rows()) && ((size_t)a.rows() <= c.capacity());
	}

	template<typename T> FORCE_INLINE bool ValSquare(const EMat<T>& c, const EMat<T>& a) noexcept
	{
		return a.rows() == a.cols() && (size_t)a.size() <= c.capacity();
	}

	template<typename T> FORCE_INLINE bool ValSquareScalar(const EigenT<T>&, const EMat<T>& a) noexcept
	{
		return a.rows() == a.cols();
	}

	template<typename T> FORCE_INLINE bool ValQR(const EMat<T>& q, const EMat<T>& r, const EMat<T>& a) noexcept
	{
		return (size_t)(a.rows() * a.rows()) <= q.capacity() && (size_t)(a.rows() * a.cols()) <= r.capacity();
	}

	template<typename T> FORCE_INLINE bool ValVecDot(const EigenT<T>&, const EVec<T>& a, const EVec<T>& b) noexcept
	{
		return a.rows() == b.rows();
	}

	// Protected-division validations: reject if any element of b is zero

	template<typename T> FORCE_INLINE bool ValMatPDiv(const EMat<T>& c, const EMat<T>& a, const EMat<T>& b) noexcept
	{
		return ValMatSameShape(c, a, b) && !(b.view().array() == EigenT<T>(0)).any();
	}

	template<typename T> FORCE_INLINE bool ValVecPDiv(const EVec<T>& c, const EVec<T>& a, const EVec<T>& b) noexcept
	{
		return ValVecSameShape(c, a, b) && !(b.v_view().array() == EigenT<T>(0)).any();
	}

	template<typename T> FORCE_INLINE bool ValMatRandom(const EMat<T>& c, const EMat<T>& a, const EigenT<T>&, const EigenT<T>&) noexcept
	{
		return (size_t)a.size() <= c.capacity();
	}

	template<typename T> FORCE_INLINE bool ValVecRandom(const EVec<T>& c, const EVec<T>& a, const EigenT<T>&, const EigenT<T>&) noexcept
	{
		return (size_t)a.size() <= c.capacity();
	}

	template<typename T> FORCE_INLINE bool ValMatUnaryMath(const EMat<T>& c, const EMat<T>& a) noexcept
	{
		return (size_t)a.size() <= c.capacity();
	}

	template<typename T> FORCE_INLINE bool ValVecUnaryMath(const EVec<T>& c, const EVec<T>& a) noexcept
	{
		return (size_t)a.size() <= c.capacity();
	}

	// 4. Execution Functions (Specific Types)

	// Linear Algebra

	template<typename T> FORCE_INLINE void OpMatMul(EMat<T>& c, const EMat<T>& a, const EMat<T>& b) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector>)
		{
			c.resize(a.rows(), b.cols());
			c.view().noalias() = a.view() * b.view();
		}
	}

	template<typename T> FORCE_INLINE void OpMatVecMul(EVec<T>& c, const EMat<T>& a, const EVec<T>& b) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector>)
		{
			c.resize(a.rows());
			c.v_view().noalias() = a.view() * b.v_view();
		}
	}

	template<typename T> FORCE_INLINE void OpMatTranspose(EMat<T>& c, const EMat<T>& a) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector>)
		{
			c.resize(a.cols(), a.rows());
			c.view() = a.view().transpose();
		}
	}

	template<typename T> FORCE_INLINE void OpMatInverse(EMat<T>& c, const EMat<T>& a) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = a.view().inverse();
		}
	}

	template<typename T> FORCE_INLINE void OpDet(EigenT<T>& out, const EMat<T>& a) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			out = a.view().determinant();
		}
	}

	template<typename T> FORCE_INLINE void OpTrace(EigenT<T>& out, const EMat<T>& a) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector>)
		{
			out = a.view().trace();
		}
	}

	template<typename T> FORCE_INLINE void OpVecDot(EigenT<T>& out, const EVec<T>& a, const EVec<T>& b) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector>)
		{
			out = a.v_view().dot(b.v_view());
		}
	}

	template<typename T> FORCE_INLINE void OpMatQR(EMat<T>& q, EMat<T>& r, const EMat<T>& a) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			Eigen::HouseholderQR<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> qr(a.view());
			q.resize(a.rows(), a.rows());
			r.resize(a.rows(), a.cols());
			q.view() = qr.householderQ();
			r.view() = qr.matrixQR().template triangularView<Eigen::Upper>();
		}
	}

	//Arithmetic Matrix/Vector Ops

	template<typename T> FORCE_INLINE void OpMatAdd(EMat<T>& c, const EMat<T>& a, const EMat<T>& b) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = a.view() + b.view();
		}
	}

	template<typename T> FORCE_INLINE void OpVecAdd(EVec<T>& c, const EVec<T>& a, const EVec<T>& b) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector>)
		{
			c.resize(a.rows());
			c.v_view() = a.v_view() + b.v_view();
		}
	}

	template<typename T> FORCE_INLINE void OpMatSub(EMat<T>& c, const EMat<T>& a, const EMat<T>& b) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = a.view() - b.view();
		}
	}

	template<typename T> FORCE_INLINE void OpVecSub(EVec<T>& c, const EVec<T>& a, const EVec<T>& b) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector>)
		{
			c.resize(a.rows());
			c.v_view() = a.v_view() - b.v_view();
		}
	}

	template<typename T> FORCE_INLINE void OpMatElemMul(EMat<T>& c, const EMat<T>& a, const EMat<T>& b) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = a.view().cwiseProduct(b.view());
		}
	}

	template<typename T> FORCE_INLINE void OpVecElemMul(EVec<T>& c, const EVec<T>& a, const EVec<T>& b) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector>)
		{
			c.resize(a.rows());
			c.v_view() = a.v_view().cwiseProduct(b.v_view());
		}
	}

	template<typename T> FORCE_INLINE void OpMatElemDiv(EMat<T>& c, const EMat<T>& a, const EMat<T>& b) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = a.view().cwiseQuotient(b.view());
		}
	}

	template<typename T> FORCE_INLINE void OpVecElemDiv(EVec<T>& c, const EVec<T>& a, const EVec<T>& b) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows());
			c.v_view() = a.v_view().cwiseQuotient(b.v_view());
		}
	}

	// Scalar Operations

	template<typename T> FORCE_INLINE void OpMatAddScalar(EMat<T>& c, const EMat<T>& a, const EigenT<T>& s) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = a.view().array() + s;
		}
	}

	template<typename T> FORCE_INLINE void OpVecAddScalar(EVec<T>& c, const EVec<T>& a, const EigenT<T>& s) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector>)
		{
			c.resize(a.rows());
			c.v_view() = a.v_view().array() + s;
		}
	}

	template<typename T> FORCE_INLINE void OpMatSubScalar(EMat<T>& c, const EMat<T>& a, const EigenT<T>& s) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = a.view().array() - s;
		}
	}

	template<typename T> FORCE_INLINE void OpVecSubScalar(EVec<T>& c, const EVec<T>& a, const EigenT<T>& s) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector>)
		{
			c.resize(a.rows());
			c.v_view() = a.v_view().array() - s;
		}
	}

	template<typename T> FORCE_INLINE void OpMatMulScalar(EMat<T>& c, const EMat<T>& a, const EigenT<T>& s) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = a.view() * s;
		}
	}

	template<typename T> FORCE_INLINE void OpVecMulScalar(EVec<T>& c, const EVec<T>& a, const EigenT<T>& s) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector>)
		{
			c.resize(a.rows());
			c.v_view() = a.v_view() * s;
		}
	}

	template<typename T> FORCE_INLINE void OpMatDivScalar(EMat<T>& c, const EMat<T>& a, const EigenT<T>& s) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = a.view() / s;
		}
	}

	template<typename T> FORCE_INLINE void OpVecDivScalar(EVec<T>& c, const EVec<T>& a, const EigenT<T>& s) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows());
			c.v_view() = a.v_view() / s;
		}
	}

	// Math & Bitwise (Separated by Mat/Vec)

	template<typename T> FORCE_INLINE void OpMatAbs(EMat<T>& c, const EMat<T>& a) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector> && std::is_signed_v<EigenT<T>>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = a.view().cwiseAbs();
		}
	}

	template<typename T> FORCE_INLINE void OpVecAbs(EVec<T>& c, const EVec<T>& a) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector> && std::is_signed_v<EigenT<T>>)
		{
			c.resize(a.rows());
			c.v_view() = a.v_view().cwiseAbs();
		}
	}

	template<typename T> FORCE_INLINE void OpMatRelu(EMat<T>& c, const EMat<T>& a) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector> && std::is_signed_v<EigenT<T>>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = a.view().cwiseMax(static_cast<EigenT<T>>(0));
		}
	}

	template<typename T> FORCE_INLINE void OpVecRelu(EVec<T>& c, const EVec<T>& a) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector> && std::is_signed_v<EigenT<T>>)
		{
			c.resize(a.rows());
			c.v_view() = a.v_view().cwiseMax(static_cast<EigenT<T>>(0));
		}
	}

#define GEN_UNARY_MATH_OPS(Name, EigenCall) \
	template<typename T> FORCE_INLINE void OpMat##Name(EMat<T>& c, const EMat<T>& a) noexcept \
	{ \
		if constexpr (std::is_floating_point_v<T>) \
		{ \
			c.resize(a.rows(), a.cols()); \
			c.view() = a.view().array().EigenCall().matrix(); \
		} \
	} \
	template<typename T> FORCE_INLINE void OpVec##Name(EVec<T>& c, const EVec<T>& a) noexcept \
	{ \
		if constexpr (std::is_floating_point_v<T>) \
		{ \
			c.resize(a.rows()); \
			c.v_view() = a.v_view().array().EigenCall().matrix(); \
		} \
	}

	GEN_UNARY_MATH_OPS(Sin, sin)
		GEN_UNARY_MATH_OPS(Cos, cos)
		GEN_UNARY_MATH_OPS(Tan, tan)
		GEN_UNARY_MATH_OPS(Asin, asin)
		GEN_UNARY_MATH_OPS(Acos, acos)
		GEN_UNARY_MATH_OPS(Atan, atan)
		GEN_UNARY_MATH_OPS(Sinh, sinh)
		GEN_UNARY_MATH_OPS(Cosh, cosh)
		GEN_UNARY_MATH_OPS(Tanh, tanh)
		GEN_UNARY_MATH_OPS(Exp, exp)
		GEN_UNARY_MATH_OPS(Log, log)
		GEN_UNARY_MATH_OPS(Sqrt, sqrt)
		GEN_UNARY_MATH_OPS(Floor, floor)
		GEN_UNARY_MATH_OPS(Ceil, ceil)

#define GEN_BITWISE_OPS(Name, Op)\
	template<typename T> FORCE_INLINE void OpMat##Name(EMat<T>& c, const EMat<T>& a, const EMat<T>& b) noexcept\
	{\
		if constexpr (std::is_integral_v<EigenT<T>>)\
		{\
			c.resize(a.rows(), a.cols());\
			c.view() = (a.view().array() Op b.view().array()).matrix();\
		}\
	} \
	template<typename T> FORCE_INLINE void OpVec##Name(EVec<T>& c, const EVec<T>& a, const EVec<T>& b) noexcept\
	{\
		if constexpr (std::is_integral_v<EigenT<T>>)\
		{\
			c.resize(a.rows());\
			c.v_view() = (a.v_view().array() Op b.v_view().array()).matrix();\
		}\
	}

		GEN_BITWISE_OPS(BitAnd, &)
		GEN_BITWISE_OPS(BitOr, | )
		GEN_BITWISE_OPS(BitXor, ^)

		template<typename T> FORCE_INLINE void OpMatBitNot(EMat<T>& c, const EMat<T>& a) noexcept
	{
		if constexpr (std::is_integral_v<EigenT<T>>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = (~a.view().array()).matrix();
		}
	}

	template<typename T> FORCE_INLINE void OpVecBitNot(EVec<T>& c, const EVec<T>& a) noexcept
	{
		if constexpr (std::is_integral_v<EigenT<T>>)
		{
			c.resize(a.rows());
			c.v_view() = (~a.v_view().array()).matrix();
		}
	}

	// --- New ops ---

	// Nop: copy input to output unchanged, preserving shape

	template<typename T> FORCE_INLINE void OpMatNop(EMat<T>& c, const EMat<T>& a) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = a.view();
		}
	}

	template<typename T> FORCE_INLINE void OpVecNop(EVec<T>& c, const EVec<T>& a) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector>)
		{
			c.resize(a.rows());
			c.v_view() = a.v_view();
		}
	}

	// Sq2: element-wise square (a * a)

	template<typename T> FORCE_INLINE void OpMatSq2(EMat<T>& c, const EMat<T>& a) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector> && std::is_arithmetic_v<EigenT<T>>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = a.view().cwiseProduct(a.view());
		}
	}

	template<typename T> FORCE_INLINE void OpVecSq2(EVec<T>& c, const EVec<T>& a) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector> && std::is_arithmetic_v<EigenT<T>>)
		{
			c.resize(a.rows());
			c.v_view() = a.v_view().cwiseProduct(a.v_view());
		}
	}

	// Inv: element-wise reciprocal (1 / a), float only

	template<typename T> FORCE_INLINE void OpMatInv(EMat<T>& c, const EMat<T>& a) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = a.view().cwiseInverse();
		}
	}

	template<typename T> FORCE_INLINE void OpVecInv(EVec<T>& c, const EVec<T>& a) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows());
			c.v_view() = a.v_view().cwiseInverse();
		}
	}

	// Cbrt: element-wise cube root, float only
	// Eigen has no built-in cbrt array op; use unaryExpr with std::cbrt

	template<typename T> FORCE_INLINE void OpMatCbrt(EMat<T>& c, const EMat<T>& a) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = a.view().array().unaryExpr([](T x) noexcept { return std::cbrt(x); }).matrix();
		}
	}

	template<typename T> FORCE_INLINE void OpVecCbrt(EVec<T>& c, const EVec<T>& a) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows());
			c.v_view() = a.v_view().array().unaryExpr([](T x) noexcept { return std::cbrt(x); }).matrix();
		}
	}

	// PDiv: protected element-wise division; validator guarantees no zero in b

	template<typename T> FORCE_INLINE void OpMatPDiv(EMat<T>& c, const EMat<T>& a, const EMat<T>& b) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = a.view().cwiseQuotient(b.view());
		}
	}

	template<typename T> FORCE_INLINE void OpVecPDiv(EVec<T>& c, const EVec<T>& a, const EVec<T>& b) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows());
			c.v_view() = a.v_view().cwiseQuotient(b.v_view());
		}
	}

	// Pow: element-wise power, float only

	template<typename T> FORCE_INLINE void OpMatPow(EMat<T>& c, const EMat<T>& a, const EMat<T>& b) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = a.view().array().pow(b.view().array()).matrix();
		}
	}

	template<typename T> FORCE_INLINE void OpVecPow(EVec<T>& c, const EVec<T>& a, const EVec<T>& b) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows());
			c.v_view() = a.v_view().array().pow(b.v_view().array()).matrix();
		}
	}

	// Aq: element-wise atan2(a, b), float only

	template<typename T> FORCE_INLINE void OpMatAq(EMat<T>& c, const EMat<T>& a, const EMat<T>& b) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = a.view().array().binaryExpr(b.view().array(),
				[](T x, T y) noexcept { return std::atan2(x, y); }).matrix();
		}
	}

	template<typename T> FORCE_INLINE void OpVecAq(EVec<T>& c, const EVec<T>& a, const EVec<T>& b) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows());
			c.v_view() = a.v_view().array().binaryExpr(b.v_view().array(),
				[](T x, T y) noexcept { return std::atan2(x, y); }).matrix();
		}
	}

	// Max / Min: element-wise, float and signed integer types

	template<typename T> FORCE_INLINE void OpMatMax(EMat<T>& c, const EMat<T>& a, const EMat<T>& b) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector> && std::is_arithmetic_v<EigenT<T>>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = a.view().cwiseMax(b.view());
		}
	}

	template<typename T> FORCE_INLINE void OpVecMax(EVec<T>& c, const EVec<T>& a, const EVec<T>& b) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector> && std::is_arithmetic_v<EigenT<T>>)
		{
			c.resize(a.rows());
			c.v_view() = a.v_view().cwiseMax(b.v_view());
		}
	}

	template<typename T> FORCE_INLINE void OpMatMin(EMat<T>& c, const EMat<T>& a, const EMat<T>& b) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector> && std::is_arithmetic_v<EigenT<T>>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = a.view().cwiseMin(b.view());
		}
	}

	template<typename T> FORCE_INLINE void OpVecMin(EVec<T>& c, const EVec<T>& a, const EVec<T>& b) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector> && std::is_arithmetic_v<EigenT<T>>)
		{
			c.resize(a.rows());
			c.v_view() = a.v_view().cwiseMin(b.v_view());
		}
	}

	// Comparison ops: result is same shape, values 0 or 1, float only
	// (integer comparison would require a separate result type; float matches the existing type system)

#define GEN_CMP_OPS(Name, Op) \
	template<typename T> FORCE_INLINE void OpMat##Name(EMat<T>& c, const EMat<T>& a, const EMat<T>& b) noexcept \
	{ \
		if constexpr (std::is_floating_point_v<T>) \
		{ \
			c.resize(a.rows(), a.cols()); \
			c.view() = (a.view().array() Op b.view().array()).template cast<T>().matrix(); \
		} \
	} \
	template<typename T> FORCE_INLINE void OpVec##Name(EVec<T>& c, const EVec<T>& a, const EVec<T>& b) noexcept \
	{ \
		if constexpr (std::is_floating_point_v<T>) \
		{ \
			c.resize(a.rows()); \
			c.v_view() = (a.v_view().array() Op b.v_view().array()).template cast<T>().matrix(); \
		} \
	}

	GEN_CMP_OPS(Lt, < )
		GEN_CMP_OPS(Gt, > )
		GEN_CMP_OPS(Lte, <= )
		GEN_CMP_OPS(Gte, >= )

	// --- Generation ops ---

	template<typename T> FORCE_INLINE void OpMatIdentity(EMat<T>& c, const EMat<T>& a) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector>)
		{
			c.resize(a.rows(), a.cols());
			c.view().setIdentity();
		}
	}

	template<typename T> FORCE_INLINE void OpMatConstant(EMat<T>& c, const EMat<T>& a, const EigenT<T>& val) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector>)
		{
			c.resize(a.rows(), a.cols());
			c.view().setConstant(val);
		}
	}

	template<typename T> FORCE_INLINE void OpVecConstant(EVec<T>& c, const EVec<T>& a, const EigenT<T>& val) noexcept
	{
		if constexpr (!std::is_same_v<T, BitVector>)
		{
			c.resize(a.rows());
			c.v_view().setConstant(val);
		}
	}

	template<typename T> FORCE_INLINE void OpMatRandom(EMat<T>& c, const EMat<T>& a, const EigenT<T>& min, const EigenT<T>& max) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows(), a.cols());
			c.view().setRandom();
			c.view() = ((c.view().array() + T(1)) * (max - min) / T(2) + min).matrix();
		}
		else if constexpr (std::is_integral_v<EigenT<T>> && !std::is_same_v<T, BitVector>)
		{
			c.resize(a.rows(), a.cols());
			for (Eigen::Index j = 0; j < a.cols(); ++j)
				for (Eigen::Index i = 0; i < a.rows(); ++i)
					c.view()(i, j) = min + static_cast<EigenT<T>>(std::rand() % (static_cast<long long>(max) - min + 1));
		}
	}

	template<typename T> FORCE_INLINE void OpVecRandom(EVec<T>& c, const EVec<T>& a, const EigenT<T>& min, const EigenT<T>& max) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows());
			c.v_view().setRandom();
			c.v_view() = ((c.v_view().array() + T(1)) * (max - min) / T(2) + min).matrix();
		}
		else if constexpr (std::is_integral_v<EigenT<T>> && !std::is_same_v<T, BitVector>)
		{
			c.resize(a.rows());
			for (Eigen::Index i = 0; i < a.rows(); ++i)
				c.v_view()(i) = min + static_cast<EigenT<T>>(std::rand() % (static_cast<long long>(max) - min + 1));
		}
	}

	// --- ML Activations ---

	template<typename T> FORCE_INLINE void OpMatSigmoid(EMat<T>& c, const EMat<T>& a) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = (T(1) / (T(1) + (-a.view().array()).exp())).matrix();
		}
	}

	template<typename T> FORCE_INLINE void OpVecSigmoid(EVec<T>& c, const EVec<T>& a) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows());
			c.v_view() = (T(1) / (T(1) + (-a.v_view().array()).exp())).matrix();
		}
	}

	template<typename T> FORCE_INLINE void OpMatSoftmax(EMat<T>& c, const EMat<T>& a) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows(), a.cols());
			auto m = a.view().array();
			auto max_coeffs = m.rowwise().maxCoeff();
			auto exp_m = (m.colwise() - max_coeffs).exp();
			auto sums = exp_m.rowwise().sum();
			c.view() = (exp_m.colwise() / sums).matrix();
		}
	}

	template<typename T> FORCE_INLINE void OpVecSoftmax(EVec<T>& c, const EVec<T>& a) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows());
			auto v = a.v_view().array();
			auto exp_v = (v - v.maxCoeff()).exp();
			c.v_view() = (exp_v / exp_v.sum()).matrix();
		}
	}

		// 5. Instruction Set Providers

		template<typename T>
	struct MatOpProvider
	{
		[[nodiscard]] static bool AddTo(InstructionSet& iset, const AbstractVM::TypeRegister& reg, const std::set<const char*>* selection = nullptr)
		{
			bool ret = true;
			if constexpr (!std::is_same_v<T, BitVector>)
			{
				static const AbstractVM::OpDescriptor baseOps[] = {
					OP_ENTRY("MatMul",       OpMatMul,       ValMatMul),
					OP_ENTRY("MatVecMul",    OpMatVecMul,    ValMatVec),
					OP_ENTRY("MatTranspose", OpMatTranspose, ValMatUnary),
					OP_ENTRY("MatAdd",       OpMatAdd,       ValMatSameShape),
					OP_ENTRY("MatSub",       OpMatSub,       ValMatSameShape),
					OP_ENTRY("MatElemMul",   OpMatElemMul,   ValMatSameShape),
					OP_ENTRY("MatAddScalar", OpMatAddScalar, ValMatScalar),
					OP_ENTRY("MatSubScalar", OpMatSubScalar, ValMatScalar),
					OP_ENTRY("MatMulScalar", OpMatMulScalar, ValMatScalar),
					OP_ENTRY("Trace",        OpTrace,        ValSquareScalar),
					OP_ENTRY("MatNop",       OpMatNop,       ValMatUnary),
					OP_ENTRY("MatSq2",       OpMatSq2,       ValMatUnary),
					OP_ENTRY("MatMax",       OpMatMax,       ValMatSameShape),
					OP_ENTRY("MatMin",       OpMatMin,       ValMatSameShape),
					OP_ENTRY("MatIdentity",  OpMatIdentity,  ValSquare),
					OP_ENTRY("MatConstant",  OpMatConstant,  ValMatScalar),
					OP_ENTRY("MatRandom",    OpMatRandom,    ValMatRandom),
				};
				ret &= iset.Add(baseOps, reg, selection);
			}

			if constexpr (std::is_integral_v<EigenT<T>>)
			{
				static const AbstractVM::OpDescriptor bitOps[] = {
					OP_ENTRY("MatBitAnd", OpMatBitAnd, ValMatSameShape),
					OP_ENTRY("MatBitOr",  OpMatBitOr,  ValMatSameShape),
					OP_ENTRY("MatBitXor", OpMatBitXor, ValMatSameShape),
					OP_ENTRY("MatBitNot", OpMatBitNot, ValMatUnary),
				};
				ret &= iset.Add(bitOps, reg, selection);
			}

			if constexpr (std::is_floating_point_v<T> || std::is_signed_v<EigenT<T>>)
			{
				static const AbstractVM::OpDescriptor signOps[] = {
					OP_ENTRY("MatAbs",  OpMatAbs,  ValMatUnary),
					OP_ENTRY("MatRelu", OpMatRelu, ValMatUnary),
				};
				ret &= iset.Add(signOps, reg, selection);
			}

			if constexpr (std::is_floating_point_v<T>)
			{
				static const AbstractVM::OpDescriptor floatOps[] = {
					OP_ENTRY("MatElemDiv",   OpMatElemDiv,   ValMatSameShape),
					OP_ENTRY("MatPDiv",      OpMatPDiv,      ValMatPDiv),
					OP_ENTRY("MatDivScalar", OpMatDivScalar, ValMatScalar),
					OP_ENTRY("MatSin",       OpMatSin,       ValMatUnary),
					OP_ENTRY("MatCos",       OpMatCos,       ValMatUnary),
					OP_ENTRY("MatTan",       OpMatTan,       ValMatUnary),
					OP_ENTRY("MatAsin",      OpMatAsin,      ValMatUnary),
					OP_ENTRY("MatAcos",      OpMatAcos,      ValMatUnary),
					OP_ENTRY("MatAtan",      OpMatAtan,      ValMatUnary),
					OP_ENTRY("MatSinh",      OpMatSinh,      ValMatUnary),
					OP_ENTRY("MatCosh",      OpMatCosh,      ValMatUnary),
					OP_ENTRY("MatTanh",      OpMatTanh,      ValMatUnary),
					OP_ENTRY("MatExp",       OpMatExp,       ValMatUnary),
					OP_ENTRY("MatLog",       OpMatLog,       ValMatUnary),
					OP_ENTRY("MatSqrt",      OpMatSqrt,      ValMatUnary),
					OP_ENTRY("MatCbrt",      OpMatCbrt,      ValMatUnary),
					OP_ENTRY("MatFloor",     OpMatFloor,     ValMatUnary),
					OP_ENTRY("MatCeil",      OpMatCeil,      ValMatUnary),
					OP_ENTRY("MatInverse",   OpMatInverse,   ValSquare),
					OP_ENTRY("MatMinv",      OpMatInverse,   ValSquare),
					OP_ENTRY("MatDet",       OpDet,          ValSquareScalar),
					OP_ENTRY("MatInv",       OpMatInv,       ValMatUnary),
					OP_ENTRY("MatPow",       OpMatPow,       ValMatSameShape),
					OP_ENTRY("MatAq",        OpMatAq,        ValMatSameShape),
					OP_ENTRY("MatLt",        OpMatLt,        ValMatSameShape),
					OP_ENTRY("MatGt",        OpMatGt,        ValMatSameShape),
					OP_ENTRY("MatLte",       OpMatLte,       ValMatSameShape),
					OP_ENTRY("MatGte",       OpMatGte,       ValMatSameShape),
					OP_ENTRY("MatSigmoid",   OpMatSigmoid,   ValMatUnaryMath),
					OP_ENTRY("MatSoftmax",   OpMatSoftmax,   ValMatUnaryMath),
					OP_ENTRY_N("MatQR",      OpMatQR,        ValQR, 2),
				};
				ret &= iset.Add(floatOps, reg, selection);
			}
			return ret;
		}
	};

	template<typename T>
	struct VecOpProvider
	{
		[[nodiscard]] static bool AddTo(InstructionSet& iset, const AbstractVM::TypeRegister& reg, const std::set<const char*>* selection = nullptr)
		{
			bool ret = true;
			if constexpr (!std::is_same_v<T, BitVector>)
			{
				static const AbstractVM::OpDescriptor baseOps[] = {
					OP_ENTRY("VecAdd",       OpVecAdd,       ValVecSameShape),
					OP_ENTRY("VecSub",       OpVecSub,       ValVecSameShape),
					OP_ENTRY("VecElemMul",   OpVecElemMul,   ValVecSameShape),
					OP_ENTRY("VecDot",       OpVecDot,       ValVecDot),
					OP_ENTRY("VecAddScalar", OpVecAddScalar, ValVecScalar),
					OP_ENTRY("VecSubScalar", OpVecSubScalar, ValVecScalar),
					OP_ENTRY("VecMulScalar", OpVecMulScalar, ValVecScalar),
					OP_ENTRY("VecNop",       OpVecNop,       ValVecUnary),
					OP_ENTRY("VecSq2",       OpVecSq2,       ValVecUnary),
					OP_ENTRY("VecMax",       OpVecMax,       ValVecSameShape),
					OP_ENTRY("VecMin",       OpVecMin,       ValVecSameShape),
					OP_ENTRY("VecConstant",  OpVecConstant,  ValVecScalar),
					OP_ENTRY("VecRandom",    OpVecRandom,    ValVecRandom),
				};
				ret &= iset.Add(baseOps, reg, selection);
			}

			if constexpr (std::is_integral_v<EigenT<T>>)
			{
				static const AbstractVM::OpDescriptor bitOps[] = {
					OP_ENTRY("VecBitAnd", OpVecBitAnd, ValVecSameShape),
					OP_ENTRY("VecBitOr",  OpVecBitOr,  ValVecSameShape),
					OP_ENTRY("VecBitXor", OpVecBitXor, ValVecSameShape),
					OP_ENTRY("VecBitNot", OpVecBitNot, ValVecUnary),
				};
				ret &= iset.Add(bitOps, reg, selection);
			}

			if constexpr (std::is_floating_point_v<T> || std::is_signed_v<EigenT<T>>)
			{
				static const AbstractVM::OpDescriptor signOps[] = {
					OP_ENTRY("VecAbs",  OpVecAbs,  ValVecUnary),
					OP_ENTRY("VecRelu", OpVecRelu, ValVecUnary),
				};
				ret &= iset.Add(signOps, reg, selection);
			}

			if constexpr (std::is_floating_point_v<T>)
			{
				static const AbstractVM::OpDescriptor floatOps[] = {
					OP_ENTRY("VecElemDiv",   OpVecElemDiv,   ValVecSameShape),
					OP_ENTRY("VecPDiv",      OpVecPDiv,      ValVecPDiv),
					OP_ENTRY("VecDivScalar", OpVecDivScalar, ValVecScalar),
					OP_ENTRY("VecSin",       OpVecSin,       ValVecUnary),
					OP_ENTRY("VecCos",       OpVecCos,       ValVecUnary),
					OP_ENTRY("VecTan",       OpVecTan,       ValVecUnary),
					OP_ENTRY("VecAsin",      OpVecAsin,      ValVecUnary),
					OP_ENTRY("VecAcos",      OpVecAcos,      ValVecUnary),
					OP_ENTRY("VecAtan",      OpVecAtan,      ValVecUnary),
					OP_ENTRY("VecSinh",      OpVecSinh,      ValVecUnary),
					OP_ENTRY("VecCosh",      OpVecCosh,      ValVecUnary),
					OP_ENTRY("VecTanh",      OpVecTanh,      ValVecUnary),
					OP_ENTRY("VecExp",       OpVecExp,       ValVecUnary),
					OP_ENTRY("VecLog",       OpVecLog,       ValVecUnary),
					OP_ENTRY("VecSqrt",      OpVecSqrt,      ValVecUnary),
					OP_ENTRY("VecCbrt",      OpVecCbrt,      ValVecUnary),
					OP_ENTRY("VecFloor",     OpVecFloor,     ValVecUnary),
					OP_ENTRY("VecCeil",      OpVecCeil,      ValVecUnary),
					OP_ENTRY("VecInv",       OpVecInv,       ValVecUnary),
					OP_ENTRY("VecPow",       OpVecPow,       ValVecSameShape),
					OP_ENTRY("VecAq",        OpVecAq,        ValVecSameShape),
					OP_ENTRY("VecLt",        OpVecLt,        ValVecSameShape),
					OP_ENTRY("VecGt",        OpVecGt,        ValVecSameShape),
					OP_ENTRY("VecLte",       OpVecLte,       ValVecSameShape),
					OP_ENTRY("VecGte",       OpVecGte,       ValVecSameShape),
					OP_ENTRY("VecSigmoid",   OpVecSigmoid,   ValVecUnaryMath),
					OP_ENTRY("VecSoftmax",   OpVecSoftmax,   ValVecUnaryMath),
				};
				ret &= iset.Add(floatOps, reg, selection);
			}
			return ret;
		}
	};

	// 6. Fuzzy Op Execution Functions
	// Dyadic operators based on a Hyperbolic Paraboloid.
	// Inputs are assumed to be in [0, 1]. All ops are float-only.

	// f_and(a, b) = a * b

	template<typename T> FORCE_INLINE void OpMatFAnd(EMat<T>& c, const EMat<T>& a, const EMat<T>& b) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = a.view().cwiseProduct(b.view());
		}
	}

	template<typename T> FORCE_INLINE void OpVecFAnd(EVec<T>& c, const EVec<T>& a, const EVec<T>& b) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows());
			c.v_view() = a.v_view().cwiseProduct(b.v_view());
		}
	}

	// f_or(a, b) = a + b - a * b

	template<typename T> FORCE_INLINE void OpMatFOr(EMat<T>& c, const EMat<T>& a, const EMat<T>& b) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = (a.view().array() + b.view().array() - a.view().array() * b.view().array()).matrix();
		}
	}

	template<typename T> FORCE_INLINE void OpVecFOr(EVec<T>& c, const EVec<T>& a, const EVec<T>& b) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows());
			c.v_view() = (a.v_view().array() + b.v_view().array() - a.v_view().array() * b.v_view().array()).matrix();
		}
	}

	// f_xor(a, b) = a + b - 2 * a * b

	template<typename T> FORCE_INLINE void OpMatFXor(EMat<T>& c, const EMat<T>& a, const EMat<T>& b) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = (a.view().array() + b.view().array() - T(2) * a.view().array() * b.view().array()).matrix();
		}
	}

	template<typename T> FORCE_INLINE void OpVecFXor(EVec<T>& c, const EVec<T>& a, const EVec<T>& b) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows());
			c.v_view() = (a.v_view().array() + b.v_view().array() - T(2) * a.v_view().array() * b.v_view().array()).matrix();
		}
	}

	// f_impl(a, b) = 1 - a + a * b

	template<typename T> FORCE_INLINE void OpMatFImpl(EMat<T>& c, const EMat<T>& a, const EMat<T>& b) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = (T(1) - a.view().array() + a.view().array() * b.view().array()).matrix();
		}
	}

	template<typename T> FORCE_INLINE void OpVecFImpl(EVec<T>& c, const EVec<T>& a, const EVec<T>& b) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows());
			c.v_view() = (T(1) - a.v_view().array() + a.v_view().array() * b.v_view().array()).matrix();
		}
	}

	// f_not(a) = 1 - a

	template<typename T> FORCE_INLINE void OpMatFNot(EMat<T>& c, const EMat<T>& a) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = (T(1) - a.view().array()).matrix();
		}
	}

	template<typename T> FORCE_INLINE void OpVecFNot(EVec<T>& c, const EVec<T>& a) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows());
			c.v_view() = (T(1) - a.v_view().array()).matrix();
		}
	}

	// f_nand(a, b) = 1 - a * b

	// f_nand(a, b) = 1 - a * b

	template<typename T> FORCE_INLINE void OpMatFNand(EMat<T>& c, const EMat<T>& a, const EMat<T>& b) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = (T(1) - a.view().array() * b.view().array()).matrix();
		}
	}

	template<typename T> FORCE_INLINE void OpVecFNand(EVec<T>& c, const EVec<T>& a, const EVec<T>& b) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows());
			c.v_view() = (T(1) - a.v_view().array() * b.v_view().array()).matrix();
		}
	}

	// f_nor(a, b) = 1 - a - b + a * b

	template<typename T> FORCE_INLINE void OpMatFNor(EMat<T>& c, const EMat<T>& a, const EMat<T>& b) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = (T(1) - a.view().array() - b.view().array() + a.view().array() * b.view().array()).matrix();
		}
	}

	template<typename T> FORCE_INLINE void OpVecFNor(EVec<T>& c, const EVec<T>& a, const EVec<T>& b) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows());
			c.v_view() = (T(1) - a.v_view().array() - b.v_view().array() + a.v_view().array() * b.v_view().array()).matrix();
		}
	}

	// f_nxor(a, b) = 1 - a - b + 2 * a * b

	template<typename T> FORCE_INLINE void OpMatFNxor(EMat<T>& c, const EMat<T>& a, const EMat<T>& b) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = (T(1) - a.view().array() - b.view().array() + T(2) * a.view().array() * b.view().array()).matrix();
		}
	}

	template<typename T> FORCE_INLINE void OpVecFNxor(EVec<T>& c, const EVec<T>& a, const EVec<T>& b) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows());
			c.v_view() = (T(1) - a.v_view().array() - b.v_view().array() + T(2) * a.v_view().array() * b.v_view().array()).matrix();
		}
	}

	// f_nimpl(a, b) = a - a * b

	template<typename T> FORCE_INLINE void OpMatFNimpl(EMat<T>& c, const EMat<T>& a, const EMat<T>& b) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows(), a.cols());
			c.view() = (a.view().array() - a.view().array() * b.view().array()).matrix();
		}
	}

	template<typename T> FORCE_INLINE void OpVecFNimpl(EVec<T>& c, const EVec<T>& a, const EVec<T>& b) noexcept
	{
		if constexpr (std::is_floating_point_v<T>)
		{
			c.resize(a.rows());
			c.v_view() = (a.v_view().array() - a.v_view().array() * b.v_view().array()).matrix();
		}
	}

	// 7. Fuzzy Instruction Set Provider (separate set, float-only)

	template<typename T>
	struct FuzzyMatOpProvider
	{
		[[nodiscard]] static bool AddTo(InstructionSet& iset, const AbstractVM::TypeRegister& reg, const std::set<const char*>* selection = nullptr)
		{
			if constexpr (!std::is_floating_point_v<T>)
			{
				return true;
			}
			else
			{
				static const AbstractVM::OpDescriptor fuzzyOps[] = {
					OP_ENTRY("MatFAnd",   OpMatFAnd,   ValMatSameShape),
					OP_ENTRY("MatFOr",    OpMatFOr,    ValMatSameShape),
					OP_ENTRY("MatFXor",   OpMatFXor,   ValMatSameShape),
					OP_ENTRY("MatFImpl",  OpMatFImpl,  ValMatSameShape),
					OP_ENTRY("MatFNot",   OpMatFNot,   ValMatUnary),
					OP_ENTRY("MatFNand",  OpMatFNand,  ValMatSameShape),
					OP_ENTRY("MatFNor",   OpMatFNor,   ValMatSameShape),
					OP_ENTRY("MatFNxor",  OpMatFNxor,  ValMatSameShape),
					OP_ENTRY("MatFNimpl", OpMatFNimpl, ValMatSameShape),
				};
				return iset.Add(fuzzyOps, reg, selection);
			}
		}
	};

	template<typename T>
	struct FuzzyVecOpProvider
	{
		[[nodiscard]] static bool AddTo(InstructionSet& iset, const AbstractVM::TypeRegister& reg, const std::set<const char*>* selection = nullptr)
		{
			if constexpr (!std::is_floating_point_v<T>)
			{
				return true;
			}
			else
			{
				static const AbstractVM::OpDescriptor fuzzyOps[] = {
					OP_ENTRY("VecFAnd",   OpVecFAnd,   ValVecSameShape),
					OP_ENTRY("VecFOr",    OpVecFOr,    ValVecSameShape),
					OP_ENTRY("VecFXor",   OpVecFXor,   ValVecSameShape),
					OP_ENTRY("VecFImpl",  OpVecFImpl,  ValVecSameShape),
					OP_ENTRY("VecFNot",   OpVecFNot,   ValVecUnary),
					OP_ENTRY("VecFNand",  OpVecFNand,  ValVecSameShape),
					OP_ENTRY("VecFNor",   OpVecFNor,   ValVecSameShape),
					OP_ENTRY("VecFNxor",  OpVecFNxor,  ValVecSameShape),
					OP_ENTRY("VecFNimpl", OpVecFNimpl, ValVecSameShape),
				};
				return iset.Add(fuzzyOps, reg, selection);
			}
		}
	};

	// --- Type Aliases ---
	using EVecF = EVec<float>;   using EMatF = EMat<float>;
	using EVecD = EVec<double>;  using EMatD = EMat<double>;
	using EVecI32 = EVec<int32_t>; using EMatI32 = EMat<int32_t>;
	using EVecI64 = EVec<int64_t>; using EMatI64 = EMat<int64_t>;
	using EVecU32 = EVec<uint32_t>; using EMatU32 = EMat<uint32_t>;
	using EVecU64 = EVec<uint64_t>; using EMatU64 = EMat<uint64_t>;
	using EVecB = EVec<BitVector>; using EMatB = EMat<BitVector>;

	template<typename... Ts>
	[[nodiscard]] static bool RegisterMatOps(InstructionSet& iset, const AbstractVM::TypeRegister& reg, const std::set<const char*>* selection = nullptr)
	{
		return (MatOpProvider<Ts>::AddTo(iset, reg, selection) && ...);
	}

	template<typename... Ts>
	[[nodiscard]] static bool RegisterVecOps(InstructionSet& iset, const AbstractVM::TypeRegister& reg, const std::set<const char*>* selection = nullptr)
	{
		return (VecOpProvider<Ts>::AddTo(iset, reg, selection) && ...);
	}

	template<typename... Ts>
	[[nodiscard]] static bool RegisterAllOps(InstructionSet& iset, const AbstractVM::TypeRegister& reg, const std::set<const char*>* selection = nullptr)
	{
		return RegisterMatOps<Ts...>(iset, reg, selection) && RegisterVecOps<Ts...>(iset, reg, selection);
	}

	template<typename... Ts>
	[[nodiscard]] static bool RegisterFuzzyOps(InstructionSet& iset, const AbstractVM::TypeRegister& reg, const std::set<const char*>* selection = nullptr)
	{
		return (FuzzyMatOpProvider<Ts>::AddTo(iset, reg, selection) && ...)
			&& (FuzzyVecOpProvider<Ts>::AddTo(iset, reg, selection) && ...);
	}
}