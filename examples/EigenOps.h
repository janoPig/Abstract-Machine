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
	GEN_UNARY_MATH_OPS(Exp, exp)
	GEN_UNARY_MATH_OPS(Log, log)
	GEN_UNARY_MATH_OPS(Sqrt, sqrt)

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
					OP_ENTRY("MatMul",       OpMatMul,      ValMatMul),
					OP_ENTRY("MatVecMul",    OpMatVecMul,   ValMatVec),
					OP_ENTRY("MatTranspose", OpMatTranspose, ValMatUnary),
					OP_ENTRY("MatAdd",       OpMatAdd,      ValMatSameShape),
					OP_ENTRY("MatSub",       OpMatSub,      ValMatSameShape),
					OP_ENTRY("MatElemMul",   OpMatElemMul,  ValMatSameShape),
					OP_ENTRY("MatAddScalar", OpMatAddScalar, ValMatScalar),
					OP_ENTRY("MatSubScalar", OpMatSubScalar, ValMatScalar),
					OP_ENTRY("MatMulScalar", OpMatMulScalar, ValMatScalar),
					OP_ENTRY("Trace",        OpTrace,       ValSquareScalar),
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
					OP_ENTRY("MatDivScalar", OpMatDivScalar, ValMatScalar),
					OP_ENTRY("MatSin",       OpMatSin,       ValMatUnary),
					OP_ENTRY("MatCos",       OpMatCos,       ValMatUnary),
					OP_ENTRY("MatExp",       OpMatExp,       ValMatUnary),
					OP_ENTRY("MatLog",       OpMatLog,       ValMatUnary),
					OP_ENTRY("MatSqrt",      OpMatSqrt,      ValMatUnary),
					OP_ENTRY("MatInverse",   OpMatInverse,   ValSquare),
					OP_ENTRY("MatDet",       OpDet,          ValSquareScalar),
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
					OP_ENTRY("VecAdd",       OpVecAdd,      ValVecSameShape),
					OP_ENTRY("VecSub",       OpVecSub,      ValVecSameShape),
					OP_ENTRY("VecElemMul",   OpVecElemMul,  ValVecSameShape),
					OP_ENTRY("VecDot",       OpVecDot,      ValVecDot),
					OP_ENTRY("VecAddScalar", OpVecAddScalar, ValVecScalar),
					OP_ENTRY("VecSubScalar", OpVecSubScalar, ValVecScalar),
					OP_ENTRY("VecMulScalar", OpVecMulScalar, ValVecScalar),
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
					OP_ENTRY("VecDivScalar", OpVecDivScalar, ValVecScalar),
					OP_ENTRY("VecSin",       OpVecSin,       ValVecUnary),
					OP_ENTRY("VecCos",       OpVecCos,       ValVecUnary),
					OP_ENTRY("VecExp",       OpVecExp,       ValVecUnary),
					OP_ENTRY("VecLog",       OpVecLog,       ValVecUnary),
					OP_ENTRY("VecSqrt",      OpVecSqrt,      ValVecUnary),
				};
				ret &= iset.Add(floatOps, reg, selection);
			}
			return ret;
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
		return (MatOpProvider<Ts>::AddTo(iset, reg, selection), ...);
	}

	template<typename... Ts>
	[[nodiscard]] static bool RegisterVecOps(InstructionSet& iset, const AbstractVM::TypeRegister& reg, const std::set<const char*>* selection = nullptr)
	{
		return (VecOpProvider<Ts>::AddTo(iset, reg, selection), ...);
	}

	template<typename... Ts>
	[[nodiscard]] static bool RegisterAllOps(InstructionSet& iset, const AbstractVM::TypeRegister& reg, const std::set<const char*>* selection = nullptr)
	{
		return RegisterMatOps<Ts...>(iset, reg, selection)
			&& RegisterVecOps<Ts...>(iset, reg, selection);
	}
}
