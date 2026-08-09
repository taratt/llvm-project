//===-- AMDGPUInteriorTileSplit.cpp - Split tiled boundary paths ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Clones a deliberately narrow GEMM staging shape behind a workgroup-uniform
/// interior dispatch, then rewrites proven-interior global→LDS staging into
/// unguarded <4 x float> traffic (and merges adjacent scalar float chains).
/// The same machinery specializes conv halo footprints: CTA-uniform
/// Base(+Extent)<=Bound selectors strip per-element H/W staging guards.
/// The proof rejects shapes which could accidentally clone an outer loop or a
/// barrier.
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/SimplifyQuery.h"
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <optional>

#define DEBUG_TYPE "amdgpu-interior-tile-split"

using namespace llvm;

STATISTIC(NumInteriorTileCandidates,
          "Number of divergent tile-boundary checks found");
STATISTIC(NumPreparedInteriorTileSplits,
          "Number of canonical interior-tile splits cloned");
STATISTIC(NumPreparedInteriorKLoopSplits,
          "Number of canonical interior K loops versioned");
STATISTIC(NumInteriorStagingVectorWidens,
          "Number of interior cooperative staging loops widened to <4 x float>");
STATISTIC(NumInteriorFloatChainsVectorized,
          "Number of interior adjacent float load/store chains vectorized");

namespace {

// Cloning a large pre-barrier graph duplicates live ranges.  Real BMM A+B
// staging is typically ~8 blocks; keep headroom for similar GEMM shapes.
static cl::opt<unsigned> MaxStagingCloneBlocks(
    "amdgpu-interior-tile-max-staging-blocks",
    cl::desc("Max blocks allowed when cloning an interior staging region"),
    cl::init(192), cl::Hidden);

static cl::opt<unsigned> MinConvFootprintStrippedBranches(
    "amdgpu-interior-conv-min-stripped-branches",
    cl::desc("Minimum removable bounds branches required before cloning a "
             "conv footprint interior staging region"),
    cl::init(2), cl::Hidden);

/// Optional peel/offset diagnostics for unmatched conv footprint guards.
/// Off by default — enabling this on huge HIP kernels floods stderr and can
/// OOM remote sessions. Use -amdgpu-interior-conv-diag with -debug-only.
static cl::opt<bool> ConvFootprintDiag(
    "amdgpu-interior-conv-diag",
    cl::desc("Print peel chains for unmatched conv footprint offsets to errs()"),
    cl::init(false), cl::Hidden);

constexpr unsigned VectorWidth = 4;

static cl::opt<bool> EnableInteriorTileVectorize(
    "amdgpu-interior-tile-vectorize",
    cl::desc("Rewrite proven-interior GEMM staging to <4 x float> loads/stores"),
    cl::init(false), cl::Hidden);

static cl::opt<int> InteriorVectorizeLimit(
    "amdgpu-interior-tile-vectorize-limit",
    cl::desc("Widen at most N staging loops per region (-1 = no limit). For "
             "bisecting which staging loop a bad widen came from"),
    cl::init(-1), cl::Hidden);

enum IDDependency : unsigned {
  DependsOnNone = 0,
  DependsOnWorkgroupID = 1u << 0,
  DependsOnWorkitemID = 1u << 1,
};

static unsigned getIDDependencies(Value *V,
                                  SmallPtrSetImpl<Value *> &Visited) {
  if (!Visited.insert(V).second)
    return DependsOnNone;

  if (auto *II = dyn_cast<IntrinsicInst>(V)) {
    StringRef Name = II->getCalledFunction()->getName();
    if (Name.starts_with("llvm.amdgcn.workgroup.id."))
      return DependsOnWorkgroupID;
    if (Name.starts_with("llvm.amdgcn.workitem.id."))
      return DependsOnWorkitemID;
  }

  auto *I = dyn_cast<Instruction>(V);
  if (!I)
    return DependsOnNone;

  unsigned Dependencies = DependsOnNone;
  for (Value *Operand : I->operands())
    Dependencies |= getIDDependencies(Operand, Visited);
  return Dependencies;
}

/// CTA-uniform footprint bases are workgroup-tiled (`in_x0 = ox_tile*... - PAD`).
/// UniformityAnalysis often misses `add/sub` of those; accept any integer that
/// does not depend on workitem.id (kernel args / workgroup ids / pure affine).
static bool isCTAUniformFootprintBase(Value *V, UniformityInfo &UI) {
  if (UI.isUniformAtDef(V))
    return true;
  if (!V->getType()->isIntegerTy() || isa<Constant>(V))
    return false;
  SmallPtrSet<Value *, 16> Visited;
  return (getIDDependencies(V, Visited) & DependsOnWorkitemID) == 0;
}

static bool isWorkgroupID(Value *V, unsigned Dimension) {
  auto *II = dyn_cast<IntrinsicInst>(V);
  if (!II)
    return false;

  StringRef Name = II->getCalledFunction()->getName();
  return (Dimension == 0 && Name == "llvm.amdgcn.workgroup.id.x") ||
         (Dimension == 1 && Name == "llvm.amdgcn.workgroup.id.y");
}

/// Supported CTA tile extents in the M or N dimension. BMM / tall-skinny use
/// 128; square and large-K cuda_only GEMMs use 32.
static bool isSupportedTileMN(uint64_t TileMN) {
  return TileMN == 32 || TileMN == 64 || TileMN == 128;
}

static bool isWorkgroupShiftByLogTile(Value *V, unsigned Dimension,
                                      uint64_t &TileMN) {
  auto *Shift = dyn_cast<BinaryOperator>(V);
  if (!Shift || Shift->getOpcode() != Instruction::Shl ||
      !isWorkgroupID(Shift->getOperand(0), Dimension))
    return false;

  auto *Amount = dyn_cast<ConstantInt>(Shift->getOperand(1));
  if (!Amount || Amount->getZExtValue() > 7)
    return false;
  TileMN = 1ull << Amount->getZExtValue();
  return isSupportedTileMN(TileMN);
}

/// Recover a CTA-uniform tile base `workgroup.id * TileMN` (or biased forms)
/// and the matched TileMN. Accepts shl-by-log2 and mul-by-TileMN spellings.
static Value *getWorkgroupTileBase(Value *V, unsigned Dimension,
                                   uint64_t &TileMN) {
  if (isWorkgroupShiftByLogTile(V, Dimension, TileMN))
    return V;
  if (auto *Cast = dyn_cast<CastInst>(V))
    if (isWorkgroupShiftByLogTile(Cast->getOperand(0), Dimension, TileMN))
      return V;

  // Host launches sometimes bias the workgroup id. Accept:
  //   (wg.id + offset) << log2(TileMN)
  //   (wg.id << log2(TileMN)) + offset
  //   (wg.id + offset) * TileMN
  Value *Raw = V;
  if (auto *Cast = dyn_cast<CastInst>(V))
    Raw = Cast->getOperand(0);
  if (auto *BO = dyn_cast<BinaryOperator>(Raw)) {
    if (BO->getOpcode() == Instruction::Shl) {
      auto *Amount = dyn_cast<ConstantInt>(BO->getOperand(1));
      if (Amount && Amount->getZExtValue() <= 7) {
        uint64_t Candidate = 1ull << Amount->getZExtValue();
        if (isSupportedTileMN(Candidate)) {
          if (auto *Add = dyn_cast<BinaryOperator>(BO->getOperand(0)))
            if (Add->getOpcode() == Instruction::Add &&
                (isWorkgroupID(Add->getOperand(0), Dimension) ||
                 isWorkgroupID(Add->getOperand(1), Dimension))) {
              TileMN = Candidate;
              return V;
            }
        }
      }
    }
    if (BO->getOpcode() == Instruction::Add) {
      uint64_t Candidate = 0;
      if (isWorkgroupShiftByLogTile(BO->getOperand(0), Dimension, Candidate) ||
          isWorkgroupShiftByLogTile(BO->getOperand(1), Dimension, Candidate)) {
        TileMN = Candidate;
        return V;
      }
    }
    if (BO->getOpcode() == Instruction::Mul) {
      Value *Other = nullptr;
      if (auto *C = dyn_cast<ConstantInt>(BO->getOperand(1))) {
        if (isSupportedTileMN(C->getZExtValue()))
          Other = BO->getOperand(0);
      } else if (auto *C = dyn_cast<ConstantInt>(BO->getOperand(0))) {
        if (isSupportedTileMN(C->getZExtValue()))
          Other = BO->getOperand(1);
      }
      if (Other) {
        auto *C = dyn_cast<ConstantInt>(
            BO->getOperand(0) == Other ? BO->getOperand(1) : BO->getOperand(0));
        uint64_t Candidate = C->getZExtValue();
        if (isWorkgroupID(Other, Dimension)) {
          TileMN = Candidate;
          return V;
        }
        if (auto *Add = dyn_cast<BinaryOperator>(Other))
          if (Add->getOpcode() == Instruction::Add &&
              (isWorkgroupID(Add->getOperand(0), Dimension) ||
               isWorkgroupID(Add->getOperand(1), Dimension))) {
            TileMN = Candidate;
            return V;
          }
      }
    }
  }
  return nullptr;
}

static Value *getShiftPlusLastLaneBase(Value *V, unsigned Dimension,
                                       uint64_t &TileMN) {
  auto *Add = dyn_cast<BinaryOperator>(V);
  if (!Add || Add->getOpcode() != Instruction::Add)
    return nullptr;

  Value *LHS = Add->getOperand(0);
  Value *RHS = Add->getOperand(1);
  ConstantInt *LastLane = dyn_cast<ConstantInt>(LHS);
  Value *BaseOperand = RHS;
  if (!LastLane) {
    LastLane = dyn_cast<ConstantInt>(RHS);
    BaseOperand = LHS;
  }
  if (!LastLane)
    return nullptr;
  uint64_t Last = LastLane->getZExtValue();
  if (!isSupportedTileMN(Last + 1))
    return nullptr;
  if (!getWorkgroupTileBase(BaseOperand, Dimension, TileMN))
    return nullptr;
  if (TileMN != Last + 1)
    return nullptr;
  return getWorkgroupTileBase(BaseOperand, Dimension, TileMN);
}

struct FullTileBoundCheck {
  unsigned Dimension;
  Value *Bound;
  Value *Base;
  bool IsSigned;
  uint64_t TileMN = 128;
};

struct CanonicalTileRemainder {
  unsigned Dimension;
  Value *Bound;
  Value *Base;
  Value *Difference;
  uint64_t TileMN = 128;
  bool IsSigned = true;
};

static bool getFullTileBoundCheck(Value *V, unsigned Dimension,
                                  FullTileBoundCheck &Check) {
  auto *Cmp = dyn_cast<ICmpInst>(V);
  if (!Cmp || Cmp->getPredicate() != ICmpInst::ICMP_ULT)
    return false;
  uint64_t TileMN = 0;
  Value *Base = getShiftPlusLastLaneBase(Cmp->getOperand(0), Dimension, TileMN);
  if (!Base)
    return false;

  SmallPtrSet<Value *, 16> Visited;
  if (getIDDependencies(Cmp->getOperand(1), Visited) != DependsOnNone)
    return false;
  Check = {Dimension, Cmp->getOperand(1), Base, false, TileMN};
  return true;
}

/// Return a conservative maximum for a lane-only expression.  A workitem
/// intrinsic has no sufficiently small architectural maximum by itself, so it
/// must first be narrowed or masked.
static bool getAffineLaneMaximum(Value *V, uint64_t &Maximum) {
  if (auto *C = dyn_cast<ConstantInt>(V)) {
    if (C->isNegative())
      return false;
    Maximum = C->getZExtValue();
    return Maximum < 128;
  }
  if (auto *Cast = dyn_cast<CastInst>(V)) {
    uint64_t OperandMaximum;
    if (!getAffineLaneMaximum(Cast->getOperand(0), OperandMaximum))
      return false;
    unsigned Width = Cast->getOperand(0)->getType()->getIntegerBitWidth();
    if (Cast->getOpcode() == Instruction::SExt &&
        (Width > 64 || OperandMaximum >= (uint64_t(1) << (Width - 1))))
      return false;
    Maximum = OperandMaximum;
    return Maximum < 128;
  }

  auto *I = dyn_cast<BinaryOperator>(V);
  if (!I)
    return false;
  if (I->getOpcode() == Instruction::And) {
    Value *Other = I->getOperand(0);
    auto *Mask = dyn_cast<ConstantInt>(I->getOperand(1));
    if (!Mask) {
      Other = I->getOperand(1);
      Mask = dyn_cast<ConstantInt>(I->getOperand(0));
    }
    SmallPtrSet<Value *, 8> Visited;
    if (!Mask || Mask->getZExtValue() >= 128 ||
        !(getIDDependencies(Other, Visited) & DependsOnWorkitemID))
      return false;
    Maximum = Mask->getZExtValue();
    return true;
  }

  uint64_t LHSMaximum, RHSMaximum;
  if (I->getOpcode() == Instruction::Add) {
    if (!getAffineLaneMaximum(I->getOperand(0), LHSMaximum) ||
        !getAffineLaneMaximum(I->getOperand(1), RHSMaximum) ||
        LHSMaximum > 127 - RHSMaximum)
      return false;
    Maximum = LHSMaximum + RHSMaximum;
    return true;
  }

  Value *Variable = I->getOperand(0);
  auto *Scale = dyn_cast<ConstantInt>(I->getOperand(1));
  if (!Scale) {
    Variable = I->getOperand(1);
    Scale = dyn_cast<ConstantInt>(I->getOperand(0));
  }
  if (!Scale || Scale->isNegative() ||
      !getAffineLaneMaximum(Variable, LHSMaximum))
    return false;
  uint64_t Factor = Scale->getZExtValue();
  if (I->getOpcode() == Instruction::Shl) {
    if (Factor >= 7 || LHSMaximum > (127 >> Factor))
      return false;
    Maximum = LHSMaximum << Factor;
    return true;
  }
  if (I->getOpcode() == Instruction::Mul) {
    if (Factor > 127 || (Factor && LHSMaximum > 127 / Factor))
      return false;
    Maximum = LHSMaximum * Factor;
    return true;
  }
  return false;
}

/// Match a signed or unsigned `Base + LaneOffset < Bound`.  Prove that Offset
/// is nonnegative and strictly less than Limit via SCEV (and affine fallbacks).
static bool isUnsignedOffsetBelow(Value *Offset, uint64_t Limit,
                                  ScalarEvolution &SE) {
  if (!Offset->getType()->isIntegerTy())
    return false;
  APInt Maximum = SE.getUnsignedRangeMax(SE.getSCEV(Offset));
  if (Maximum.getBitWidth() < 64 &&
      Limit >= (uint64_t(1) << Maximum.getBitWidth()))
    return true;
  return Maximum.ult(APInt(Maximum.getBitWidth(), Limit));
}

/// Match Clang/InstCombine's expanded `X % D` as either:
///   X - (udiv X, D) * D
///   X - ((X * Magic) >> S) * D   (possibly through zext/trunc of the mul/lshr)
/// Returns Max = D-1 when Divisor is in [2, min(Limit,128)].
static bool matchExpandedURemMaximum(Value *V, uint64_t Limit,
                                     uint64_t &Maximum) {
  using namespace llvm::PatternMatch;
  Value *X = nullptr;
  Value *Quot = nullptr;
  const APInt *D = nullptr;
  if (!match(V, m_Sub(m_Value(X),
                      m_c_Mul(m_Value(Quot), m_APInt(D)))))
    return false;
  if (!D || D->isZero() || D->isNegative())
    return false;
  uint64_t Divisor = D->getZExtValue();
  if (Divisor < 2 || Divisor > Limit || Divisor > 128)
    return false;

  auto StripCasts = [](Value *Op) -> Value * {
    while (auto *Cast = dyn_cast<CastInst>(Op)) {
      unsigned OpCode = Cast->getOpcode();
      if (OpCode != Instruction::ZExt && OpCode != Instruction::SExt &&
          OpCode != Instruction::Trunc)
        break;
      Op = Cast->getOperand(0);
    }
    return Op;
  };

  Value *Q = StripCasts(Quot);
  // Exact: quot = udiv X, D
  if (match(Q, m_UDiv(m_Specific(X), m_SpecificInt(Divisor))) ||
      match(Q, m_UDiv(m_Specific(StripCasts(X)), m_SpecificInt(Divisor)))) {
    Maximum = Divisor - 1;
    return true;
  }

  // Magic: quot = lshr (mul X, Magic), S  (X may be zext'd inside the mul).
  Value *ShiftOp = nullptr;
  if (!match(Q, m_LShr(m_Value(ShiftOp), m_ConstantInt())) &&
      !match(Q, m_AShr(m_Value(ShiftOp), m_ConstantInt())))
    return false;
  ShiftOp = StripCasts(ShiftOp);
  Value *MulLHS = nullptr;
  Value *MulRHS = nullptr;
  if (!match(ShiftOp, m_Mul(m_Value(MulLHS), m_Value(MulRHS))))
    return false;
  Value *XS = StripCasts(X);
  Value *A = StripCasts(MulLHS);
  Value *B = StripCasts(MulRHS);
  if (A != XS && B != XS && A != X && B != X)
    return false;
  Maximum = Divisor - 1;
  return true;
}

static bool getUnsignedOffsetMaximumBelow(Value *Offset, uint64_t Limit,
                                          ScalarEvolution &SE,
                                          uint64_t &Maximum) {
  if (!Offset->getType()->isIntegerTy())
    return false;

  APInt SCEVMaximum = SE.getUnsignedRangeMax(SE.getSCEV(Offset));
  unsigned Width = SCEVMaximum.getBitWidth();
  if (Width < 64 && Limit >= (uint64_t(1) << Width)) {
    Maximum = SCEVMaximum.getZExtValue();
    return true;
  }
  if (Width <= 64 &&
      SCEVMaximum.ult(APInt(SCEVMaximum.getBitWidth(), Limit))) {
    Maximum = SCEVMaximum.getZExtValue();
    return true;
  }

  if (auto *Cast = dyn_cast<CastInst>(Offset)) {
    if (Cast->getOpcode() != Instruction::ZExt &&
        Cast->getOpcode() != Instruction::SExt &&
        Cast->getOpcode() != Instruction::Trunc)
      return false;
    uint64_t OperandMaximum;
    if (!getUnsignedOffsetMaximumBelow(Cast->getOperand(0), Limit, SE,
                                       OperandMaximum))
      return false;
    if (Cast->getOpcode() == Instruction::SExt) {
      unsigned Width = Cast->getOperand(0)->getType()->getIntegerBitWidth();
      if (Width > 64 ||
          OperandMaximum >= (uint64_t(1) << (Width - 1)))
        return false;
    }
    if (Cast->getOpcode() == Instruction::Trunc) {
      // Hipcc often does `zext i16 (trunc (shl (urem ...)))`; prove through
      // the narrow value so guard stripping matches footprint matching.
      unsigned DestWidth = Cast->getType()->getIntegerBitWidth();
      if (DestWidth >= 64)
        return false;
      uint64_t TruncCap = (uint64_t(1) << DestWidth) - 1;
      Maximum = std::min(OperandMaximum, TruncCap);
      return Maximum < Limit;
    }
    Maximum = OperandMaximum;
    return true;
  }

  // Rem-induction PHIs (hipcc LSR of `ix4 = vid % C`): bound from urem/and
  // init incomings; skip back-edges that reference the PHI.
  if (auto *PN = dyn_cast<PHINode>(Offset)) {
    auto UsesPN = [&](Value *In) -> bool {
      SmallPtrSet<Value *, 8> Seen;
      SmallVector<Value *, 4> Stack{In};
      while (!Stack.empty()) {
        Value *Cur = Stack.pop_back_val();
        if (Cur == PN)
          return true;
        auto *I = dyn_cast<Instruction>(Cur);
        if (!I || !Seen.insert(Cur).second)
          continue;
        if (!isa<BinaryOperator>(I) && !isa<SelectInst>(I) &&
            !isa<CastInst>(I) && !isa<FreezeInst>(I))
          continue;
        for (Value *Op : I->operands())
          Stack.push_back(Op);
      }
      return false;
    };
    uint64_t Worst = 0;
    bool Any = false;
    for (Value *In : PN->incoming_values()) {
      if (UsesPN(In))
        continue;
      uint64_t Part = 0;
      if (!getUnsignedOffsetMaximumBelow(In, Limit, SE, Part))
        return false;
      Worst = std::max(Worst, Part);
      Any = true;
    }
    if (!Any)
      return false;
    Maximum = Worst;
    return Maximum < Limit;
  }

  if (auto *Shift = dyn_cast<BinaryOperator>(Offset)) {
    if (Shift->getOpcode() == Instruction::LShr) {
      auto *Amount = dyn_cast<ConstantInt>(Shift->getOperand(1));
      if (!Amount || Amount->getZExtValue() >= 64)
        return false;
      uint64_t ShiftAmount = Amount->getZExtValue();
      if (Limit > (UINT64_MAX >> ShiftAmount))
        return false;
      uint64_t OperandMaximum;
      if (!getUnsignedOffsetMaximumBelow(Shift->getOperand(0),
                                         Limit << ShiftAmount, SE,
                                         OperandMaximum))
        return false;
      Maximum = OperandMaximum >> ShiftAmount;
      return true;
    }
    if (Shift->getOpcode() == Instruction::Shl) {
      // Same shape as conv matching: `shl nuw nsw (urem %lane, C), 2`.
      auto *Amount = dyn_cast<ConstantInt>(Shift->getOperand(1));
      if (!Amount || Amount->getZExtValue() >= 10)
        return false;
      uint64_t ShiftAmount = Amount->getZExtValue();
      uint64_t OperandMaximum = 0;
      if (!getUnsignedOffsetMaximumBelow(Shift->getOperand(0), Limit, SE,
                                         OperandMaximum))
        return false;
      if (OperandMaximum > ((Limit - 1) >> ShiftAmount))
        return false;
      Maximum = OperandMaximum << ShiftAmount;
      return Maximum < Limit;
    }
    if (Shift->getOpcode() == Instruction::URem) {
      auto *Divisor = dyn_cast<ConstantInt>(Shift->getOperand(1));
      if (!Divisor || Divisor->isZero())
        return false;
      uint64_t D = Divisor->getZExtValue();
      if (D < 2 || D > Limit)
        return false;
      Maximum = D - 1;
      return true;
    }
    if (Shift->getOpcode() == Instruction::And) {
      // Hipcc often does `ix = vid & (C-1)` for power-of-two tile dims; also
      // appears after InstCombine of `urem` by a power of two.
      auto *Mask = dyn_cast<ConstantInt>(Shift->getOperand(1));
      if (!Mask)
        Mask = dyn_cast<ConstantInt>(Shift->getOperand(0));
      if (!Mask)
        return false;
      Maximum = Mask->getZExtValue();
      return Maximum >= 1 && Maximum < Limit && Maximum < 128;
    }
    if (Shift->getOpcode() == Instruction::Mul) {
      Value *Var = Shift->getOperand(0);
      auto *FactorC = dyn_cast<ConstantInt>(Shift->getOperand(1));
      if (!FactorC) {
        Var = Shift->getOperand(1);
        FactorC = dyn_cast<ConstantInt>(Shift->getOperand(0));
      }
      if (!FactorC || FactorC->isZero() || FactorC->getZExtValue() > Limit - 1)
        return false;
      uint64_t Factor = FactorC->getZExtValue();
      uint64_t OperandLimit = (Limit - 1) / Factor + 1;
      uint64_t OperandMaximum = 0;
      if (!getUnsignedOffsetMaximumBelow(Var, OperandLimit, SE, OperandMaximum) ||
          OperandMaximum > (Limit - 1) / Factor)
        return false;
      Maximum = OperandMaximum * Factor;
      return true;
    }
    if (Shift->getOpcode() == Instruction::Sub) {
      // Clang expands `x % C` to mul/lshr magic or udiv; accept both.
      if (matchExpandedURemMaximum(Shift, Limit, Maximum))
        return true;
    }
  }

  auto *Add = dyn_cast<BinaryOperator>(Offset);
  if (!Add || Add->getOpcode() != Instruction::Add)
    return false;
  uint64_t LHSMaximum, RHSMaximum;
  if (!getUnsignedOffsetMaximumBelow(Add->getOperand(0), Limit, SE,
                                     LHSMaximum) ||
      !getUnsignedOffsetMaximumBelow(Add->getOperand(1), Limit, SE,
                                     RHSMaximum) ||
      LHSMaximum > Limit - 1 - RHSMaximum)
    return false;
  Maximum = LHSMaximum + RHSMaximum;
  return true;
}

static bool dependsOnValue(Value *V, Value *Needle,
                           SmallPtrSetImpl<Value *> &Visited) {
  if (V == Needle)
    return true;
  if (!Visited.insert(V).second)
    return false;
  auto *I = dyn_cast<Instruction>(V);
  if (!I)
    return false;
  for (Value *Operand : I->operands())
    if (dependsOnValue(Operand, Needle, Visited))
      return true;
  return false;
}

/// Recover the range of a Clang-expanded staging induction variable.  LTO can
/// obscure this recurrence from ScalarEvolution, but the loop itself still
/// proves it: the PHI starts from a workitem ID, advances by a positive
/// constant, and its in-loop successor is guarded by an unsigned/signed
/// constant upper bound.
static bool getLoopBoundedShiftedOffsetMaximum(Value *Offset, uint64_t Limit,
                                                LoopInfo &LI,
                                                uint64_t &Maximum) {
  auto *Shift = dyn_cast<BinaryOperator>(Offset);
  if (!Shift)
    return false;
  PHINode *Phi = nullptr;
  uint64_t ShiftAmount = 0;
  if (Shift->getOpcode() == Instruction::LShr) {
    auto *Amount = dyn_cast<ConstantInt>(Shift->getOperand(1));
    Phi = dyn_cast<PHINode>(Shift->getOperand(0));
    if (!Amount || !Phi || Amount->getZExtValue() >= 64)
      return false;
    ShiftAmount = Amount->getZExtValue();
  } else if (Shift->getOpcode() == Instruction::UDiv) {
    // `idx / TileK` is the same row index as `idx >> log2(TileK)` when TileK
    // is a power of two (gemm_small_k uses TileK=8).
    auto *Amount = dyn_cast<ConstantInt>(Shift->getOperand(1));
    Phi = dyn_cast<PHINode>(Shift->getOperand(0));
    if (!Amount || !Phi || !Amount->getValue().isPowerOf2() ||
        Amount->getValue().countr_zero() >= 64)
      return false;
    ShiftAmount = Amount->getValue().countr_zero();
  } else {
    return false;
  }
  if (Limit > (UINT64_MAX >> ShiftAmount))
    return false;
  uint64_t PreShiftLimit = Limit << ShiftAmount;

  Loop *L = LI.getLoopFor(Phi->getParent());
  if (!L)
    return false;
  bool HasWorkitemStart = false;
  bool HasPositiveStep = false;
  for (Value *Incoming : Phi->incoming_values()) {
    SmallPtrSet<Value *, 16> Visited;
    HasWorkitemStart |=
        getIDDependencies(Incoming, Visited) == DependsOnWorkitemID;
    auto *Add = dyn_cast<BinaryOperator>(Incoming);
    if (!Add || Add->getOpcode() != Instruction::Add)
      continue;
    Value *Other = Add->getOperand(0) == Phi ? Add->getOperand(1)
                                             : Add->getOperand(0);
    if (Add->getOperand(0) != Phi && Add->getOperand(1) != Phi)
      continue;
    auto *Step = dyn_cast<ConstantInt>(Other);
    HasPositiveStep |= Step && !Step->isNegative() && !Step->isZero();
  }
  if (!HasWorkitemStart || !HasPositiveStep)
    return false;

  SmallVector<BasicBlock *, 4> ExitingBlocks;
  L->getExitingBlocks(ExitingBlocks);
  for (BasicBlock *Exiting : ExitingBlocks) {
    auto *Branch = dyn_cast<BranchInst>(Exiting->getTerminator());
    if (!Branch || !Branch->isConditional() ||
        !L->contains(Branch->getSuccessor(0)))
      continue;
    auto *Cmp = dyn_cast<ICmpInst>(Branch->getCondition());
    if (!Cmp || (Cmp->getPredicate() != ICmpInst::ICMP_ULT &&
                 Cmp->getPredicate() != ICmpInst::ICMP_SLT))
      continue;
    auto *Bound = dyn_cast<ConstantInt>(Cmp->getOperand(1));
    if (!Bound || Bound->isNegative())
      continue;
    uint64_t BoundValue = Bound->getZExtValue();
    SmallPtrSet<Value *, 32> Visited;
    if (BoundValue <= 1023 || BoundValue > PreShiftLimit ||
        !dependsOnValue(Cmp->getOperand(0), Phi, Visited))
      continue;
    Maximum = (BoundValue - 1) >> ShiftAmount;
    return Maximum < Limit;
  }
  return false;
}

/// Match `Base + Offset < Bound` for the exact Base/Bound pair selected by
/// Full.  Direct lane forms use the architectural lane proof; nested staging
/// forms use SCEV's unsigned range for their affine offset.
static bool isPerLaneTileBoundCheck(Value *V, const FullTileBoundCheck &Full,
                                    ScalarEvolution &SE) {
  auto *Cmp = dyn_cast<ICmpInst>(V);
  if (!Cmp)
    return false;
  ICmpInst::Predicate Pred = Cmp->getPredicate();
  Value *Index = Cmp->getOperand(0);
  Value *Bound = Cmp->getOperand(1);
  if (Pred == ICmpInst::ICMP_UGT || Pred == ICmpInst::ICMP_SGT ||
      Pred == ICmpInst::ICMP_UGE || Pred == ICmpInst::ICMP_SGE) {
    Pred = ICmpInst::getSwappedPredicate(Pred);
    std::swap(Index, Bound);
  }
  bool Inclusive = Pred == ICmpInst::ICMP_ULE || Pred == ICmpInst::ICMP_SLE;
  if (Pred != ICmpInst::ICMP_SLT && Pred != ICmpInst::ICMP_ULT && !Inclusive)
    return false;

  // InstCombine rewrites `ult (base+off+c), W` into `ult (base+off), (W-c)`.
  // The CTA selector uses the peeled Bound=W, so accept either spelling.
  uint64_t BoundExtra = Inclusive ? 1 : 0;
  if (auto *Cast = dyn_cast<CastInst>(Bound))
    if (Cast->getOpcode() == Instruction::SExt ||
        Cast->getOpcode() == Instruction::ZExt)
      Bound = Cast->getOperand(0);
  if (Bound != Full.Bound) {
    auto *Sub = dyn_cast<BinaryOperator>(Bound);
    auto *C = Sub && Sub->getOpcode() == Instruction::Sub
                 ? dyn_cast<ConstantInt>(Sub->getOperand(1))
                 : nullptr;
    if (!Sub || !C || C->isNegative() || Sub->getOperand(0) != Full.Bound)
      return false;
    BoundExtra += C->getZExtValue();
  }

  // Peel nested add/or/casts toward Full.Base so conv footprints like
  // `in_x0 + (ix4*4 + 3) < W` strip correctly.
  SmallPtrSet<Value *, 8> Visited;
  std::function<bool(Value *, uint64_t &)> MatchesBase =
      [&](Value *Cur, uint64_t &OffMax) -> bool {
    if (!Visited.insert(Cur).second)
      return false;
    if (auto *Cast = dyn_cast<CastInst>(Cur))
      if (Cast->getOpcode() == Instruction::SExt ||
          Cast->getOpcode() == Instruction::ZExt ||
          Cast->getOpcode() == Instruction::Trunc)
        return MatchesBase(Cast->getOperand(0), OffMax);
    if (auto *Fr = dyn_cast<FreezeInst>(Cur))
      return MatchesBase(Fr->getOperand(0), OffMax);
    if (Cur == Full.Base) {
      OffMax = 0;
      return true;
    }
    auto *Add = dyn_cast<BinaryOperator>(Cur);
    if (!Add || (Add->getOpcode() != Instruction::Add &&
                 Add->getOpcode() != Instruction::Or))
      return false;
    uint64_t InnerOff = 0, SideOff = 0;
    // Prefer the same structural offset proofs used by conv footprint matching
    // (zext/trunc/shl/urem). Affine/SCEV remain as fallbacks for GEMM lanes.
    if (MatchesBase(Add->getOperand(0), InnerOff) &&
        ((getAffineLaneMaximum(Add->getOperand(1), SideOff) &&
          SideOff < Full.TileMN) ||
         getUnsignedOffsetMaximumBelow(Add->getOperand(1), Full.TileMN, SE,
                                       SideOff)) &&
        InnerOff <= Full.TileMN - 1 - SideOff) {
      OffMax = InnerOff + SideOff;
      return true;
    }
    Visited.erase(Add->getOperand(0));
    if (MatchesBase(Add->getOperand(1), InnerOff) &&
        ((getAffineLaneMaximum(Add->getOperand(0), SideOff) &&
          SideOff < Full.TileMN) ||
         getUnsignedOffsetMaximumBelow(Add->getOperand(0), Full.TileMN, SE,
                                       SideOff)) &&
        InnerOff <= Full.TileMN - 1 - SideOff) {
      OffMax = InnerOff + SideOff;
      return true;
    }
    return false;
  };

  uint64_t OffMax = 0;
  if (!MatchesBase(Index, OffMax))
    return false;
  if (OffMax > Full.TileMN - 1 - BoundExtra)
    return false;
  return OffMax + BoundExtra < Full.TileMN;
}

/// Recover a full-tile M/N proof directly from a per-lane guard.  Some HIP
/// GEMMs do not materialize a min/max tile extent: their staging loops retain
/// only `workgroup.id * TileMN + offset < extent`.  The same bounded-offset
/// proof used to remove the guard makes `base <= bound - TileMN` sufficient
/// for every such lane.  Keep the base spelling exact and require a uniform
/// bound so this cannot turn a lane-varying predicate into a CTA dispatch.
static bool getDirectTileBoundCheck(Value *V, unsigned Dimension,
                                    UniformityInfo &UI, ScalarEvolution &SE,
                                    LoopInfo &LI,
                                    CanonicalTileRemainder &Remainder) {
  // InstCombine represents a short-circuit conjunction as
  // `select guard, bound-check, false`.  Plain `and i1` is equally common
  // (gemm_small_k / tall_skinny).  The tile-bound conjunct remains a
  // sufficient source for the uniform M/N dispatch proof, but the other
  // conjunct must still be retained in the cloned staging path.
  if (auto *Select = dyn_cast<SelectInst>(V)) {
    auto *False = dyn_cast<ConstantInt>(Select->getFalseValue());
    if (False && False->isZero())
      return getDirectTileBoundCheck(Select->getTrueValue(), Dimension, UI,
                                     SE, LI, Remainder) ||
             getDirectTileBoundCheck(Select->getCondition(), Dimension, UI,
                                     SE, LI, Remainder);
  }
  if (auto *And = dyn_cast<BinaryOperator>(V)) {
    if (And->getOpcode() == Instruction::And &&
        And->getType()->isIntegerTy(1))
      return getDirectTileBoundCheck(And->getOperand(0), Dimension, UI, SE, LI,
                                     Remainder) ||
             getDirectTileBoundCheck(And->getOperand(1), Dimension, UI, SE, LI,
                                     Remainder);
  }

  auto *Cmp = dyn_cast<ICmpInst>(V);
  if (!Cmp)
    return false;

  ICmpInst::Predicate Pred = Cmp->getPredicate();
  Value *Index = Cmp->getOperand(0);
  Value *Bound = Cmp->getOperand(1);
  if (Pred == ICmpInst::ICMP_UGT || Pred == ICmpInst::ICMP_SGT) {
    Pred = ICmpInst::getSwappedPredicate(Pred);
    std::swap(Index, Bound);
  }
  if (Pred != ICmpInst::ICMP_ULT && Pred != ICmpInst::ICMP_SLT ||
      !Bound->getType()->isIntegerTy() || !UI.isUniformAtDef(Bound))
    return false;

  // Peel a widening cast on the compared index so
  // `sext(base + lane) < extent` still matches. Keep Bound as the icmp's
  // operand so later guard stripping can identity-match the same compare.
  Value *RawIndex = Index;
  if (auto *Cast = dyn_cast<CastInst>(Index))
    if (Cast->getOpcode() == Instruction::SExt ||
        Cast->getOpcode() == Instruction::ZExt)
      RawIndex = Cast->getOperand(0);

  auto *Add = dyn_cast<BinaryOperator>(RawIndex);
  if (!Add || (Add->getOpcode() != Instruction::Add &&
               Add->getOpcode() != Instruction::Or))
    return false;
  Value *LHS = Add->getOperand(0);
  Value *RHS = Add->getOperand(1);
  uint64_t TileMN = 0;
  Value *Base = getWorkgroupTileBase(LHS, Dimension, TileMN);
  Value *Offset = RHS;
  if (!Base) {
    Base = getWorkgroupTileBase(RHS, Dimension, TileMN);
    Offset = LHS;
  }
  uint64_t Maximum;
  if (!Base ||
      !((getAffineLaneMaximum(Offset, Maximum) && Maximum < TileMN) ||
        getUnsignedOffsetMaximumBelow(Offset, TileMN, SE, Maximum) ||
        getLoopBoundedShiftedOffsetMaximum(Offset, TileMN, LI, Maximum)))
    return false;

  Remainder = {Dimension, Bound, Base, Cmp, TileMN,
               Pred == ICmpInst::ICMP_SLT};
  return true;
}
/// `block_row + TILE_M <= M` / `Base <= Bound - TileMN`.  cuda_only large-K
/// GEMMs expose this form instead of (or in addition to) per-lane guards.
static bool getUniformFullTileBoundCheck(Value *V, unsigned Dimension,
                                          UniformityInfo &UI,
                                          CanonicalTileRemainder &Remainder) {
  if (auto *Select = dyn_cast<SelectInst>(V)) {
    auto *False = dyn_cast<ConstantInt>(Select->getFalseValue());
    if (False && False->isZero())
      return getUniformFullTileBoundCheck(Select->getTrueValue(), Dimension,
                                           UI, Remainder) ||
             getUniformFullTileBoundCheck(Select->getCondition(), Dimension,
                                           UI, Remainder);
  }
  if (auto *And = dyn_cast<BinaryOperator>(V)) {
    if (And->getOpcode() == Instruction::And &&
        And->getType()->isIntegerTy(1))
      return getUniformFullTileBoundCheck(And->getOperand(0), Dimension, UI,
                                           Remainder) ||
             getUniformFullTileBoundCheck(And->getOperand(1), Dimension, UI,
                                           Remainder);
  }

  auto *Cmp = dyn_cast<ICmpInst>(V);
  if (!Cmp || !UI.isUniformAtDef(Cmp))
    return false;

  ICmpInst::Predicate Pred = Cmp->getPredicate();
  Value *LHS = Cmp->getOperand(0);
  Value *RHS = Cmp->getOperand(1);

  auto MatchBasePlusTile = [&](Value *Sum, Value *Bound,
                               ICmpInst::Predicate P) -> bool {
    if (P != ICmpInst::ICMP_ULE && P != ICmpInst::ICMP_SLE &&
        P != ICmpInst::ICMP_ULT && P != ICmpInst::ICMP_SLT)
      return false;
    auto *Add = dyn_cast<BinaryOperator>(Sum);
    if (!Add || Add->getOpcode() != Instruction::Add)
      return false;
    uint64_t TileMN = 0;
    Value *Base = getWorkgroupTileBase(Add->getOperand(0), Dimension, TileMN);
    auto *TileC = dyn_cast<ConstantInt>(Add->getOperand(1));
    if (!Base) {
      Base = getWorkgroupTileBase(Add->getOperand(1), Dimension, TileMN);
      TileC = dyn_cast<ConstantInt>(Add->getOperand(0));
    }
    if (!Base || !TileC || TileC->getZExtValue() != TileMN)
      return false;
    // base+TileMN <= Bound  (or < Bound+1 — reject the strict form unless
    // Bound is adjusted; only accept non-strict <= / signed equivalents).
    if (P == ICmpInst::ICMP_ULT || P == ICmpInst::ICMP_SLT)
      return false;
    Remainder = {Dimension, Bound, Base, Cmp, TileMN,
                 P == ICmpInst::ICMP_SLE};
    return true;
  };

  auto MatchBaseLeBoundMinusTile = [&](Value *BaseCand, Value *BoundMinus,
                                        ICmpInst::Predicate P) -> bool {
    if (P != ICmpInst::ICMP_ULE && P != ICmpInst::ICMP_SLE)
      return false;
    auto *Sub = dyn_cast<BinaryOperator>(BoundMinus);
    if (!Sub || Sub->getOpcode() != Instruction::Sub)
      return false;
    auto *TileC = dyn_cast<ConstantInt>(Sub->getOperand(1));
    uint64_t TileMN = 0;
    Value *Base = getWorkgroupTileBase(BaseCand, Dimension, TileMN);
    if (!Base || !TileC || TileC->getZExtValue() != TileMN)
      return false;
    Remainder = {Dimension, Sub->getOperand(0), Base, Cmp, TileMN,
                 P == ICmpInst::ICMP_SLE};
    return true;
  };

  if (MatchBasePlusTile(LHS, RHS, Pred))
    return true;
  if (MatchBasePlusTile(RHS, LHS, ICmpInst::getSwappedPredicate(Pred)))
    return true;
  if (MatchBaseLeBoundMinusTile(LHS, RHS, Pred))
    return true;
  if (MatchBaseLeBoundMinusTile(RHS, LHS, ICmpInst::getSwappedPredicate(Pred)))
    return true;
  return false;
}

/// The successor selected by a predicate proven in the cloned prefix.  Keep
/// this distinct from a plain match: some edge predicates are false for a full
/// tile, while the ordinary lane and K guards are true.
enum class GuardOutcome : unsigned {
  True = 0,
  False = 1,
};

static bool isUniformFullTileTautology(Value *V,
                                         const FullTileBoundCheck &Full) {
  auto *Cmp = dyn_cast<ICmpInst>(V);
  if (!Cmp)
    return false;
  ICmpInst::Predicate Pred = Cmp->getPredicate();
  Value *LHS = Cmp->getOperand(0);
  Value *RHS = Cmp->getOperand(1);

  auto MatchAdd = [&](Value *Sum, Value *Bound, ICmpInst::Predicate P) {
    if (P != ICmpInst::ICMP_ULE && P != ICmpInst::ICMP_SLE)
      return false;
    auto *Add = dyn_cast<BinaryOperator>(Sum);
    if (!Add || Add->getOpcode() != Instruction::Add)
      return false;
    auto *TileC = dyn_cast<ConstantInt>(Add->getOperand(1));
    Value *Base = Add->getOperand(0);
    if (!TileC) {
      TileC = dyn_cast<ConstantInt>(Add->getOperand(0));
      Base = Add->getOperand(1);
    }
    return Base == Full.Base && Bound == Full.Bound && TileC &&
           TileC->getZExtValue() == Full.TileMN;
  };
  auto MatchSub = [&](Value *BaseCand, Value *BoundMinus,
                      ICmpInst::Predicate P) {
    if (P != ICmpInst::ICMP_ULE && P != ICmpInst::ICMP_SLE)
      return false;
    auto *Sub = dyn_cast<BinaryOperator>(BoundMinus);
    if (!Sub || Sub->getOpcode() != Instruction::Sub)
      return false;
    auto *TileC = dyn_cast<ConstantInt>(Sub->getOperand(1));
    return BaseCand == Full.Base && Sub->getOperand(0) == Full.Bound &&
           TileC && TileC->getZExtValue() == Full.TileMN;
  };
  return MatchAdd(LHS, RHS, Pred) ||
         MatchAdd(RHS, LHS, ICmpInst::getSwappedPredicate(Pred)) ||
         MatchSub(LHS, RHS, Pred) ||
         MatchSub(RHS, LHS, ICmpInst::getSwappedPredicate(Pred));
}

static bool isPerLaneTileBoundCheck(Value *V, const FullTileBoundCheck &Full,
                                    ScalarEvolution &SE);

static void collectConjuncts(Value *V, SmallVectorImpl<Value *> &Conjuncts);

static bool isRemovableFootprintConjunct(Value *V,
                                         ArrayRef<FullTileBoundCheck> FullChecks,
                                         ScalarEvolution &SE) {
  for (const FullTileBoundCheck &Full : FullChecks)
    if (isPerLaneTileBoundCheck(V, Full, SE) ||
        isUniformFullTileTautology(V, Full))
      return true;
  return false;
}

static bool getRemovableSafetyBranchOutcome(
    const BranchInst *Branch, ArrayRef<FullTileBoundCheck> FullChecks,
    ScalarEvolution &SE, GuardOutcome &Outcome) {
  if (!Branch->isConditional())
    return false;
  Value *Cond = Branch->getCondition();
  if (isRemovableFootprintConjunct(Cond, FullChecks, SE)) {
    Outcome = GuardOutcome::True;
    return true;
  }

  // Conv staging often ANDs independent H/W (or x0..x3) validity tests.
  // If every conjunct is proven by the CTA-uniform footprint selector, the
  // whole branch is unconditionally taken on the interior clone.
  SmallVector<Value *, 8> Conjuncts;
  collectConjuncts(Cond, Conjuncts);
  if (Conjuncts.size() < 2)
    return false;
  for (Value *Conjunct : Conjuncts)
    if (!isRemovableFootprintConjunct(Conjunct, FullChecks, SE))
      return false;
  Outcome = GuardOutcome::True;
  return true;
}

static bool isWorkgroupTileBase(Value *V, unsigned Dimension) {
  uint64_t TileMN = 0;
  return getWorkgroupTileBase(V, Dimension, TileMN) != nullptr;
}

/// Generic scalar optimization can lower min/max to selects, but the
/// tile-relative subtraction itself remains available.
static bool isTileRelativeDifference(Value *V, unsigned Dimension) {
  auto *Sub = dyn_cast<BinaryOperator>(V);
  return Sub && Sub->getOpcode() == Instruction::Sub &&
         isWorkgroupTileBase(Sub->getOperand(1), Dimension);
}

static bool isNamedCall(Value *V, StringRef Name) {
  auto *Call = dyn_cast<CallBase>(V);
  return Call && Call->getCalledFunction() &&
         Call->getCalledFunction()->getName().starts_with(Name);
}

/// Match the select lowering of `smin(smax(Difference, 0), TileMN)`.  Scalar
/// optimization commonly replaces the intrinsic form before this pass runs;
/// accept only its exact signed, ordered select spelling.
static bool getSelectClampedTileExtent(Value *V, Value *&Difference,
                                       uint64_t &TileMN) {
  auto *Min = dyn_cast<SelectInst>(V);
  auto *MinCmp = Min ? dyn_cast<ICmpInst>(Min->getCondition()) : nullptr;
  auto *TileSize =
      Min ? dyn_cast<ConstantInt>(Min->getFalseValue()) : nullptr;
  if (!MinCmp || !TileSize || !isSupportedTileMN(TileSize->getZExtValue()) ||
      MinCmp->getPredicate() != ICmpInst::ICMP_SLT ||
      MinCmp->getOperand(1) != TileSize ||
      Min->getTrueValue() != MinCmp->getOperand(0))
    return false;

  auto *Max = dyn_cast<SelectInst>(Min->getTrueValue());
  auto *MaxCmp = Max ? dyn_cast<ICmpInst>(Max->getCondition()) : nullptr;
  auto *Zero =
      Max ? dyn_cast<ConstantInt>(Max->getFalseValue()) : nullptr;
  if (!MaxCmp || !Zero || !Zero->isZero() ||
      MaxCmp->getPredicate() != ICmpInst::ICMP_SGT ||
      MaxCmp->getOperand(1) != Zero ||
      Max->getTrueValue() != MaxCmp->getOperand(0))
    return false;

  Difference = Max->getTrueValue();
  TileMN = TileSize->getZExtValue();
  return true;
}

/// Match min(max(Bound - (workgroup.id * TileMN), 0), TileMN), the clamp form
/// emitted by Clang for a TileMN-row or TileMN-column tile extent.
static bool getClampedTileExtent(Value *V, unsigned Dimension,
                                 CanonicalTileRemainder &Remainder) {
  Value *Difference = nullptr;
  uint64_t TileMN = 0;
  if (isNamedCall(V, "llvm.smin.") || isNamedCall(V, "llvm.umin.")) {
    auto *Min = cast<CallBase>(V);
    Value *Extent = nullptr;
    bool HasTileSize = false;
    for (Value *Operand : Min->args()) {
      if (auto *C = dyn_cast<ConstantInt>(Operand)) {
        if (!isSupportedTileMN(C->getZExtValue()) || HasTileSize)
          return false;
        HasTileSize = true;
        TileMN = C->getZExtValue();
      } else {
        if (Extent)
          return false;
        Extent = Operand;
      }
    }
    if (!HasTileSize || !Extent || !isNamedCall(Extent, "llvm.smax."))
      return false;

    auto *Max = cast<CallBase>(Extent);
    bool HasZero = false;
    for (Value *Operand : Max->args()) {
      if (auto *C = dyn_cast<ConstantInt>(Operand)) {
        if (!C->isZero() || HasZero)
          return false;
        HasZero = true;
      } else {
        if (Difference)
          return false;
        Difference = Operand;
      }
    }
    if (!HasZero)
      return false;
  } else if (!getSelectClampedTileExtent(V, Difference, TileMN)) {
    return false;
  }

  auto *Sub = dyn_cast<BinaryOperator>(Difference);
  uint64_t BaseTileMN = 0;
  if (!Sub || Sub->getOpcode() != Instruction::Sub ||
      !getWorkgroupTileBase(Sub->getOperand(1), Dimension, BaseTileMN) ||
      BaseTileMN != TileMN)
    return false;

  Remainder = {Dimension, Sub->getOperand(0), Sub->getOperand(1), Sub, TileMN,
               /*IsSigned=*/true};
  return true;
}

static bool isClampedTileExtent(Value *V, unsigned Dimension) {
  CanonicalTileRemainder Remainder;
  return getClampedTileExtent(V, Dimension, Remainder);
}

/// Match min(K - k0, 32), the full-K-tile extent form emitted by Clang.
static bool isClampedKTileExtent(Value *V) {
  if (!isNamedCall(V, "llvm.smin."))
    return false;
  auto *Min = cast<CallBase>(V);
  for (Value *Operand : Min->args())
    if (auto *C = dyn_cast<ConstantInt>(Operand))
      if (C->equalsInt(32))
        return true;
  return false;
}

static bool isKTileCheck(Value *V) {
  auto *Cmp = dyn_cast<ICmpInst>(V);
  if (!Cmp)
    return false;

  auto IsK = [](Value *V) {
    SmallPtrSet<Value *, 16> Visited;
    return getIDDependencies(V, Visited) == DependsOnNone;
  };
  auto *RHS = dyn_cast<ConstantInt>(Cmp->getOperand(1));
  if (RHS && RHS->equalsInt(32) &&
      (Cmp->getPredicate() == ICmpInst::ICMP_UGE ||
       Cmp->getPredicate() == ICmpInst::ICMP_SGE))
    return IsK(Cmp->getOperand(0));

  auto *LHS = dyn_cast<ConstantInt>(Cmp->getOperand(0));
  return LHS && LHS->equalsInt(32) &&
         (Cmp->getPredicate() == ICmpInst::ICMP_ULE ||
          Cmp->getPredicate() == ICmpInst::ICMP_SLE) &&
         IsK(Cmp->getOperand(1));
}

/// Match `umin/smin(K - k0, 32) >= 32` (or its equivalent operand order).
/// The cloned prefix has an exact multiple-of-32 bound, so this is true on
/// every cloned iteration only when k0 is the loop induction value and K is
/// the recurrence exit bound.  The prefix dispatch separately requires
/// K >= 32 signed, making the unsigned recurrence and signed remainder agree.
static bool isFullKTileGuard(Value *V, Value *IV, Value *Bound) {
  auto *Cmp = dyn_cast<ICmpInst>(V);
  if (!Cmp)
    return false;

  Value *Extent = nullptr;
  if (auto *C = dyn_cast<ConstantInt>(Cmp->getOperand(1))) {
    if (!C->equalsInt(32) ||
        (Cmp->getPredicate() != ICmpInst::ICMP_UGE &&
         Cmp->getPredicate() != ICmpInst::ICMP_SGE))
      return false;
    Extent = Cmp->getOperand(0);
  } else if (auto *C = dyn_cast<ConstantInt>(Cmp->getOperand(0))) {
    if (!C->equalsInt(32) ||
        (Cmp->getPredicate() != ICmpInst::ICMP_ULE &&
         Cmp->getPredicate() != ICmpInst::ICMP_SLE))
      return false;
    Extent = Cmp->getOperand(1);
  } else {
    return false;
  }

  bool IsUnsignedMin = isNamedCall(Extent, "llvm.umin.");
  bool IsSignedMin = isNamedCall(Extent, "llvm.smin.");
  if (!IsUnsignedMin && !IsSignedMin)
    return false;
  if (IsSignedMin && Cmp->getPredicate() != ICmpInst::ICMP_SGE &&
      Cmp->getPredicate() != ICmpInst::ICMP_SLE)
    return false;
  auto *Min = cast<CallBase>(Extent);
  Value *Difference = nullptr;
  bool HasTileSize = false;
  for (Value *Operand : Min->args()) {
    if (auto *C = dyn_cast<ConstantInt>(Operand)) {
      if (!C->equalsInt(32) || HasTileSize)
        return false;
      HasTileSize = true;
    } else {
      if (Difference)
        return false;
      Difference = Operand;
    }
  }
  auto *Sub = dyn_cast_or_null<BinaryOperator>(Difference);
  return HasTileSize && Sub && Sub->getOpcode() == Instruction::Sub &&
         Sub->getOperand(0) == Bound && Sub->getOperand(1) == IV;
}

/// Match the scalar guard emitted in the nested staging loop:
///   k0 + nested.offset < K
/// The outer K induction and bound must be exactly the recurrence proven by
/// splitInteriorKLoop; only the nested offset is discharged through SCEV.
static bool isNestedKBoundCheck(Value *V, Value *IV, Value *Bound,
                                ScalarEvolution &SE) {
  auto *Cmp = dyn_cast<ICmpInst>(V);
  if (!Cmp || Cmp->getPredicate() != ICmpInst::ICMP_SLT ||
      Cmp->getOperand(1) != Bound)
    return false;

  auto *Add = dyn_cast<BinaryOperator>(Cmp->getOperand(0));
  if (!Add || Add->getOpcode() != Instruction::Add)
    return false;

  Value *LHS = Add->getOperand(0);
  Value *RHS = Add->getOperand(1);
  if (LHS == IV)
    return isUnsignedOffsetBelow(RHS, 32, SE);
  if (RHS == IV)
    return isUnsignedOffsetBelow(LHS, 32, SE);
  return false;
}

/// Match the one quotient/remainder spelling used by gemm_ts_outer:
///   div = udiv idx, divisor
///   rem = idx - div * divisor
///   select (k0 + trunc(div) < K), (NBase + sext(cond + trunc(rem)) < N), 0
///
/// This is deliberately not a general div/rem recognizer.  In particular,
/// the divisor must be a positive constant, the index must be nonnegative and
/// below divisor * 32, and every reconstructed arithmetic operation must be
/// marked no-wrap.  Together these local facts prove trunc(div) < 32 and
/// cond + trunc(rem) < 128.
static bool getPositiveNoWrapConstantSub(Value *V, uint64_t &Result) {
  auto *Sub = dyn_cast<BinaryOperator>(V);
  if (!Sub || Sub->getOpcode() != Instruction::Sub || !Sub->hasNoSignedWrap())
    return false;
  auto *LHS = dyn_cast<ConstantInt>(Sub->getOperand(0));
  auto *RHS = dyn_cast<ConstantInt>(Sub->getOperand(1));
  if (!LHS || !RHS || LHS->isNegative() || RHS->isNegative() ||
      LHS->getValue().ule(RHS->getValue()))
    return false;
  Result = LHS->getZExtValue() - RHS->getZExtValue();
  return Result != 0;
}

static bool isGemmTSOuterQuotientRemainderCheck(
    Value *KOffset, Value *NCheck, ArrayRef<FullTileBoundCheck> FullChecks,
    ScalarEvolution &SE) {
  auto *QuotientTrunc = dyn_cast<TruncInst>(KOffset);
  auto *Div = QuotientTrunc
                  ? dyn_cast<BinaryOperator>(QuotientTrunc->getOperand(0))
                  : nullptr;
  if (!Div || Div->getOpcode() != Instruction::UDiv)
    return false;

  Value *Index = Div->getOperand(0);
  Value *Divisor = Div->getOperand(1);
  uint64_t DivisorValue;
  if (!getPositiveNoWrapConstantSub(Divisor, DivisorValue) ||
      !SE.isKnownNonNegative(SE.getSCEV(Index)))
    return false;

  if (DivisorValue > UINT64_MAX / 32)
    return false;
  uint64_t IndexLimit = DivisorValue * 32;
  if (!isUnsignedOffsetBelow(Index, IndexLimit, SE))
    return false;

  auto *NComparison = dyn_cast<ICmpInst>(NCheck);
  if (!NComparison)
    return false;
  auto *NIndexAdd = dyn_cast<BinaryOperator>(NComparison->getOperand(0));
  if (!NIndexAdd || NIndexAdd->getOpcode() != Instruction::Add ||
      !NIndexAdd->hasNoSignedWrap())
    return false;
  Value *NOffset = NIndexAdd->getOperand(0);
  for (const FullTileBoundCheck &Full : FullChecks) {
    if (Full.Dimension != 0 || NIndexAdd->getOperand(1) != Full.Base)
      continue;
    if (NComparison->getOperand(1) != Full.Bound ||
        (Full.IsSigned ? NComparison->getPredicate() != ICmpInst::ICMP_SLT
                       : NComparison->getPredicate() != ICmpInst::ICMP_ULT))
      continue;

    auto *OffsetSExt = dyn_cast<SExtInst>(NOffset);
    auto *OffsetAdd =
        OffsetSExt ? dyn_cast<BinaryOperator>(OffsetSExt->getOperand(0))
                   : nullptr;
    if (!OffsetAdd || OffsetAdd->getOpcode() != Instruction::Add ||
        !OffsetAdd->hasNoSignedWrap())
      continue;

    Value *Cond = OffsetAdd->getOperand(0);
    auto *RemainderTrunc =
        dyn_cast<TruncInst>(OffsetAdd->getOperand(1));
    if (!RemainderTrunc)
      continue;
    if (RemainderTrunc->getType() != QuotientTrunc->getType())
      continue;

    auto *Remainder =
        dyn_cast<BinaryOperator>(RemainderTrunc->getOperand(0));
    if (!Remainder || Remainder->getOpcode() != Instruction::Sub ||
        !Remainder->hasNoUnsignedWrap() || Remainder->getOperand(0) != Index)
      continue;
    auto *Product = dyn_cast<BinaryOperator>(Remainder->getOperand(1));
    if (!Product || Product->getOpcode() != Instruction::Mul ||
        !Product->hasNoUnsignedWrap() || Product->getOperand(0) != Div ||
        Product->getOperand(1) != Divisor)
      continue;

    uint64_t CondMaximum;
    if (!getUnsignedOffsetMaximumBelow(Cond, 128, SE, CondMaximum) ||
        DivisorValue - 1 > 127 - CondMaximum)
      continue;
    return true;
  }
  return false;
}

/// Match the select form of the exact gemm_ts_outer nested K/N guard.  The
/// select is intentionally accepted only in this polarity, so replacing its
/// branch with successor zero removes only the proven true-path guard in the
/// cloned prefix.
static bool isNestedKAndNBoundCheck(Value *V, Value *IV, Value *Bound,
                                    ArrayRef<FullTileBoundCheck> FullChecks,
                                    ScalarEvolution &SE) {
  auto *Select = dyn_cast<SelectInst>(V);
  auto *FalseValue = Select ? dyn_cast<ConstantInt>(Select->getFalseValue())
                            : nullptr;
  if (!FalseValue || !FalseValue->isZero())
    return false;

  auto *KCmp = dyn_cast<ICmpInst>(Select->getCondition());
  auto *NCmp = dyn_cast<ICmpInst>(Select->getTrueValue());
  if (!KCmp || !NCmp || KCmp->getPredicate() != ICmpInst::ICMP_SLT ||
      KCmp->getOperand(1) != Bound)
    return false;
  auto *KAdd = dyn_cast<BinaryOperator>(KCmp->getOperand(0));
  if (!KAdd || KAdd->getOpcode() != Instruction::Add ||
      !KAdd->hasNoSignedWrap())
    return false;
  if (KAdd->getOperand(1) != IV)
    return false;
  Value *Offset = KAdd->getOperand(0);

  return isGemmTSOuterQuotientRemainderCheck(Offset, NCmp, FullChecks, SE);
}

static bool isAlignmentCheck(Value *V, UniformityInfo &UI) {
  auto *Cmp = dyn_cast<ICmpInst>(V);
  if (!Cmp || Cmp->getPredicate() != ICmpInst::ICMP_EQ ||
      !isa<ConstantInt>(Cmp->getOperand(1)) ||
      !cast<ConstantInt>(Cmp->getOperand(1))->isZero() ||
      !UI.isUniformAtDef(Cmp))
    return false;

  auto *Mask = dyn_cast<BinaryOperator>(Cmp->getOperand(0));
  if (!Mask || Mask->getOpcode() != Instruction::And ||
      !isa<ConstantInt>(Mask->getOperand(1)))
    return false;
  return !cast<ConstantInt>(Mask->getOperand(1))->isZero();
}

/// Match the real-GEMM N-edge predicate:
///   select alignment, (trunc(clamp(N - nbase, 0, TileMN)) & ~3), 0 < TileMN
///
/// The synthesized full-N selector proves the clamp is exactly TileMN, and the
/// select's exact alignment input is one of the selector's conjuncts.  Thus
/// the selected value is TileMN, not below TileMN, so the conditional branch
/// takes its false successor in the cloned prefix.
static bool isFullNEdgeFallbackCheck(
    Value *V, ArrayRef<FullTileBoundCheck> FullChecks,
    ArrayRef<Value *> Alignments) {
  auto *Cmp = dyn_cast<ICmpInst>(V);
  if (!Cmp || Cmp->getPredicate() != ICmpInst::ICMP_ULT)
    return false;
  auto *Limit = dyn_cast<ConstantInt>(Cmp->getOperand(1));
  auto *Select = dyn_cast<SelectInst>(Cmp->getOperand(0));
  if (!Limit || !Select || !isSupportedTileMN(Limit->getZExtValue()))
    return false;
  const uint64_t TileMN = Limit->getZExtValue();
  bool IsSelectorAlignment = false;
  for (Value *Alignment : Alignments)
    IsSelectorAlignment |= Alignment == Select->getCondition();
  if (!IsSelectorAlignment)
    return false;

  auto *Zero = dyn_cast<ConstantInt>(Select->getFalseValue());
  auto *Masked = dyn_cast<BinaryOperator>(Select->getTrueValue());
  if (!Zero || !Zero->isZero() || !Masked ||
      Masked->getOpcode() != Instruction::And)
    return false;
  Value *ExtentTrunc = Masked->getOperand(0);
  auto *Mask = dyn_cast<ConstantInt>(Masked->getOperand(1));
  if (!Mask) {
    ExtentTrunc = Masked->getOperand(1);
    Mask = dyn_cast<ConstantInt>(Masked->getOperand(0));
  }
  auto *Trunc = dyn_cast<TruncInst>(ExtentTrunc);
  // 252 = TileMN-4 for TileMN=128; for smaller tiles accept any mask that
  // clears the low 2 bits of a value known to equal TileMN.
  if (!Mask || (Mask->getZExtValue() & 3) != 0 || !Trunc)
    return false;

  for (const FullTileBoundCheck &Full : FullChecks) {
    if (Full.Dimension != 0 || Full.TileMN != TileMN)
      continue;
    CanonicalTileRemainder NExtent;
    if (getClampedTileExtent(Trunc->getOperand(0), 0, NExtent) &&
        NExtent.Bound == Full.Bound && NExtent.Base == Full.Base &&
        NExtent.TileMN == TileMN)
      return true;
  }
  return false;
}

static bool getRemovablePrefixGuardOutcome(
    const BranchInst *Branch, Value *IV, Value *Bound,
    ArrayRef<FullTileBoundCheck> FullChecks, ArrayRef<Value *> Alignments,
    ScalarEvolution &SE, GuardOutcome &Outcome) {
  if (!Branch->isConditional())
    return false;
  if (isFullKTileGuard(Branch->getCondition(), IV, Bound) ||
      isNestedKBoundCheck(Branch->getCondition(), IV, Bound, SE) ||
      isNestedKAndNBoundCheck(Branch->getCondition(), IV, Bound, FullChecks,
                               SE)) {
    Outcome = GuardOutcome::True;
    return true;
  }
  if (isFullNEdgeFallbackCheck(Branch->getCondition(), FullChecks,
                               Alignments)) {
    Outcome = GuardOutcome::False;
    return true;
  }
  return getRemovableSafetyBranchOutcome(Branch, FullChecks, SE, Outcome);
}

/// Return the non-M/N conjunct of a staging safety predicate.  Direct-bound
/// analysis has already proved every value in DirectGuards is a tile-bound
/// check covered by the uniform interior dispatch; the remaining conjunct
/// (normally the K check) must stay in the cloned path.
///
/// Accepts both InstCombine spellings of a 2-way conjunction:
///   select direct, other, false
///   and i1 direct, other
static Value *getInteriorConjunctionRemainder(
    Value *V, ArrayRef<Value *> DirectGuards) {
  auto IsDirectGuard = [&](Value *Candidate) {
    for (Value *Guard : DirectGuards)
      if (Candidate == Guard)
        return true;
    return false;
  };

  if (auto *Select = dyn_cast<SelectInst>(V)) {
    auto *FalseValue = dyn_cast<ConstantInt>(Select->getFalseValue());
    if (!FalseValue || !FalseValue->isZero())
      return nullptr;
    if (IsDirectGuard(Select->getCondition()))
      return Select->getTrueValue();
    if (IsDirectGuard(Select->getTrueValue()))
      return Select->getCondition();
    return nullptr;
  }

  auto *And = dyn_cast<BinaryOperator>(V);
  if (!And || And->getOpcode() != Instruction::And ||
      !And->getType()->isIntegerTy(1))
    return nullptr;
  if (IsDirectGuard(And->getOperand(0)))
    return And->getOperand(1);
  if (IsDirectGuard(And->getOperand(1)))
    return And->getOperand(0);
  return nullptr;
}

static void collectConjuncts(Value *V, SmallVectorImpl<Value *> &Conjuncts) {
  auto *And = dyn_cast<BinaryOperator>(V);
  if (!And || And->getOpcode() != Instruction::And ||
      And->getType() != Type::getInt1Ty(V->getContext())) {
    Conjuncts.push_back(V);
    return;
  }

  collectConjuncts(And->getOperand(0), Conjuncts);
  collectConjuncts(And->getOperand(1), Conjuncts);
}

static bool isBarrierBlock(const BasicBlock *BB) {
  for (const Instruction &I : *BB)
    if (const auto *II = dyn_cast<IntrinsicInst>(&I))
      if (II->getCalledFunction()->getName() == "llvm.amdgcn.s.barrier")
        return true;
  return false;
}

static bool containsBarrier(const Loop *L) {
  for (const BasicBlock *BB : L->blocks())
    if (isBarrierBlock(BB))
      return true;
  return false;
}

/// Clone a canonical full-K prefix before the original guarded tail.
[[maybe_unused]] static bool
splitInteriorKLoop(Loop *L, BasicBlock *Dispatch,
                   Value *StaticFullTileCondition,
                   ArrayRef<FullTileBoundCheck> FullChecks,
                   ArrayRef<Value *> Alignments, LoopInfo &LI,
                   DominatorTree &DT, ScalarEvolution &SE) {
  BasicBlock *Preheader = L->getLoopPreheader();
  BasicBlock *Exit = L->getUniqueExitBlock();
  BasicBlock *Exiting = L->getExitingBlock();
  auto *DispatchBranch = dyn_cast<BranchInst>(Dispatch->getTerminator());
  if (!L->isLoopSimplifyForm() || !L->isLCSSAForm(DT) ||
      !L->isSafeToClone() || !Preheader || !Exit || !Exiting ||
      Exiting != L->getLoopLatch() || !containsBarrier(L) ||
      Preheader->getSinglePredecessor() != Dispatch ||
      !DispatchBranch || !DispatchBranch->isUnconditional() ||
      DispatchBranch->getSuccessor(0) != Preheader ||
      StaticFullTileCondition->getType() !=
          Type::getInt1Ty(L->getHeader()->getContext()))
    return false;

  // This deliberately accepts only the canonical unsigned K recurrence:
  //   %k = phi [ 0, %preheader ], [ %k.next, %latch ]
  //   %k.next = add %k, 32
  //   br i1 (%k.next < %K), header, exit
  // A loop-entry available bound and a SCEV step of exactly 32 make
  // `%K & -32` a safe sequential prefix bound.
  auto *ExitBranch = dyn_cast<BranchInst>(Exiting->getTerminator());
  if (!ExitBranch || !ExitBranch->isConditional())
    return false;
  auto *ExitCmp = dyn_cast<ICmpInst>(ExitBranch->getCondition());
  if (!ExitCmp)
    return false;
  BasicBlock *Continue = ExitBranch->getSuccessor(0) == L->getHeader()
                             ? ExitBranch->getSuccessor(0)
                             : ExitBranch->getSuccessor(1) == L->getHeader()
                                   ? ExitBranch->getSuccessor(1)
                                   : nullptr;
  if (!Continue)
    return false;
  ICmpInst::Predicate Pred = ExitBranch->getSuccessor(0) == Continue
                                  ? ExitCmp->getPredicate()
                                  : ExitCmp->getInversePredicate();
  if (Pred != ICmpInst::ICMP_ULT && Pred != ICmpInst::ICMP_SLT)
    return false;

  Value *IVValue = ExitCmp->getOperand(0);
  Value *Bound = ExitCmp->getOperand(1);
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(IVValue));
  if (!AR) {
    std::swap(IVValue, Bound);
    AR = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(IVValue));
    if (!AR)
      return false;
  }
  auto *Step = dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE));
  if (!Step || Step->getAPInt().getZExtValue() != 32 ||
      !L->isLoopInvariant(Bound) ||
      AR->getLoop() != L || !SE.isAvailableAtLoopEntry(SE.getSCEV(Bound), L) ||
      !Bound->getType()->isIntegerTy())
    return false;
  PHINode *IV = nullptr;
  for (PHINode &PN : L->getHeader()->phis()) {
    if (PN.getIncomingValueForBlock(Exiting) != IVValue)
      continue;
    if (!isa<ConstantInt>(PN.getIncomingValueForBlock(Preheader)) ||
        !cast<ConstantInt>(PN.getIncomingValueForBlock(Preheader))->isZero())
      return false;
    IV = &PN;
    break;
  }
  if (!IV)
    return false;
  auto *Inc = dyn_cast<BinaryOperator>(IVValue);
  if (!Inc || Inc->getOpcode() != Instruction::Add)
    return false;

  // The zero-K path in the real kernel can enter the postlude directly.
  // Isolate the loop edge before cloning so prefix/tail live-outs meet at a
  // dedicated exit block.
  if (Exit->getSinglePredecessor() != Exiting) {
    formDedicatedExitBlocks(L, &DT, &LI, nullptr, true);
    Exit = L->getUniqueExitBlock();
    if (!Exit || Exit->getSinglePredecessor() != Exiting)
      return false;
  }

  // Do not version a loop unless the cloned prefix will actually become less
  // guarded.  Both predicates are proven against the original loop here and
  // rechecked after remapping below.
  bool HasRemovableGuard = false;
  for (BasicBlock *BB : L->blocks()) {
    GuardOutcome Outcome;
    HasRemovableGuard |= getRemovablePrefixGuardOutcome(
        cast<BranchInst>(BB->getTerminator()), IV, Bound, FullChecks,
        Alignments, SE, Outcome);
  }
  if (!HasRemovableGuard)
    return false;

  // The prefix-to-tail merge below carries header PHI backedge values.  An
  // arbitrary loop live-out would need its own edge PHI and is deliberately
  // outside this narrow first implementation.
  SmallPtrSet<Value *, 8> HeaderBackedgeValues;
  for (PHINode &PN : L->getHeader()->phis())
    HeaderBackedgeValues.insert(PN.getIncomingValueForBlock(Exiting));
  for (PHINode &PN : Exit->phis())
    if (!HeaderBackedgeValues.contains(PN.getIncomingValueForBlock(Exiting)))
      return false;

  IRBuilder<> Builder(DispatchBranch);
  Value *PrefixBound =
      Builder.CreateAnd(Bound, ConstantInt::getSigned(Bound->getType(), -32),
                        "interior.k.prefix.bound");
  Value *HasPrefixBits = Builder.CreateICmpNE(
      PrefixBound, ConstantInt::getNullValue(Bound->getType()),
      "interior.k.has.prefix");
  // Signed K extents are valid only for a nonnegative K.  This also keeps the
  // existing unsigned recurrence away from values with the sign bit set.
  Value *HasPrefix = Builder.CreateAnd(
      HasPrefixBits,
      Builder.CreateICmpSGE(Bound, ConstantInt::get(Bound->getType(), 32),
                            "interior.k.nonnegative"),
      "interior.k.has.prefix");
  Value *RunPrefix =
      Builder.CreateAnd(StaticFullTileCondition, HasPrefix, "interior.k.full");

  ValueToValueMapTy VMap;
  SmallVector<BasicBlock *, 8> ClonedBlocks;
  Loop *PrefixLoop =
      cloneLoopWithPreheader(Exit, Dispatch, L, VMap, ".interior", &LI, &DT,
                             ClonedBlocks);
  remapInstructionsInBlocks(ClonedBlocks, VMap);

  BasicBlock *PrefixPreheader = PrefixLoop->getLoopPreheader();
  BasicBlock *PrefixExiting = cast<BasicBlock>(VMap[Exiting]);
  auto *PrefixCmp = cast<ICmpInst>(VMap[ExitCmp]);
  if (PrefixCmp->getOperand(0) == VMap.lookup(IVValue))
    PrefixCmp->setOperand(1, PrefixBound);
  else if (PrefixCmp->getOperand(1) == VMap.lookup(IVValue))
    PrefixCmp->setOperand(0, PrefixBound);
  else
    llvm_unreachable("cloned K exit compare must use cloned induction");

  unsigned RemovedSafetyBranches = 0;
  Value *PrefixIV = VMap.lookup(IV);
  for (BasicBlock *BB : L->blocks()) {
    BasicBlock *PrefixBB = cast<BasicBlock>(VMap[BB]);
    auto *OriginalBranch = cast<BranchInst>(BB->getTerminator());
    GuardOutcome Outcome;
    if (!getRemovablePrefixGuardOutcome(OriginalBranch, IV, Bound, FullChecks,
                                        Alignments, SE, Outcome))
      continue;

    auto *PrefixBranch = cast<BranchInst>(PrefixBB->getTerminator());
    GuardOutcome PrefixOutcome;
    if (!getRemovablePrefixGuardOutcome(PrefixBranch, PrefixIV, Bound,
                                        FullChecks, Alignments, SE,
                                        PrefixOutcome) ||
        PrefixOutcome != Outcome)
      llvm_unreachable("cloned prefix guard must retain canonical shape");
    BranchInst::Create(
        PrefixBranch->getSuccessor(static_cast<unsigned>(Outcome)),
        PrefixBranch);
    PrefixBranch->eraseFromParent();
    ++RemovedSafetyBranches;
  }

  BranchInst::Create(PrefixPreheader, Preheader, RunPrefix,
                     DispatchBranch);
  DispatchBranch->eraseFromParent();

  // The original loop is the guarded tail.  It receives either zero (when no
  // prefix ran) or every corresponding cloned backedge value.  Carry every
  // header PHI, not only the induction, before entering the guarded tail.
  IRBuilder<> TailBuilder(Preheader->getTerminator());
  SmallVector<std::pair<Value *, Value *>, 8> PrefixLiveOutMerges;
  for (PHINode &PN : L->getHeader()->phis()) {
    PHINode *PrefixPN = cast<PHINode>(VMap[&PN]);
    Value *Initial = PN.getIncomingValueForBlock(Preheader);
    Value *PrefixFinal = PrefixPN->getIncomingValueForBlock(PrefixExiting);
    PHINode *TailStart =
        TailBuilder.CreatePHI(PN.getType(), 2, PN.getName() + ".tail.start");
    TailStart->addIncoming(Initial, Dispatch);
    TailStart->addIncoming(PrefixFinal, PrefixExiting);
    PN.setIncomingValueForBlock(Preheader, TailStart);
    PrefixLiveOutMerges.push_back({PrefixFinal, TailStart});
  }
  PrefixExiting->getTerminator()->replaceSuccessorWith(Exit, Preheader);
  Value *PrefixLeavesTail =
      TailBuilder.CreateICmpNE(PrefixBound, Bound, "interior.k.has.tail");
  // If the CTA cannot take the full-tile prefix, the original guarded loop
  // must execute from zero even when K is an exact multiple of 32.
  Value *HasTail = TailBuilder.CreateOr(
      TailBuilder.CreateNot(RunPrefix), PrefixLeavesTail, "interior.k.has.tail");
  BranchInst::Create(L->getHeader(), Exit, HasTail, Preheader->getTerminator());
  Preheader->getTerminator()->eraseFromParent();

  // Preserve the existing LCSSA merge: skipping the tail contributes the
  // cloned prefix live-out; executing it contributes the original live-out.
  for (PHINode &PN : Exit->phis()) {
    Value *Incoming = PN.getIncomingValueForBlock(Exiting);
    if (auto It = VMap.find(Incoming); It != VMap.end())
      Incoming = It->second;
    for (const auto &[PrefixValue, MergedValue] : PrefixLiveOutMerges)
      if (Incoming == PrefixValue) {
        Incoming = MergedValue;
        break;
      }
    PN.addIncoming(Incoming, Preheader);
  }

  DT.changeImmediateDominator(Preheader, Dispatch);
  DT.changeImmediateDominator(Exit, Preheader);
  SE.forgetLoop(L);
  SE.forgetLoop(PrefixLoop);
  ++NumPreparedInteriorKLoopSplits;
  LLVM_DEBUG(dbgs() << "Split canonical interior K loop in "
                    << L->getHeader()->getParent()->getName() << " at "
                    << Dispatch->getName()
                    << "; prefix loop " << PrefixLoop->getHeader()->getName()
                    << ", guarded tail " << L->getHeader()->getName()
                    << "; removed " << RemovedSafetyBranches
                    << " proven guard(s)"
                    << ", shared live-out exit " << Exit->getName() << '\n');
  return true;
}

/// Collect the canonical setup values immediately before a K loop.  Clang may
/// put them in the loop preheader or in its sole predecessor when it splits
/// the unconditional edge into the loop.  Do not search arbitrary dominators:
/// an unrelated tiled region could otherwise provide a plausible M/N pair.
static bool collectInteriorTileSetup(
    BasicBlock *Preheader, UniformityInfo &UI, CanonicalTileRemainder &M,
    CanonicalTileRemainder &N, SmallVectorImpl<Value *> &Alignments,
    bool &HasM, bool &HasN) {
  if (!Preheader)
    return false;

  SmallVector<BasicBlock *, 2> SetupBlocks;
  if (BasicBlock *Setup = Preheader->getSinglePredecessor())
    SetupBlocks.push_back(Setup);
  SetupBlocks.push_back(Preheader);
  for (BasicBlock *BB : SetupBlocks) {
    for (Instruction &I : *BB) {
      CanonicalTileRemainder Remainder;
      if (getClampedTileExtent(&I, 1, Remainder)) {
        if (HasM)
          return false;
        M = Remainder;
        HasM = true;
      }
      if (getClampedTileExtent(&I, 0, Remainder)) {
        if (HasN)
          return false;
        N = Remainder;
        HasN = true;
      }
      if (isAlignmentCheck(&I, UI))
        Alignments.push_back(&I);
    }
  }
  return true;
}

/// Supplement clamp setup with the direct M/N guards retained by simple HIP
/// staging loops.  Do not search outside the candidate outer K loop, and
/// reject competing base/bound pairs rather than choosing one by order.
static bool collectDirectInteriorTileBounds(
    Loop *L, UniformityInfo &UI, ScalarEvolution &SE, LoopInfo &LI,
    CanonicalTileRemainder &M, CanonicalTileRemainder &N, bool &HasM,
    bool &HasN, bool &HasDirectM, bool &HasDirectN) {
  auto Record = [](const CanonicalTileRemainder &Remainder,
                   CanonicalTileRemainder &Recorded, bool &HasRecorded) {
    if (HasRecorded)
      return Recorded.Bound == Remainder.Bound &&
             Recorded.Base == Remainder.Base &&
             Recorded.TileMN == Remainder.TileMN;
    Recorded = Remainder;
    HasRecorded = true;
    return true;
  };
  for (BasicBlock *BB : L->blocks()) {
    auto *Branch = dyn_cast<BranchInst>(BB->getTerminator());
    if (!Branch || !Branch->isConditional())
      continue;
    CanonicalTileRemainder Remainder;
    if (getDirectTileBoundCheck(Branch->getCondition(), 1, UI, SE, LI,
                                Remainder) ||
        getUniformFullTileBoundCheck(Branch->getCondition(), 1, UI,
                                      Remainder)) {
      HasDirectM = true;
      if (!Record(Remainder, M, HasM))
        return false;
    }
    if (getDirectTileBoundCheck(Branch->getCondition(), 0, UI, SE, LI,
                                Remainder) ||
        getUniformFullTileBoundCheck(Branch->getCondition(), 0, UI,
                                      Remainder)) {
      HasDirectN = true;
      if (!Record(Remainder, N, HasN))
        return false;
    }
  }
  return true;
}

/// Build the CTA-uniform M/N/alignment selector from the canonical setup
/// values for the K loop.
static Value *synthesizeInteriorTileSelector(
    BasicBlock *Preheader, const CanonicalTileRemainder &M,
    const CanonicalTileRemainder &N, ArrayRef<Value *> Alignments) {
  if (M.Bound->getType() != N.Bound->getType() || !M.TileMN || !N.TileMN)
    return nullptr;

  IRBuilder<> Builder(Preheader->getTerminator());
  // The source extents use signed min/max.  Do not infer signed arithmetic
  // facts from their wrapping subtraction: require a nonnegative bound and
  // prove Base <= Bound - TileMN directly.  This makes Base + every matched
  // nonnegative lane offset below TileMN a defined signed in-bounds index.
  Value *MPositive = Builder.CreateICmpSGE(
      M.Bound, ConstantInt::get(M.Bound->getType(), M.TileMN),
      "interior.m.nonnegative");
  Value *MFull = Builder.CreateICmpSLE(
      M.Base, Builder.CreateSub(M.Bound,
                                ConstantInt::get(M.Bound->getType(), M.TileMN)),
      "interior.m.full");
  MFull = Builder.CreateAnd(MPositive, MFull, "interior.m.tile");
  Value *NPositive = Builder.CreateICmpSGE(
      N.Bound, ConstantInt::get(N.Bound->getType(), N.TileMN),
      "interior.n.nonnegative");
  Value *NFull = Builder.CreateICmpSLE(
      N.Base, Builder.CreateSub(N.Bound,
                                ConstantInt::get(N.Bound->getType(), N.TileMN)),
      "interior.n.full");
  NFull = Builder.CreateAnd(NPositive, NFull, "interior.n.tile");
  Value *MNFull = Builder.CreateAnd(MFull, NFull, "interior.mn.full");
  if (Alignments.empty())
    return MNFull;
  Value *Alignment = Alignments.front();
  for (unsigned I = 1; I != Alignments.size(); ++I)
    Alignment = Builder.CreateAnd(Alignment, Alignments[I],
                                  "interior.aligned");
  return Builder.CreateAnd(MNFull, Alignment, "interior.full");
}

/// Check the pieces that must hold before splitting a preheader.  The
/// subsequent split only changes its predecessor; the remaining checks in
/// splitInteriorKLoop are consequently guaranteed by this preflight.
[[maybe_unused]] static bool
canSplitInteriorKLoop(Loop *L, ArrayRef<FullTileBoundCheck> FullChecks,
                      ArrayRef<Value *> Alignments, LoopInfo &LI,
                      DominatorTree &DT, ScalarEvolution &SE) {
  BasicBlock *Preheader = L->getLoopPreheader();
  BasicBlock *Exit = L->getUniqueExitBlock();
  BasicBlock *Exiting = L->getExitingBlock();
  const bool HasBasicShape = L->isLoopSimplifyForm() &&
                             L->isLCSSAForm(DT) && L->isSafeToClone() &&
                             Preheader && Exit && Exiting &&
                             Exiting == L->getLoopLatch() &&
                             containsBarrier(L) && L->getHeader();
  if (!HasBasicShape) {
    LLVM_DEBUG(dbgs() << "Interior K-loop preflight rejected "
                      << L->getHeader()->getName()
                      << ": simplify=" << L->isLoopSimplifyForm()
                      << " lcssa=" << L->isLCSSAForm(DT)
                      << " cloneable=" << L->isSafeToClone()
                      << " preheader=" << static_cast<bool>(Preheader)
                      << " exit=" << static_cast<bool>(Exit)
                      << " latch=" << (Exiting == L->getLoopLatch())
                      << " barrier=" << containsBarrier(L) << '\n');
    return false;
  }

  auto *ExitBranch = dyn_cast<BranchInst>(Exiting->getTerminator());
  auto *ExitCmp = ExitBranch && ExitBranch->isConditional()
                      ? dyn_cast<ICmpInst>(ExitBranch->getCondition())
                      : nullptr;
  if (!ExitCmp) {
    LLVM_DEBUG(dbgs() << "Interior K-loop preflight rejected "
                      << L->getHeader()->getName()
                      << ": no conditional exit compare\n");
    return false;
  }
  BasicBlock *Continue = ExitBranch->getSuccessor(0) == L->getHeader()
                             ? ExitBranch->getSuccessor(0)
                             : ExitBranch->getSuccessor(1) == L->getHeader()
                                   ? ExitBranch->getSuccessor(1)
                                   : nullptr;
  if (!Continue) {
    LLVM_DEBUG(dbgs() << "Interior K-loop preflight rejected "
                      << L->getHeader()->getName()
                      << ": no loop-header continue successor\n");
    return false;
  }
  ICmpInst::Predicate Pred = ExitBranch->getSuccessor(0) == Continue
                                  ? ExitCmp->getPredicate()
                                  : ExitCmp->getInversePredicate();
  if (Pred != ICmpInst::ICMP_ULT && Pred != ICmpInst::ICMP_SLT) {
    LLVM_DEBUG(dbgs() << "Interior K-loop preflight rejected "
                      << L->getHeader()->getName()
                      << ": unsupported continue predicate\n");
    return false;
  }

  Value *IVValue = ExitCmp->getOperand(0);
  Value *Bound = ExitCmp->getOperand(1);
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(IVValue));
  if (!AR) {
    std::swap(IVValue, Bound);
    AR = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(IVValue));
  }
  auto *Step = AR ? dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE))
                  : nullptr;
  if (!Step || Step->getAPInt().getZExtValue() != 32 ||
      !L->isLoopInvariant(Bound) || AR->getLoop() != L ||
      !SE.isAvailableAtLoopEntry(SE.getSCEV(Bound), L) ||
      !Bound->getType()->isIntegerTy()) {
    LLVM_DEBUG(dbgs() << "Interior K-loop preflight rejected "
                      << L->getHeader()->getName()
                      << ": unsupported SCEV recurrence, step, or bound\n");
    return false;
  }
  PHINode *IV = nullptr;
  for (PHINode &PN : L->getHeader()->phis()) {
    if (PN.getIncomingValueForBlock(Exiting) != IVValue)
      continue;
    if (!isa<ConstantInt>(PN.getIncomingValueForBlock(Preheader)) ||
        !cast<ConstantInt>(PN.getIncomingValueForBlock(Preheader))->isZero()) {
      LLVM_DEBUG(dbgs() << "Interior K-loop preflight rejected "
                        << L->getHeader()->getName()
                        << ": unsupported induction PHI\n");
      return false;
    }
    IV = &PN;
    break;
  }
  if (!IV) {
    LLVM_DEBUG(dbgs() << "Interior K-loop preflight rejected "
                      << L->getHeader()->getName()
                      << ": no matching induction PHI\n");
    return false;
  }
  auto *Inc = dyn_cast<BinaryOperator>(IVValue);
  if (!Inc || Inc->getOpcode() != Instruction::Add) {
    LLVM_DEBUG(dbgs() << "Interior K-loop preflight rejected "
                      << L->getHeader()->getName()
                      << ": unsupported induction increment\n");
    return false;
  }

  SmallPtrSet<Value *, 8> HeaderBackedgeValues;
  for (PHINode &PN : L->getHeader()->phis())
    HeaderBackedgeValues.insert(PN.getIncomingValueForBlock(Exiting));
  for (PHINode &PN : Exit->phis())
    if (!HeaderBackedgeValues.contains(PN.getIncomingValueForBlock(Exiting))) {
      LLVM_DEBUG(dbgs() << "Interior K-loop preflight rejected "
                        << L->getHeader()->getName()
                        << ": non-header LCSSA liveout\n");
      return false;
    }

  bool HasRemovableGuard = false;
  for (BasicBlock *BB : L->blocks()) {
    GuardOutcome Outcome;
    HasRemovableGuard |= getRemovablePrefixGuardOutcome(
        cast<BranchInst>(BB->getTerminator()), IV, Bound, FullChecks,
        Alignments, SE, Outcome);
  }
  if (!HasRemovableGuard)
    LLVM_DEBUG(dbgs() << "Interior K-loop preflight rejected "
                      << L->getHeader()->getName()
                      << ": no removable guard\n");
  return HasRemovableGuard;
}

/// Returns a closed, single-entry/single-exit staging region.  The barrier is
/// intentionally outside the region: it must remain shared by the fast and
/// edge paths.  The staging-only outer-K path may opt into nested staging
/// loops, but never clones the enclosing outer-K loop.
static bool findClosedStagingRegion(BasicBlock *Entry, BasicBlock *Dispatch,
                                    SmallPtrSetImpl<BasicBlock *> &Region,
                                    BasicBlock *&Barrier,
                                    bool AllowNestedLoops = false,
                                    bool *ContainsLoop = nullptr,
                                    bool AllowEntryExternalPredecessors = false) {
  if (isBarrierBlock(Entry) ||
      (!AllowEntryExternalPredecessors &&
       Entry->getSinglePredecessor() != Dispatch))
    return false;

  SmallVector<BasicBlock *, 8> Worklist{Entry};
  Region.insert(Entry);
  while (!Worklist.empty()) {
    BasicBlock *BB = Worklist.pop_back_val();
    for (BasicBlock *Successor : successors(BB)) {
      if (isBarrierBlock(Successor)) {
        if (Barrier && Barrier != Successor)
          return false;
        Barrier = Successor;
        continue;
      }
      if (Region.insert(Successor).second)
        Worklist.push_back(Successor);
    }
  }

  if (!Barrier)
    return false;

  bool HasSafetyBranch = false;
  for (BasicBlock *BB : Region) {
    if (isBarrierBlock(BB) || !isa<BranchInst>(BB->getTerminator()))
      return false;

    for (BasicBlock *Predecessor : predecessors(BB))
      if (Predecessor != Dispatch && !Region.contains(Predecessor) &&
          !(AllowEntryExternalPredecessors && BB == Entry))
        return false;

    if (auto *Br = dyn_cast<BranchInst>(BB->getTerminator()))
      if (Br->isConditional())
        HasSafetyBranch = true;

    for (BasicBlock *Successor : successors(BB)) {
      if (Successor == Barrier)
        continue;
      if (!Region.contains(Successor))
        return false;
    }
  }

  // Every cloned block must retain a path to the shared barrier.  In
  // particular, accepting an isolated nested cycle here would make the fast
  // path fail to reconverge even though the region has no CFG exit besides the
  // barrier.
  SmallPtrSet<BasicBlock *, 8> ReachesBarrier;
  SmallVector<BasicBlock *, 8> ReverseWorklist;
  for (BasicBlock *Predecessor : predecessors(Barrier))
    if (Region.contains(Predecessor) &&
        ReachesBarrier.insert(Predecessor).second)
      ReverseWorklist.push_back(Predecessor);
  while (!ReverseWorklist.empty()) {
    BasicBlock *BB = ReverseWorklist.pop_back_val();
    for (BasicBlock *Predecessor : predecessors(BB))
      if (Region.contains(Predecessor) &&
          ReachesBarrier.insert(Predecessor).second)
        ReverseWorklist.push_back(Predecessor);
  }
  if (ReachesBarrier.size() != Region.size()) {
    LLVM_DEBUG(dbgs() << "Interior staging preflight rejected "
                      << Entry->getName()
                      << ": not every staging block reaches the shared barrier\n");
    return false;
  }

  // The normal staging path remains acyclic.  The outer-K staging path permits
  // cycles because real GEMM staging contains nested load loops; the closed
  // region proof still prevents the enclosing K loop, barrier, compute, and
  // latch from entering the clone.
  SmallPtrSet<BasicBlock *, 8> Visiting;
  SmallPtrSet<BasicBlock *, 8> Visited;
  auto HasCycle = [&](auto &&Self, BasicBlock *BB) -> bool {
    if (!Visiting.insert(BB).second)
      return true;
    if (Visited.insert(BB).second) {
      for (BasicBlock *Successor : successors(BB))
        if (Region.contains(Successor) && Self(Self, Successor))
          return true;
    }
    Visiting.erase(BB);
    return false;
  };
  bool HasCycleInRegion = HasCycle(HasCycle, Entry);
  if (ContainsLoop)
    *ContainsLoop = HasCycleInRegion;
  return HasSafetyBranch && (AllowNestedLoops || !HasCycleInRegion);
}

/// The shared barrier may fan out to compute and a skip-compute path, but no
/// successor may leave the outer K iteration or re-enter cloned staging.
static bool hasOnlySharedBarrierExits(
    const Loop *L, const SmallPtrSetImpl<BasicBlock *> &Region,
    const BasicBlock *Barrier) {
  bool HasSuccessor = false;
  for (const BasicBlock *Successor : successors(Barrier)) {
    HasSuccessor = true;
    if (!L->contains(Successor) || Region.contains(Successor))
      return false;
  }
  return HasSuccessor;
}

/// Prove that sharing Barrier does not require a merge of a value produced by
/// the cloned staging graph.  A direct use in the barrier or compute phase
/// would otherwise refer only to the fallback definition on the cloned path.
static bool canCloneStagingRegion(
    BasicBlock *Entry, const SmallPtrSetImpl<BasicBlock *> &Region,
    BasicBlock *Barrier, ArrayRef<FullTileBoundCheck> FullChecks,
    ArrayRef<Value *> DirectGuards,
    ArrayRef<Value *> Alignments, PHINode *IV, Value *Bound,
    ScalarEvolution &SE, bool IgnoreEntryInstructions = false) {
  if (isa<PHINode>(&Barrier->front())) {
    LLVM_DEBUG(dbgs() << "Interior staging preflight rejected "
                      << Entry->getName()
                      << ": shared barrier PHI requires a path merge\n");
    return false;
  }

  for (BasicBlock *BB : Region) {
    for (Instruction &I : *BB) {
      // Header-anchored staging is split only at the header terminator.  All
      // header definitions therefore stay in the shared dispatch, so their
      // existing uses in the latch, barrier successors, and compute phase do
      // not need path merges.  The subsequent post-split check validates the
      // actual staging-only region without this exception.
      if (IgnoreEntryInstructions && BB == Entry)
        continue;
      for (User *U : I.users()) {
        auto *UseI = dyn_cast<Instruction>(U);
        if (UseI && !Region.contains(UseI->getParent())) {
          LLVM_DEBUG(dbgs() << "Interior staging preflight rejected "
                            << Entry->getName() << ": staging live-out "
                            << I.getName() << " reaches "
                            << UseI->getParent()->getName()
                            << " and requires a path merge\n");
          return false;
        }
      }
    }
  }

  bool HasRemovableSafetyBranch = false;
  for (BasicBlock *BB : Region) {
    auto *Branch = dyn_cast<BranchInst>(BB->getTerminator());
    if (!Branch || !Branch->isConditional())
      continue;
    GuardOutcome Outcome;
    HasRemovableSafetyBranch |= getRemovablePrefixGuardOutcome(
        Branch, IV, Bound, FullChecks, Alignments, SE, Outcome);
    if (!HasRemovableSafetyBranch)
      HasRemovableSafetyBranch = getInteriorConjunctionRemainder(
          Branch->getCondition(), DirectGuards);
  }
  if (!HasRemovableSafetyBranch)
    LLVM_DEBUG(dbgs() << "Interior staging preflight rejected "
                      << Entry->getName()
                      << ": no removable staging safety branch\n");
  return HasRemovableSafetyBranch;
}

/// Supported K-tile widths. BMM uses 32; gemm_small_k / tall_skinny use 8.
static bool isSupportedTileK(uint64_t TileK) {
  return TileK == 8 || TileK == 16 || TileK == 32 || TileK == 64 ||
         TileK == 128;
}

enum class OuterKForm : unsigned {
  /// `for (k0 = 0; k0 < K; k0 += TileK)` — IV is the K offset.
  OffsetStep = 0,
  /// `for (t = 0; t < ceil(K/TileK); ++t)` — IV is the tile index; k0 = t*TileK.
  TileCount = 1,
};

struct OuterKLoopInfo {
  PHINode *IV = nullptr;
  Value *Bound = nullptr;
  uint64_t TileK = 0;
  OuterKForm Form = OuterKForm::OffsetStep;
};

/// Recover TileK from `IV * C` / `IV << log2(C)` uses inside the loop body.
/// The scale often sits behind a widening cast, because the tile offset is used
/// to index with 64-bit arithmetic. Also accept `(IV + const) * C`, which is
/// how double-buffered GEMMs write the next-tile column (`(t+1) * TILE_K`).
static uint64_t inferTileKFromIndexUses(PHINode *IV) {
  SmallVector<Value *, 8> Sources = {IV};
  for (User *U : IV->users()) {
    if (isa<SExtInst>(U) || isa<ZExtInst>(U) || isa<TruncInst>(U))
      Sources.push_back(U);
    // Peel `iv + C` / `C + iv` so `(t+1)*TileK` still reveals TileK.
    if (auto *Add = dyn_cast<BinaryOperator>(U))
      if (Add->getOpcode() == Instruction::Add &&
          (isa<ConstantInt>(Add->getOperand(0)) ||
           isa<ConstantInt>(Add->getOperand(1))))
        Sources.push_back(Add);
  }

  for (Value *Src : Sources) {
    for (User *U : Src->users()) {
      if (isa<SExtInst>(U) || isa<ZExtInst>(U) || isa<TruncInst>(U)) {
        for (User *UU : U->users()) {
          auto *BO = dyn_cast<BinaryOperator>(UU);
          if (!BO)
            continue;
          if (BO->getOpcode() == Instruction::Mul) {
            Value *Other = BO->getOperand(0) == U ? BO->getOperand(1)
                                                  : BO->getOperand(0);
            if (auto *C = dyn_cast<ConstantInt>(Other))
              if (isSupportedTileK(C->getZExtValue()))
                return C->getZExtValue();
          }
        }
      }
      auto *BO = dyn_cast<BinaryOperator>(U);
      if (!BO)
        continue;
      if (BO->getOpcode() == Instruction::Mul) {
        Value *Other =
            BO->getOperand(0) == Src ? BO->getOperand(1) : BO->getOperand(0);
        if (auto *C = dyn_cast<ConstantInt>(Other))
          if (isSupportedTileK(C->getZExtValue()))
            return C->getZExtValue();
      }
      if (BO->getOpcode() == Instruction::Shl && BO->getOperand(0) == Src)
        if (auto *C = dyn_cast<ConstantInt>(BO->getOperand(1))) {
          uint64_t Shift = C->getZExtValue();
          if (Shift >= 3 && Shift <= 7) {
            uint64_t TileK = 1ull << Shift;
            if (isSupportedTileK(TileK))
              return TileK;
          }
        }
    }
  }
  return 0;
}

/// Return the induction, bound, and tile width for an accepted outer K loop.
/// Two forms are recognized:
///   OffsetStep: `k0 += TileK`, Bound is K; full-tile when `k0 <= K - TileK`
///   TileCount:  `t += 1`, Bound is ceil(K/TileK); full-tile when `t+1 < Bound`
///               (skips the possibly-partial last tile — always safe).
static bool getCanonicalOuterKLoop(Loop *L, ScalarEvolution &SE,
                                   OuterKLoopInfo &Info) {
  // Every rejection is reported: without it, "not a canonical outer K loop"
  // gives no way to tell an unsupported loop shape from a matcher bug.
  auto Reject = [&](const char *Why) {
    LLVM_DEBUG(dbgs() << "Outer K loop rejected at "
                      << L->getHeader()->getName() << ": " << Why << '\n');
    return false;
  };

  BasicBlock *Preheader = L->getLoopPreheader();
  BasicBlock *Exiting = L->getExitingBlock();
  if (!L->isLoopSimplifyForm() || !Preheader || !Exiting ||
      Exiting != L->getLoopLatch())
    return Reject("not simplify form, or exiting block is not the latch");

  auto *ExitBranch = dyn_cast<BranchInst>(Exiting->getTerminator());
  auto *ExitCmp =
      ExitBranch && ExitBranch->isConditional()
          ? dyn_cast<ICmpInst>(ExitBranch->getCondition())
          : nullptr;
  if (!ExitCmp)
    return Reject("latch does not branch on an icmp");
  BasicBlock *Continue = ExitBranch->getSuccessor(0) == L->getHeader()
                             ? ExitBranch->getSuccessor(0)
                             : ExitBranch->getSuccessor(1) == L->getHeader()
                                   ? ExitBranch->getSuccessor(1)
                                   : nullptr;
  if (!Continue)
    return Reject("latch does not branch back to the header");
  ICmpInst::Predicate Pred = ExitBranch->getSuccessor(0) == Continue
                                  ? ExitCmp->getPredicate()
                                  : ExitCmp->getInversePredicate();
  // A unit-step counted loop is canonicalized to `continue while iv.next !=
  // bound`, since equality is provable there. Treat that as the less-than it
  // came from, but only once SCEV agrees the loop is finite — `!=` alone does
  // not bound the counter.
  const bool IsNotEqualForm = Pred == ICmpInst::ICMP_NE;
  if (Pred != ICmpInst::ICMP_ULT && Pred != ICmpInst::ICMP_SLT &&
      !IsNotEqualForm)
    return Reject("latch predicate is not a less-than or not-equal");

  Value *IVNext = ExitCmp->getOperand(0);
  Value *Bound = ExitCmp->getOperand(1);
  // Latch compares are often on a widened copy of the IV bump
  // (`sext i32 %k.next to i64`), especially in cuda_only GEMMs with 64-bit
  // index arithmetic. Peel those casts before asking SCEV for an addrec.
  auto PeelWideningCast = [](Value *V) -> Value * {
    while (auto *Cast = dyn_cast<CastInst>(V)) {
      if (Cast->getOpcode() != Instruction::SExt &&
          Cast->getOpcode() != Instruction::ZExt)
        break;
      V = Cast->getOperand(0);
    }
    return V;
  };
  Value *IVNextRaw = PeelWideningCast(IVNext);
  Value *BoundRaw = PeelWideningCast(Bound);
  const SCEVAddRecExpr *AR =
      dyn_cast<SCEVAddRecExpr>(SE.getSCEV(IVNextRaw));
  if (!AR) {
    std::swap(IVNext, Bound);
    std::swap(IVNextRaw, BoundRaw);
    AR = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(IVNextRaw));
  }
  auto *Step = AR ? dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE))
                  : nullptr;
  if (!Step || AR->getLoop() != L)
    return Reject("latch compares no affine IV of this loop");
  if (!L->isLoopInvariant(Bound) ||
      !SE.isAvailableAtLoopEntry(SE.getSCEV(Bound), L) ||
      !Bound->getType()->isIntegerTy())
    return Reject("trip bound is not an integer available at loop entry");
  const uint64_t StepVal = Step->getAPInt().getZExtValue();
  if (IsNotEqualForm) {
    if (StepVal != 1)
      return Reject("not-equal latch on a non-unit step");
    if (isa<SCEVCouldNotCompute>(SE.getBackedgeTakenCount(L)))
      return Reject("not-equal latch with an unknown backedge count");
  }

  PHINode *IV = nullptr;
  for (PHINode &PN : L->getHeader()->phis()) {
    Value *Incoming = PN.getIncomingValueForBlock(Exiting);
    if (Incoming != IVNext && Incoming != IVNextRaw &&
        PeelWideningCast(Incoming) != IVNextRaw)
      continue;
    auto *Initial =
        dyn_cast<ConstantInt>(PN.getIncomingValueForBlock(Preheader));
    if (!Initial || !Initial->isZero())
      return Reject("K induction variable does not start at zero");
    Value *IncVal = PeelWideningCast(Incoming);
    auto *Inc = dyn_cast<BinaryOperator>(IncVal);
    if (!Inc || Inc->getOpcode() != Instruction::Add)
      return Reject("K induction variable is not advanced by an add");
    IV = &PN;
    break;
  }
  if (!IV)
    return Reject("no header phi feeds the latch compare");

  // Prefer the raw integer trip bound when the icmp widened it.
  if (Bound != BoundRaw && BoundRaw->getType() == IV->getType())
    Bound = BoundRaw;
  if (isSupportedTileK(StepVal)) {
    Info = {IV, Bound, StepVal, OuterKForm::OffsetStep};
    return true;
  }
  if (StepVal == 1) {
    uint64_t TileK = inferTileKFromIndexUses(IV);
    if (!TileK)
      return Reject("unit-step K counter with no recognizable tile-K scale");
    Info = {IV, Bound, TileK, OuterKForm::TileCount};
    return true;
  }
  LLVM_DEBUG(dbgs() << "Outer K loop rejected at " << L->getHeader()->getName()
                    << ": step " << StepVal << " is not a supported tile K\n");
  return false;
}

static Type *getFloatTy(LLVMContext &Ctx) { return Type::getFloatTy(Ctx); }

static FixedVectorType *getFloat4Ty(LLVMContext &Ctx) {
  return FixedVectorType::get(getFloatTy(Ctx), VectorWidth);
}

static bool matchUnitStrideDiv(Value *V, Value *Src, unsigned &Divisor);

static bool isZeroFloat(Value *V) {
  if (auto *C = dyn_cast<ConstantFP>(V))
    return C->isZero();
  if (auto *CI = dyn_cast<ConstantInt>(V))
    return CI->isZero();
  return false;
}

/// Peel `select(c, load, 0)` / `select(c, 0, load)` left after guard folding.
static Value *peelLoadThroughTrivialSelects(Value *V, LoadInst *LI) {
  while (auto *Sel = dyn_cast<SelectInst>(V)) {
    Value *T = Sel->getTrueValue();
    Value *F = Sel->getFalseValue();
    if (isZeroFloat(F))
      V = T;
    else if (isZeroFloat(T))
      V = F;
    else
      break;
  }
  if (auto *Cast = dyn_cast<CastInst>(V))
    if (Cast->getOperand(0) == LI)
      return Cast;
  return V;
}

static bool matchUnitStrideDiv(Value *V, Value *Src, unsigned &Divisor) {
  if (auto *BO = dyn_cast<BinaryOperator>(V)) {
    if ((BO->getOpcode() == Instruction::UDiv ||
         BO->getOpcode() == Instruction::SDiv) &&
        BO->getOperand(0) == Src) {
      if (auto *C = dyn_cast<ConstantInt>(BO->getOperand(1))) {
        Divisor = C->getZExtValue();
        return Divisor >= VectorWidth && (Divisor % VectorWidth) == 0;
      }
    }
    if ((BO->getOpcode() == Instruction::LShr ||
         BO->getOpcode() == Instruction::AShr) &&
        BO->getOperand(0) == Src) {
      if (auto *C = dyn_cast<ConstantInt>(BO->getOperand(1))) {
        unsigned Shift = C->getZExtValue();
        if (Shift >= 2 && Shift < 31) {
          Divisor = 1u << Shift;
          return (Divisor % VectorWidth) == 0;
        }
      }
    }
  }
  return false;
}

/// Return C if V is (Src urem C), (Src and (C-1)), or
/// `Src - (Src/C)*C` / `Src - ((Src>>log2(C))<<log2(C))` with C a power of two.
static bool matchUnitStrideRem(Value *V, Value *Src, unsigned &Modulus) {
  if (auto *BO = dyn_cast<BinaryOperator>(V)) {
    if ((BO->getOpcode() == Instruction::URem ||
         BO->getOpcode() == Instruction::SRem) &&
        BO->getOperand(0) == Src) {
      if (auto *C = dyn_cast<ConstantInt>(BO->getOperand(1))) {
        Modulus = C->getZExtValue();
        return Modulus >= VectorWidth && (Modulus % VectorWidth) == 0;
      }
    }
    if (BO->getOpcode() == Instruction::And) {
      Value *Masked = BO->getOperand(0);
      Value *MaskV = BO->getOperand(1);
      if (Masked != Src)
        std::swap(Masked, MaskV);
      if (Masked == Src) {
        if (auto *C = dyn_cast<ConstantInt>(MaskV)) {
          uint64_t Mask = C->getZExtValue();
          if (Mask && ((Mask + 1) & Mask) == 0) {
            Modulus = static_cast<unsigned>(Mask + 1);
            return Modulus >= VectorWidth && (Modulus % VectorWidth) == 0;
          }
        }
      }
    }
    // Real clang HIP: `lk = idx - lm * C` with `lm = idx / C` (or lshr/shl).
    if (BO->getOpcode() == Instruction::Sub && BO->getOperand(0) == Src) {
      Value *Scaled = BO->getOperand(1);
      if (auto *ScaleBO = dyn_cast<BinaryOperator>(Scaled)) {
        Value *Lm = nullptr;
        unsigned C = 0;
        if (ScaleBO->getOpcode() == Instruction::Mul) {
          Value *Op0 = ScaleBO->getOperand(0);
          Value *Op1 = ScaleBO->getOperand(1);
          if (auto *CI = dyn_cast<ConstantInt>(Op1)) {
            Lm = Op0;
            C = CI->getZExtValue();
          } else if (auto *CI = dyn_cast<ConstantInt>(Op0)) {
            Lm = Op1;
            C = CI->getZExtValue();
          }
        } else if (ScaleBO->getOpcode() == Instruction::Shl) {
          if (auto *CI = dyn_cast<ConstantInt>(ScaleBO->getOperand(1))) {
            unsigned Shift = CI->getZExtValue();
            if (Shift >= 2 && Shift < 31) {
              Lm = ScaleBO->getOperand(0);
              C = 1u << Shift;
            }
          }
        }
        if (Lm && C >= VectorWidth && (C % VectorWidth) == 0) {
          unsigned D = 0;
          if (matchUnitStrideDiv(Lm, Src, D) && D == C) {
            Modulus = C;
            return true;
          }
        }
      }
    }
  }
  return false;
}

static bool isLocalOrLDSPointer(Value *Ptr) {
  return Ptr->getType()->getPointerAddressSpace() == AMDGPUAS::LOCAL_ADDRESS;
}

static bool isGlobalPointer(Value *Ptr) {
  unsigned AS = Ptr->getType()->getPointerAddressSpace();
  return AS == AMDGPUAS::GLOBAL_ADDRESS || AS == AMDGPUAS::CONSTANT_ADDRESS;
}

/// Collect rem/div ops eligible for float4 index rewrite. Rem may sit on the
/// loop-invariant Init when Step is a multiple of Modulus. Rem may also be
/// absent entirely in that case (LDS linearized on IV; column rem unused).
static bool collectFloat4IndexOps(PHINode *IV, Value *Init, unsigned Modulus,
                                  uint64_t Step,
                                  SmallVectorImpl<Instruction *> &Rems,
                                  SmallVectorImpl<Instruction *> &Divs) {
  Rems.clear();
  Divs.clear();
  for (User *U : IV->users()) {
    unsigned M = 0, D = 0;
    if (matchUnitStrideRem(U, IV, M) && M == Modulus)
      Rems.push_back(cast<Instruction>(U));
    if (matchUnitStrideDiv(U, IV, D) && D == Modulus)
      Divs.push_back(cast<Instruction>(U));
  }
  if (Rems.empty() && Init && Modulus && (Step % Modulus) == 0) {
    for (User *U : Init->users()) {
      unsigned M = 0;
      if (matchUnitStrideRem(U, Init, M) && M == Modulus)
        Rems.push_back(cast<Instruction>(U));
    }
  }
  if (Divs.empty())
    return false;
  // Need an explicit rem, or a step that makes rem loop-invariant so HIP may
  // omit / sink it away from the staging IV.
  return !Rems.empty() || (Init && (Step % Modulus) == 0);
}

/// Collect the rem ops that define this loop's per-thread column, including
/// the invariant `Init % Modulus` form that LICM sinks out of the loop when
/// Step is a multiple of Modulus (BMM: tid%32 with step 256).
static void collectColumnRems(PHINode *IV, Value *Init, unsigned Modulus,
                              uint64_t Step,
                              const SmallPtrSetImpl<BasicBlock *> &LoopBlocks,
                              SmallVectorImpl<Instruction *> &Rems) {
  for (User *U : IV->users()) {
    unsigned M = 0;
    if (matchUnitStrideRem(U, IV, M) && M == Modulus)
      Rems.push_back(cast<Instruction>(U));
  }
  if (!Init || Modulus == 0 || (Step % Modulus) != 0)
    return;
  for (User *U : Init->users()) {
    unsigned M = 0;
    if (matchUnitStrideRem(U, Init, M) && M == Modulus &&
        !llvm::is_contained(Rems, cast<Instruction>(U)))
      Rems.push_back(cast<Instruction>(U));
  }
  for (BasicBlock *BB : LoopBlocks) {
    for (Instruction &I : *BB) {
      unsigned M = 0;
      if ((matchUnitStrideRem(&I, Init, M) || matchUnitStrideRem(&I, IV, M)) &&
          M == Modulus && !llvm::is_contained(Rems, &I))
        Rems.push_back(&I);
    }
  }
}

/// Retarget the in-loop uses of `IV / Modulus` (the staging row) to
/// `IV / (Modulus / 4)`. Float4 re-tiling keeps the row pitch but gives each
/// thread four contiguous columns, so the row index advances 4x per step.
/// Returns the number of uses retargeted.
static unsigned rewriteRowForFloat4(PHINode *IV, unsigned Modulus,
                                    const SmallPtrSetImpl<BasicBlock *> &LoopBlocks,
                                    SmallVectorImpl<Instruction *> &ToErase) {
  SmallVector<Instruction *, 8> Divs;
  for (User *U : IV->users()) {
    unsigned D = 0;
    if (matchUnitStrideDiv(U, IV, D) && D == Modulus)
      Divs.push_back(cast<Instruction>(U));
  }
  if (Divs.empty())
    return 0;

  IRBuilder<> B(&*IV->getParent()->getFirstInsertionPt());
  Value *NewDiv = B.CreateUDiv(
      IV, ConstantInt::get(IV->getType(), Modulus / VectorWidth),
      IV->getName() + ".row4");

  unsigned Replaced = 0;
  for (Instruction *Div : Divs) {
    SmallVector<Use *, 8> Uses;
    for (Use &U : Div->uses())
      Uses.push_back(&U);
    for (Use *U : Uses) {
      auto *UserI = dyn_cast<Instruction>(U->getUser());
      if (!UserI || !LoopBlocks.count(UserI->getParent()))
        continue;
      U->set(NewDiv);
      ++Replaced;
    }
    if (Div->use_empty())
      ToErase.push_back(Div);
  }
  return Replaced;
}

/// Node budget for the address-expression walks below. Exceeding it is
/// reported as "inconclusive" so callers refuse the transform.
static constexpr unsigned AddressWalkBudget = 256;
static constexpr unsigned InconclusiveCount = ~0u;

/// Count occurrences of `Target` anywhere in the expression rooted at `V`.
static unsigned countOccurrencesAnywhere(Value *V, Value *Target) {
  SmallVector<Value *, 32> Worklist = {V};
  unsigned Count = 0, Visited = 0;
  while (!Worklist.empty()) {
    Value *Cur = Worklist.pop_back_val();
    if (Cur == Target) {
      ++Count;
      continue;
    }
    if (++Visited > AddressWalkBudget)
      return InconclusiveCount;
    if (auto *I = dyn_cast<Instruction>(Cur))
      if (!isa<PHINode>(I) && !I->mayReadOrWriteMemory())
        for (Value *Op : I->operands())
          Worklist.push_back(Op);
  }
  return Count;
}

/// Count occurrences of `Target` reachable from `V` through operations that
/// contribute it to the address with coefficient exactly +1: GEP indices,
/// add, disjoint-or, and widening casts. Anything else (mul, shl, sub, and)
/// scales or negates, so it is deliberately not traversed.
static unsigned countUnitCoefficientOccurrences(Value *V, Value *Target) {
  SmallVector<Value *, 32> Worklist = {V};
  unsigned Count = 0, Visited = 0;
  while (!Worklist.empty()) {
    Value *Cur = Worklist.pop_back_val();
    if (Cur == Target) {
      ++Count;
      continue;
    }
    if (++Visited > AddressWalkBudget)
      return InconclusiveCount;
    auto *I = dyn_cast<Instruction>(Cur);
    if (!I)
      continue;
    switch (I->getOpcode()) {
    case Instruction::GetElementPtr:
    case Instruction::Add:
    case Instruction::Or:
    case Instruction::SExt:
    case Instruction::ZExt:
    case Instruction::Trunc:
    case Instruction::BitCast:
    case Instruction::AddrSpaceCast:
      for (Value *Op : I->operands())
        Worklist.push_back(Op);
      break;
    default:
      break;
    }
  }
  return Count;
}

/// The float4 re-tiling replaces the per-thread column by correcting the final
/// address with `col4 - ColOld`. That cancels only if `ColOld` contributes to
/// the address exactly once and with coefficient +1.
static bool columnCancelsInAddress(Value *Ptr, Value *ColOld) {
  unsigned Total = countOccurrencesAnywhere(Ptr, ColOld);
  if (Total != 1)
    return false;
  return countUnitCoefficientOccurrences(Ptr, ColOld) == 1;
}

/// Rebase a staging memory op by `Delta` float elements.
static void offsetMemoryPointer(Instruction *MemI, Value *Delta) {
  IRBuilder<> B(MemI);
  auto *LI = dyn_cast<LoadInst>(MemI);
  auto *SI = dyn_cast<StoreInst>(MemI);
  Value *Ptr = LI ? LI->getPointerOperand() : SI->getPointerOperand();
  Value *NewPtr = B.CreateGEP(getFloatTy(MemI->getContext()), Ptr, Delta,
                              Ptr->getName() + ".col4");
  if (LI)
    LI->setOperand(LoadInst::getPointerOperandIndex(), NewPtr);
  else
    SI->setOperand(StoreInst::getPointerOperandIndex(), NewPtr);
}

/// Walk select/phi/cast/zero to find the unique float load feeding V, if any.
static LoadInst *findUniqueFloatLoad(Value *V) {
  SmallPtrSet<Value *, 8> Seen;
  SmallVector<Value *, 8> Worklist = {V};
  LoadInst *Found = nullptr;
  while (!Worklist.empty()) {
    Value *Cur = Worklist.pop_back_val();
    if (!Seen.insert(Cur).second)
      continue;
    if (isZeroFloat(Cur) || isa<UndefValue>(Cur) || isa<PoisonValue>(Cur))
      continue;
    if (auto *LI = dyn_cast<LoadInst>(Cur)) {
      if (!LI->getType()->isFloatTy())
        return nullptr;
      if (Found && Found != LI)
        return nullptr;
      Found = LI;
      continue;
    }
    if (auto *Cast = dyn_cast<CastInst>(Cur)) {
      Worklist.push_back(Cast->getOperand(0));
      continue;
    }
    if (auto *Sel = dyn_cast<SelectInst>(Cur)) {
      Worklist.push_back(Sel->getTrueValue());
      Worklist.push_back(Sel->getFalseValue());
      continue;
    }
    if (auto *Phi = dyn_cast<PHINode>(Cur)) {
      for (Value *Inc : Phi->incoming_values())
        Worklist.push_back(Inc);
      continue;
    }
    return nullptr;
  }
  return Found;
}

/// Clone address math that does not dominate Before into Before's block, then
/// emit a new load there. Needed for the HIP diamond:
///   load_bb: ptr = gep ...; v = load ptr; br merge
///   merge:   p = phi [v, 0]; store p
/// where neither the load nor its gep dominate the store.
static LoadInst *reconstituteLoadBefore(LoadInst *LI, Instruction *Before,
                                        DominatorTree &DT,
                                        const SmallPtrSetImpl<BasicBlock *> &Region) {
  DenseMap<Value *, Value *> Mapped;
  std::function<Value *(Value *)> MapValue = [&](Value *V) -> Value * {
    if (auto It = Mapped.find(V); It != Mapped.end())
      return It->second;
    if (auto *I = dyn_cast<Instruction>(V)) {
      if (DT.dominates(I, Before))
        return Mapped[V] = I;
      if (!Region.count(I->getParent()))
        return nullptr;
      SmallVector<Value *, 4> NewOps;
      NewOps.reserve(I->getNumOperands());
      for (Value *Op : I->operands()) {
        Value *MappedOp = MapValue(Op);
        if (!MappedOp)
          return nullptr;
        NewOps.push_back(MappedOp);
      }
      Instruction *Cloned = I->clone();
      for (unsigned Idx = 0, E = NewOps.size(); Idx != E; ++Idx)
        Cloned->setOperand(Idx, NewOps[Idx]);
      Cloned->insertBefore(Before);
      Cloned->setName(I->getName() + ".atstore");
      return Mapped[V] = Cloned;
    }
    return Mapped[V] = V;
  };

  Value *NewPtr = MapValue(LI->getPointerOperand());
  if (!NewPtr)
    return nullptr;
  IRBuilder<> B(Before);
  LoadInst *NewLI = B.CreateAlignedLoad(LI->getType(), NewPtr, LI->getAlign(),
                                        LI->getName() + ".atstore");
  NewLI->setOrdering(LI->getOrdering());
  NewLI->setSyncScopeID(LI->getSyncScopeID());
  return NewLI;
}

static bool valueDependsOn(Value *V, Value *Target,
                           SmallPtrSetImpl<Value *> &Seen) {
  if (V == Target)
    return true;
  auto *I = dyn_cast<Instruction>(V);
  if (!I || !Seen.insert(V).second)
    return false;
  for (Value *Op : I->operands())
    if (valueDependsOn(Op, Target, Seen))
      return true;
  return false;
}

static bool addrDependsOnIV(Instruction *MemI, Value *IV, Value *Init) {
  SmallPtrSet<Value *, 16> Seen;
  Value *Ptr = nullptr;
  if (auto *LI = dyn_cast<LoadInst>(MemI))
    Ptr = LI->getPointerOperand();
  else if (auto *SI = dyn_cast<StoreInst>(MemI))
    Ptr = SI->getPointerOperand();
  if (!Ptr)
    return false;
  if (valueDependsOn(Ptr, IV, Seen))
    return true;
  if (Init) {
    Seen.clear();
    if (valueDependsOn(Ptr, Init, Seen))
      return true;
  }
  return false;
}

static bool widenLoadStoreToFloat4(LoadInst *LI, StoreInst *SI,
                                   bool MaterializeAtStore) {
  if (!LI || !SI)
    return false;
  if (!LI->getType()->isFloatTy() ||
      !SI->getValueOperand()->getType()->isFloatTy())
    return false;
  if (!MaterializeAtStore) {
    if (SI->getValueOperand() != LI &&
        !(isa<CastInst>(SI->getValueOperand()) &&
          cast<CastInst>(SI->getValueOperand())->getOperand(0) == LI))
      return false;
  }
  if (!isGlobalPointer(LI->getPointerOperand()) ||
      !isLocalOrLDSPointer(SI->getPointerOperand()))
    return false;

  // Only claim alignment we can prove. The global row stride is a runtime K,
  // so the global base need not be 16-byte aligned and the load has to keep
  // the original alignment; claiming more would be a miscompile (gfx9+ still
  // selects a b128 load for an unaligned <4 x float> when unaligned access is
  // enabled). The LDS side is different: the offset is
  // row * Modulus + column with both terms multiples of VectorWidth, so once
  // the LDS object itself is 16-byte aligned every access is too. Raising the
  // alignment of an LDS object we own only affects its layout.
  Align LoadAlign = LI->getAlign();
  Align StoreAlign = SI->getAlign();
  if (auto *Obj = dyn_cast<GlobalVariable>(
          getUnderlyingObject(SI->getPointerOperand()))) {
    if (Obj->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS) {
      if (Obj->getAlign().valueOrOne() < Align(16))
        Obj->setAlignment(Align(16));
      StoreAlign = std::max(StoreAlign, Align(16));
    }
  }

  Instruction *LoadIP = MaterializeAtStore ? static_cast<Instruction *>(SI)
                                           : static_cast<Instruction *>(LI);
  IRBuilder<> B(LoadIP);
  Type *VecTy = getFloat4Ty(LI->getContext());
  LoadInst *NewLoad =
      B.CreateAlignedLoad(VecTy, LI->getPointerOperand(), LoadAlign,
                          LI->getName() + ".v4");
  NewLoad->setOrdering(LI->getOrdering());
  NewLoad->setSyncScopeID(LI->getSyncScopeID());

  B.SetInsertPoint(SI);
  StoreInst *NewStore = B.CreateAlignedStore(NewLoad, SI->getPointerOperand(),
                                             StoreAlign);
  NewStore->setOrdering(SI->getOrdering());
  NewStore->setSyncScopeID(SI->getSyncScopeID());

  SI->eraseFromParent();
  if (LI->use_empty())
    LI->eraseFromParent();
  return true;
}

/// Fair-ablation payoff: turn a scalar cooperative global→LDS staging loop
/// into a float4 loop on the proven interior path.
///
/// Recognizes IV += Step, IV < Bound with Bound%4==0, and indexing of the form
/// (IV / Modulus, IV % Modulus) feeding one float load (global) and one float
/// store (LDS). Rewrites the trip count to Bound/4 and replaces rem/div so each
/// iteration owns a unique <4 x float> chunk.
///
/// \p RequireUnguarded rejects any loop whose staging access is predicated.
/// Widening makes one load cover four elements, so it is only sound where all
/// four are known in bounds.  A cloned interior region has that by
/// construction; anywhere else the loop must be proven guard-free.
static bool widenOneCooperativeStagingLoop(PHINode *IV, DominatorTree &DT,
                                           bool RequireUnguarded) {
  BasicBlock *Header = IV->getParent();
  if (IV->getNumIncomingValues() != 2)
    return false;

  // Find latch: incoming that is Add IV, StepC.
  BasicBlock *Latch = nullptr;
  BinaryOperator *Add = nullptr;
  ConstantInt *StepC = nullptr;
  for (unsigned I = 0; I != IV->getNumIncomingValues(); ++I) {
    if (auto *BO = dyn_cast<BinaryOperator>(IV->getIncomingValue(I))) {
      if (BO->getOpcode() == Instruction::Add && BO->getOperand(0) == IV) {
        if (auto *C = dyn_cast<ConstantInt>(BO->getOperand(1))) {
          Latch = IV->getIncomingBlock(I);
          Add = BO;
          StepC = C;
          break;
        }
      }
      if (BO->getOpcode() == Instruction::Add && BO->getOperand(1) == IV) {
        if (auto *C = dyn_cast<ConstantInt>(BO->getOperand(0))) {
          Latch = IV->getIncomingBlock(I);
          Add = BO;
          StepC = C;
          break;
        }
      }
    }
  }
  if (!Latch || !Add || !StepC || StepC->isZero())
    return false;

  auto *LatchBr = dyn_cast<BranchInst>(Latch->getTerminator());
  if (!LatchBr || !LatchBr->isConditional())
    return false;
  auto *Cmp = dyn_cast<ICmpInst>(LatchBr->getCondition());
  if (!Cmp ||
      (Cmp->getPredicate() != ICmpInst::ICMP_ULT &&
       Cmp->getPredicate() != ICmpInst::ICMP_SLT))
    return false;

  Value *CmpIV = Cmp->getOperand(0);
  Value *CmpBound = Cmp->getOperand(1);
  if (CmpIV != Add && CmpIV != IV)
    std::swap(CmpIV, CmpBound);
  if (CmpIV != Add && CmpIV != IV)
    return false;
  auto *BoundC = dyn_cast<ConstantInt>(CmpBound);
  if (!BoundC)
    return false;
  const uint64_t Step = StepC->getZExtValue();
  uint64_t CmpLimit = BoundC->getZExtValue();
  // HIP often emits `idx + Step < N` as `idx < N-Step` (e.g. 3840 for N=4096,
  // Step=256). Float4 rewrite must use the true element limit N.
  uint64_t ElementLimit = CmpLimit;
  if (CmpIV == IV && Step != 0 && CmpLimit % Step == 0)
    ElementLimit = CmpLimit + Step;
  else if (CmpIV == Add)
    ElementLimit = CmpLimit;
  if (ElementLimit < VectorWidth || (ElementLimit % VectorWidth) != 0)
    return false;
  if (Step % VectorWidth != 0)
    return false;
  const uint64_t Float4Bound = ElementLimit / VectorWidth;
  // The replacement constant has to keep the form of the original compare. For
  // `idx < N-Step` it is Float4Bound-Step; using Float4Bound there would buy an
  // extra trip that stages a row past the end of the tile.
  if (CmpIV == IV && Float4Bound < Step)
    return false;
  const uint64_t NewCmpLimit =
      CmpIV == IV ? Float4Bound - Step : Float4Bound;

  Value *Init = nullptr;
  for (unsigned I = 0; I != IV->getNumIncomingValues(); ++I)
    if (IV->getIncomingBlock(I) != Latch)
      Init = IV->getIncomingValue(I);

  // Discover modulus: prefer rem/and/sub of IV; if clang sunk rem because
  // Step is a multiple of the tile dimension (BMM: step 256, mod 32/128),
  // take modulus from div/lshr of IV instead.
  unsigned Modulus = 0;
  for (User *U : IV->users()) {
    unsigned M = 0;
    if (matchUnitStrideRem(U, IV, M)) {
      Modulus = M;
      break;
    }
  }
  if (!Modulus) {
    for (User *U : IV->users()) {
      unsigned D = 0;
      if (matchUnitStrideDiv(U, IV, D)) {
        Modulus = D;
        break;
      }
    }
  }
  if (!Modulus) {
    LLVM_DEBUG(dbgs() << "Widen skipped (no rem/div modulus) for IV in "
                      << Header->getName() << '\n');
    return false;
  }

  bool HasDiv = false;
  for (User *U : IV->users()) {
    unsigned D = 0;
    if (matchUnitStrideDiv(U, IV, D) && D == Modulus) {
      HasDiv = true;
      break;
    }
  }
  if (!HasDiv) {
    LLVM_DEBUG(dbgs() << "Widen skipped (no matching div/lshr) for IV in "
                      << Header->getName() << " modulus " << Modulus << '\n');
    return false;
  }

  // Rem of IV is ideal. When step is a multiple of the tile dim, clang sinks
  // rem onto tid and the interior clone may not reference it at all (LDS is
  // often linearized on the raw IV). Allow that shape.
  bool HasRem = false;
  for (User *U : IV->users()) {
    unsigned M = 0;
    if (matchUnitStrideRem(U, IV, M) && M == Modulus) {
      HasRem = true;
      break;
    }
  }
  if (!HasRem && Init && (Step % Modulus) == 0)
    HasRem = true;
  if (!HasRem) {
    LLVM_DEBUG(dbgs() << "Widen skipped (no rem on IV or invariant Init) for IV in "
                      << Header->getName() << " modulus " << Modulus
                      << " step " << Step << '\n');
    return false;
  }

  // Pair an LDS float store with the unique global float load that feeds it
  // (through phi/select/cast). Do not pick "last load" and "last store"
  // independently — that mismatches sibling staging loops.
  LoadInst *LI = nullptr;
  StoreInst *SI = nullptr;
  PHINode *StorePhi = nullptr;
  SmallVector<BasicBlock *, 8> Worklist = {Header};
  SmallPtrSet<BasicBlock *, 8> Visited;
  while (!Worklist.empty()) {
    BasicBlock *BB = Worklist.pop_back_val();
    if (!Visited.insert(BB).second)
      continue;
    for (Instruction &Inst : *BB) {
      auto *S = dyn_cast<StoreInst>(&Inst);
      if (!S || !S->getValueOperand()->getType()->isFloatTy() ||
          !isLocalOrLDSPointer(S->getPointerOperand()))
        continue;
      LoadInst *Cand = findUniqueFloatLoad(S->getValueOperand());
      if (!Cand || !isGlobalPointer(Cand->getPointerOperand()))
        continue;
      // Must belong to this cooperative IV (not a sibling A/B staging loop).
      if (!addrDependsOnIV(Cand, IV, Init) && !addrDependsOnIV(S, IV, Init))
        continue;
      LI = Cand;
      SI = S;
    }
    if (BB == Latch)
      continue;
    for (BasicBlock *Succ : successors(BB))
      if (Succ != Header)
        Worklist.push_back(Succ);
  }
  if (!LI || !SI || !Visited.count(LI->getParent()))
    return false;

  // Positive proof of an already-interior staging loop: the only conditional
  // branch is the loop's own back edge, so nothing predicates the access. A
  // kernel whose bounds checks simply do not match the M/N tile matcher must
  // not be mistaken for one that has no bounds checks at all.
  if (RequireUnguarded) {
    for (BasicBlock *BB : Visited) {
      if (BB == Latch)
        continue;
      auto *Br = dyn_cast<BranchInst>(BB->getTerminator());
      if (!Br || Br->isConditional()) {
        LLVM_DEBUG(dbgs()
                   << "Widen skipped (staging access is predicated) for IV in "
                   << Header->getName() << '\n');
        return false;
      }
    }
  }

  // Validate the stored value without mutating.
  Value *Stored = peelLoadThroughTrivialSelects(SI->getValueOperand(), LI);
  if (auto *Phi = dyn_cast<PHINode>(Stored)) {
    StorePhi = Phi;
    Value *IncomingLoad = nullptr;
    for (Value *Inc : Phi->incoming_values()) {
      if (isa<Constant>(Inc) || isZeroFloat(Inc))
        continue;
      Value *Src = peelLoadThroughTrivialSelects(Inc, LI);
      if (auto *Cast = dyn_cast<CastInst>(Src))
        Src = Cast->getOperand(0);
      if (Src != LI)
        return false;
      IncomingLoad = LI;
    }
    if (!IncomingLoad)
      return false;
  } else if (Stored != LI) {
    StorePhi = nullptr;
    if (!(isa<CastInst>(Stored) &&
          cast<CastInst>(Stored)->getOperand(0) == LI)) {
      if (findUniqueFloatLoad(SI->getValueOperand()) != LI)
        return false;
    }
  } else {
    StorePhi = nullptr;
  }

  // A merge with zero, or a load that does not already reach the store, both
  // mean the value was predicated. Reconstituting the load would speculate it.
  if (RequireUnguarded && (StorePhi || !DT.dominates(LI, SI))) {
    LLVM_DEBUG(dbgs()
               << "Widen skipped (staged value is predicated) for IV in "
               << Header->getName() << '\n');
    return false;
  }

  // Diamond CFG after guard strip: load/gep live in a side block and feed a
  // phi at the store. Reconstitute address+load immediately before the store
  // so the widen has a dominated scalar load to replace.
  if (!DT.dominates(LI, SI)) {
    Instruction *InsertBefore = StorePhi ? static_cast<Instruction *>(StorePhi)
                                         : static_cast<Instruction *>(SI);
    LoadInst *LocalLI =
        reconstituteLoadBefore(LI, InsertBefore, DT, Visited);
    if (!LocalLI) {
      LLVM_DEBUG(dbgs() << "Widen skipped (cannot reconstitute load at store) for IV in "
                        << Header->getName() << '\n');
      return false;
    }
    if (StorePhi) {
      StorePhi->replaceAllUsesWith(LocalLI);
      if (StorePhi->use_empty()) {
        StorePhi->eraseFromParent();
        StorePhi = nullptr;
      }
    } else if (SI->getValueOperand() != LocalLI) {
      SI->setOperand(0, LocalLI);
    }
    if (LI->use_empty())
      LI->eraseFromParent();
    LI = LocalLI;
    DT.recalculate(*Header->getParent());
    if (!DT.dominates(LI, SI)) {
      LLVM_DEBUG(dbgs() << "Widen skipped (reconstituted load still does not dominate) for IV in "
                        << Header->getName() << '\n');
      return false;
    }
  }

  SmallVector<Instruction *, 8> Rems;
  SmallVector<Instruction *, 8> Divs;
  if (!collectFloat4IndexOps(IV, Init, Modulus, Step, Rems, Divs)) {
    LLVM_DEBUG(dbgs() << "Widen skipped (collect rem/div failed) for IV in "
                      << Header->getName() << '\n');
    return false;
  }
  // The source gives each thread a single column (`Init % Modulus`), and LICM
  // normally sinks that invariant column into the staging base pointers, so
  // there is nothing left inside the loop to rewrite. Rather than decompose
  // those hoisted bases (which are shared with the guarded fallback path),
  // re-tile by correcting the final addresses: each thread instead takes the
  // four contiguous columns at `(IV % (Modulus/4)) * 4`.
  Rems.clear();
  collectColumnRems(IV, Init, Modulus, Step, Visited, Rems);
  // Cloning reshaped the CFG, so the cached tree may be stale here.
  DT.recalculate(*Header->getParent());
  Instruction *ColOld = nullptr;
  for (Instruction *Rem : Rems) {
    if (Rem->getType() != IV->getType())
      continue;
    if (!DT.dominates(Rem, LI) || !DT.dominates(Rem, SI))
      continue;
    // Both addresses must contain exactly this column, exactly once, with
    // coefficient +1, or the correction will not cancel.
    if (!columnCancelsInAddress(LI->getPointerOperand(), Rem) ||
        !columnCancelsInAddress(SI->getPointerOperand(), Rem))
      continue;
    ColOld = Rem;
    break;
  }
  if (!Rems.empty() && !ColOld) {
    LLVM_DEBUG(dbgs() << "Widen skipped (no column rem cancels in both staging addresses) for IV in "
                      << Header->getName() << '\n');
    return false;
  }

  // Re-tiling redefines the row as IV/(Modulus/4), so a row index that escapes
  // the staging loop would be read with the wrong meaning.
  for (Instruction *Div : Divs)
    for (User *U : Div->users()) {
      auto *UserI = dyn_cast<Instruction>(U);
      if (!UserI || !Visited.count(UserI->getParent())) {
        LLVM_DEBUG(dbgs() << "Widen skipped (row index escapes staging loop) for IV in "
                          << Header->getName() << '\n');
        return false;
      }
    }

  // After retargeting, the load should only feed this store (and a transient
  // phi/select we are about to drop).
  for (User *U : LI->users()) {
    if (U == SI || U == StorePhi)
      continue;
    if (auto *Cast = dyn_cast<CastInst>(U)) {
      if (Cast->hasOneUse() &&
          (Cast->user_back() == SI || Cast->user_back() == StorePhi))
        continue;
    }
    if (auto *Sel = dyn_cast<SelectInst>(U)) {
      if (Sel->hasOneUse() && Sel->user_back() == SI)
        continue;
    }
    LLVM_DEBUG(dbgs() << "Widen skipped (load has extra users) for IV in "
                      << Header->getName() << '\n');
    return false;
  }

  // --- Commit mutations (no early return after this point) ---
  if (SI->getValueOperand() != LI) {
    Value *Op = SI->getValueOperand();
    if (Op == StorePhi || peelLoadThroughTrivialSelects(Op, LI) == LI ||
        (isa<CastInst>(Op) && cast<CastInst>(Op)->getOperand(0) == LI) ||
        findUniqueFloatLoad(Op) == LI)
      SI->setOperand(0, LI);
  }
  if (StorePhi && StorePhi->use_empty())
    StorePhi->eraseFromParent();

  SmallVector<Instruction *, 8> DeadIdx;
  const unsigned Inner4 = Modulus / VectorWidth;
  rewriteRowForFloat4(IV, Modulus, Visited, DeadIdx);

  // Shrink the trip count, keeping the compare in its original form.
  Constant *NewBound = ConstantInt::get(BoundC->getType(), NewCmpLimit);
  if (Cmp->getOperand(0) == CmpBound)
    Cmp->setOperand(0, NewBound);
  else
    Cmp->setOperand(1, NewBound);

  if (ColOld) {
    // Swap the hoisted per-thread column for the float4 column at each memory
    // op: delta = (IV % Inner4) * 4 - ColOld. Correcting the address leaves the
    // shared base pointers (and the guarded fallback path) untouched.
    for (Instruction *MemI : {static_cast<Instruction *>(LI),
                              static_cast<Instruction *>(SI)}) {
      IRBuilder<> B(MemI);
      Value *ColNew = B.CreateMul(
          B.CreateURem(IV, ConstantInt::get(IV->getType(), Inner4),
                       IV->getName() + ".col4"),
          ConstantInt::get(IV->getType(), VectorWidth),
          IV->getName() + ".col4.scaled", /*HasNUW=*/true, /*HasNSW=*/true);
      offsetMemoryPointer(
          MemI, B.CreateSub(ColNew, ColOld, "interior.col4.delta"));
    }
  } else {
    // No column rem at all: the staging address is linearized on the raw IV,
    // so switch it to float4 element offsets (elem = IV * 4).
    IRBuilder<> ScaleB(&*IV->getParent()->getFirstInsertionPt());
    Value *ScaledIV = ScaleB.CreateMul(
        IV, ConstantInt::get(IV->getType(), VectorWidth), IV->getName() + ".elem",
        /*HasNUW=*/true, /*HasNSW=*/true);
    SmallVector<Use *, 8> IVUses;
    for (Use &U : IV->uses())
      IVUses.push_back(&U);
    for (Use *U : IVUses) {
      Instruction *UserI = dyn_cast<Instruction>(U->getUser());
      if (!UserI || UserI == Add || UserI == ScaledIV)
        continue;
      if (isa<GetElementPtrInst>(UserI)) {
        U->set(ScaledIV);
        continue;
      }
      // Scale address arithmetic that feeds GEPs, but not the index helpers we
      // just inserted.
      if (UserI->getName().ends_with(".row4") ||
          UserI->getName().ends_with(".elem"))
        continue;
      if (auto *BO = dyn_cast<BinaryOperator>(UserI)) {
        if (BO->getOpcode() == Instruction::Add ||
            BO->getOpcode() == Instruction::Or ||
            BO->getOpcode() == Instruction::Mul) {
          bool FeedsGEP = false;
          for (User *UU : BO->users())
            if (isa<GetElementPtrInst>(UU))
              FeedsGEP = true;
          if (FeedsGEP)
            U->set(ScaledIV);
        }
      }
    }
  }

  bool Widened = widenLoadStoreToFloat4(LI, SI, /*MaterializeAtStore=*/false);
  assert(Widened && "prevalidated load/store widen failed");
  (void)Widened;
  if (StorePhi && StorePhi->use_empty())
    StorePhi->eraseFromParent();
  if (LI->getParent() && LI->use_empty())
    LI->eraseFromParent();

  for (Instruction *I : DeadIdx) {
    if (I->use_empty())
      I->eraseFromParent();
  }

  ++NumInteriorStagingVectorWidens;
  LLVM_DEBUG(dbgs() << "Widened interior cooperative staging loop with IV "
                    << IV->getName() << " modulus " << Modulus
                    << " element-limit " << ElementLimit << " -> "
                    << Float4Bound
                    << (ColOld ? " (column re-tiled)" : " (linearized IV)")
                    << '\n');
  return true;
}

static unsigned widenCooperativeStagingLoops(ArrayRef<BasicBlock *> Blocks,
                                             bool RequireUnguarded) {
  if (Blocks.empty())
    return 0;
  // Cloned interior blocks are not in the pass's DT yet; build a fresh tree.
  DominatorTree DT(*Blocks.front()->getParent());
  unsigned Widened = 0;
  SmallPtrSet<PHINode *, 8> Seen;
  for (BasicBlock *BB : Blocks) {
    for (PHINode &PN : BB->phis()) {
      if (!Seen.insert(&PN).second)
        continue;
      if (!PN.getType()->isIntegerTy())
        continue;
      if (InteriorVectorizeLimit >= 0 &&
          Widened >= static_cast<unsigned>(InteriorVectorizeLimit)) {
        LLVM_DEBUG(dbgs() << "Widen limit reached; leaving remaining staging "
                             "loops scalar\n");
        return Widened;
      }
      if (widenOneCooperativeStagingLoop(&PN, DT, RequireUnguarded))
        ++Widened;
    }
  }
  return Widened;
}

static std::optional<int64_t> constantByteOffsetBetween(Value *PtrA, Value *PtrB,
                                                        const DataLayout &DL) {
  unsigned AS = cast<PointerType>(PtrA->getType())->getAddressSpace();
  // Index width is per address space on AMDGPU (64-bit global, 32-bit LDS), so
  // two pointers are only comparable when they share one.
  if (cast<PointerType>(PtrB->getType())->getAddressSpace() != AS)
    return std::nullopt;
  unsigned IndexWidth = DL.getIndexSizeInBits(AS);
  APInt OffA(IndexWidth, 0), OffB(IndexWidth, 0);
  Value *BaseA = PtrA->stripAndAccumulateConstantOffsets(DL, OffA,
                                                         /*AllowNonInbounds=*/true);
  Value *BaseB = PtrB->stripAndAccumulateConstantOffsets(DL, OffB,
                                                         /*AllowNonInbounds=*/true);
  if (BaseA != BaseB)
    return std::nullopt;
  return (OffB - OffA).getSExtValue();
}

/// Merge four contiguous scalar float loads/stores in one BB into <4 x float>.
static unsigned vectorizeAdjacentFloatChains(ArrayRef<BasicBlock *> Blocks) {
  unsigned Vectorized = 0;
  for (BasicBlock *BB : Blocks) {
    const DataLayout &DL = BB->getModule()->getDataLayout();
    SmallVector<LoadInst *, 8> Loads;
    SmallVector<StoreInst *, 8> Stores;
    for (Instruction &I : *BB) {
      if (auto *LI = dyn_cast<LoadInst>(&I))
        if (LI->getType()->isFloatTy())
          Loads.push_back(LI);
      if (auto *SI = dyn_cast<StoreInst>(&I))
        if (SI->getValueOperand()->getType()->isFloatTy())
          Stores.push_back(SI);
    }

    auto ConsumeChain =
        [&](auto &Ops, bool IsLoad) {
          SmallPtrSet<Instruction *, 8> Used;
          for (unsigned I = 0; I + VectorWidth <= Ops.size(); ++I) {
            if (Used.count(Ops[I]))
              continue;
            SmallVector<Instruction *, 4> Chain = {Ops[I]};
            for (unsigned J = I + 1; J < Ops.size() && Chain.size() < VectorWidth;
                 ++J) {
              if (Used.count(Ops[J]))
                continue;
              auto Off = constantByteOffsetBetween(
                  IsLoad ? cast<LoadInst>(Chain.front())->getPointerOperand()
                         : cast<StoreInst>(Chain.front())->getPointerOperand(),
                  IsLoad ? cast<LoadInst>(Ops[J])->getPointerOperand()
                         : cast<StoreInst>(Ops[J])->getPointerOperand(),
                  DL);
              if (!Off || *Off != static_cast<int64_t>(Chain.size() * 4))
                continue;
              // Require program order and no intervening side-effect between
              // consecutive chain members.
              Instruction *Prev = Chain.back();
              Instruction *Next = Ops[J];
              if (!Prev->comesBefore(Next))
                continue;
              bool Clean = true;
              for (Instruction *It = Prev->getNextNode(); It && It != Next;
                   It = It->getNextNode()) {
                if (It->mayReadOrWriteMemory() || It->mayHaveSideEffects()) {
                  Clean = false;
                  break;
                }
              }
              if (!Clean)
                continue;
              Chain.push_back(Ops[J]);
            }
            if (Chain.size() != VectorWidth)
              continue;

            Type *VecTy = getFloat4Ty(BB->getContext());
            if (IsLoad) {
              IRBuilder<> B(Chain.front());
              auto *First = cast<LoadInst>(Chain.front());
              // The merged access starts at the first element's address, so it
              // is only as aligned as that pointer was already known to be.
              LoadInst *NewLI =
                  B.CreateAlignedLoad(VecTy, First->getPointerOperand(),
                                      First->getAlign(), "interior.f32x4");
              for (unsigned K = 0; K != VectorWidth; ++K) {
                Value *Ext = B.CreateExtractElement(NewLI, B.getInt32(K));
                Chain[K]->replaceAllUsesWith(Ext);
                Used.insert(Chain[K]);
              }
              for (Instruction *C : Chain)
                cast<LoadInst>(C)->eraseFromParent();
            } else {
              // Build the vector at the last store: the values stored by the
              // later chain members are not defined yet at the first one.
              IRBuilder<> B(Chain.back());
              auto *First = cast<StoreInst>(Chain.front());
              Value *Vec = PoisonValue::get(VecTy);
              for (unsigned K = 0; K != VectorWidth; ++K) {
                Vec = B.CreateInsertElement(
                    Vec, cast<StoreInst>(Chain[K])->getValueOperand(),
                    B.getInt32(K));
                Used.insert(Chain[K]);
              }
              B.CreateAlignedStore(Vec, First->getPointerOperand(),
                                   First->getAlign());
              for (Instruction *C : llvm::reverse(Chain))
                cast<StoreInst>(C)->eraseFromParent();
            }
            ++Vectorized;
            ++NumInteriorFloatChainsVectorized;
          }
        };

    ConsumeChain(Loads, true);
    ConsumeChain(Stores, false);
  }
  return Vectorized;
}

/// \p RequireUnguarded must be set whenever \p Blocks are not a region whose
/// bounds checks this pass already proved away.
/// \p WidenedOut receives the number of cooperative staging loops rewritten
/// to float4 (as opposed to adjacent-chain vectorization).
static unsigned optimizeInteriorMemoryPaths(ArrayRef<BasicBlock *> Blocks,
                                            bool RequireUnguarded,
                                            unsigned *WidenedOut = nullptr) {
  if (!EnableInteriorTileVectorize)
    return 0;
  unsigned Widened = widenCooperativeStagingLoops(Blocks, RequireUnguarded);
  if (WidenedOut)
    *WidenedOut = Widened;
  unsigned Changed = Widened;
  // Merging adjacent scalar accesses is only in scope for a region this pass
  // proved interior. Run over a whole arbitrary function it just duplicates
  // the generic load/store vectorizer on kernels that are not GEMMs at all.
  if (!RequireUnguarded)
    Changed += vectorizeAdjacentFloatChains(Blocks);
  return Changed;
}

/// Clone a staging region behind a CTA-uniform dispatch.  The original entry
/// remains reachable through the other dispatch edge and is therefore the
/// fallback path.  The shared barrier is not cloned.
static bool cloneStagingRegion(Function &F, BranchInst *Dispatch,
                               BasicBlock *Entry,
                               const SmallPtrSetImpl<BasicBlock *> &Region,
                               BasicBlock *Barrier,
                               ArrayRef<FullTileBoundCheck> FullChecks,
                               ArrayRef<Value *> DirectGuards,
                               ArrayRef<Value *> Alignments, PHINode *IV,
                               Value *Bound, ScalarEvolution &SE) {
  if (!canCloneStagingRegion(Entry, Region, Barrier, FullChecks, DirectGuards,
                             Alignments, IV, Bound, SE))
    return false;

  unsigned EntrySuccessor = 0;
  while (Dispatch->getSuccessor(EntrySuccessor) != Entry)
    if (++EntrySuccessor == Dispatch->getNumSuccessors())
      return false;

  SmallVector<BasicBlock *, 8> RegionBlocks;
  for (BasicBlock &BB : F)
    if (Region.contains(&BB))
      RegionBlocks.push_back(&BB);

  ValueToValueMapTy VMap;
  for (BasicBlock *BB : RegionBlocks) {
    BasicBlock *Clone = CloneBasicBlock(BB, VMap, ".interior", &F);
    VMap[BB] = Clone;
  }

  for (BasicBlock *BB : RegionBlocks) {
    BasicBlock *Clone = cast<BasicBlock>(VMap[BB]);
    for (Instruction &I : *Clone)
      // Definitions outside the staging region (workgroup/workitem IDs and
      // tile bases) deliberately remain shared with the clone.
      RemapInstruction(&I, VMap, RF_IgnoreMissingLocals);
  }

  unsigned RemovedSafetyBranches = 0;
  for (BasicBlock *BB : RegionBlocks) {
    auto *OriginalBranch = dyn_cast<BranchInst>(BB->getTerminator());
    if (!OriginalBranch || !OriginalBranch->isConditional())
      continue;
    Value *Remainder =
        getInteriorConjunctionRemainder(OriginalBranch->getCondition(),
                                        DirectGuards);
    if (Remainder) {
      auto *ClonedBranch =
          cast<BranchInst>(cast<BasicBlock>(VMap[BB])->getTerminator());
      Value *ClonedRemainder = Remainder;
      if (Value *Mapped = VMap.lookup(Remainder))
        ClonedRemainder = Mapped;
      ClonedBranch->setCondition(ClonedRemainder);
      ++RemovedSafetyBranches;
      continue;
    }
    GuardOutcome Outcome;
    if (!getRemovablePrefixGuardOutcome(OriginalBranch, IV, Bound, FullChecks,
                                        Alignments, SE, Outcome))
      continue;

    auto *ClonedBranch =
        cast<BranchInst>(cast<BasicBlock>(VMap[BB])->getTerminator());
    BranchInst::Create(
        ClonedBranch->getSuccessor(static_cast<unsigned>(Outcome)),
        ClonedBranch);
    ClonedBranch->eraseFromParent();
    ++RemovedSafetyBranches;
  }

  Dispatch->setSuccessor(EntrySuccessor, cast<BasicBlock>(VMap[Entry]));

  SmallVector<BasicBlock *, 8> InteriorBlocks;
  for (BasicBlock *BB : RegionBlocks)
    InteriorBlocks.push_back(cast<BasicBlock>(VMap[BB]));
  const unsigned MemoryOpts =
      optimizeInteriorMemoryPaths(InteriorBlocks, /*RequireUnguarded=*/false);
#ifndef NDEBUG
  if (MemoryOpts && verifyFunction(F, &dbgs()))
    report_fatal_error(
        "AMDGPUInteriorTileSplit: memory opts produced invalid IR");
#endif

  ++NumPreparedInteriorTileSplits;
  LLVM_DEBUG(dbgs() << "Cloned canonical interior-tile staging region in "
                    << F.getName() << " at "
                    << Dispatch->getParent()->getName()
                    << "; removed " << RemovedSafetyBranches
                    << " proven lane bounds branch(es)"
                    << "; memory opts " << MemoryOpts
                    << "; fast staging entry "
                    << cast<BasicBlock>(VMap[Entry])->getName()
                    << ", fallback " << Entry->getName() << ", shared barrier "
                    << Barrier->getName() << '\n');
  return true;
}

static bool specializeInteriorEpilogueStores(
    Function &F, Loop *L, Value *MNFull,
    ArrayRef<FullTileBoundCheck> FullChecks, ArrayRef<Value *> DirectGuards,
    DominatorTree &DT, LoopInfo &LI, ScalarEvolution &SE);

/// Version only the guarded staging prefix of one outer-K iteration.  The
/// shared barrier is the reconvergence point: both CTA-uniform dispatch paths
/// execute it, then flow to the original compute and outer latch.  This is
/// intentionally not a loop versioning transform.
static bool splitOuterKStaging(Function &F, Loop *L, UniformityInfo &UI,
                               LoopInfo &LI, DominatorTree &DT,
                               ScalarEvolution &SE) {
  BasicBlock *Preheader = L->getLoopPreheader();
  OuterKLoopInfo KInfo;
  if (!getCanonicalOuterKLoop(L, SE, KInfo) || !Preheader) {
    LLVM_DEBUG(dbgs() << "Interior staging preflight rejected "
                      << (L->getHeader() ? L->getHeader()->getName()
                                         : "<no-header>")
                      << ": not a canonical outer K loop\n");
    return false;
  }
  PHINode *IV = KInfo.IV;
  Value *Bound = KInfo.Bound;
  const uint64_t TileK = KInfo.TileK;

  CanonicalTileRemainder M, N;
  SmallVector<Value *, 4> Alignments;
  bool HasM = false, HasN = false;
  bool HasDirectM = false, HasDirectN = false;
  bool HasCanonicalSetup =
      collectInteriorTileSetup(Preheader, UI, M, N, Alignments, HasM, HasN);
  // Direct lane bounds are enough on their own (no preheader clamps needed).
  bool HasDirectSetup =
      collectDirectInteriorTileBounds(L, UI, SE, LI, M, N, HasM, HasN,
                                      HasDirectM, HasDirectN);
  LLVM_DEBUG(dbgs() << "Interior staging setup for "
                    << L->getHeader()->getName()
                    << ": canonical=" << HasCanonicalSetup
                    << " M=" << HasM << " N=" << HasN
                    << " direct-M=" << HasDirectM
                    << " direct-N=" << HasDirectN
                    << " alignments=" << Alignments.size()
                    << " tileK=" << TileK
                    << " k-form="
                    << (KInfo.Form == OuterKForm::TileCount ? "tile-count"
                                                            : "offset-step")
                    << '\n');
  // Accept either clamped extents or direct lane M/N proofs. Alignments are
  // optional when both direct M and N are present.
  if ((!HasM || !HasN) ||
      (Alignments.empty() && !(HasDirectM && HasDirectN) &&
       !(HasCanonicalSetup && HasM && HasN))) {
    LLVM_DEBUG(dbgs() << "Interior staging preflight rejected "
                      << L->getHeader()->getName()
                      << ": missing or ambiguous M/N tile/alignment setup\n");
    return false;
  }
  if (!HasDirectSetup) {
    LLVM_DEBUG(dbgs() << "Interior staging preflight rejected "
                      << L->getHeader()->getName()
                      << ": conflicting direct M/N tile bounds\n");
    return false;
  }
  SmallVector<FullTileBoundCheck, 2> FullChecks{
      {M.Dimension, M.Bound, M.Base, M.IsSigned, M.TileMN},
      {N.Dimension, N.Bound, N.Base, N.IsSigned, N.TileMN}};
  SmallVector<Value *, 2> DirectGuards{M.Difference, N.Difference};

  // First prove the old CFG, before changing it.  The candidate must end at a
  // shared barrier whose exits remain in the outer K iteration and outside
  // staging.  Real GEMM has both compute and skip-compute barrier exits.
  BasicBlock *StagingEntry = nullptr;
  BasicBlock *StagingPredecessor = nullptr;
  BasicBlock *Barrier = nullptr;
  bool HasNestedStagingLoop = false;
  bool SplitHeaderForDispatch = false;
  for (BasicBlock *BB : L->blocks()) {
    const bool IsHeader = BB == L->getHeader();
    Instruction *FirstNonPHI = BB->getFirstNonPHI();
    if (!FirstNonPHI)
      continue;
    BasicBlock *Dispatch = IsHeader ? nullptr : BB->getSinglePredecessor();
    if (!IsHeader && (!Dispatch || !L->contains(Dispatch)))
      continue;
    SmallPtrSet<BasicBlock *, 8> Candidate;
    BasicBlock *CandidateBarrier = nullptr;
    bool CandidateHasLoop = false;
    if (!findClosedStagingRegion(BB, Dispatch, Candidate, CandidateBarrier,
                                 /*AllowNestedLoops=*/true,
                                 &CandidateHasLoop,
                                 /*AllowEntryExternalPredecessors=*/IsHeader) ||
        !L->contains(CandidateBarrier) ||
        !hasOnlySharedBarrierExits(L, Candidate, CandidateBarrier) ||
        !canCloneStagingRegion(BB, Candidate, CandidateBarrier, FullChecks,
                               DirectGuards, Alignments, IV, Bound, SE,
                               /*IgnoreEntryInstructions=*/IsHeader))
      continue;
    if (Candidate.size() > MaxStagingCloneBlocks) {
      LLVM_DEBUG(dbgs() << "Interior staging candidate rejected at "
                        << BB->getName() << ": " << Candidate.size()
                        << " blocks exceeds clone limit "
                        << MaxStagingCloneBlocks << '\n');
      continue;
    }
    StagingEntry = BB;
    StagingPredecessor = Dispatch;
    Barrier = CandidateBarrier;
    HasNestedStagingLoop = CandidateHasLoop;
    SplitHeaderForDispatch = IsHeader;
    break;
  }
  if (!StagingEntry) {
    LLVM_DEBUG(dbgs() << "Interior staging preflight rejected "
                      << L->getHeader()->getName()
                      << ": no closed pre-barrier staging region\n");
    return false;
  }

  // A real GEMM puts staging control in the outer K header.  Split the header
  // only at its terminator, retaining every PHI and setup instruction in the
  // shared dispatch.  This keeps header values used after the shared barrier
  // single-defined on both paths.  For an ordinary staging entry, use the
  // incoming-edge split as before.  In both cases the original entry is the
  // fallback and only the true edge is redirected to the clone.
  BasicBlock *Dispatch = nullptr;
  if (SplitHeaderForDispatch) {
    Dispatch = StagingEntry;
    StagingEntry = SplitBlock(Dispatch, Dispatch->getTerminator(), &DT, &LI,
                              nullptr, "staging");
    StagingPredecessor = Dispatch;
  } else {
    Dispatch = SplitEdge(StagingPredecessor, StagingEntry, &DT, &LI);
    if (!Dispatch)
      return false;
  }
  auto *DispatchBranch = cast<BranchInst>(Dispatch->getTerminator());
  SmallPtrSet<BasicBlock *, 8> Region;
  BasicBlock *SharedBarrier = nullptr;
  // SplitBlock / SplitEdge can leave a CFG that no longer matches the
  // preflight view: header-anchored candidates ignore entry live-outs and
  // external predecessors until the terminator is peeled off, and nested
  // staging graphs (tall/skinny GEMM) may then fail the closed-region or
  // clone-safety recheck.  The inserted dispatch block is a no-op fallthrough,
  // so bail out cleanly instead of aborting the compiler.
  const bool RegionOk = findClosedStagingRegion(
      StagingEntry, Dispatch, Region, SharedBarrier,
      /*AllowNestedLoops=*/true);
  const bool BarrierOk = RegionOk && SharedBarrier == Barrier;
  const bool ExitsOk =
      BarrierOk && hasOnlySharedBarrierExits(L, Region, SharedBarrier);
  const bool CloneOk =
      ExitsOk && canCloneStagingRegion(StagingEntry, Region, SharedBarrier,
                                       FullChecks, DirectGuards, Alignments, IV,
                                       Bound, SE);
  if (!CloneOk) {
    LLVM_DEBUG(dbgs() << "Interior staging aborted after dispatch split at "
                      << StagingEntry->getName()
                      << ": region=" << RegionOk << " barrier=" << BarrierOk
                      << " exits=" << ExitsOk << " clone=" << CloneOk << '\n');
    // Undo the fallthrough split when it is still a trivial single-pred /
    // single-succ edge so we do not leave a mutated CFG after declining.
    if (StagingEntry->getSinglePredecessor() == Dispatch &&
        Dispatch->getSingleSuccessor() == StagingEntry)
      MergeBlockIntoPredecessor(StagingEntry, /*DTU=*/nullptr, &LI,
                                /*MSSAU=*/nullptr, /*MemDep=*/nullptr,
                                /*PredecessorWithTwoSuccessors=*/false, &DT);
    return false;
  }

  Value *MNFull = synthesizeInteriorTileSelector(Preheader, M, N, Alignments);
  if (!MNFull) {
    LLVM_DEBUG(dbgs() << "Interior staging preflight rejected "
                      << L->getHeader()->getName()
                      << ": could not synthesize full M/N selector\n");
    return false;
  }

  // The new block is inside the outer loop, so this full-K predicate is
  // evaluated once per iteration rather than only at loop entry.
  IRBuilder<> Builder(DispatchBranch);
  Value *KFull = nullptr;
  if (KInfo.Form == OuterKForm::OffsetStep) {
    // IV is k0; a full tile needs k0 + TileK <= Bound.
    Value *KAtLeastTile = Builder.CreateICmpSGE(
        Bound, ConstantInt::get(Bound->getType(), TileK),
        "interior.k.nonnegative");
    KFull = Builder.CreateICmpSLE(
        IV,
        Builder.CreateSub(Bound, ConstantInt::get(Bound->getType(), TileK)),
        "interior.k.tile");
    KFull = Builder.CreateAnd(KAtLeastTile, KFull, "interior.k.full");
  } else {
    // IV is the tile index; skip the last (possibly partial) tile.
    KFull = Builder.CreateICmpULT(
        Builder.CreateAdd(IV, ConstantInt::get(IV->getType(), 1),
                          "interior.k.next", /*HasNUW=*/true),
        Bound, "interior.k.full");
  }
  Value *RunFast =
      Builder.CreateAnd(MNFull, KFull, "interior.staging.full");
  // cloneStagingRegion replaces the first edge to StagingEntry with the clone,
  // yielding `RunFast ? staging.interior : staging`.
  BranchInst::Create(StagingEntry, StagingEntry, RunFast, DispatchBranch);
  DispatchBranch->eraseFromParent();

  bool Changed = cloneStagingRegion(
      F, cast<BranchInst>(Dispatch->getTerminator()), StagingEntry, Region,
      SharedBarrier, FullChecks, DirectGuards, Alignments, IV, Bound, SE);
  if (Changed && HasNestedStagingLoop)
    LLVM_DEBUG(dbgs() << "Cloned nested-loop staging region in " << F.getName()
                      << "; outer K latch and shared barrier were retained\n");
  // Specialize the post-loop store epilogue behind the same CTA-uniform M/N
  // selector. Staging clone alone leaves per-element exec-masked C stores on
  // the common path — the gap the SASS comparisons rank as Priority-1 after
  // staging is cleaned up.
  if (Changed)
    Changed |= specializeInteriorEpilogueStores(F, L, MNFull, FullChecks,
                                                DirectGuards, DT, LI, SE);
  return Changed;
}

/// Halo / input-tile footprints are typically tens to low hundreds of elements
/// (e.g. IN_TILE_H = TILE_OY*STRIDE+KH-1), not the GEMM 32/64/128 tile set.
static bool isSupportedConvFootprintExtent(uint64_t Extent) {
  return Extent >= 2 && Extent <= 256;
}

/// Strip casts/freeze/add-0 so `in_x0` copies share one Base key.
static Value *peelTrivialBase(Value *V) {
  while (V) {
    if (auto *Cast = dyn_cast<CastInst>(V)) {
      if (Cast->getOpcode() == Instruction::SExt ||
          Cast->getOpcode() == Instruction::ZExt ||
          Cast->getOpcode() == Instruction::Trunc) {
        V = Cast->getOperand(0);
        continue;
      }
    }
    if (auto *Fr = dyn_cast<FreezeInst>(V)) {
      V = Fr->getOperand(0);
      continue;
    }
    if (auto *BO = dyn_cast<BinaryOperator>(V)) {
      if (BO->getOpcode() == Instruction::Add ||
          BO->getOpcode() == Instruction::Or) {
        if (auto *C = dyn_cast<ConstantInt>(BO->getOperand(1))) {
          if (C->isZero()) {
            V = BO->getOperand(0);
            continue;
          }
        } else if (auto *C = dyn_cast<ConstantInt>(BO->getOperand(0))) {
          if (C->isZero()) {
            V = BO->getOperand(1);
            continue;
          }
        }
      }
    }
    break;
  }
  return V;
}

/// Strip integer casts/freeze so SCEV/KnownBits see the narrow rem/shl, not a
/// full-width `zext i16` range of 65535 (conv11: `%74 = zext i16 %73`).
static Value *peelIntegerCastsForOffset(Value *V) {
  while (V) {
    if (auto *Cast = dyn_cast<CastInst>(V)) {
      unsigned Op = Cast->getOpcode();
      if (Op == Instruction::ZExt || Op == Instruction::SExt ||
          Op == Instruction::Trunc) {
        V = Cast->getOperand(0);
        continue;
      }
    }
    if (auto *Fr = dyn_cast<FreezeInst>(V)) {
      V = Fr->getOperand(0);
      continue;
    }
    break;
  }
  return V;
}

/// Prove an unsigned maximum strictly below Limit for conv staging offsets.
/// Prefer structural urem/and/mul/add bounds from Clang's
/// `iy = tmp % IN_TILE_H` / `ix4*4` lowering. Do NOT use loose SCEV/workitem
/// ranges (those report 255 for a tid and make H=224 selectors unreachable).
static bool getConvOffsetMaximumBelow(Value *Offset, uint64_t Limit,
                                      ScalarEvolution &SE, LoopInfo &LI,
                                      uint64_t &Maximum) {
  if (!Offset->getType()->isIntegerTy() || Limit < 2)
    return false;

  // Try the unpeeled value first (zext nneg / !range on the cast itself).
  if (getUnsignedOffsetMaximumBelow(Offset, Limit, SE, Maximum) &&
      Maximum < 128)
    return true;

  // Hipcc narrows `ix4*4` / `iy` to i16/i8 then zexts back; prove the source.
  Value *Peeled = peelIntegerCastsForOffset(Offset);
  if (Peeled != Offset) {
    if (getUnsignedOffsetMaximumBelow(Peeled, Limit, SE, Maximum) &&
        Maximum < 128)
      return true;
  }
  Offset = Peeled;

  SmallPtrSet<Value *, 16> Visited;
  std::function<bool(Value *, uint64_t &)> Structural =
      [&](Value *V, uint64_t &Max) -> bool {
    if (!Visited.insert(V).second)
      return false;
    if (auto *C = dyn_cast<ConstantInt>(V)) {
      if (C->isNegative())
        return false;
      Max = C->getZExtValue();
      return Max < Limit;
    }
    if (auto *Cast = dyn_cast<CastInst>(V)) {
      if (Cast->getOpcode() != Instruction::ZExt &&
          Cast->getOpcode() != Instruction::SExt &&
          Cast->getOpcode() != Instruction::Trunc)
        return false;
      return Structural(Cast->getOperand(0), Max);
    }
    if (auto *Fr = dyn_cast<FreezeInst>(V))
      return Structural(Fr->getOperand(0), Max);
    if (auto *Sel = dyn_cast<SelectInst>(V)) {
      uint64_t TMax = 0, FMax = 0;
      if (!Structural(Sel->getTrueValue(), TMax) ||
          !Structural(Sel->getFalseValue(), FMax))
        return false;
      Max = std::max(TMax, FMax);
      return Max < Limit;
    }
    auto *BO = dyn_cast<BinaryOperator>(V);
    if (!BO) {
      // Inductive ix4 PHIs are self-referential; Visited recursion always fails.
      // Prefer SCEV; if that is loose, accept a rem/and init incoming and treat
      // PN-using updates as rem-induction backedges (hipcc cooperative loops).
      if (auto *PN = dyn_cast<PHINode>(V)) {
        if (SE.isSCEVable(PN->getType())) {
          APInt UMax = SE.getUnsignedRangeMax(SE.getSCEV(PN));
          if (UMax.getBitWidth() <= 64 && UMax.ult(Limit) && UMax.ult(128)) {
            Max = UMax.getZExtValue();
            return true;
          }
        }
        auto UsesPN = [&](Value *In) -> bool {
          SmallPtrSet<Value *, 8> Seen;
          SmallVector<Value *, 4> Stack{In};
          while (!Stack.empty()) {
            Value *Cur = Stack.pop_back_val();
            if (Cur == PN)
              return true;
            auto *I = dyn_cast<Instruction>(Cur);
            if (!I || !Seen.insert(Cur).second)
              continue;
            if (!isa<BinaryOperator>(I) && !isa<SelectInst>(I) &&
                !isa<CastInst>(I) && !isa<FreezeInst>(I))
              continue;
            for (Value *Op : I->operands())
              Stack.push_back(Op);
          }
          return false;
        };
        uint64_t Worst = 0;
        bool Any = false;
        for (Value *In : PN->incoming_values()) {
          if (UsesPN(In))
            continue;
          uint64_t Part = 0;
          if (!Structural(In, Part))
            return false;
          Worst = std::max(Worst, Part);
          Any = true;
        }
        if (!Any)
          return false;
        Max = Worst;
        return Max < Limit && Max < 128;
      }
      // AMDGPU bitfield extract of a lane index: ubfe(src, 0, Width) < 2^Width.
      if (auto *II = dyn_cast<IntrinsicInst>(V)) {
        Intrinsic::ID Id = II->getIntrinsicID();
        if (Id == Intrinsic::amdgcn_ubfe) {
          auto *WidthC = dyn_cast<ConstantInt>(II->getArgOperand(2));
          if (!WidthC || WidthC->getZExtValue() == 0 ||
              WidthC->getZExtValue() >= 8)
            return false;
          Max = (uint64_t(1) << WidthC->getZExtValue()) - 1;
          return Max < Limit && Max < 128;
        }
        if (Id == Intrinsic::umin) {
          uint64_t A = 0, B = 0;
          if (!Structural(II->getArgOperand(0), A) ||
              !Structural(II->getArgOperand(1), B))
            return false;
          Max = std::min(A, B);
          return Max < Limit;
        }
      }
      return false;
    }
    switch (BO->getOpcode()) {
    case Instruction::URem:
    case Instruction::SRem: {
      auto *Divisor = dyn_cast<ConstantInt>(BO->getOperand(1));
      if (!Divisor || Divisor->isZero() || Divisor->isNegative())
        return false;
      uint64_t D = Divisor->getZExtValue();
      if (D < 2 || D > Limit)
        return false;
      Max = D - 1;
      return true;
    }
    case Instruction::And: {
      auto *Mask = dyn_cast<ConstantInt>(BO->getOperand(1));
      if (!Mask)
        Mask = dyn_cast<ConstantInt>(BO->getOperand(0));
      if (!Mask)
        return false;
      Max = Mask->getZExtValue();
      // Reject full-byte tid masks (255) that make H=224 selectors unreachable;
      // real staging tiles use urem or small and-masks.  Also reject the
      // identity-ish case Max==0.
      return Max >= 1 && Max < Limit && Max < 128;
    }
    case Instruction::UDiv: {
      auto *Divisor = dyn_cast<ConstantInt>(BO->getOperand(1));
      if (!Divisor || Divisor->isZero())
        return false;
      uint64_t D = Divisor->getZExtValue();
      if (D < 2)
        return false;
      uint64_t OperandMaximum = 0;
      if (!Structural(BO->getOperand(0), OperandMaximum))
        return false;
      Max = OperandMaximum / D;
      return Max < Limit;
    }
    case Instruction::Mul: {
      Value *Var = BO->getOperand(0);
      auto *FactorC = dyn_cast<ConstantInt>(BO->getOperand(1));
      if (!FactorC) {
        Var = BO->getOperand(1);
        FactorC = dyn_cast<ConstantInt>(BO->getOperand(0));
      }
      if (!FactorC || FactorC->isZero() || FactorC->isNegative())
        return false;
      uint64_t Factor = FactorC->getZExtValue();
      if (Factor > Limit - 1)
        return false;
      uint64_t OperandMaximum = 0;
      if (!Structural(Var, OperandMaximum) ||
          OperandMaximum > (Limit - 1) / Factor)
        return false;
      Max = OperandMaximum * Factor;
      return true;
    }
    case Instruction::Shl: {
      auto *Amount = dyn_cast<ConstantInt>(BO->getOperand(1));
      if (!Amount || Amount->getZExtValue() >= 10)
        return false;
      uint64_t Shift = Amount->getZExtValue();
      uint64_t OperandMaximum = 0;
      if (!Structural(BO->getOperand(0), OperandMaximum))
        return false;
      if (OperandMaximum > ((Limit - 1) >> Shift))
        return false;
      Max = OperandMaximum << Shift;
      return Max < Limit;
    }
    case Instruction::LShr: {
      auto *Amount = dyn_cast<ConstantInt>(BO->getOperand(1));
      if (!Amount || Amount->getZExtValue() >= 32)
        return false;
      uint64_t Shift = Amount->getZExtValue();
      uint64_t OperandMaximum = 0;
      if (!Structural(BO->getOperand(0), OperandMaximum))
        return false;
      Max = OperandMaximum >> Shift;
      return Max < Limit;
    }
    case Instruction::Sub: {
      // Clang expands `x % C` as `x - (x/C)*C` or mul/lshr magic.
      if (matchExpandedURemMaximum(BO, Limit, Max))
        return true;
      return false;
    }
    case Instruction::Add:
    case Instruction::Or: {
      uint64_t LHSMaximum = 0, RHSMaximum = 0;
      if (!Structural(BO->getOperand(0), LHSMaximum) ||
          !Structural(BO->getOperand(1), RHSMaximum) ||
          LHSMaximum > Limit - 1 - RHSMaximum)
        return false;
      Max = LHSMaximum + RHSMaximum;
      return true;
    }
    default:
      return false;
    }
  };

  if (Structural(Offset, Maximum))
    return true;

  // KnownBits after narrowing (trunc/urem) often beats opaque SCEV.
  {
    const DataLayout &DL = SE.getDataLayout();
    KnownBits KB = computeKnownBits(Offset, DL);
    if (!KB.hasConflict()) {
      APInt MaxAP = KB.getMaxValue();
      if (MaxAP.ult(Limit) && MaxAP.ult(128)) {
        Maximum = MaxAP.getZExtValue();
        return true;
      }
    }
  }

  // ConstantRange (incl. !range metadata / trunc of urem) after peel.
  {
    const DataLayout &DL = SE.getDataLayout();
    SimplifyQuery SQ(DL);
    ConstantRange CR = computeConstantRange(Offset, /*ForSigned=*/false, SQ);
    if (!CR.isFullSet() && !CR.isWrappedSet()) {
      const APInt &UMax = CR.getUnsignedMax();
      if (UMax.ult(Limit) && UMax.ult(128)) {
        Maximum = UMax.getZExtValue();
        return true;
      }
    }
  }

  // Loop-bounded shifted offsets (GEMM-style), still structural in the latch.
  return getLoopBoundedShiftedOffsetMaximum(Offset, Limit, LI, Maximum);
}

/// Prove Index = Base + Off with Off < Limit, for a known CTA-uniform Base.
static bool matchOffsetFromKnownBase(Value *Index, Value *Base, uint64_t Limit,
                                     ScalarEvolution &SE, LoopInfo &LI,
                                     uint64_t &OffMaxOut) {
  auto SameBase = [&](Value *V) -> bool {
    if (V == Base || peelTrivialBase(V) == peelTrivialBase(Base))
      return true;
    if (!SE.isSCEVable(V->getType()) || !SE.isSCEVable(Base->getType()))
      return false;
    return SE.getSCEV(V) == SE.getSCEV(Base);
  };
  SmallPtrSet<Value *, 8> Visited;
  std::function<bool(Value *, uint64_t &)> Recurse =
      [&](Value *V, uint64_t &OffMax) -> bool {
    if (!Visited.insert(V).second)
      return false;
    if (auto *Cast = dyn_cast<CastInst>(V))
      if (Cast->getOpcode() == Instruction::SExt ||
          Cast->getOpcode() == Instruction::ZExt ||
          Cast->getOpcode() == Instruction::Trunc)
        return Recurse(Cast->getOperand(0), OffMax);
    if (auto *Fr = dyn_cast<FreezeInst>(V))
      return Recurse(Fr->getOperand(0), OffMax);
    if (SameBase(V)) {
      OffMax = 0;
      return true;
    }
    auto *BO = dyn_cast<BinaryOperator>(V);
    if (!BO || (BO->getOpcode() != Instruction::Add &&
                BO->getOpcode() != Instruction::Or))
      return false;
    uint64_t InnerOff = 0, SideOff = 0;
    if (Recurse(BO->getOperand(0), InnerOff) &&
        getConvOffsetMaximumBelow(BO->getOperand(1), Limit, SE, LI, SideOff) &&
        InnerOff <= Limit - 1 - SideOff) {
      OffMax = InnerOff + SideOff;
      return true;
    }
    Visited.erase(BO->getOperand(0));
    if (Recurse(BO->getOperand(1), InnerOff) &&
        getConvOffsetMaximumBelow(BO->getOperand(0), Limit, SE, LI, SideOff) &&
        InnerOff <= Limit - 1 - SideOff) {
      OffMax = InnerOff + SideOff;
      return true;
    }
    return false;
  };
  return Recurse(Index, OffMaxOut);
}

/// Decompose Index into a CTA-uniform Base plus a bounded nonnegative offset.
/// Peels nested `add`/`or` and widening casts so
/// `in_x0 + (ix4*4 + 3)` still recovers Base=`in_x0`.
static bool decomposeUniformBaseOffset(Value *Index, UniformityInfo &UI,
                                       ScalarEvolution &SE, LoopInfo &LI,
                                       uint64_t Limit, Value *&BaseOut,
                                       uint64_t &OffMaxOut) {
  SmallPtrSet<Value *, 8> Visited;
  std::function<bool(Value *, Value *&, uint64_t &)> Recurse =
      [&](Value *V, Value *&Base, uint64_t &OffMax) -> bool {
    if (!Visited.insert(V).second)
      return false;
    if (auto *Cast = dyn_cast<CastInst>(V))
      if (Cast->getOpcode() == Instruction::SExt ||
          Cast->getOpcode() == Instruction::ZExt ||
          Cast->getOpcode() == Instruction::Trunc)
        return Recurse(Cast->getOperand(0), Base, OffMax);
    if (auto *Fr = dyn_cast<FreezeInst>(V))
      return Recurse(Fr->getOperand(0), Base, OffMax);

    if (isCTAUniformFootprintBase(V, UI)) {
      // Constants are uniform but are never a CTA footprint Base.
      if (isa<Constant>(V))
        return false;
      Base = peelTrivialBase(V);
      OffMax = 0;
      return true;
    }

    auto *BO = dyn_cast<BinaryOperator>(V);
    // Pad bases look like `add/sub %tile, C` (C may be negative). UI sometimes
    // fails to mark the whole add uniform even when both inputs are; treat
    // uniform±const as a CTA-uniform Base so `Base + zext(ix*4)` still matches.
    if (BO && (BO->getOpcode() == Instruction::Add ||
               BO->getOpcode() == Instruction::Sub)) {
      Value *U = BO->getOperand(0);
      auto *C = dyn_cast<ConstantInt>(BO->getOperand(1));
      if (!C && BO->getOpcode() == Instruction::Add) {
        U = BO->getOperand(1);
        C = dyn_cast<ConstantInt>(BO->getOperand(0));
      }
      if (C && !isa<Constant>(U) && isCTAUniformFootprintBase(U, UI)) {
        Base = BO;
        OffMax = 0;
        return true;
      }
    }

    if (!BO || (BO->getOpcode() != Instruction::Add &&
                BO->getOpcode() != Instruction::Or))
      return false;

    Value *LHS = BO->getOperand(0);
    Value *RHS = BO->getOperand(1);
    Value *InnerBase = nullptr;
    uint64_t InnerOff = 0;
    uint64_t SideOff = 0;
    if (Recurse(LHS, InnerBase, InnerOff) &&
        getConvOffsetMaximumBelow(RHS, Limit, SE, LI, SideOff) &&
        InnerOff <= Limit - 1 - SideOff) {
      Base = InnerBase;
      OffMax = InnerOff + SideOff;
      return true;
    }
    Visited.erase(LHS);
    if (Recurse(RHS, InnerBase, InnerOff) &&
        getConvOffsetMaximumBelow(LHS, Limit, SE, LI, SideOff) &&
        InnerOff <= Limit - 1 - SideOff) {
      Base = InnerBase;
      OffMax = InnerOff + SideOff;
      return true;
    }
    return false;
  };

  return Recurse(Index, BaseOut, OffMaxOut);
}

/// Match a CTA-uniform footprint lane guard `Base(+Off) < Bound`.
static bool matchConvFootprintLaneGuard(Value *V, UniformityInfo &UI,
                                        ScalarEvolution &SE, LoopInfo &LI,
                                        Value *&BaseOut, Value *&BoundOut,
                                        uint64_t &OffMaxOut) {
  auto *Cmp = dyn_cast<ICmpInst>(V);
  if (!Cmp)
    return false;
  ICmpInst::Predicate Pred = Cmp->getPredicate();
  Value *Index = Cmp->getOperand(0);
  Value *Bound = Cmp->getOperand(1);
  if (Pred == ICmpInst::ICMP_UGT || Pred == ICmpInst::ICMP_SGT ||
      Pred == ICmpInst::ICMP_UGE || Pred == ICmpInst::ICMP_SGE) {
    Pred = ICmpInst::getSwappedPredicate(Pred);
    std::swap(Index, Bound);
  }
  // ule/sle Index, Bound ≡ ult/slt Index, Bound+1 when Bound is rewritten as
  // W-c by InstCombine; handle the additive form via ExtraOff below.
  bool Inclusive = Pred == ICmpInst::ICMP_ULE || Pred == ICmpInst::ICMP_SLE;
  if (Pred != ICmpInst::ICMP_ULT && Pred != ICmpInst::ICMP_SLT && !Inclusive)
    return false;

  // Peel widening casts and InstCombine's `ult (base+off), (W-c)`.
  uint64_t ExtraOff = 0;
  Value *BoundRaw = Bound;
  if (auto *Cast = dyn_cast<CastInst>(BoundRaw))
    if (Cast->getOpcode() == Instruction::SExt ||
        Cast->getOpcode() == Instruction::ZExt)
      BoundRaw = Cast->getOperand(0);
  if (auto *Sub = dyn_cast<BinaryOperator>(BoundRaw)) {
    if (Sub->getOpcode() == Instruction::Sub) {
      if (auto *C = dyn_cast<ConstantInt>(Sub->getOperand(1))) {
        if (UI.isUniformAtDef(Sub->getOperand(0)) && !C->isNegative()) {
          ExtraOff = C->getZExtValue();
          BoundRaw = Sub->getOperand(0);
        }
      }
    }
  }
  if (!BoundRaw->getType()->isIntegerTy() || !UI.isUniformAtDef(BoundRaw))
    return false;

  constexpr uint64_t FootprintLimit = 256;
  if (!decomposeUniformBaseOffset(Index, UI, SE, LI, FootprintLimit, BaseOut,
                                  OffMaxOut))
    return false;
  BaseOut = peelTrivialBase(BaseOut);
  if (Inclusive)
    ExtraOff += 1;
  if (OffMaxOut > FootprintLimit - 1 - ExtraOff)
    return false;
  OffMaxOut += ExtraOff;
  // Prefer the peeled uniform extent (W) so H/W checks share one Bound SSA
  // after InstCombine rewrites per-lane `+c` into `W-c`.
  BoundOut = BoundRaw;
  return true;
}

/// Dump peel chains for failed conv offsets. errs() always reaches HIP device
/// compile logs when -amdgpu-interior-conv-diag is on (default). Also mirror to
/// dbgs() so -debug-only captures the same blocks.
static void dumpConvOffsetFail(Value *LHS, Value *RHS, Value *PeeledR,
                               Value *PeeledL, bool BaseOK, bool OffOK,
                               uint64_t TmpOff) {
  if (!ConvFootprintDiag)
    return;
  auto Emit = [&](raw_ostream &OS) {
    OS << "  CONV-OFFSET-FAIL\n    lhs: " << *LHS << "\n    rhs: " << *RHS
       << "\n    CONV-DIAG base-lhs=" << BaseOK << " off-rhs=" << OffOK;
    if (OffOK)
      OS << " off-max=" << TmpOff;
    OS << "\n    PEEL-rhs: " << *PeeledR << '\n';
    if (auto *PI = dyn_cast<Instruction>(PeeledR)) {
      OS << "    PEEL-rhs-op: " << PI->getOpcodeName() << '\n';
      if (auto *PN = dyn_cast<PHINode>(PI)) {
        for (unsigned I = 0, E = PN->getNumIncomingValues(); I < E; ++I)
          OS << "      phi-in" << I << ": " << *PN->getIncomingValue(I) << '\n';
      }
      unsigned N = 0;
      for (Value *Op : PI->operands()) {
        OS << "      peel-opnd" << N++ << ": " << *Op << '\n';
        if (auto *OI = dyn_cast<Instruction>(Op)) {
          OS << "        peel-opnd-def: " << OI->getOpcodeName();
          if (auto *OPN = dyn_cast<PHINode>(OI)) {
            for (unsigned I = 0, E = OPN->getNumIncomingValues(); I < E; ++I)
              OS << "\n          phi-in" << I << ": "
                 << *OPN->getIncomingValue(I);
          } else {
            unsigned M = 0;
            for (Value *Op2 : OI->operands()) {
              OS << "\n          op" << M++ << ": " << *Op2;
              if (auto *OI2 = dyn_cast<Instruction>(Op2))
                OS << " [" << OI2->getOpcodeName() << "]";
              if (M >= 4)
                break;
            }
          }
          OS << '\n';
        }
        if (N >= 4)
          break;
      }
    }
    OS << "    PEEL-lhs: " << *PeeledL << '\n';
  };
  Emit(errs());
  LLVM_DEBUG(Emit(dbgs()));
}

/// Print one bounded definition chain for the first failing i16 offset.
/// HIP device cc1 does not reliably inherit dump env vars, and forwarding a
/// custom -mllvm path to ld.lld aborts the link. A single small stderr record
/// is reliable and cannot create the earlier debug flood.
static void dumpFirstI16OffsetFail(Value *Offset) {
  static bool Dumped = false;
  if (Dumped)
    return;
  auto *Outer = dyn_cast<CastInst>(Offset);
  if (!Outer || (Outer->getOpcode() != Instruction::ZExt &&
                 Outer->getOpcode() != Instruction::SExt))
    return;
  // Stop at the immediate narrow source (`%73`), rather than peeling its
  // trunc too and accidentally reaching the original i32 expression.
  Value *V = Outer->getOperand(0);
  auto *Ty = dyn_cast<IntegerType>(V->getType());
  if (!Ty || Ty->getBitWidth() != 16)
    return;
  Dumped = true;

  std::string Record;
  raw_string_ostream OS(Record);
  OS << "ITS-I16-OFFSET-FAIL\n  offset: " << *Offset
     << "\n  peeled: " << *V << '\n';
  SmallVector<Value *, 8> Queue{V};
  SmallPtrSet<Value *, 16> Seen;
  unsigned Lines = 0;
  for (unsigned Depth = 0; Depth < 5 && !Queue.empty() && Lines < 40; ++Depth) {
    SmallVector<Value *, 8> Next;
    for (Value *Cur : Queue) {
      if (!Seen.insert(Cur).second)
        continue;
      OS << "  d" << Depth << ": " << *Cur << '\n';
      ++Lines;
      auto *I = dyn_cast<Instruction>(Cur);
      if (!I)
        continue;
      for (Value *Op : I->operands())
        if (isa<Instruction>(Op) || isa<Argument>(Op))
          Next.push_back(Op);
    }
    Queue.swap(Next);
  }
  OS << "ITS-I16-OFFSET-END\n";
  OS.flush();
  errs() << Record;
  // HIP reliably forwards DEBUG_TYPE output when -debug-only is requested,
  // unlike arbitrary errs() from its device cc1 child.
  LLVM_DEBUG(dbgs() << Record);
}

static void
recordConvFootprintGuard(Value *V, UniformityInfo &UI, ScalarEvolution &SE,
                         LoopInfo &LI,
                         DenseMap<Value *, uint64_t> &ExtentByBase,
                         DenseMap<Value *, Value *> &BoundByBase) {
  // Peel trivial inversions / select-form conjunctions down to icmps.
  SmallVector<Value *, 8> Worklist{V};
  SmallPtrSet<Value *, 16> Seen;
  while (!Worklist.empty()) {
    Value *Cur = Worklist.pop_back_val();
    if (!Seen.insert(Cur).second)
      continue;
    if (auto *Xor = dyn_cast<BinaryOperator>(Cur)) {
      if (Xor->getOpcode() == Instruction::Xor &&
          Xor->getType()->isIntegerTy(1)) {
        if (auto *C = dyn_cast<ConstantInt>(Xor->getOperand(1))) {
          if (C->isOne()) {
            Worklist.push_back(Xor->getOperand(0));
            continue;
          }
        } else if (auto *C = dyn_cast<ConstantInt>(Xor->getOperand(0))) {
          if (C->isOne()) {
            Worklist.push_back(Xor->getOperand(1));
            continue;
          }
        }
      }
    }
    if (auto *Select = dyn_cast<SelectInst>(Cur)) {
      auto *False = dyn_cast<ConstantInt>(Select->getFalseValue());
      if (False && False->isZero()) {
        Worklist.push_back(Select->getCondition());
        Worklist.push_back(Select->getTrueValue());
        continue;
      }
    }
    SmallVector<Value *, 8> Conjuncts;
    collectConjuncts(Cur, Conjuncts);
    if (Conjuncts.size() > 1) {
      for (Value *Conjunct : Conjuncts)
        Worklist.push_back(Conjunct);
      continue;
    }

    Value *Base = nullptr;
    Value *Bound = nullptr;
    uint64_t OffMax = 0;
    if (!matchConvFootprintLaneGuard(Cur, UI, SE, LI, Base, Bound, OffMax)) {
      if (auto *Cmp = dyn_cast<ICmpInst>(Cur)) {
        Value *IdxVal = Cmp->getOperand(0);
        LLVM_DEBUG(dbgs() << "  conv-footprint unmatched: " << *Cmp
                          << "\n    index: " << *IdxVal << '\n');
        if (auto *BO = dyn_cast<BinaryOperator>(IdxVal))
          if (BO->getOpcode() == Instruction::Add ||
              BO->getOpcode() == Instruction::Or) {
            Value *LHS = BO->getOperand(0);
            Value *RHS = BO->getOperand(1);
            // One bounded stderr dump for the first failing i16 offset.
            if (isa<CastInst>(RHS))
              dumpFirstI16OffsetFail(RHS);
            else if (isa<CastInst>(LHS))
              dumpFirstI16OffsetFail(LHS);
            if (ConvFootprintDiag) {
              Value *PeeledR = peelIntegerCastsForOffset(RHS);
              Value *PeeledL = peelIntegerCastsForOffset(LHS);
              uint64_t TmpOff = 0;
              bool OffOK = getConvOffsetMaximumBelow(RHS, 256, SE, LI, TmpOff);
              bool BaseOK = isCTAUniformFootprintBase(LHS, UI);
              if (!OffOK && !BaseOK) {
                uint64_t Tmp2 = 0;
                if (getConvOffsetMaximumBelow(LHS, 256, SE, LI, Tmp2) &&
                    isCTAUniformFootprintBase(RHS, UI)) {
                  OffOK = true;
                  TmpOff = Tmp2;
                  BaseOK = true;
                  std::swap(PeeledL, PeeledR);
                }
              }
              dumpConvOffsetFail(LHS, RHS, PeeledR, PeeledL, BaseOK, OffOK,
                                 TmpOff);
            }
          }
      } else {
        LLVM_DEBUG(dbgs() << "  conv-footprint unmatched: " << *Cur << '\n');
      }
      continue;
    }
    Base = peelTrivialBase(Base);
    uint64_t Extent = OffMax + 1;
    auto &Slot = ExtentByBase[Base];
    if (Extent > Slot) {
      Slot = Extent;
      BoundByBase[Base] = Bound;
    }
    LLVM_DEBUG(dbgs() << "  conv-footprint matched base=" << Base->getName()
                      << " bound=" << Bound->getName() << " extent=" << Extent
                      << " from " << *Cur << '\n');
  }
}

static bool
collectConvFootprintChecks(const SmallPtrSetImpl<BasicBlock *> &Region,
                           UniformityInfo &UI, ScalarEvolution &SE, LoopInfo &LI,
                           SmallVectorImpl<FullTileBoundCheck> &FullChecks) {
  DenseMap<Value *, uint64_t> ExtentByBase;
  DenseMap<Value *, Value *> BoundByBase;
  LLVM_DEBUG(dbgs() << "Conv footprint scanning " << Region.size()
                    << " staging blocks\n");
  for (BasicBlock *BB : Region) {
    // Branch conditions (including nested ANDs / selects / xor-not).
    if (auto *Branch = dyn_cast<BranchInst>(BB->getTerminator()))
      if (Branch->isConditional())
        recordConvFootprintGuard(Branch->getCondition(), UI, SE, LI,
                                 ExtentByBase, BoundByBase);
  }

  FullChecks.clear();
  // Second pass: x0..x3 checks often share an already-matched W base
  // (`icmp ult in_x0, W`) but use `in_x0 + off` that the first decompose
  // missed when Off was not yet tied to that Base.
  for (BasicBlock *BB : Region) {
    auto *Branch = dyn_cast<BranchInst>(BB->getTerminator());
    if (!Branch || !Branch->isConditional())
      continue;
    SmallVector<Value *, 8> Conjuncts;
    collectConjuncts(Branch->getCondition(), Conjuncts);
    for (Value *Conjunct : Conjuncts) {
      auto *Cmp = dyn_cast<ICmpInst>(Conjunct);
      if (!Cmp)
        continue;
      ICmpInst::Predicate Pred = Cmp->getPredicate();
      Value *Index = Cmp->getOperand(0);
      Value *Bound = Cmp->getOperand(1);
      if (Pred == ICmpInst::ICMP_UGT || Pred == ICmpInst::ICMP_SGT ||
          Pred == ICmpInst::ICMP_UGE || Pred == ICmpInst::ICMP_SGE) {
        Pred = ICmpInst::getSwappedPredicate(Pred);
        std::swap(Index, Bound);
      }
      if (Pred != ICmpInst::ICMP_ULT && Pred != ICmpInst::ICMP_SLT &&
          Pred != ICmpInst::ICMP_ULE && Pred != ICmpInst::ICMP_SLE)
        continue;
      uint64_t ExtraOff = Pred == ICmpInst::ICMP_ULE || Pred == ICmpInst::ICMP_SLE
                              ? 1
                              : 0;
      Value *BoundRaw = Bound;
      if (auto *Cast = dyn_cast<CastInst>(BoundRaw))
        if (Cast->getOpcode() == Instruction::SExt ||
            Cast->getOpcode() == Instruction::ZExt)
          BoundRaw = Cast->getOperand(0);
      if (auto *Sub = dyn_cast<BinaryOperator>(BoundRaw))
        if (Sub->getOpcode() == Instruction::Sub)
          if (auto *C = dyn_cast<ConstantInt>(Sub->getOperand(1)))
            if (!C->isNegative() && UI.isUniformAtDef(Sub->getOperand(0))) {
              ExtraOff += C->getZExtValue();
              BoundRaw = Sub->getOperand(0);
            }
      constexpr uint64_t FootprintLimit = 256;
      for (auto &BaseEntry : BoundByBase) {
        if (BaseEntry.second != BoundRaw)
          continue;
        uint64_t OffMax = 0;
        if (!matchOffsetFromKnownBase(Index, BaseEntry.first, FootprintLimit, SE,
                                      LI, OffMax))
          continue;
        if (OffMax > FootprintLimit - 1 - ExtraOff)
          continue;
        uint64_t Extent = OffMax + ExtraOff + 1;
        auto &Slot = ExtentByBase[BaseEntry.first];
        if (Extent > Slot) {
          Slot = Extent;
          LLVM_DEBUG(dbgs() << "  conv-footprint known-base matched extent="
                            << Extent << " from " << *Cmp << '\n');
        }
      }
    }
  }

  for (const auto &Entry : ExtentByBase) {
    if (!isSupportedConvFootprintExtent(Entry.second)) {
      LLVM_DEBUG(dbgs() << "  conv-footprint base " << Entry.first->getName()
                        << " extent " << Entry.second
                        << " outside supported range\n");
      continue;
    }
    Value *Bound = BoundByBase.lookup(Entry.first);
    if (!Bound)
      continue;
    FullChecks.push_back({/*Dimension=*/0, Bound, Entry.first,
                          /*IsSigned=*/true, Entry.second});
  }
  LLVM_DEBUG(dbgs() << "Conv footprint recovered " << FullChecks.size()
                    << " dims from " << ExtentByBase.size() << " bases\n");
  // Need independent H and W (or analogous) footprint proofs.
  if (FullChecks.size() < 2)
    return false;
  SmallPtrSet<Value *, 4> Bases;
  for (const FullTileBoundCheck &Check : FullChecks)
    Bases.insert(Check.Base);
  return Bases.size() >= 2;
}

static unsigned countRemovableFootprintBranches(
    const SmallPtrSetImpl<BasicBlock *> &Region,
    ArrayRef<FullTileBoundCheck> FullChecks, ScalarEvolution &SE) {
  unsigned Count = 0;
  for (BasicBlock *BB : Region) {
    auto *Branch = dyn_cast<BranchInst>(BB->getTerminator());
    if (!Branch || !Branch->isConditional())
      continue;
    GuardOutcome Outcome;
    if (getRemovableSafetyBranchOutcome(Branch, FullChecks, SE, Outcome))
      ++Count;
  }
  return Count;
}

static Value *
synthesizeConvFootprintSelector(BasicBlock *InsertBB,
                                ArrayRef<FullTileBoundCheck> Checks) {
  if (Checks.empty())
    return nullptr;
  IRBuilder<> Builder(InsertBB->getTerminator());
  Value *All = ConstantInt::getTrue(InsertBB->getContext());
  for (const FullTileBoundCheck &C : Checks) {
    if (C.Bound->getType() != C.Base->getType() || !C.TileMN)
      return nullptr;
    Type *Ty = C.Base->getType();
    Value *Extent = ConstantInt::get(Ty, C.TileMN);
    Value *Zero = ConstantInt::get(Ty, 0);
    // Pad makes Base negative on halo CTAs; unsigned lane guards treat that as
    // out-of-bounds, so the interior selector must reject Base < 0.
    Value *NonNeg =
        Builder.CreateICmpSGE(C.Base, Zero, "interior.conv.base.nonneg");
    Value *BoundOk =
        Builder.CreateICmpSGE(C.Bound, Extent, "interior.conv.bound.ok");
    Value *Full = Builder.CreateICmpSLE(
        C.Base, Builder.CreateSub(C.Bound, Extent), "interior.conv.dim.full");
    Value *Dim = Builder.CreateAnd(NonNeg, BoundOk, "interior.conv.dim.prep");
    Dim = Builder.CreateAnd(Dim, Full, "interior.conv.dim");
    All = Builder.CreateAnd(All, Dim, "interior.conv.full");
  }
  return All;
}

/// Barrier exits must leave the cloned staging region (compute / epilogue).
/// A barrier block that returns directly is also fine: there is simply no
/// post-barrier CFG inside the clone.
static bool hasOnlySharedBarrierExitsOutsideRegion(
    const SmallPtrSetImpl<BasicBlock *> &Region, const BasicBlock *Barrier) {
  for (const BasicBlock *Successor : successors(Barrier)) {
    if (Region.contains(Successor))
      return false;
  }
  return true;
}

/// Specialize conv (and similar) cooperative staging: insert a CTA-uniform
/// "whole input footprint in bounds" selector, clone the pre-barrier staging
/// region, and strip proven per-element H/W guards on the interior path.
static bool splitInteriorConvFootprint(Function &F, UniformityInfo &UI,
                                       LoopInfo &LI, DominatorTree &DT,
                                       ScalarEvolution &SE) {
  BasicBlock *StagingEntry = nullptr;
  BasicBlock *StagingPredecessor = nullptr;
  BasicBlock *Barrier = nullptr;
  bool SplitEntryForDispatch = false;
  SmallVector<FullTileBoundCheck, 4> FullChecks;
  SmallPtrSet<BasicBlock *, 8> SelectedRegion;
  unsigned BestStripped = 0;
  unsigned BestScore = 0;

  auto Consider = [&](BasicBlock *BB, BasicBlock *Dispatch, bool AllowExternal,
                      bool SplitAtTerminator) {
    SmallPtrSet<BasicBlock *, 8> Candidate;
    BasicBlock *CandidateBarrier = nullptr;
    bool CandidateHasLoop = false;
    if (!findClosedStagingRegion(BB, Dispatch, Candidate, CandidateBarrier,
                                 /*AllowNestedLoops=*/true, &CandidateHasLoop,
                                 AllowExternal))
      return;
    if (!hasOnlySharedBarrierExitsOutsideRegion(Candidate, CandidateBarrier)) {
      LLVM_DEBUG(dbgs() << "Conv footprint candidate rejected at "
                        << BB->getName()
                        << ": barrier re-enters staging region\n");
      return;
    }
    if (Candidate.size() > MaxStagingCloneBlocks) {
      LLVM_DEBUG(dbgs() << "Conv footprint candidate rejected at "
                        << BB->getName() << ": " << Candidate.size()
                        << " blocks exceeds clone limit\n");
      return;
    }

    SmallVector<FullTileBoundCheck, 4> CandidateChecks;
    if (!collectConvFootprintChecks(Candidate, UI, SE, LI, CandidateChecks)) {
      LLVM_DEBUG(dbgs() << "Conv footprint candidate rejected at "
                        << BB->getName()
                        << ": need >=2 CTA-uniform footprint dims\n");
      return;
    }
    if (!canCloneStagingRegion(BB, Candidate, CandidateBarrier, CandidateChecks,
                               /*DirectGuards=*/{}, /*Alignments=*/{},
                               /*IV=*/nullptr, /*Bound=*/nullptr, SE,
                               /*IgnoreEntryInstructions=*/SplitAtTerminator)) {
      LLVM_DEBUG(dbgs() << "Conv footprint candidate rejected at "
                        << BB->getName() << ": clone preflight failed\n");
      return;
    }

    unsigned Stripped =
        countRemovableFootprintBranches(Candidate, CandidateChecks, SE);
    if (Stripped < MinConvFootprintStrippedBranches) {
      LLVM_DEBUG(dbgs() << "Conv footprint candidate rejected at "
                        << BB->getName() << ": only " << Stripped
                        << " removable branches (min "
                        << MinConvFootprintStrippedBranches << ")\n");
      return;
    }
    const bool IsEntry = BB == &F.getEntryBlock();
    unsigned Score =
        Stripped * 4 + (CandidateHasLoop ? 2 : 0) + (IsEntry ? 0 : 1);
    if (StagingEntry && Score <= BestScore)
      return;

    StagingEntry = BB;
    StagingPredecessor = Dispatch;
    Barrier = CandidateBarrier;
    SplitEntryForDispatch = SplitAtTerminator;
    FullChecks = std::move(CandidateChecks);
    SelectedRegion.clear();
    SelectedRegion.insert_range(Candidate);
    BestStripped = Stripped;
    BestScore = Score;
  };

  for (BasicBlock &BBRef : F) {
    BasicBlock *BB = &BBRef;
    const bool IsEntry = BB == &F.getEntryBlock();
    BasicBlock *Dispatch = IsEntry ? nullptr : BB->getSinglePredecessor();
    if (!IsEntry && !Dispatch)
      continue;
    Consider(BB, Dispatch, /*AllowExternal=*/IsEntry,
             /*SplitAtTerminator=*/IsEntry);
  }

  // Cooperative staging lives in loops whose headers have a latch + preheader.
  // Split the preheader edge (not the header terminator) so the latch can keep
  // jumping to the shared header while the interior path clones the loop.
  for (Loop *L : LI.getLoopsInPreorder()) {
    BasicBlock *Header = L->getHeader();
    BasicBlock *Preheader = L->getLoopPreheader();
    if (!Header || !Preheader)
      continue;
    Consider(Header, Preheader, /*AllowExternal=*/true,
             /*SplitAtTerminator=*/false);
  }

  if (!StagingEntry) {
    LLVM_DEBUG(dbgs() << "Conv footprint preflight rejected " << F.getName()
                      << ": no profitable closed staging footprint region\n");
    return false;
  }

  BasicBlock *Dispatch = nullptr;
  BasicBlock *OriginalEntry = StagingEntry;
  if (SplitEntryForDispatch) {
    Dispatch = StagingEntry;
    StagingEntry = SplitBlock(Dispatch, Dispatch->getTerminator(), &DT, &LI,
                              nullptr, "conv.staging");
    StagingPredecessor = Dispatch;
  } else {
    Dispatch = SplitEdge(StagingPredecessor, StagingEntry, &DT, &LI);
    if (!Dispatch)
      return false;
  }

  auto *DispatchBranch = cast<BranchInst>(Dispatch->getTerminator());
  SmallPtrSet<BasicBlock *, 8> Region;
  Region.insert_range(SelectedRegion);
  if (SplitEntryForDispatch) {
    // SplitBlock leaves setup instructions in the shared dispatch and moves
    // the old terminator to the new staging entry. The preflight region
    // therefore changes only by replacing the original entry with that block.
    Region.erase(OriginalEntry);
    Region.insert(StagingEntry);
  }
  BasicBlock *SharedBarrier = Barrier;
  // The selected region already passed the complete closure/reachability
  // proof. SplitEdge only replaces its external preheader edge; SplitBlock
  // performs the entry substitution above. Re-running graph discovery here
  // incorrectly walks through the new dispatch shape on real HIP loop headers.
  const bool RegionOk = !Region.empty();
  const bool BarrierOk = RegionOk && SharedBarrier == Barrier;
  const bool ExitsOk =
      BarrierOk &&
      hasOnlySharedBarrierExitsOutsideRegion(Region, SharedBarrier);
  const bool CloneOk =
      ExitsOk &&
      canCloneStagingRegion(StagingEntry, Region, SharedBarrier, FullChecks, {},
                            {}, nullptr, nullptr, SE);
  if (!CloneOk) {
    LLVM_DEBUG(dbgs() << "Conv footprint aborted after dispatch split at "
                      << StagingEntry->getName() << " region=" << RegionOk
                      << " barrier=" << BarrierOk << " exits=" << ExitsOk
                      << " clone=" << CloneOk << '\n');
    if (StagingEntry->getSinglePredecessor() == Dispatch &&
        Dispatch->getSingleSuccessor() == StagingEntry)
      MergeBlockIntoPredecessor(StagingEntry, /*DTU=*/nullptr, &LI,
                                /*MSSAU=*/nullptr, /*MemDep=*/nullptr,
                                /*PredecessorWithTwoSuccessors=*/false, &DT);
    return false;
  }

  // Re-collect after the split so Base SSA values still match region guards.
  FullChecks.clear();
  if (!collectConvFootprintChecks(Region, UI, SE, LI, FullChecks) ||
      countRemovableFootprintBranches(Region, FullChecks, SE) <
          MinConvFootprintStrippedBranches) {
    LLVM_DEBUG(dbgs() << "Conv footprint aborted after recollect at "
                      << StagingEntry->getName() << '\n');
    if (StagingEntry->getSinglePredecessor() == Dispatch &&
        Dispatch->getSingleSuccessor() == StagingEntry)
      MergeBlockIntoPredecessor(StagingEntry, /*DTU=*/nullptr, &LI,
                                /*MSSAU=*/nullptr, /*MemDep=*/nullptr,
                                /*PredecessorWithTwoSuccessors=*/false, &DT);
    return false;
  }

  Value *RunFast = synthesizeConvFootprintSelector(Dispatch, FullChecks);
  if (!RunFast) {
    if (StagingEntry->getSinglePredecessor() == Dispatch &&
        Dispatch->getSingleSuccessor() == StagingEntry)
      MergeBlockIntoPredecessor(StagingEntry, /*DTU=*/nullptr, &LI,
                                /*MSSAU=*/nullptr, /*MemDep=*/nullptr,
                                /*PredecessorWithTwoSuccessors=*/false, &DT);
    return false;
  }

  BranchInst::Create(StagingEntry, StagingEntry, RunFast, DispatchBranch);
  DispatchBranch->eraseFromParent();

  bool Changed = cloneStagingRegion(
      F, cast<BranchInst>(Dispatch->getTerminator()), StagingEntry, Region,
      SharedBarrier, FullChecks, {}, {}, nullptr, nullptr, SE);
  if (Changed) {
#ifndef NDEBUG
    if (verifyFunction(F, &dbgs()))
      report_fatal_error(
          "AMDGPUInteriorTileSplit: conv staging clone produced invalid IR");
#endif
    // SplitEdge/SplitBlock updated DT for the dispatch insertion, but
    // cloneStagingRegion then appended blocks and redirected an edge without
    // a DomTreeUpdater. Legacy LTO may reuse the required DT analysis in the
    // following CodeSink pass, so leave it internally consistent.
    DT.recalculate(F);
    LLVM_DEBUG(dbgs() << "Cloned conv footprint interior staging in "
                      << F.getName() << "; footprint dims " << FullChecks.size()
                      << "; preflight stripped-branch estimate " << BestStripped
                      << '\n');
  }
  return Changed;
}

/// Clone the store epilogue after the outer K loop behind \p MNFull and strip
/// proven M/N lane guards on the interior copy.
static bool specializeInteriorEpilogueStores(
    Function &F, Loop *L, Value *MNFull,
    ArrayRef<FullTileBoundCheck> FullChecks, ArrayRef<Value *> DirectGuards,
    DominatorTree &DT, LoopInfo &LI, ScalarEvolution &SE) {
  BasicBlock *Exit = L->getUniqueExitBlock();
  if (!Exit || !MNFull)
    return false;

  // Epilogue starts at the unique successor of the loop exit, when that edge
  // is unambiguous.
  auto *ExitBr = dyn_cast<BranchInst>(Exit->getTerminator());
  if (!ExitBr || ExitBr->isConditional() || ExitBr->getNumSuccessors() != 1)
    return false;
  BasicBlock *EpilogueEntry = ExitBr->getSuccessor(0);
  if (L->contains(EpilogueEntry))
    return false;

  SmallPtrSet<BasicBlock *, 8> Region;
  SmallVector<BasicBlock *, 8> Worklist{EpilogueEntry};
  Region.insert(EpilogueEntry);
  bool HasStore = false;
  bool HasRemovableGuard = false;
  while (!Worklist.empty()) {
    BasicBlock *BB = Worklist.pop_back_val();
    for (Instruction &I : *BB) {
      if (auto *SI = dyn_cast<StoreInst>(&I))
        if (SI->getValueOperand()->getType()->isFloatTy() ||
            SI->getValueOperand()->getType()->isVectorTy())
          HasStore = true;
    }
    auto *Br = dyn_cast<BranchInst>(BB->getTerminator());
    if (Br && Br->isConditional()) {
      GuardOutcome Outcome;
      if (getRemovableSafetyBranchOutcome(Br, FullChecks, SE, Outcome) ||
          getInteriorConjunctionRemainder(Br->getCondition(), DirectGuards))
        HasRemovableGuard = true;
    }
    for (BasicBlock *Succ : successors(BB)) {
      if (L->contains(Succ) || isa<ReturnInst>(Succ->getTerminator()))
        continue;
      // Stay within a simple forward region: no entries from outside.
      bool ExternalPred = false;
      for (BasicBlock *Pred : predecessors(Succ))
        if (Pred != BB && !Region.contains(Pred) && Pred != Exit)
          ExternalPred = true;
      if (ExternalPred)
        continue;
      if (Region.insert(Succ).second)
        Worklist.push_back(Succ);
    }
  }
  if (!HasStore || !HasRemovableGuard || Region.size() > MaxStagingCloneBlocks)
    return false;

  // Reject if any region block has a PHI needing path merges from outside.
  for (BasicBlock *BB : Region)
    for (PHINode &PN : BB->phis())
      for (unsigned I = 0, E = PN.getNumIncomingValues(); I != E; ++I)
        if (!Region.contains(PN.getIncomingBlock(I)) &&
            PN.getIncomingBlock(I) != Exit)
          return false;

  BasicBlock *Dispatch = SplitEdge(Exit, EpilogueEntry, &DT, &LI);
  if (!Dispatch)
    return false;
  auto *DispatchBr = cast<BranchInst>(Dispatch->getTerminator());
  BranchInst::Create(EpilogueEntry, EpilogueEntry, MNFull, DispatchBr);
  DispatchBr->eraseFromParent();
  auto *NewDispatch = cast<BranchInst>(Dispatch->getTerminator());

  ValueToValueMapTy VMap;
  SmallVector<BasicBlock *, 8> RegionBlocks(Region.begin(), Region.end());
  // Stable order: function order.
  RegionBlocks.clear();
  for (BasicBlock &BB : F)
    if (Region.contains(&BB))
      RegionBlocks.push_back(&BB);

  for (BasicBlock *BB : RegionBlocks) {
    BasicBlock *Clone = CloneBasicBlock(BB, VMap, ".interior", &F);
    VMap[BB] = Clone;
  }
  for (BasicBlock *BB : RegionBlocks) {
    BasicBlock *Clone = cast<BasicBlock>(VMap[BB]);
    for (Instruction &I : *Clone)
      RemapInstruction(&I, VMap, RF_IgnoreMissingLocals);
  }

  unsigned Removed = 0;
  for (BasicBlock *BB : RegionBlocks) {
    auto *OriginalBranch = dyn_cast<BranchInst>(BB->getTerminator());
    if (!OriginalBranch || !OriginalBranch->isConditional())
      continue;
    if (Value *Remainder = getInteriorConjunctionRemainder(
            OriginalBranch->getCondition(), DirectGuards)) {
      auto *ClonedBranch =
          cast<BranchInst>(cast<BasicBlock>(VMap[BB])->getTerminator());
      Value *ClonedRemainder = Remainder;
      if (Value *Mapped = VMap.lookup(Remainder))
        ClonedRemainder = Mapped;
      ClonedBranch->setCondition(ClonedRemainder);
      ++Removed;
      continue;
    }
    GuardOutcome Outcome;
    if (!getRemovableSafetyBranchOutcome(OriginalBranch, FullChecks, SE,
                                         Outcome))
      continue;
    auto *ClonedBranch =
        cast<BranchInst>(cast<BasicBlock>(VMap[BB])->getTerminator());
    BranchInst::Create(
        ClonedBranch->getSuccessor(static_cast<unsigned>(Outcome)),
        ClonedBranch);
    ClonedBranch->eraseFromParent();
    ++Removed;
  }
  if (!Removed) {
    // No guards stripped — delete clones and leave the fallthrough dispatch.
    for (BasicBlock *BB : RegionBlocks)
      if (auto *Clone = dyn_cast_or_null<BasicBlock>(VMap[BB]))
        Clone->eraseFromParent();
    if (EpilogueEntry->getSinglePredecessor() == Dispatch &&
        Dispatch->getSingleSuccessor() == EpilogueEntry)
      MergeBlockIntoPredecessor(EpilogueEntry, /*DTU=*/nullptr, &LI,
                                /*MSSAU=*/nullptr, /*MemDep=*/nullptr,
                                /*PredecessorWithTwoSuccessors=*/false, &DT);
    return false;
  }

  NewDispatch->setSuccessor(0, cast<BasicBlock>(VMap[EpilogueEntry]));
  LLVM_DEBUG(dbgs() << "Specialized interior store epilogue in " << F.getName()
                    << "; removed " << Removed
                    << " proven M/N store guard(s)\n");
  return true;
}

static bool splitCanonicalInteriorTile(Function &F, UniformityInfo &UI,
                                       LoopInfo &LI, DominatorTree &DT,
                                       ScalarEvolution &SE) {
  bool HasMClamp = false, HasNClamp = false, HasKClamp = false;
  bool HasCanonicalAlignment = false;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB) {
      HasMClamp |= isClampedTileExtent(&I, 1) ||
                   isTileRelativeDifference(&I, 1);
      HasNClamp |= isClampedTileExtent(&I, 0) ||
                   isTileRelativeDifference(&I, 0);
      HasKClamp |= isClampedKTileExtent(&I);
      HasCanonicalAlignment |= isAlignmentCheck(&I, UI);
    }

  LLVM_DEBUG(dbgs() << "Interior-tile proof for " << F.getName()
                    << ": M-clamp=" << HasMClamp
                    << " N-clamp=" << HasNClamp
                    << " K-clamp=" << HasKClamp
                    << " alignment=" << HasCanonicalAlignment << '\n');

  auto GetFullChecks = [&](Value *Condition,
                           SmallVectorImpl<FullTileBoundCheck> &FullChecks) {
    if (!HasMClamp || !HasNClamp || !HasKClamp || !HasCanonicalAlignment)
      return false;

    SmallVector<Value *, 8> Conjuncts;
    collectConjuncts(Condition, Conjuncts);
    bool HasM = false, HasN = false, HasK = false, HasAlignment = false;
    for (Value *Conjunct : Conjuncts) {
      FullTileBoundCheck MCheck, NCheck;
      if (getFullTileBoundCheck(Conjunct, 1, MCheck)) {
        HasM = true;
        FullChecks.push_back(MCheck);
      }
      if (getFullTileBoundCheck(Conjunct, 0, NCheck)) {
        HasN = true;
        FullChecks.push_back(NCheck);
      }
      HasK |= isKTileCheck(Conjunct);
      HasAlignment |= isAlignmentCheck(Conjunct, UI);
    }
    return HasM && HasN && HasK && HasAlignment;
  };

  // The staging-only POC deliberately does not clone an outer K loop.  It
  // inserts a CTA-uniform dispatch inside that loop and clones only the
  // guarded staging graph (including any nested staging loops) up to its
  // first shared barrier.
  SmallVector<Loop *, 4> Loops = LI.getLoopsInPreorder();
  for (Loop *L : Loops) {
    if (splitOuterKStaging(F, L, UI, LI, DT, SE))
      return true;
  }

  // Conv / halo staging: no outer-K tile setup, but CTA-uniform input
  // footprint bases with per-element H/W guards before the shared barrier.
  if (splitInteriorConvFootprint(F, UI, LI, DT, SE))
    return true;

  for (BasicBlock &BB : F) {
    auto *Branch = dyn_cast<BranchInst>(BB.getTerminator());
    if (!Branch || !Branch->isConditional() || !UI.isUniformTerminator(Branch))
      continue;

    SmallPtrSet<Value *, 16> Visited;
    if (getIDDependencies(Branch->getCondition(), Visited) &
        DependsOnWorkitemID)
      continue;

    SmallVector<FullTileBoundCheck, 2> FullChecks;
    // Require both the canonical clamped extents and a full-tile dispatch.
    // The former proves the staging shape; the latter is the CTA-uniform
    // condition which selects the cloned interior path.
    if (!GetFullChecks(Branch->getCondition(), FullChecks))
      continue;

    for (BasicBlock *Successor : successors(&BB)) {
      SmallPtrSet<BasicBlock *, 8> Region;
      BasicBlock *Barrier = nullptr;
      if (!findClosedStagingRegion(Successor, &BB, Region, Barrier))
        continue;

      if (cloneStagingRegion(F, Branch, Successor, Region, Barrier,
                             FullChecks, {}, {}, nullptr, nullptr, SE))
        return true;
    }
  }
  return false;
}

static bool findInteriorTileCandidates(Function &F, UniformityInfo &UI,
                                       LoopInfo &LI, DominatorTree &DT,
                                       ScalarEvolution &SE) {
  // HIP may sandbox absolute /tmp writes during device compile. Prefer a
  // relative path; fall back to printing a short note (peel dumps go to errs).
  if (const char *DumpPath = std::getenv("AMDGPU_INTERIOR_TILE_SPLIT_DUMP_IR")) {
    if (DumpPath[0] != '\0') {
      errs() << "ITS-DUMP: pass running on " << F.getName()
             << " cc=" << F.getCallingConv() << " path=" << DumpPath << '\n';
      auto TryWrite = [&](StringRef Path) -> bool {
        std::error_code EC;
        raw_fd_ostream OS(Path, EC, sys::fs::OF_TextWithCRLF | sys::fs::OF_Append);
        if (EC) {
          errs() << "ITS-DUMP: open failed path=" << Path
                 << " ec=" << EC.value() << " msg=" << EC.message() << '\n';
          return false;
        }
        OS << "; *** IR Dump Before amdgpu-interior-tile-split on "
           << F.getName() << " ***\n";
        F.print(OS);
        OS << '\n';
        OS.flush();
        errs() << "ITS-DUMP: wrote " << Path << '\n';
        return true;
      };
      if (!TryWrite(DumpPath)) {
        // Sandbox often allows CWD.
        TryWrite("amdgpu-its-before.ll");
      }
    }
  }

  if (!AMDGPU::isEntryFunctionCC(F.getCallingConv()))
    return false;

  bool Changed = splitCanonicalInteriorTile(F, UI, LI, DT, SE);

  // Already-interior kernels (e.g. interior_split_stress_interior) have no
  // M/N lane guards to strip, so the clone never runs. Still try float4 widen
  // on the whole function when vectorize is enabled; each candidate loop has to
  // prove for itself that its staging access is unpredicated.
  if (EnableInteriorTileVectorize) {
    SmallVector<BasicBlock *, 16> Blocks;
    for (BasicBlock &BB : F)
      Blocks.push_back(&BB);
    unsigned Widened = 0;
    if (optimizeInteriorMemoryPaths(Blocks, /*RequireUnguarded=*/true,
                                    &Widened)) {
      Changed = true;
      if (Widened)
        LLVM_DEBUG(dbgs() << "Widened already-interior staging in "
                          << F.getName() << " without a prior clone\n");
#ifndef NDEBUG
      if (verifyFunction(F, &dbgs()))
        report_fatal_error(
            "AMDGPUInteriorTileSplit: no-clone widen produced invalid IR");
#endif
    }
  }

  for (BasicBlock &BB : F) {
    auto *Branch = dyn_cast<BranchInst>(BB.getTerminator());
    if (!Branch || !Branch->isConditional() || UI.isUniformTerminator(Branch))
      continue;

    SmallPtrSet<Value *, 16> Visited;
    const unsigned Dependencies =
        getIDDependencies(Branch->getCondition(), Visited);
    if ((Dependencies & (DependsOnWorkgroupID | DependsOnWorkitemID)) !=
        (DependsOnWorkgroupID | DependsOnWorkitemID))
      continue;

    ++NumInteriorTileCandidates;
    LLVM_DEBUG(dbgs() << "Potential interior-tile boundary check in "
                      << F.getName() << ":\n"
                      << *Branch << '\n');
  }

  // The existing broad candidate diagnostic remains useful for kernels that
  // are close to, but do not yet meet, the canonical shape above.
  return Changed;
}

static bool isBMMDeviceName(StringRef Name) {
  // Accept both the demangled POC name and the Itanium HIP mangling
  // `_Z10bmm_device...` that real hipcc device IR uses.
  return Name == "bmm_device" || Name.starts_with("_Z10bmm_device");
}

static bool isBMMInteriorABI(const Function &F) {
  if (!isBMMDeviceName(F.getName()) ||
      !AMDGPU::isEntryFunctionCC(F.getCallingConv()) || F.arg_size() != 10)
    return false;
  auto Arg = F.arg_begin();
  for (unsigned I = 0; I != 3; ++I, ++Arg)
    if (!Arg->getType()->isPointerTy())
      return false;
  for (unsigned I = 0; I != 4; ++I, ++Arg)
    if (!Arg->getType()->isIntegerTy(32))
      return false;
  for (unsigned I = 0; I != 3; ++I, ++Arg)
    if (!Arg->getType()->isIntegerTy(64))
      return false;
  return true;
}

static Function *findBMMInteriorCandidate(Module &M) {
  for (Function &F : M)
    if (!F.isDeclaration() && isBMMInteriorABI(F))
      return &F;
  return nullptr;
}

static bool dependsOnBMMExtent(Value *V, ArrayRef<Argument *> Extents,
                               SmallPtrSetImpl<Value *> &Visited) {
  if (!Visited.insert(V).second)
    return false;
  for (Argument *Extent : Extents)
    if (Extent == V)
      return true;
  if (auto *I = dyn_cast<Instruction>(V))
    for (Value *Operand : I->operands())
      if (dependsOnBMMExtent(Operand, Extents, Visited))
        return true;
  return false;
}

/// Remove a comparison only where a positive full-tile extent proves which
/// edge of the conditional is safe.  Require a workgroup/workitem dependent
/// operand: this excludes scalar K-loop termination tests, whose direction
/// cannot be replaced without changing trip counts.
static bool removeFullTileChecks(Function &F) {
  SmallVector<Argument *, 3> Extents;
  auto Arg = F.arg_begin();
  std::advance(Arg, 4); // B, M, N, K are arguments 3..6.
  Extents.push_back(&*Arg++); // M
  Extents.push_back(&*Arg++); // N
  Extents.push_back(&*Arg);   // K

  SmallVector<BranchInst *, 16> Checks;
  for (BasicBlock &BB : F) {
    auto *BI = dyn_cast<BranchInst>(BB.getTerminator());
    auto *Cmp = BI && BI->isConditional()
                    ? dyn_cast<ICmpInst>(BI->getCondition())
                    : nullptr;
    if (!Cmp)
      continue;
    SmallPtrSet<Value *, 32> Visited;
    SmallPtrSet<Value *, 32> LeftIDs;
    SmallPtrSet<Value *, 32> RightIDs;
    unsigned IDs = getIDDependencies(Cmp->getOperand(0), LeftIDs) |
                   getIDDependencies(Cmp->getOperand(1), RightIDs);
    if (IDs != DependsOnNone && dependsOnBMMExtent(Cmp, Extents, Visited))
      Checks.push_back(BI);
  }

  bool Changed = false;
  for (BranchInst *BI : Checks) {
    auto *Cmp = cast<ICmpInst>(BI->getCondition());
    unsigned SafeSuccessor;
    switch (Cmp->getPredicate()) {
    case CmpInst::ICMP_SLT:
    case CmpInst::ICMP_ULT:
    case CmpInst::ICMP_SLE:
    case CmpInst::ICMP_ULE:
      SafeSuccessor = 0;
      break;
    case CmpInst::ICMP_SGE:
    case CmpInst::ICMP_UGE:
    case CmpInst::ICMP_SGT:
    case CmpInst::ICMP_UGT:
      SafeSuccessor = 1;
      break;
    default:
      continue;
    }
    BranchInst::Create(BI->getSuccessor(SafeSuccessor), BI);
    BI->eraseFromParent();
    Changed = true;
  }

  if (Changed) {
    SmallVector<BasicBlock *, 16> Blocks;
    for (BasicBlock &BB : F)
      Blocks.push_back(&BB);
    optimizeInteriorMemoryPaths(Blocks, /*RequireUnguarded=*/true);
#ifndef NDEBUG
    if (verifyFunction(F, &dbgs()))
      report_fatal_error(
          "AMDGPUInteriorTileSplit: full-tile check removal produced invalid IR");
#endif
  }
  return Changed;
}

class AMDGPUInteriorTileSplitLegacy : public FunctionPass {
public:
  static char ID;

  AMDGPUInteriorTileSplitLegacy() : FunctionPass(ID) {}

  StringRef getPassName() const override { return "AMDGPU Interior Tile Split"; }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    UniformityInfo &UI =
        getAnalysis<UniformityInfoWrapperPass>().getUniformityInfo();
    LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    ScalarEvolution &SE =
        getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    return findInteriorTileCandidates(F, UI, LI, DT, SE);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<UniformityInfoWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
  }
};

} // end anonymous namespace

PreservedAnalyses AMDGPUBMMInteriorSpecializationPass::run(
    Module &M, ModuleAnalysisManager &) {
  Function *Original = findBMMInteriorCandidate(M);
  if (!Original)
    return PreservedAnalyses::all();

  // Host registration looks up `<device-side-name>.interior`, which for HIP is
  // the mangled kernel name plus the suffix.
  std::string InteriorName = (Original->getName() + ".interior").str();
  if (M.getFunction(InteriorName))
    return PreservedAnalyses::all();

  ValueToValueMapTy VMap;
  Function *Interior = CloneFunction(Original, VMap);
  Interior->setName(InteriorName);
  // Host registration references this name from another compilation unit, so
  // retain it even if no direct device-side call names the clone.
  appendToCompilerUsed(M, {Interior});
  removeFullTileChecks(*Interior);
  return PreservedAnalyses::none();
}

PreservedAnalyses
AMDGPUInteriorTileSplitPass::run(Function &F, FunctionAnalysisManager &FAM) {
  UniformityInfo &UI = FAM.getResult<UniformityInfoAnalysis>(F);
  LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);
  DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  ScalarEvolution &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
  if (!findInteriorTileCandidates(F, UI, LI, DT, SE))
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}

INITIALIZE_PASS_BEGIN(AMDGPUInteriorTileSplitLegacy, DEBUG_TYPE,
                      "Find AMDGPU interior tile split candidates", false,
                      true)
INITIALIZE_PASS_DEPENDENCY(UniformityInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_END(AMDGPUInteriorTileSplitLegacy, DEBUG_TYPE,
                    "Find AMDGPU interior tile split candidates", false, true)

char AMDGPUInteriorTileSplitLegacy::ID = 0;

FunctionPass *llvm::createAMDGPUInteriorTileSplitLegacy() {
  return new AMDGPUInteriorTileSplitLegacy();
}
