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
/// interior dispatch.  The proof rejects shapes which could accidentally clone
/// an outer loop or a barrier.
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <cstdint>

#define DEBUG_TYPE "amdgpu-interior-tile-split"

using namespace llvm;

STATISTIC(NumInteriorTileCandidates,
          "Number of divergent tile-boundary checks found");
STATISTIC(NumPreparedInteriorTileSplits,
          "Number of canonical interior-tile splits cloned");
STATISTIC(NumPreparedInteriorKLoopSplits,
          "Number of canonical interior K loops versioned");

namespace {

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

static bool isWorkgroupID(Value *V, unsigned Dimension) {
  auto *II = dyn_cast<IntrinsicInst>(V);
  if (!II)
    return false;

  StringRef Name = II->getCalledFunction()->getName();
  return (Dimension == 0 && Name == "llvm.amdgcn.workgroup.id.x") ||
         (Dimension == 1 && Name == "llvm.amdgcn.workgroup.id.y");
}

static bool isWorkgroupShiftBy7(Value *V, unsigned Dimension) {
  auto *Shift = dyn_cast<BinaryOperator>(V);
  if (!Shift || Shift->getOpcode() != Instruction::Shl ||
      !isWorkgroupID(Shift->getOperand(0), Dimension))
    return false;

  auto *Amount = dyn_cast<ConstantInt>(Shift->getOperand(1));
  return Amount && Amount->equalsInt(7);
}

static Value *getWorkgroupShiftBy7OrExtend(Value *V, unsigned Dimension) {
  if (isWorkgroupShiftBy7(V, Dimension))
    return V;
  if (auto *Cast = dyn_cast<CastInst>(V))
    if (isWorkgroupShiftBy7(Cast->getOperand(0), Dimension))
      return V;
  return nullptr;
}

static Value *getShiftPlusLastLaneBase(Value *V, unsigned Dimension) {
  auto *Add = dyn_cast<BinaryOperator>(V);
  if (!Add || Add->getOpcode() != Instruction::Add)
    return nullptr;

  Value *LHS = Add->getOperand(0);
  Value *RHS = Add->getOperand(1);
  if (auto *C = dyn_cast<ConstantInt>(LHS)) {
    if (!C->equalsInt(127))
      return nullptr;
    LHS = RHS;
  } else {
    auto *LastLane = dyn_cast<ConstantInt>(RHS);
    if (!LastLane || !LastLane->equalsInt(127))
      return nullptr;
  }
  return getWorkgroupShiftBy7OrExtend(LHS, Dimension);
}

struct FullTileBoundCheck {
  unsigned Dimension;
  Value *Bound;
  Value *Base;
  bool IsSigned;
};

struct CanonicalTileRemainder {
  unsigned Dimension;
  Value *Bound;
  Value *Base;
  Value *Difference;
};

static bool getFullTileBoundCheck(Value *V, unsigned Dimension,
                                  FullTileBoundCheck &Check) {
  auto *Cmp = dyn_cast<ICmpInst>(V);
  if (!Cmp || Cmp->getPredicate() != ICmpInst::ICMP_ULT)
    return false;
  Value *Base = getShiftPlusLastLaneBase(Cmp->getOperand(0), Dimension);
  if (!Base)
    return false;

  SmallPtrSet<Value *, 16> Visited;
  if (getIDDependencies(Cmp->getOperand(1), Visited) != DependsOnNone)
    return false;
  Check = {Dimension, Cmp->getOperand(1), Base, false};
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

/// Match a signed or unsigned `Base + LaneOffset < Bound`.  The static M/N
/// selector is built from the same Base/Bound pair and proves a 128-wide tile.
/// Prove that Offset is nonnegative and strictly less than Limit.  This uses
/// SCEV rather than a syntactic loop match so the proof survives the casts and
/// affine adds that Clang emits for the nested staging loops.  A full range is
/// deliberately not accepted.
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
  if (Pred == ICmpInst::ICMP_UGT || Pred == ICmpInst::ICMP_SGT) {
    Pred = ICmpInst::getSwappedPredicate(Pred);
    std::swap(Index, Bound);
  }
  if (Bound != Full.Bound ||
      (Full.IsSigned ? Pred != ICmpInst::ICMP_SLT
                     : Pred != ICmpInst::ICMP_ULT))
    return false;

  auto *Add = dyn_cast<BinaryOperator>(Index);
  if (!Add || Add->getOpcode() != Instruction::Add)
    return false;

  Value *LHS = Add->getOperand(0);
  Value *RHS = Add->getOperand(1);
  auto IsBoundedOffset = [&](Value *Offset) {
    uint64_t Maximum;
    return (getAffineLaneMaximum(Offset, Maximum) && Maximum < 128) ||
           isUnsignedOffsetBelow(Offset, 128, SE);
  };
  return (LHS == Full.Base && IsBoundedOffset(RHS)) ||
         (RHS == Full.Base && IsBoundedOffset(LHS));
}

static bool isRemovableSafetyBranch(
    const BranchInst *Branch, ArrayRef<FullTileBoundCheck> FullChecks,
    ScalarEvolution &SE) {
  if (!Branch->isConditional())
    return false;
  for (const FullTileBoundCheck &Full : FullChecks)
    if (isPerLaneTileBoundCheck(Branch->getCondition(), Full, SE))
      return true;
  return false;
}

static bool isWorkgroupShiftBy7OrExtend(Value *V, unsigned Dimension) {
  return getWorkgroupShiftBy7OrExtend(V, Dimension);
}

/// Generic scalar optimization can lower min/max to selects, but the
/// tile-relative subtraction itself remains available.
static bool isTileRelativeDifference(Value *V, unsigned Dimension) {
  auto *Sub = dyn_cast<BinaryOperator>(V);
  return Sub && Sub->getOpcode() == Instruction::Sub &&
         isWorkgroupShiftBy7OrExtend(Sub->getOperand(1), Dimension);
}

static bool isNamedCall(Value *V, StringRef Name) {
  auto *Call = dyn_cast<CallBase>(V);
  return Call && Call->getCalledFunction() &&
         Call->getCalledFunction()->getName().starts_with(Name);
}

/// Match min(max(Bound - (workgroup.id << 7), 0), 128), which is the
/// clamp form emitted by Clang for a 128-row or 128-column tile extent.
static bool getClampedTileExtent(Value *V, unsigned Dimension,
                                 CanonicalTileRemainder &Remainder) {
  if (!isNamedCall(V, "llvm.smin.") && !isNamedCall(V, "llvm.umin."))
    return false;

  auto *Min = cast<CallBase>(V);
  Value *Extent = nullptr;
  bool HasTileSize = false;
  for (Value *Operand : Min->args()) {
    if (auto *C = dyn_cast<ConstantInt>(Operand)) {
      if (!C->equalsInt(128) || HasTileSize)
        return false;
      HasTileSize = true;
    } else {
      if (Extent)
        return false;
      Extent = Operand;
    }
  }
  if (!HasTileSize || !Extent || !isNamedCall(Extent, "llvm.smax."))
    return false;

  auto *Max = cast<CallBase>(Extent);
  Value *Difference = nullptr;
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
  auto *Sub = dyn_cast_or_null<BinaryOperator>(Difference);
  if (!HasZero || !Sub || Sub->getOpcode() != Instruction::Sub ||
      !isWorkgroupShiftBy7OrExtend(Sub->getOperand(1), Dimension))
    return false;

  Remainder = {Dimension, Sub->getOperand(0), Sub->getOperand(1), Sub};
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

static bool isRemovablePrefixGuard(
    const BranchInst *Branch, Value *IV, Value *Bound,
    ArrayRef<FullTileBoundCheck> FullChecks, ScalarEvolution &SE) {
  return Branch->isConditional() &&
         (isFullKTileGuard(Branch->getCondition(), IV, Bound) ||
          isNestedKBoundCheck(Branch->getCondition(), IV, Bound, SE) ||
          isRemovableSafetyBranch(Branch, FullChecks, SE));
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
static bool splitInteriorKLoop(Loop *L, BasicBlock *Dispatch,
                               Value *StaticFullTileCondition,
                               ArrayRef<FullTileBoundCheck> FullChecks,
                               LoopInfo &LI, DominatorTree &DT,
                               ScalarEvolution &SE) {
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
  for (BasicBlock *BB : L->blocks())
    HasRemovableGuard |= isRemovablePrefixGuard(
        cast<BranchInst>(BB->getTerminator()), IV, Bound, FullChecks, SE);
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
    if (!isRemovablePrefixGuard(OriginalBranch, IV, Bound, FullChecks, SE))
      continue;

    auto *PrefixBranch = cast<BranchInst>(PrefixBB->getTerminator());
    if (!isRemovablePrefixGuard(PrefixBranch, PrefixIV, Bound,
                                FullChecks, SE))
      llvm_unreachable("cloned prefix guard must retain canonical shape");
    BranchInst::Create(PrefixBranch->getSuccessor(0), PrefixBranch);
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

/// Build the CTA-uniform M/N/alignment selector from the canonical setup
/// values for the K loop.
static Value *synthesizeInteriorTileSelector(BasicBlock *Preheader,
                                              UniformityInfo &UI) {
  CanonicalTileRemainder M, N;
  SmallVector<Value *, 4> Alignments;
  bool HasM = false, HasN = false;
  if (!collectInteriorTileSetup(Preheader, UI, M, N, Alignments, HasM, HasN) ||
      !HasM || !HasN || Alignments.empty() ||
      M.Bound->getType() != N.Bound->getType())
    return nullptr;

  IRBuilder<> Builder(Preheader->getTerminator());
  // The source extents use signed min/max.  Do not infer signed arithmetic
  // facts from their wrapping subtraction: require a nonnegative bound and
  // prove Base <= Bound - 128 directly.  This makes Base + every matched
  // nonnegative lane offset below 128 a defined signed in-bounds index.
  Value *MPositive = Builder.CreateICmpSGE(
      M.Bound, ConstantInt::get(M.Bound->getType(), 128),
      "interior.m.nonnegative");
  Value *MFull = Builder.CreateICmpSLE(
      M.Base, Builder.CreateSub(M.Bound,
                                ConstantInt::get(M.Bound->getType(), 128)),
      "interior.m.full");
  MFull = Builder.CreateAnd(MPositive, MFull, "interior.m.tile");
  Value *NPositive = Builder.CreateICmpSGE(
      N.Bound, ConstantInt::get(N.Bound->getType(), 128),
      "interior.n.nonnegative");
  Value *NFull = Builder.CreateICmpSLE(
      N.Base, Builder.CreateSub(N.Bound,
                                ConstantInt::get(N.Bound->getType(), 128)),
      "interior.n.full");
  NFull = Builder.CreateAnd(NPositive, NFull, "interior.n.tile");
  Value *MNFull = Builder.CreateAnd(MFull, NFull, "interior.mn.full");
  Value *Alignment = Alignments.front();
  for (unsigned I = 1; I != Alignments.size(); ++I)
    Alignment = Builder.CreateAnd(Alignment, Alignments[I],
                                  "interior.aligned");
  return Builder.CreateAnd(MNFull, Alignment, "interior.full");
}

/// Check the pieces that must hold before splitting a preheader.  The
/// subsequent split only changes its predecessor; the remaining checks in
/// splitInteriorKLoop are consequently guaranteed by this preflight.
static bool canSplitInteriorKLoop(Loop *L,
                                  ArrayRef<FullTileBoundCheck> FullChecks,
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
  }
  auto *Step = AR ? dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE))
                  : nullptr;
  if (!Step || Step->getAPInt().getZExtValue() != 32 ||
      !L->isLoopInvariant(Bound) || AR->getLoop() != L ||
      !SE.isAvailableAtLoopEntry(SE.getSCEV(Bound), L) ||
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

  SmallPtrSet<Value *, 8> HeaderBackedgeValues;
  for (PHINode &PN : L->getHeader()->phis())
    HeaderBackedgeValues.insert(PN.getIncomingValueForBlock(Exiting));
  for (PHINode &PN : Exit->phis())
    if (!HeaderBackedgeValues.contains(PN.getIncomingValueForBlock(Exiting)))
      return false;

  bool HasRemovableGuard = false;
  for (BasicBlock *BB : L->blocks())
    HasRemovableGuard |= isRemovablePrefixGuard(
        cast<BranchInst>(BB->getTerminator()), IV, Bound, FullChecks, SE);
  return HasRemovableGuard;
}

/// Returns a closed, acyclic, single-entry/single-exit staging region.  The
/// barrier is intentionally outside the region: it must remain shared by the
/// fast and edge paths.
static bool findClosedStagingRegion(BasicBlock *Entry, BasicBlock *Dispatch,
                                    SmallPtrSetImpl<BasicBlock *> &Region,
                                    BasicBlock *&Barrier) {
  if (isBarrierBlock(Entry) || Entry->getSinglePredecessor() != Dispatch)
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
      if (Predecessor != Dispatch && !Region.contains(Predecessor))
        return false;

    if (isa<CondBrInst>(BB->getTerminator()))
      HasSafetyBranch = true;

    for (BasicBlock *Successor : successors(BB)) {
      if (Successor == Barrier)
        continue;
      if (!Region.contains(Successor))
        return false;
    }
  }

  // A cycle would be part of the outer K loop (or another loop) and must not
  // be cloned by this pass.
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
  return HasSafetyBranch && !HasCycle(HasCycle, Entry);
}

/// Clone a staging region behind a CTA-uniform dispatch.  The original entry
/// remains reachable through the other dispatch edge and is therefore the
/// fallback path.  The shared barrier is not cloned.
static bool cloneStagingRegion(Function &F, BranchInst *Dispatch,
                               BasicBlock *Entry,
                               const SmallPtrSetImpl<BasicBlock *> &Region,
                               BasicBlock *Barrier,
                               ArrayRef<FullTileBoundCheck> FullChecks,
                               ScalarEvolution &SE) {
  // Cloning incoming values into a PHI in the shared barrier would require
  // proving the corresponding values are valid on both paths.  Reject that
  // shape rather than manufacture a potentially invalid incoming value.
  if (isa<PHINode>(&Barrier->front()))
    return false;

  bool HasRemovableSafetyBranch = false;
  for (BasicBlock *BB : Region)
    HasRemovableSafetyBranch |= isRemovableSafetyBranch(
        cast<BranchInst>(BB->getTerminator()), FullChecks, SE);
  if (!HasRemovableSafetyBranch)
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
    auto *OriginalBranch = cast<BranchInst>(BB->getTerminator());
    if (!isRemovableSafetyBranch(OriginalBranch, FullChecks, SE))
      continue;

    auto *ClonedBranch =
        cast<BranchInst>(cast<BasicBlock>(VMap[BB])->getTerminator());
    BranchInst::Create(ClonedBranch->getSuccessor(0), ClonedBranch);
    ClonedBranch->eraseFromParent();
    ++RemovedSafetyBranches;
  }

  Dispatch->setSuccessor(EntrySuccessor, cast<BasicBlock>(VMap[Entry]));
  ++NumPreparedInteriorTileSplits;
  LLVM_DEBUG(dbgs() << "Cloned canonical interior-tile staging region in "
                    << F.getName() << " at "
                    << Dispatch->getParent()->getName()
                    << "; removed " << RemovedSafetyBranches
                    << " proven lane bounds branch(es)"
                    << "; fast staging entry "
                    << cast<BasicBlock>(VMap[Entry])->getName()
                    << ", fallback " << Entry->getName() << ", shared barrier "
                    << Barrier->getName() << '\n');
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

  // A loop version is introduced for an existing canonical CTA-uniform
  // selector.  The selector need not yet be the terminator: versioning makes
  // it the choice between the cloned interior loop and the original edge loop.
  SmallVector<Loop *, 4> Loops = LI.getLoopsInPreorder();
  for (Loop *L : Loops) {
    BasicBlock *Preheader = L->getLoopPreheader();
    BasicBlock *Dispatch =
        Preheader ? Preheader->getSinglePredecessor() : nullptr;
    if (!Dispatch)
      continue;
    for (Instruction &I : *Dispatch) {
      if (!I.getType()->isIntegerTy(1) || !UI.isUniformAtDef(&I))
        continue;
      SmallVector<FullTileBoundCheck, 2> FullChecks;
      if (GetFullChecks(&I, FullChecks) &&
          splitInteriorKLoop(L, Dispatch, &I, FullChecks, LI, DT, SE))
        return true;
    }
  }

  // Real GEMM IR commonly reaches the K loop through an unconditional
  // preheader.  Materialize the M/N/alignment dispatch immediately there,
  // then split the preheader so the original block becomes the dispatch.
  for (Loop *L : Loops) {
    BasicBlock *Preheader = L->getLoopPreheader();
    SmallVector<FullTileBoundCheck, 2> FullChecks;
    bool HasM = false, HasN = false;
    CanonicalTileRemainder M, N;
    SmallVector<Value *, 4> Alignments;
    const bool HasCanonicalSetup =
        collectInteriorTileSetup(Preheader, UI, M, N, Alignments, HasM, HasN);
    if (HasCanonicalSetup && HasM && HasN) {
      FullChecks.push_back({1, M.Bound, M.Base, /*IsSigned=*/true});
      FullChecks.push_back({0, N.Bound, N.Base, /*IsSigned=*/true});
    }
    const bool CanSplit = Preheader && Preheader->getSinglePredecessor() &&
                          HasCanonicalSetup && HasM && HasN &&
                          canSplitInteriorKLoop(L, FullChecks, DT, SE);
    LLVM_DEBUG(dbgs() << "Interior K-loop candidate "
                      << (L->getHeader() ? L->getHeader()->getName()
                                         : "<none>")
                      << ": preheader="
                      << (Preheader ? Preheader->getName() : "<none>")
                      << " HasM=" << HasM << " HasN=" << HasN
                      << " canSplit=" << CanSplit << '\n');
    if (!CanSplit)
      continue;

    Value *Selector = synthesizeInteriorTileSelector(Preheader, UI);
    if (!Selector)
      continue;
    SplitBlock(Preheader, Preheader->getTerminator(), &DT, &LI);
    if (splitInteriorKLoop(L, Preheader, Selector, FullChecks, LI, DT, SE))
      return true;
    llvm_unreachable("preflighted interior K loop must split");
  }

  for (BasicBlock &BB : F) {
    auto *Branch = dyn_cast<CondBrInst>(BB.getTerminator());
    if (!Branch || !UI.isUniformTerminator(Branch))
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
                             FullChecks, SE))
        return true;
    }
  }
  return false;
}

static bool findInteriorTileCandidates(Function &F, UniformityInfo &UI,
                                       LoopInfo &LI, DominatorTree &DT,
                                       ScalarEvolution &SE) {
  if (!AMDGPU::isEntryFunctionCC(F.getCallingConv()))
    return false;

  const bool Changed = splitCanonicalInteriorTile(F, UI, LI, DT, SE);

  for (BasicBlock &BB : F) {
    auto *Branch = dyn_cast<CondBrInst>(BB.getTerminator());
    if (!Branch || UI.isUniformTerminator(Branch))
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
