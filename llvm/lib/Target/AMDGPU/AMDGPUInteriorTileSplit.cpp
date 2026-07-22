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
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

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

static bool isWorkitemID(Value *V) {
  auto *II = dyn_cast<IntrinsicInst>(V);
  return II &&
         II->getCalledFunction()->getName().starts_with(
             "llvm.amdgcn.workitem.id.");
}

static bool isWorkgroupShiftBy7(Value *V, unsigned Dimension) {
  auto *Shift = dyn_cast<BinaryOperator>(V);
  if (!Shift || Shift->getOpcode() != Instruction::Shl ||
      !isWorkgroupID(Shift->getOperand(0), Dimension))
    return false;

  auto *Amount = dyn_cast<ConstantInt>(Shift->getOperand(1));
  return Amount && Amount->equalsInt(7);
}

static bool isShiftPlusLastLane(Value *V, unsigned Dimension) {
  auto *Add = dyn_cast<BinaryOperator>(V);
  if (!Add || Add->getOpcode() != Instruction::Add)
    return false;

  Value *LHS = Add->getOperand(0);
  Value *RHS = Add->getOperand(1);
  if (auto *C = dyn_cast<ConstantInt>(LHS)) {
    if (!C->equalsInt(127))
      return false;
    LHS = RHS;
  } else {
    auto *LastLane = dyn_cast<ConstantInt>(RHS);
    if (!LastLane || !LastLane->equalsInt(127))
      return false;
  }
  return isWorkgroupShiftBy7(LHS, Dimension);
}

static bool isWorkgroupShiftBy7OrExtend(Value *V, unsigned Dimension);

struct FullTileBoundCheck {
  unsigned Dimension;
  Value *Bound;
};

static bool getFullTileBoundCheck(Value *V, unsigned Dimension,
                                  FullTileBoundCheck &Check) {
  auto *Cmp = dyn_cast<ICmpInst>(V);
  if (!Cmp || Cmp->getPredicate() != ICmpInst::ICMP_ULT ||
      !isShiftPlusLastLane(Cmp->getOperand(0), Dimension))
    return false;

  SmallPtrSet<Value *, 16> Visited;
  if (getIDDependencies(Cmp->getOperand(1), Visited) != DependsOnNone)
    return false;
  Check = {Dimension, Cmp->getOperand(1)};
  return true;
}

/// Match the lane-varying form `workgroup.id * 128 + workitem.id < Bound`.
/// A matching full-tile check proves this condition true for every lane.
static bool isPerLaneTileBoundCheck(Value *V, const FullTileBoundCheck &Full) {
  auto *Cmp = dyn_cast<ICmpInst>(V);
  if (!Cmp || Cmp->getPredicate() != ICmpInst::ICMP_ULT ||
      Cmp->getOperand(1) != Full.Bound)
    return false;

  auto *Add = dyn_cast<BinaryOperator>(Cmp->getOperand(0));
  if (!Add || Add->getOpcode() != Instruction::Add)
    return false;

  Value *LHS = Add->getOperand(0);
  Value *RHS = Add->getOperand(1);
  return (isWorkgroupShiftBy7OrExtend(LHS, Full.Dimension) &&
          isWorkitemID(RHS)) ||
         (isWorkitemID(LHS) &&
          isWorkgroupShiftBy7OrExtend(RHS, Full.Dimension));
}

static bool isRemovableSafetyBranch(
    const BranchInst *Branch, ArrayRef<FullTileBoundCheck> FullChecks) {
  if (!Branch->isConditional())
    return false;
  for (const FullTileBoundCheck &Full : FullChecks)
    if (isPerLaneTileBoundCheck(Branch->getCondition(), Full))
      return true;
  return false;
}

static bool isWorkgroupShiftBy7OrExtend(Value *V, unsigned Dimension) {
  if (isWorkgroupShiftBy7(V, Dimension))
    return true;
  if (auto *Cast = dyn_cast<CastInst>(V))
    return isWorkgroupShiftBy7(Cast->getOperand(0), Dimension);
  return false;
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
static bool isClampedTileExtent(Value *V, unsigned Dimension) {
  if (!isNamedCall(V, "llvm.smin."))
    return false;

  auto *Min = cast<CallBase>(V);
  Value *Extent = nullptr;
  for (Value *Operand : Min->args()) {
    if (auto *C = dyn_cast<ConstantInt>(Operand)) {
      if (!C->equalsInt(128))
        return false;
    } else {
      Extent = Operand;
    }
  }
  if (!Extent || !isNamedCall(Extent, "llvm.smax."))
    return false;

  auto *Max = cast<CallBase>(Extent);
  Value *Difference = nullptr;
  for (Value *Operand : Max->args()) {
    if (auto *C = dyn_cast<ConstantInt>(Operand)) {
      if (!C->isZero())
        return false;
    } else {
      Difference = Operand;
    }
  }
  return isTileRelativeDifference(Difference, Dimension);
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

/// Clone an innermost K loop into a full-tile and an edge version.  This is
/// deliberately stricter than the acyclic staging-region transform: the loop
/// must already be in the forms required by cloneLoopWithPreheader, and its
/// sole dedicated exit must contain the LCSSA PHIs which merge the live-outs.
static bool versionInteriorKLoop(Loop *L, BasicBlock *Dispatch,
                                 Value *DispatchCondition, LoopInfo &LI,
                                 DominatorTree &DT) {
  BasicBlock *Preheader = L->getLoopPreheader();
  BasicBlock *Exit = L->getUniqueExitBlock();
  BasicBlock *Exiting = L->getExitingBlock();
  auto *DispatchBranch = dyn_cast<BranchInst>(Dispatch->getTerminator());
  if (!L->isInnermost() || !L->isLoopSimplifyForm() || !L->isLCSSAForm(DT) ||
      !Preheader || !Exit || !Exiting || !containsBarrier(L) ||
      Preheader->getSinglePredecessor() != Dispatch ||
      Exit->getSinglePredecessor() != Exiting || !DispatchBranch ||
      !DispatchBranch->isUnconditional() ||
      DispatchBranch->getSuccessor(0) != Preheader ||
      DispatchCondition->getType() !=
          Type::getInt1Ty(L->getHeader()->getContext()))
    return false;

  // The dedicated exit is intentionally required to have only the LCSSA
  // incoming edge.  This lets the existing PHIs merge the cloned live-outs
  // without trying to reconstruct arbitrary exit CFG.
  SmallVector<Instruction *, 8> DefsUsedOutside =
      findDefsUsedOutsideOfLoop(L);
  for (Instruction *Def : DefsUsedOutside)
    for (User *U : Def->users()) {
      auto *Use = dyn_cast<Instruction>(U);
      if (!Use || L->contains(Use->getParent()))
        continue;
      auto *PN = dyn_cast<PHINode>(Use);
      if (!PN || PN->getParent() != Exit ||
          PN->getIncomingValueForBlock(Exiting) != Def)
        return false;
    }

  ValueToValueMapTy VMap;
  SmallVector<BasicBlock *, 8> ClonedBlocks;
  Loop *InteriorLoop =
      cloneLoopWithPreheader(Exit, Dispatch, L, VMap, ".interior", &LI, &DT,
                             ClonedBlocks);
  remapInstructionsInBlocks(ClonedBlocks, VMap);

  BasicBlock *InteriorPreheader = InteriorLoop->getLoopPreheader();
  BasicBlock *InteriorExiting = cast<BasicBlock>(VMap[Exiting]);
  BranchInst::Create(InteriorPreheader, Preheader, DispatchCondition,
                     DispatchBranch);
  DispatchBranch->eraseFromParent();

  for (PHINode &PN : Exit->phis()) {
    Value *Incoming = PN.getIncomingValueForBlock(Exiting);
    if (auto It = VMap.find(Incoming); It != VMap.end())
      Incoming = It->second;
    PN.addIncoming(Incoming, InteriorExiting);
  }

  // Both loop exits are now reached from the dispatch block.  This matches the
  // LoopVersioning update and keeps the cloned LoopInfo/DT state consistent.
  DT.changeImmediateDominator(Exit, Dispatch);
  ++NumPreparedInteriorKLoopSplits;
  LLVM_DEBUG(dbgs() << "Versioned canonical interior K loop in "
                    << L->getHeader()->getParent()->getName() << " at "
                    << Dispatch->getName()
                    << "; interior loop " << InteriorLoop->getHeader()->getName()
                    << ", edge loop " << L->getHeader()->getName()
                    << ", shared live-out exit " << Exit->getName() << '\n');
  return true;
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
                               ArrayRef<FullTileBoundCheck> FullChecks) {
  // Cloning incoming values into a PHI in the shared barrier would require
  // proving the corresponding values are valid on both paths.  Reject that
  // shape rather than manufacture a potentially invalid incoming value.
  if (isa<PHINode>(&Barrier->front()))
    return false;

  bool HasRemovableSafetyBranch = false;
  for (BasicBlock *BB : Region)
    HasRemovableSafetyBranch |= isRemovableSafetyBranch(
        cast<BranchInst>(BB->getTerminator()), FullChecks);
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
    if (!isRemovableSafetyBranch(OriginalBranch, FullChecks))
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
                                       LoopInfo &LI, DominatorTree &DT) {
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

  // A loop version is only introduced when a canonical CTA-uniform selector
  // has already been materialized in its preheader's dispatch block.  The
  // selector need not yet be the terminator: versioning makes it the choice
  // between the cloned interior loop and the original edge loop.
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
          versionInteriorKLoop(L, Dispatch, &I, LI, DT))
        return true;
    }
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
                             FullChecks))
        return true;
    }
  }
  return false;
}

static bool findInteriorTileCandidates(Function &F, UniformityInfo &UI,
                                       LoopInfo &LI, DominatorTree &DT) {
  if (!AMDGPU::isEntryFunctionCC(F.getCallingConv()))
    return false;

  const bool Changed = splitCanonicalInteriorTile(F, UI, LI, DT);

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
    return findInteriorTileCandidates(F, UI, LI, DT);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<UniformityInfoWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
  }
};

} // end anonymous namespace

PreservedAnalyses
AMDGPUInteriorTileSplitPass::run(Function &F, FunctionAnalysisManager &FAM) {
  UniformityInfo &UI = FAM.getResult<UniformityInfoAnalysis>(F);
  LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);
  DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  if (!findInteriorTileCandidates(F, UI, LI, DT))
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}

INITIALIZE_PASS_BEGIN(AMDGPUInteriorTileSplitLegacy, DEBUG_TYPE,
                      "Find AMDGPU interior tile split candidates", false,
                      true)
INITIALIZE_PASS_DEPENDENCY(UniformityInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(AMDGPUInteriorTileSplitLegacy, DEBUG_TYPE,
                    "Find AMDGPU interior tile split candidates", false, true)

char AMDGPUInteriorTileSplitLegacy::ID = 0;

FunctionPass *llvm::createAMDGPUInteriorTileSplitLegacy() {
  return new AMDGPUInteriorTileSplitLegacy();
}
